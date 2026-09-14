from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .config import Config, save_config_snapshot, select_device, set_seed
from .pose_conventions import DEFAULT_CONVENTION
from .validation import DatasetContract, save_validation, validate_inputs


def _reconstruction_config(cfg: Config) -> Config:
    """Return the configuration visible to reconstruction stages.

    Ground truth is deliberately removed at this boundary.  Camera fitting,
    visual-hull generation, and optimization receive this copy, so a future
    implementation cannot accidentally use the evaluation path as a shape
    input.  The original configuration remains available only to the final,
    post-export evaluation block in :func:`run_demo`.
    """
    return replace(cfg, ground_truth_mesh_path=None)


def _selected_best_state(reconstruction_dir: Path) -> dict:
    """Read the optimizer-selected held-out state without exposing GT data."""
    import csv

    losses_path = reconstruction_dir / "losses.csv"
    if not losses_path.exists():
        return {}
    with losses_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = next((row for row in reversed(rows) if row.get("is_best") == "1"), None)
    if selected is None:
        return {}
    # Preserve the useful quality/validity fields while keeping metadata small.
    keys = (
        "epoch", "best_epoch", "validation_silhouette", "validation_iou",
        "validity_is_valid", "validity_degenerate_face_count",
        "validity_extreme_edge_count", "validity_spike_vertex_count",
        "validity_disconnected_fragment_count",
    )
    return {key: selected[key] for key in keys if key in selected}


def _write_reconstruction_metrics(
    path: Path, cfg: Config, camera_result, reconstruction_dir: Path, *,
    coarse_stats: dict | None = None,
) -> dict:
    """Write run metadata that is safe for selection and diagnostics only."""
    split_path = reconstruction_dir / "view_split.json"
    view_split = json.loads(split_path.read_text()) if split_path.exists() else {}
    camera_metrics = camera_result.metrics if camera_result else {"enabled": False, "skipped": True}
    metrics = {
        "reconstruction_mode": cfg.reconstruction_mode,
        "camera_source": camera_metrics.get("camera_source", "cad_silhouette_fit"),
        "camera_calibration": camera_metrics,
        "coarse_geometry": {
            "volume_center": cfg.coarse_volume_center,
            "volume_scale": cfg.coarse_volume_scale,
            "volume_padding_ratio": cfg.coarse_volume_padding_ratio,
            "voxel_resolution": cfg.coarse_voxel_resolution,
            "min_silhouette_support": cfg.coarse_min_silhouette_support,
            "occupancy_threshold": cfg.coarse_occupancy_threshold,
            "min_component_voxels": cfg.coarse_min_component_voxels,
            "marching_cubes_step": cfg.coarse_marching_cubes_step,
            "result": coarse_stats,
        },
        "refinement": {
            "enabled": cfg.image_only_refinement_enabled if cfg.reconstruction_mode == "image_only" else True,
            "epochs": cfg.num_epochs,
            "vertex_learning_rate": cfg.opt_lr_verts,
            "silhouette_weight": cfg.loss_silhouette_weight,
            "rgb_weight": cfg.loss_rgb_weight,
            "normal_weight": cfg.loss_normal_weight,
            "laplacian_weight": cfg.loss_laplacian_weight,
            "edge_weight": cfg.loss_edge_weight,
            "image_only_coarse_anchor_weight": cfg.image_only_coarse_anchor_weight,
        },
        "train_validation_split": view_split,
        "selected_best_state": _selected_best_state(reconstruction_dir),
    }
    path.write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def _build_geometry_model(contract: DatasetContract, cfg: Config, device, camera_result=None):
    from .openscan_pose import OpenScanCameraModel

    if camera_result is None:
        camera = {
            "distance": cfg.camera_distance_initial,
            "fov_deg": cfg.camera_fov_initial,
            "principal_point_x_ndc": 0.0,
            "principal_point_y_ndc": 0.0,
        }
        convention = None
        theta_deltas = [0.0] * len(contract.frames)
        phi_deltas = [0.0] * len(contract.frames)
    else:
        camera = camera_result.camera.as_dict()
        convention = camera_result.convention
        theta_deltas = camera_result.theta_deltas
        phi_deltas = camera_result.phi_deltas
    model = OpenScanCameraModel(
        theta_deg=[frame.theta_deg for frame in contract.frames],
        phi_deg=[frame.phi_deg for frame in contract.frames],
        distance_initial=float(camera["distance"]),
        fov_initial=float(camera["fov_deg"]),
        max_theta_delta_deg=cfg.max_theta_delta_deg,
        max_phi_delta_deg=cfg.max_phi_delta_deg,
        pose_refine=False,
        device=device,
        convention=convention or DEFAULT_CONVENTION,
    )
    model.set_fitted_parameters(camera, theta_deltas, phi_deltas)
    model.freeze()
    return model


def _fit_global_alignment(mesh, dataset, camera_model, renderer, cfg, device, output_dir):
    """Fit only global R/T/S in normalized coordinates with frozen cameras."""
    import torch
    from .alignment import GlobalSimilarityAlignment, SimilarityTransform
    from .losses import alpha_silhouette_loss
    from .optimizer import select_geometry_view_indices, _batch_indices

    camera_model.freeze()
    vertices = mesh.verts_padded().detach()
    alignment = GlobalSimilarityAlignment(
        SimilarityTransform.identity(dtype=vertices.dtype, device=device)
    )
    optimizer = torch.optim.Adam(alignment.parameters(), lr=cfg.alignment_lr)
    for epoch in range(1, cfg.alignment_epochs + 1) if cfg.alignment_enabled else ():
        indices = select_geometry_view_indices(
            dataset.frames, cfg.geometry_views_per_epoch, epoch, cfg.seed
        )
        # Accumulate all selected views before a global update, keeping memory
        # bounded by the existing render batch size.
        optimizer.zero_grad(set_to_none=True)
        for batch in _batch_indices(indices, cfg.geometry_view_batch_size):
            target = torch.stack([dataset[i].alpha for i in batch]).to(device)
            aligned = mesh.update_padded(alignment(vertices))
            rendered = renderer.render_mask(aligned, camera_model.cameras(batch))
            loss = alpha_silhouette_loss(rendered, target)
            if not torch.isfinite(loss) or rendered.detach().sum() <= 0:
                raise RuntimeError("Global alignment failed: non-finite loss or empty silhouette.")
            (loss * len(batch) / len(indices)).backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in alignment.parameters()):
            raise RuntimeError("Global alignment failed: missing or non-finite gradient.")
        optimizer.step()
    alignment.requires_grad_(False)
    fitted = alignment.as_transform()
    metadata = {
        "enabled": cfg.alignment_enabled,
        "coordinate_frame": "normalized_object",
        "rotation": fitted.rotation.cpu().tolist(),
        "translation": fitted.translation.cpu().tolist(),
        "scale": float(fitted.scale.cpu()),
    }
    (output_dir / "global_alignment.json").write_text(json.dumps(metadata, indent=2))
    return mesh.update_padded(fitted.apply(vertices).detach())


def run_demo(
    cfg: Config,
    contract: DatasetContract,
    camera_fit_only: bool = False,
    skip_camera_fit: bool = False,
) -> dict:
    from .camera_fitting import fit_camera

    set_seed(cfg.seed)
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_validation(contract, output_dir / "validation.json")
    save_config_snapshot(cfg, output_dir / "run_config.json")
    geometry_device = None if camera_fit_only else select_device(cfg.device)
    if geometry_device is not None:
        run_config = cfg.as_dict()
        run_config["actual_device"] = str(geometry_device)
        (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    # This is the strict ground-truth isolation boundary for every
    # reconstruction-stage call below.  Do not replace this with ``cfg``.
    reconstruction_cfg = _reconstruction_config(cfg)
    camera_result = None if skip_camera_fit else fit_camera(
        cfg.initial_mesh_path,
        contract,
        reconstruction_cfg,
        output_dir / "camera_fit",
    )
    if camera_fit_only:
        summary = {
            "status": "camera_fit_only",
            "camera_fit": camera_result.metrics if camera_result else {"enabled": False, "skipped": True},
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    # PyTorch3D is intentionally imported only after the fast CPU Camera Fit Gate.
    from .config import ensure_pytorch3d
    from .mesh_io import (
        export_mesh_stl, load_initial_mesh, mesh_from_arrays, normalize_vertices,
        restore_vertices, save_transform,
    )
    from .renderer import MPSRendererUnsupported, ReconstructionRenderer
    from .optimizer import train_geometry
    import torch
    import numpy as np

    ensure_pytorch3d()
    device = geometry_device or select_device(cfg.device)
    from .rgba_dataset import RGBADataset

    dataset = RGBADataset(contract, cfg.max_image_dimension)
    base_mesh, transform = load_initial_mesh(
        cfg.initial_mesh_path,
        device,
        center_normalize=cfg.center_normalize,
        scale_normalize=cfg.scale_normalize,
    )
    reconstruction_dir = output_dir / "reconstruction"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    save_transform(transform, reconstruction_dir / "mesh_transform.json")
    # This artifact always records the supplied initialization, even when the
    # following image-only stages replace its topology entirely.
    if cfg.reconstruction_mode == "image_only":
        export_mesh_stl(base_mesh, reconstruction_dir / "initial_sphere.stl", transform)

    # Geometry is optimized on a topology-aware proxy in CAD mode. Image-only
    # mode instead creates a fresh visual-hull mesh below.
    from .fast_silhouette import FastMesh, create_camera_proxy, save_fast_mesh_stl

    camera_model = _build_geometry_model(contract, reconstruction_cfg, device, camera_result)
    renderer = ReconstructionRenderer(dataset.canvas_size, device, reconstruction_cfg.silhouette_faces_per_pixel)
    initial_mesh = base_mesh
    coarse_mesh = None
    coarse_stats = None
    if cfg.reconstruction_mode == "image_only":
        from .coarse_geometry import generate_coarse_geometry
        from .fast_silhouette import CameraParameters

        calibration = camera_result.camera if camera_result else CameraParameters(
            cfg.camera_distance_initial, cfg.camera_fov_initial,
        )
        coarse_result = generate_coarse_geometry(
            contract, calibration, camera_result.convention if camera_result else DEFAULT_CONVENTION,
            reconstruction_cfg, device, initial_mesh_hint=base_mesh,
            theta_deltas=camera_result.theta_deltas if camera_result else None,
            phi_deltas=camera_result.phi_deltas if camera_result else None,
        )
        geometry_mesh = coarse_result.mesh
        coarse_mesh = geometry_mesh.detach()
        coarse_stats = coarse_result.stats
        export_mesh_stl(coarse_mesh, reconstruction_dir / "image_coarse_mesh.stl", transform)
        proxy_stats = {"geometry_vertices": coarse_stats["vertex_count"], "geometry_faces": coarse_stats["face_count"]}
    else:
        raw_vertices = restore_vertices(base_mesh.verts_packed(), transform).detach().cpu().numpy()
        raw_faces = base_mesh.faces_packed().detach().cpu().numpy().astype(np.int32)
        reference_mesh = FastMesh(raw_vertices, raw_faces)
        geometry_proxy, proxy_method = create_camera_proxy(reference_mesh, cfg.geometry_max_faces)
        save_fast_mesh_stl(geometry_proxy, reconstruction_dir / "geometry_proxy.stl")
        proxy_stats = {
            "original_vertices": int(len(reference_mesh.vertices)), "original_faces": int(len(reference_mesh.faces)),
            "geometry_vertices": int(len(geometry_proxy.vertices)), "geometry_faces": int(len(geometry_proxy.faces)),
            "reduction_ratio": float(len(geometry_proxy.faces) / max(len(reference_mesh.faces), 1)),
            "method": proxy_method, "bbox_min": geometry_proxy.vertices.min(axis=0).tolist(),
            "bbox_max": geometry_proxy.vertices.max(axis=0).tolist(),
        }
        (reconstruction_dir / "geometry_proxy_stats.json").write_text(json.dumps(proxy_stats, indent=2))
        normalized_proxy_vertices = normalize_vertices(torch.from_numpy(geometry_proxy.vertices.astype(np.float32)), transform)
        geometry_mesh = mesh_from_arrays(normalized_proxy_vertices, torch.from_numpy(geometry_proxy.faces), device)
    try:
        renderer.probe(geometry_mesh, camera_model.cameras([0]))
    except MPSRendererUnsupported:
        if device.type != "mps":
            raise
        print("Warning: PyTorch3D rasterization is unavailable on MPS; using CPU for geometry reconstruction.")
        device = torch.device("cpu")
        camera_model = _build_geometry_model(contract, reconstruction_cfg, device, camera_result)
        renderer = ReconstructionRenderer(dataset.canvas_size, device, reconstruction_cfg.silhouette_faces_per_pixel)
        geometry_mesh = mesh_from_arrays(
            geometry_mesh.verts_packed().detach(), geometry_mesh.faces_packed().detach(),
            device,
        )
    except Exception as exc:
        if device.type == "cuda" and cfg.device.lower() == "cuda":
            raise RuntimeError(
                "CUDA geometry rasterization probe failed after DEVICE=cuda was explicitly requested."
            ) from exc
        if device.type == "cuda" and cfg.device.lower() == "auto":
            print(f"Warning: CUDA geometry probe failed ({exc}); using CPU for geometry reconstruction.")
            device = torch.device("cpu")
            geometry_mesh = mesh_from_arrays(
                geometry_mesh.verts_packed().detach(), geometry_mesh.faces_packed().detach(),
                device,
            )
            camera_model = _build_geometry_model(contract, reconstruction_cfg, device, camera_result)
            renderer = ReconstructionRenderer(dataset.canvas_size, device, reconstruction_cfg.silhouette_faces_per_pixel)
            renderer.probe(geometry_mesh, camera_model.cameras([0]))
        else:
            raise
    if cfg.reconstruction_mode != "image_only":
        # Camera -> Global Alignment -> Local Deformation for the CAD-prior path.
        geometry_mesh = _fit_global_alignment(
            geometry_mesh, dataset, camera_model, renderer, reconstruction_cfg, device, reconstruction_dir
        )
    from .visualization import save_mesh_stage_previews
    preview_indices = dataset.representative_indices()
    if cfg.reconstruction_mode == "image_only":
        save_mesh_stage_previews(initial_mesh, dataset, camera_model, renderer, preview_indices, reconstruction_dir / "previews", "sphere")
        save_mesh_stage_previews(coarse_mesh, dataset, camera_model, renderer, preview_indices, reconstruction_dir / "previews", "coarse")
    geometry_result = train_geometry(
        geometry_mesh,
        transform,
        dataset,
        camera_model,
        renderer,
        reconstruction_cfg,
        device,
        output_dir,
        mesh_stats=proxy_stats,
    )
    if cfg.reconstruction_mode == "image_only":
        save_mesh_stage_previews(geometry_result.mesh, dataset, camera_model, renderer, preview_indices, reconstruction_dir / "previews", "final")
    _write_reconstruction_metrics(
        reconstruction_dir / "reconstruction_metrics.json", cfg, camera_result, reconstruction_dir,
        coarse_stats=coarse_stats,
    )
    # Ground truth is evaluation-only and is intentionally imported/read only
    # after final mesh export and metadata selection are complete.
    if cfg.ground_truth_mesh_path is not None:
        from .ground_truth_eval import evaluate_reconstruction_against_ground_truth
        evaluation = evaluate_reconstruction_against_ground_truth(
            cfg.ground_truth_mesh_path, initial_mesh,
            coarse_mesh if coarse_mesh is not None else geometry_mesh,
            geometry_result.mesh, transforms={"sphere": transform, "coarse": transform, "final": transform}, seed=cfg.seed,
        )
        (reconstruction_dir / "ground_truth_evaluation.json").write_text(json.dumps(evaluation, indent=2))
    summary = {
        "status": "completed",
        "device": str(device),
        "frames": len(dataset),
        "canvas_size": dataset.canvas_size,
        "camera_fit": camera_result.metrics if camera_result else {"enabled": False, "skipped": True},
        "final_obj": str(geometry_result.final_obj),
        "final_stl": str(geometry_result.final_stl),
        "reconstruction_metrics": str(reconstruction_dir / "reconstruction_metrics.json"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def run_from_config(
    cfg: Config,
    validate_only: bool = False,
    camera_fit_only: bool = False,
    skip_camera_fit: bool = False,
) -> dict:
    contract = validate_inputs(cfg)
    save_validation(contract, cfg.output_dir / "validation.json")
    if contract.validation.get("coverage_warning"):
        print(f"Warning: {contract.validation['coverage_warning']}")
    aliases = contract.validation.get("csv_extension_aliases", 0)
    if aliases:
        print(f"Info: matched {aliases} CSV image names to processed PNG files by filename stem.")
    if validate_only:
        return {"status": "validated", **contract.validation}
    return run_demo(cfg, contract, camera_fit_only, skip_camera_fit)
