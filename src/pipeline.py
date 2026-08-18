from __future__ import annotations

import json
from pathlib import Path

from .config import Config, save_config_snapshot, select_device, set_seed
from .pose_conventions import DEFAULT_CONVENTION
from .validation import DatasetContract, save_validation, validate_inputs


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

    camera_result = None if skip_camera_fit else fit_camera(
        cfg.initial_mesh_path,
        contract,
        cfg,
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
    from .mesh_io import load_initial_mesh, mesh_from_arrays, normalize_vertices, restore_vertices, save_transform
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

    # Geometry is optimized on a topology-aware proxy; the loaded reference is
    # retained only as the authoritative source for dimensions and transforms.
    from .fast_silhouette import FastMesh, create_camera_proxy, save_fast_mesh_stl

    raw_vertices = restore_vertices(base_mesh.verts_packed(), transform).detach().cpu().numpy()
    raw_faces = base_mesh.faces_packed().detach().cpu().numpy().astype(np.int32)
    reference_mesh = FastMesh(raw_vertices, raw_faces)
    geometry_proxy, proxy_method = create_camera_proxy(reference_mesh, cfg.geometry_max_faces)
    save_fast_mesh_stl(geometry_proxy, reconstruction_dir / "geometry_proxy.stl")
    proxy_stats = {
        "original_vertices": int(len(reference_mesh.vertices)),
        "original_faces": int(len(reference_mesh.faces)),
        "geometry_vertices": int(len(geometry_proxy.vertices)),
        "geometry_faces": int(len(geometry_proxy.faces)),
        "reduction_ratio": float(len(geometry_proxy.faces) / max(len(reference_mesh.faces), 1)),
        "method": proxy_method,
        "bbox_min": geometry_proxy.vertices.min(axis=0).tolist(),
        "bbox_max": geometry_proxy.vertices.max(axis=0).tolist(),
    }
    (reconstruction_dir / "geometry_proxy_stats.json").write_text(json.dumps(proxy_stats, indent=2))
    normalized_proxy_vertices = normalize_vertices(
        torch.from_numpy(geometry_proxy.vertices.astype(np.float32)), transform
    )
    geometry_mesh = mesh_from_arrays(
        normalized_proxy_vertices,
        torch.from_numpy(geometry_proxy.faces),
        device,
    )
    del base_mesh
    camera_model = _build_geometry_model(contract, cfg, device, camera_result)
    renderer = ReconstructionRenderer(dataset.canvas_size, device, cfg.silhouette_faces_per_pixel)
    try:
        renderer.probe(geometry_mesh, camera_model.cameras([0]))
    except MPSRendererUnsupported:
        if device.type != "mps":
            raise
        print("Warning: PyTorch3D rasterization is unavailable on MPS; using CPU for geometry reconstruction.")
        device = torch.device("cpu")
        camera_model = _build_geometry_model(contract, cfg, device, camera_result)
        renderer = ReconstructionRenderer(dataset.canvas_size, device, cfg.silhouette_faces_per_pixel)
        geometry_mesh = mesh_from_arrays(
            normalized_proxy_vertices,
            torch.from_numpy(geometry_proxy.faces),
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
                normalized_proxy_vertices,
                torch.from_numpy(geometry_proxy.faces),
                device,
            )
            camera_model = _build_geometry_model(contract, cfg, device, camera_result)
            renderer = ReconstructionRenderer(dataset.canvas_size, device, cfg.silhouette_faces_per_pixel)
            renderer.probe(geometry_mesh, camera_model.cameras([0]))
        else:
            raise
    geometry_result = train_geometry(
        geometry_mesh,
        transform,
        dataset,
        camera_model,
        renderer,
        cfg,
        device,
        output_dir,
        mesh_stats=proxy_stats,
    )
    summary = {
        "status": "completed",
        "device": str(device),
        "frames": len(dataset),
        "canvas_size": dataset.canvas_size,
        "camera_fit": camera_result.metrics if camera_result else {"enabled": False, "skipped": True},
        "final_obj": str(geometry_result.final_obj),
        "final_stl": str(geometry_result.final_stl),
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
