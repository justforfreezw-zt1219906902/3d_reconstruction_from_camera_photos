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
    from .mesh_io import load_initial_mesh, save_transform
    from .renderer import MPSRendererUnsupported, ReconstructionRenderer
    from .optimizer import train_geometry
    import torch

    ensure_pytorch3d()
    device = select_device(cfg.device)
    from .rgba_dataset import RGBADataset

    dataset = RGBADataset(contract, cfg.max_image_dimension)
    base_mesh, transform = load_initial_mesh(
        cfg.initial_mesh_path,
        device,
        center_normalize=cfg.center_normalize,
        scale_normalize=cfg.scale_normalize,
    )
    save_transform(transform, output_dir / "reconstruction" / "mesh_transform.json")
    camera_model = _build_geometry_model(contract, cfg, device, camera_result)
    renderer = ReconstructionRenderer(dataset.canvas_size, device, cfg.silhouette_faces_per_pixel)
    try:
        renderer.probe(base_mesh, camera_model.cameras([0]))
    except MPSRendererUnsupported:
        if device.type != "mps":
            raise
        print("Warning: PyTorch3D rasterization is unavailable on MPS; using CPU for geometry reconstruction.")
        device = torch.device("cpu")
        base_mesh, transform = load_initial_mesh(
            cfg.initial_mesh_path,
            device,
            center_normalize=cfg.center_normalize,
            scale_normalize=cfg.scale_normalize,
        )
        camera_model = _build_geometry_model(contract, cfg, device, camera_result)
        renderer = ReconstructionRenderer(dataset.canvas_size, device, cfg.silhouette_faces_per_pixel)
    geometry_result = train_geometry(
        base_mesh,
        transform,
        dataset,
        camera_model,
        renderer,
        cfg,
        device,
        output_dir,
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
