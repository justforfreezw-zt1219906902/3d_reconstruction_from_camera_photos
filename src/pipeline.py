from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import Config, save_config_snapshot, select_device, set_seed
from .validation import DatasetContract, save_validation, validate_inputs


def _build_camera_model(contract: DatasetContract, cfg: Config, device: torch.device) -> OpenScanCameraModel:
    from .openscan_pose import OpenScanCameraModel

    return OpenScanCameraModel(
        theta_deg=[frame.theta_deg for frame in contract.frames],
        phi_deg=[frame.phi_deg for frame in contract.frames],
        distance_initial=cfg.camera_distance_initial,
        fov_initial=cfg.camera_fov_initial,
        max_theta_delta_deg=cfg.max_theta_delta_deg,
        max_phi_delta_deg=cfg.max_phi_delta_deg,
        pose_refine=cfg.pose_refine,
        device=device,
    )


def run_demo(
    cfg: Config,
    contract: DatasetContract,
    camera_fit_only: bool = False,
    skip_camera_fit: bool = False,
    forced_device: torch.device | None = None,
) -> dict:
    from .camera_fitting import CameraFitGateError, fit_camera
    from .config import ensure_pytorch3d
    from .mesh_io import load_initial_mesh, save_transform
    from .openscan_pose import OpenScanCameraModel
    from .optimizer import ReconstructionGateError, train_geometry
    from .rgba_dataset import RGBADataset
    from .renderer import MPSRendererUnsupported, ReconstructionRenderer

    ensure_pytorch3d()
    set_seed(cfg.seed)
    device = forced_device or select_device(cfg.device)
    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_validation(contract, output_dir / "validation.json")
    save_config_snapshot(cfg, output_dir / "run_config.json")

    dataset = RGBADataset(contract, cfg.max_image_dimension)
    base_mesh, transform = load_initial_mesh(
        cfg.initial_mesh_path,
        device,
        center_normalize=cfg.center_normalize,
        scale_normalize=cfg.scale_normalize,
    )
    save_transform(transform, output_dir / "reconstruction" / "mesh_transform.json")
    camera_model = _build_camera_model(contract, cfg, device)
    renderer = ReconstructionRenderer(dataset.canvas_size, device, cfg.silhouette_faces_per_pixel)
    if skip_camera_fit:
        camera_model.freeze()
        camera_result = {"enabled": False, "skipped": True}
    else:
        camera_result = fit_camera(
            base_mesh,
            dataset,
            camera_model,
            renderer,
            cfg,
            device,
            output_dir / "camera_fit",
        ).metrics
    if camera_fit_only:
        summary = {"status": "camera_fit_only", "camera_fit": camera_result}
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary
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
        "camera_fit": camera_result,
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
    try:
        return run_demo(cfg, contract, camera_fit_only, skip_camera_fit)
    except RuntimeError as exc:
        from .renderer import MPSRendererUnsupported

        if not isinstance(exc, MPSRendererUnsupported):
            raise
        print(f"Warning: {exc} Retrying the complete demo on CPU.")
        return run_demo(cfg, contract, camera_fit_only, skip_camera_fit, torch.device("cpu"))
