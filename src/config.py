from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default)).expanduser()


@dataclass(frozen=True)
class Config:
    initial_mesh_path: Path
    rgba_dir: Path
    positions_csv: Path
    output_dir: Path
    device: str
    max_image_dimension: int
    camera_distance_initial: float
    camera_fov_initial: float
    camera_fit_enabled: bool
    camera_fit_max_dimension: int
    camera_fit_max_faces: int
    camera_fit_max_frames: int
    camera_fit_max_evaluations: int
    camera_convention_min_median_iou: float
    camera_gate_min_median_iou: float
    pose_refine: bool
    pose_refine_epochs: int
    max_theta_delta_deg: float
    max_phi_delta_deg: float
    num_epochs: int
    batch_size: int
    opt_lr_verts: float
    optimize_texture: bool
    loss_silhouette_weight: float
    loss_rgb_weight: float
    loss_laplacian_weight: float
    loss_edge_weight: float
    loss_normal_weight: float
    max_vertex_displacement_ratio: float
    min_usable_frames: int
    export_every_epochs: int
    save_preview_every_epochs: int
    early_stopping_patience: int
    seed: int
    joint_fine_tune: bool
    center_normalize: bool
    scale_normalize: bool
    silhouette_faces_per_pixel: int

    def as_dict(self) -> dict:
        data = asdict(self)
        for key in ("initial_mesh_path", "rgba_dir", "positions_csv", "output_dir"):
            data[key] = str(data[key])
        return data


def load_config(env_file: Optional[str | Path] = None) -> Config:
    load_dotenv(env_file or ".env")
    cfg = Config(
        initial_mesh_path=_path("INITIAL_MESH_PATH", "/absolute/path/to/reference.stl"),
        rgba_dir=_path("RGBA_DIR", "./dataset/rgba"),
        positions_csv=_path("POSITIONS_CSV", "./dataset/positions.csv"),
        output_dir=_path("OUTPUT_DIR", "./outputs/demo"),
        device=os.getenv("DEVICE", "auto"),
        max_image_dimension=_int("MAX_IMAGE_DIMENSION", 768),
        camera_distance_initial=_float("CAMERA_DISTANCE_INITIAL", 2.7),
        camera_fov_initial=_float("CAMERA_FOV_INITIAL", 60.0),
        camera_fit_enabled=_bool("CAMERA_FIT_ENABLED", True),
        camera_fit_max_dimension=_int("CAMERA_FIT_MAX_DIMENSION", 128),
        camera_fit_max_faces=_int("CAMERA_FIT_MAX_FACES", 3000),
        camera_fit_max_frames=_int("CAMERA_FIT_MAX_FRAMES", 12),
        camera_fit_max_evaluations=_int("CAMERA_FIT_MAX_EVALUATIONS", 200),
        camera_convention_min_median_iou=_float("CAMERA_CONVENTION_MIN_MEDIAN_IOU", 0.20),
        camera_gate_min_median_iou=_float("CAMERA_GATE_MIN_MEDIAN_IOU", 0.50),
        pose_refine=_bool("POSE_REFINE", True),
        pose_refine_epochs=_int("POSE_REFINE_EPOCHS", 3),
        max_theta_delta_deg=_float("MAX_THETA_DELTA_DEG", 2.0),
        max_phi_delta_deg=_float("MAX_PHI_DELTA_DEG", 2.0),
        num_epochs=_int("NUM_EPOCHS", 50),
        batch_size=_int("BATCH_SIZE", 1),
        opt_lr_verts=_float("OPT_LR_VERTS", 0.0005),
        optimize_texture=_bool("OPTIMIZE_TEXTURE", False),
        loss_silhouette_weight=_float("LOSS_SILHOUETTE_WEIGHT", 1.0),
        loss_rgb_weight=_float("LOSS_RGB_WEIGHT", 0.0),
        loss_laplacian_weight=_float("LOSS_LAPLACIAN_WEIGHT", 0.1),
        loss_edge_weight=_float("LOSS_EDGE_WEIGHT", 0.1),
        loss_normal_weight=_float("LOSS_NORMAL_WEIGHT", 0.01),
        max_vertex_displacement_ratio=_float("MAX_VERTEX_DISPLACEMENT_RATIO", 0.10),
        min_usable_frames=_int("MIN_USABLE_FRAMES", 20),
        export_every_epochs=_int("EXPORT_EVERY_EPOCHS", 5),
        save_preview_every_epochs=_int("SAVE_PREVIEW_EVERY_EPOCHS", 2),
        early_stopping_patience=_int("EARLY_STOPPING_PATIENCE", 0),
        seed=_int("SEED", 42),
        joint_fine_tune=_bool("JOINT_FINE_TUNE", False),
        center_normalize=_bool("MESH_CENTER_NORMALIZE", True),
        scale_normalize=_bool("MESH_SCALE_NORMALIZE", True),
        silhouette_faces_per_pixel=_int("SILHOUETTE_FACES_PER_PIXEL", 20),
    )
    if cfg.max_image_dimension <= 0:
        raise ValueError("MAX_IMAGE_DIMENSION must be positive.")
    if cfg.camera_fit_max_dimension <= 0:
        raise ValueError("CAMERA_FIT_MAX_DIMENSION must be positive.")
    if cfg.camera_fit_max_faces <= 0 or cfg.camera_fit_max_frames <= 0:
        raise ValueError("Camera-fit max faces and max frames must be positive.")
    if cfg.camera_fit_max_evaluations <= 0 or cfg.pose_refine_epochs < 0:
        raise ValueError("Camera-fit evaluation count and pose epochs must be non-negative.")
    if cfg.batch_size != 1:
        raise ValueError("The OpenScan demo currently supports BATCH_SIZE=1 only.")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def save_config_snapshot(cfg: Config, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.as_dict(), indent=2))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(requested: str = "auto") -> torch.device:
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("Warning: DEVICE=mps requested but MPS is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            print("Warning: DEVICE=cuda requested but CUDA is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_pytorch3d() -> None:
    try:
        import pytorch3d  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "PyTorch3D is required after dataset validation. Install it in the selected environment."
        ) from exc
