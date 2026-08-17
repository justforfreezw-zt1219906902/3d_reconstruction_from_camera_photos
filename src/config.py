from __future__ import annotations

import os
import random
from dataclasses import dataclass
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
    original_obj_path: Path
    dataset_synth_dir: Path
    dataset_real_1_dir: Path
    dataset_real_2_dir: Path
    output_dir: Path
    device: str
    image_size: int
    background_color: str
    synth_num_views: int
    synth_rotation_start_deg: float
    synth_rotation_end_deg: float
    synth_camera_distance: float
    synth_elevation_deg: float
    synth_azimuth_start_deg: float
    synth_azimuth_end_deg: float
    real_camera_mode: str
    real_use_masks: bool
    real_image_pattern: str
    real_mask_pattern: str
    real_azimuth_start_deg: float
    real_azimuth_end_deg: float
    real_elevation_deg: float
    real_camera_distance: float
    opt_num_iters: int
    opt_lr_verts: float
    opt_lr_texture: float
    opt_optimize_texture: bool
    loss_silhouette_weight: float
    loss_rgb_weight: float
    loss_laplacian_weight: float
    loss_edge_weight: float
    loss_normal_weight: float
    export_every: int
    save_preview_every: int
    mesh_scale_normalize: bool
    mesh_center_normalize: bool
    seed: int

    @property
    def background_rgb(self) -> tuple[float, float, float]:
        if self.background_color.lower() == "white":
            return (1.0, 1.0, 1.0)
        if self.background_color.lower() == "black":
            return (0.0, 0.0, 0.0)
        raise ValueError("BACKGROUND_COLOR must be 'white' or 'black'.")

    def dataset_dir(self, name: str) -> Path:
        if name == "synthetic":
            return self.dataset_synth_dir
        if name == "real1":
            return self.dataset_real_1_dir
        if name == "real2":
            return self.dataset_real_2_dir
        raise ValueError(f"Unknown dataset '{name}'. Expected synthetic, real1, or real2.")

    def run_dir(self, name: str) -> Path:
        return self.output_dir / "runs" / name


def load_config(env_file: Optional[str | Path] = None) -> Config:
    load_dotenv(env_file or ".env")
    cfg = Config(
        original_obj_path=_path("ORIGINAL_OBJ_PATH", "/absolute/path/to/original.obj"),
        dataset_synth_dir=_path("DATASET_SYNTH_DIR", "./outputs/synthetic_benchmark"),
        dataset_real_1_dir=_path("DATASET_REAL_1_DIR", "/absolute/path/to/real_dataset_1"),
        dataset_real_2_dir=_path("DATASET_REAL_2_DIR", "/absolute/path/to/real_dataset_2"),
        output_dir=_path("OUTPUT_DIR", "./outputs"),
        device=os.getenv("DEVICE", "auto"),
        image_size=_int("IMAGE_SIZE", 512),
        background_color=os.getenv("BACKGROUND_COLOR", "white"),
        synth_num_views=_int("SYNTH_NUM_VIEWS", 30),
        synth_rotation_start_deg=_float("SYNTH_ROTATION_START_DEG", 0.0),
        synth_rotation_end_deg=_float("SYNTH_ROTATION_END_DEG", 90.0),
        synth_camera_distance=_float("SYNTH_CAMERA_DISTANCE", 2.7),
        synth_elevation_deg=_float("SYNTH_ELEVATION_DEG", 20.0),
        synth_azimuth_start_deg=_float("SYNTH_AZIMUTH_START_DEG", 0.0),
        synth_azimuth_end_deg=_float("SYNTH_AZIMUTH_END_DEG", 90.0),
        real_camera_mode=os.getenv("REAL_CAMERA_MODE", "turntable"),
        real_use_masks=_bool("REAL_USE_MASKS", True),
        real_image_pattern=os.getenv("REAL_IMAGE_PATTERN", "*.png"),
        real_mask_pattern=os.getenv("REAL_MASK_PATTERN", "{stem}_mask.png"),
        real_azimuth_start_deg=_float("REAL_AZIMUTH_START_DEG", 0.0),
        real_azimuth_end_deg=_float("REAL_AZIMUTH_END_DEG", 90.0),
        real_elevation_deg=_float("REAL_ELEVATION_DEG", 20.0),
        real_camera_distance=_float("REAL_CAMERA_DISTANCE", 2.7),
        opt_num_iters=_int("OPT_NUM_ITERS", 2000),
        opt_lr_verts=_float("OPT_LR_VERTS", 0.0005),
        opt_lr_texture=_float("OPT_LR_TEXTURE", 0.001),
        opt_optimize_texture=_bool("OPT_OPTIMIZE_TEXTURE", True),
        loss_silhouette_weight=_float("LOSS_SILHOUETTE_WEIGHT", 1.0),
        loss_rgb_weight=_float("LOSS_RGB_WEIGHT", 1.0),
        loss_laplacian_weight=_float("LOSS_LAPLACIAN_WEIGHT", 0.1),
        loss_edge_weight=_float("LOSS_EDGE_WEIGHT", 0.1),
        loss_normal_weight=_float("LOSS_NORMAL_WEIGHT", 0.01),
        export_every=_int("EXPORT_EVERY", 200),
        save_preview_every=_int("SAVE_PREVIEW_EVERY", 100),
        mesh_scale_normalize=_bool("MESH_SCALE_NORMALIZE", True),
        mesh_center_normalize=_bool("MESH_CENTER_NORMALIZE", True),
        seed=_int("SEED", 42),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    return cfg


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
            "PyTorch3D is required but could not be imported. Install it for your "
            "Python/PyTorch version, or use a CPU-only environment where PyTorch3D builds cleanly."
        ) from exc
