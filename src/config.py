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


RUNTIME_PROFILES: dict[str, dict[str, int | float | str]] = {
    "apple_fast": {
        "device": "cpu", "geometry_max_faces": 5000, "max_image_dimension": 128,
        "geometry_views_per_epoch": 32, "num_epochs": 20,
        "silhouette_faces_per_pixel": 5, "geometry_view_batch_size": 1,
    },
    "apple_quality": {
        "device": "cpu", "geometry_max_faces": 10000, "max_image_dimension": 256,
        "geometry_views_per_epoch": 64, "num_epochs": 30,
        "silhouette_faces_per_pixel": 10, "geometry_view_batch_size": 1,
    },
    "cuda": {
        "device": "cuda", "geometry_max_faces": 30000, "max_image_dimension": 512,
        "geometry_views_per_epoch": 0, "num_epochs": 50,
        "silhouette_faces_per_pixel": 20, "geometry_view_batch_size": 4,
    },
    "custom": {},
}


def runtime_profile_defaults(name: str) -> dict:
    name = (name or "custom").strip().lower()
    if name not in RUNTIME_PROFILES:
        raise ValueError(f"Unknown RUNTIME_PROFILE '{name}'. Use apple_fast, apple_quality, cuda, or custom.")
    return dict(RUNTIME_PROFILES[name])


def _profile_value(name: str, defaults: dict, key: str, fallback):
    return os.getenv(name, str(defaults.get(key, fallback)))


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


def _optional_path(name: str) -> Optional[Path]:
    """Read an optional path without turning an unset value into ``Path('.')``."""
    value = os.getenv(name, "").strip()
    return Path(value).expanduser() if value else None


def _optional_float_triplet(name: str) -> Optional[tuple[float, float, float]]:
    """Read ``x,y,z`` volume-centre input, allowing the rough mesh centre by default."""
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        parts = tuple(float(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated x,y,z triplet.") from exc
    if len(parts) != 3 or not all(np.isfinite(part) for part in parts):
        raise ValueError(f"{name} must contain three finite values: x,y,z.")
    return parts[0], parts[1], parts[2]


@dataclass(frozen=True)
class Config:
    initial_mesh_path: Path
    rgba_dir: Path
    positions_csv: Path
    output_dir: Path
    device: str
    runtime_profile: str
    geometry_max_faces: int
    geometry_views_per_epoch: int
    geometry_view_batch_size: int
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

    alignment_enabled: bool = True
    alignment_epochs: int = 10
    alignment_lr: float = 0.01

    # Conservative local-only controls relative to the frozen aligned CAD prior:
    # unit anchor weight retains reference proximity; modest smoothness weight
    # discourages isolated offsets without dominating the reference penalty.
    loss_anchor_weight: float = 1.0
    loss_local_smoothness_weight: float = 0.1
    # Limit residual motion to 3% of object size; hold out 20% of geometry views.
    max_local_deformation_ratio: float = 0.03
    geometry_validation_fraction: float = 0.20

    # Image-only reconstruction controls.  These are defaulted so direct
    # construction used by older CAD-prior callers remains source-compatible.
    reconstruction_mode: str = "cad_prior"
    # Evaluation-only: reconstruction modules must never load or consume this.
    ground_truth_mesh_path: Optional[Path] = None
    # Both supported sources are independent of an initial mesh silhouette.
    camera_calibration_source: str = "positions_rig"
    camera_calibration_path: Optional[Path] = None
    # None means derive only a rough centre/scale from the initialization mesh.
    coarse_volume_center: Optional[tuple[float, float, float]] = None
    coarse_volume_scale: Optional[float] = None
    coarse_volume_padding_ratio: float = 0.25
    coarse_voxel_resolution: int = 128
    coarse_min_silhouette_support: float = 1.0
    coarse_occupancy_threshold: float = 0.5
    coarse_min_component_voxels: int = 64
    coarse_marching_cubes_step: int = 1
    mesh_cleanup_enabled: bool = True
    mesh_min_component_faces: int = 32
    mesh_max_edge_length_ratio: float = 10.0
    mesh_spike_ratio: float = 8.0
    image_only_refinement_enabled: bool = True
    # An optional anchor can only reference the image-derived coarse mesh.
    image_only_coarse_anchor_weight: float = 0.0

    def as_dict(self) -> dict:
        data = asdict(self)
        for key in (
            "initial_mesh_path", "rgba_dir", "positions_csv", "output_dir",
            "ground_truth_mesh_path", "camera_calibration_path",
        ):
            if data[key] is not None:
                data[key] = str(data[key])
        return data


def load_config(env_file: Optional[str | Path] = None) -> Config:
    load_dotenv(env_file or ".env")
    runtime_profile = os.getenv("RUNTIME_PROFILE", "custom").strip().lower()
    profile = runtime_profile_defaults(runtime_profile)
    reconstruction_mode = os.getenv("RECONSTRUCTION_MODE", "cad_prior").strip().lower()
    if reconstruction_mode not in {"cad_prior", "image_only"}:
        raise ValueError("RECONSTRUCTION_MODE must be either 'cad_prior' or 'image_only'.")
    camera_calibration_source = os.getenv("CAMERA_CALIBRATION_SOURCE", "positions_rig").strip().lower()
    if camera_calibration_source not in {"positions_rig", "calibration_file"}:
        raise ValueError(
            "CAMERA_CALIBRATION_SOURCE must be either 'positions_rig' or 'calibration_file'."
        )
    camera_calibration_path = _optional_path("CAMERA_CALIBRATION_PATH")
    if camera_calibration_source == "calibration_file" and camera_calibration_path is None:
        raise ValueError("CAMERA_CALIBRATION_PATH is required when CAMERA_CALIBRATION_SOURCE=calibration_file.")

    # CAD anchor and local-deformation controls intentionally have no effect in
    # image-only mode.  Retain their established defaults unchanged for CAD.
    cad_loss_anchor_weight = _float("LOSS_ANCHOR_WEIGHT", 1.0)
    cad_max_local_deformation_ratio = _float("MAX_LOCAL_DEFORMATION_RATIO", 0.03)
    cfg = Config(
        alignment_enabled=_bool("ALIGNMENT_ENABLED", True),
        alignment_epochs=_int("ALIGNMENT_EPOCHS", 10),
        alignment_lr=_float("ALIGNMENT_LR", 0.01),
        initial_mesh_path=_path("INITIAL_MESH_PATH", "/absolute/path/to/reference.stl"),
        rgba_dir=_path("RGBA_DIR", "./dataset/rgba"),
        positions_csv=_path("POSITIONS_CSV", "./dataset/positions.csv"),
        output_dir=_path("OUTPUT_DIR", "./outputs/demo"),
        device=_profile_value("DEVICE", profile, "device", "auto"),
        runtime_profile=runtime_profile,
        geometry_max_faces=int(_profile_value("GEOMETRY_MAX_FACES", profile, "geometry_max_faces", 10000)),
        geometry_views_per_epoch=int(_profile_value("GEOMETRY_VIEWS_PER_EPOCH", profile, "geometry_views_per_epoch", 0)),
        geometry_view_batch_size=int(_profile_value("GEOMETRY_VIEW_BATCH_SIZE", profile, "geometry_view_batch_size", 1)),
        max_image_dimension=int(_profile_value("MAX_IMAGE_DIMENSION", profile, "max_image_dimension", 768)),
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
        num_epochs=int(_profile_value("NUM_EPOCHS", profile, "num_epochs", 50)),
        batch_size=_int("BATCH_SIZE", int(_profile_value("GEOMETRY_VIEW_BATCH_SIZE", profile, "geometry_view_batch_size", 1))),
        opt_lr_verts=_float("OPT_LR_VERTS", 0.0005),
        optimize_texture=_bool("OPTIMIZE_TEXTURE", False),
        loss_silhouette_weight=_float("LOSS_SILHOUETTE_WEIGHT", 1.0),
        loss_rgb_weight=_float("LOSS_RGB_WEIGHT", 0.0),
        loss_laplacian_weight=_float("LOSS_LAPLACIAN_WEIGHT", 0.1),
        loss_edge_weight=_float("LOSS_EDGE_WEIGHT", 0.1),
        loss_normal_weight=_float("LOSS_NORMAL_WEIGHT", 0.01),
        loss_anchor_weight=cad_loss_anchor_weight if reconstruction_mode == "cad_prior" else 0.0,
        loss_local_smoothness_weight=_float("LOSS_LOCAL_SMOOTHNESS_WEIGHT", 0.1),
        max_local_deformation_ratio=(
            cad_max_local_deformation_ratio if reconstruction_mode == "cad_prior" else 0.0
        ),
        geometry_validation_fraction=_float("GEOMETRY_VALIDATION_FRACTION", 0.20),
        max_vertex_displacement_ratio=_float("MAX_VERTEX_DISPLACEMENT_RATIO", 0.10),
        min_usable_frames=_int("MIN_USABLE_FRAMES", 20),
        export_every_epochs=_int("EXPORT_EVERY_EPOCHS", 5),
        save_preview_every_epochs=_int("SAVE_PREVIEW_EVERY_EPOCHS", 2),
        early_stopping_patience=_int("EARLY_STOPPING_PATIENCE", 0),
        seed=_int("SEED", 42),
        joint_fine_tune=_bool("JOINT_FINE_TUNE", False),
        center_normalize=_bool("MESH_CENTER_NORMALIZE", True),
        scale_normalize=_bool("MESH_SCALE_NORMALIZE", True),
        silhouette_faces_per_pixel=int(_profile_value("SILHOUETTE_FACES_PER_PIXEL", profile, "silhouette_faces_per_pixel", 20)),
        reconstruction_mode=reconstruction_mode,
        ground_truth_mesh_path=_optional_path("GROUND_TRUTH_MESH_PATH"),
        camera_calibration_source=camera_calibration_source,
        camera_calibration_path=camera_calibration_path,
        coarse_volume_center=_optional_float_triplet("COARSE_VOLUME_CENTER"),
        coarse_volume_scale=(
            _float("COARSE_VOLUME_SCALE", 0.0) if os.getenv("COARSE_VOLUME_SCALE", "").strip() else None
        ),
        coarse_volume_padding_ratio=_float("COARSE_VOLUME_PADDING_RATIO", 0.25),
        coarse_voxel_resolution=_int("COARSE_VOXEL_RESOLUTION", 128),
        coarse_min_silhouette_support=_float("COARSE_MIN_SILHOUETTE_SUPPORT", 1.0),
        coarse_occupancy_threshold=_float("COARSE_OCCUPANCY_THRESHOLD", 0.5),
        coarse_min_component_voxels=_int("COARSE_MIN_COMPONENT_VOXELS", 64),
        coarse_marching_cubes_step=_int("COARSE_MARCHING_CUBES_STEP", 1),
        mesh_cleanup_enabled=_bool("MESH_CLEANUP_ENABLED", True),
        mesh_min_component_faces=_int("MESH_MIN_COMPONENT_FACES", 32),
        mesh_max_edge_length_ratio=_float("MESH_MAX_EDGE_LENGTH_RATIO", 10.0),
        mesh_spike_ratio=_float("MESH_SPIKE_RATIO", 8.0),
        image_only_refinement_enabled=_bool("IMAGE_ONLY_REFINEMENT_ENABLED", True),
        image_only_coarse_anchor_weight=_float("IMAGE_ONLY_COARSE_ANCHOR_WEIGHT", 0.0),
    )
    if cfg.alignment_epochs <= 0 or not np.isfinite(cfg.alignment_lr) or cfg.alignment_lr <= 0:
        raise ValueError("ALIGNMENT_EPOCHS and ALIGNMENT_LR must be positive and finite.")
    if cfg.max_image_dimension <= 0:
        raise ValueError("MAX_IMAGE_DIMENSION must be positive.")
    if cfg.camera_fit_max_dimension <= 0:
        raise ValueError("CAMERA_FIT_MAX_DIMENSION must be positive.")
    if cfg.camera_fit_max_faces <= 0 or cfg.camera_fit_max_frames <= 0:
        raise ValueError("Camera-fit max faces and max frames must be positive.")
    if cfg.camera_fit_max_evaluations <= 0 or cfg.pose_refine_epochs < 0:
        raise ValueError("Camera-fit evaluation count and pose epochs must be non-negative.")
    if cfg.geometry_max_faces <= 0:
        raise ValueError("GEOMETRY_MAX_FACES must be positive.")
    if cfg.geometry_views_per_epoch < 0 or cfg.geometry_view_batch_size <= 0:
        raise ValueError("GEOMETRY_VIEWS_PER_EPOCH must be non-negative and GEOMETRY_VIEW_BATCH_SIZE positive.")
    if cfg.coarse_volume_scale is not None and (not np.isfinite(cfg.coarse_volume_scale) or cfg.coarse_volume_scale <= 0):
        raise ValueError("COARSE_VOLUME_SCALE must be positive and finite when configured.")
    if not np.isfinite(cfg.coarse_volume_padding_ratio) or cfg.coarse_volume_padding_ratio < 0:
        raise ValueError("COARSE_VOLUME_PADDING_RATIO must be finite and non-negative.")
    if cfg.coarse_voxel_resolution < 16:
        raise ValueError("COARSE_VOXEL_RESOLUTION must be at least 16.")
    if not np.isfinite(cfg.coarse_min_silhouette_support) or not 0 < cfg.coarse_min_silhouette_support <= 1:
        raise ValueError("COARSE_MIN_SILHOUETTE_SUPPORT must be in (0, 1].")
    if not np.isfinite(cfg.coarse_occupancy_threshold) or not 0 < cfg.coarse_occupancy_threshold <= 1:
        raise ValueError("COARSE_OCCUPANCY_THRESHOLD must be in (0, 1].")
    if cfg.coarse_min_component_voxels < 0 or cfg.coarse_marching_cubes_step <= 0:
        raise ValueError("Coarse component voxels must be non-negative and marching-cubes step positive.")
    if cfg.mesh_min_component_faces < 0:
        raise ValueError("MESH_MIN_COMPONENT_FACES must be non-negative.")
    if not np.isfinite(cfg.mesh_max_edge_length_ratio) or cfg.mesh_max_edge_length_ratio <= 0:
        raise ValueError("MESH_MAX_EDGE_LENGTH_RATIO must be positive and finite.")
    if not np.isfinite(cfg.mesh_spike_ratio) or cfg.mesh_spike_ratio <= 0:
        raise ValueError("MESH_SPIKE_RATIO must be positive and finite.")
    if not np.isfinite(cfg.image_only_coarse_anchor_weight) or cfg.image_only_coarse_anchor_weight < 0:
        raise ValueError("IMAGE_ONLY_COARSE_ANCHOR_WEIGHT must be finite and non-negative.")
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
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            print("Warning: DEVICE=mps requested but MPS is unavailable. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda was explicitly requested, but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_pytorch3d() -> None:
    try:
        import pytorch3d  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "PyTorch3D is required after dataset validation. Install it in the selected environment."
        ) from exc
