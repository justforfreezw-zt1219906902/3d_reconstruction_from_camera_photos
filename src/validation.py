from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

from .config import Config


REQUIRED_COLUMNS = {"image", "position_index", "phi_deg", "theta_deg"}


class ValidationError(ValueError):
    """Gate 1 failure: the OpenScan input contract is invalid."""


@dataclass(frozen=True)
class FrameRecord:
    image: str
    image_path: Path
    position_index: int
    phi_deg: float
    theta_deg: float
    width: int
    height: int
    aspect_ratio: float


@dataclass(frozen=True)
class DatasetContract:
    rgba_dir: Path
    positions_csv: Path
    initial_mesh_path: Path
    frames: tuple[FrameRecord, ...]
    unique_phi_values: tuple[float, ...]
    theta_min: float
    theta_max: float
    maximum_theta_gap: float
    aspect_ratio: float
    validation: dict


def _resolve_csv_image(rgba_dir: Path, image_value: str) -> Path:
    raw = Path(image_value)
    direct = (rgba_dir / raw).resolve()
    if direct.exists() and direct.is_file():
        return direct
    # Processed datasets may retain the original JPG names in positions.csv
    # while the RGBA export has the same frame as a PNG.
    same_stem = [
        path.resolve()
        for path in rgba_dir.rglob(f"{raw.stem}.*")
        if path.is_file() and path.suffix.lower() == ".png"
    ]
    matches = same_stem or [path.resolve() for path in rgba_dir.rglob(raw.name) if path.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValidationError(f"CSV image '{image_value}' does not exist under {rgba_dir}.")
    raise ValidationError(f"CSV image '{image_value}' is ambiguous; multiple PNGs have that name.")


def _read_csv(path: Path, rgba_dir: Path) -> list[FrameRecord]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValidationError(f"positions.csv is missing required columns: {sorted(missing)}")
        records: list[FrameRecord] = []
        seen_names: set[str] = set()
        seen_positions: set[int] = set()
        for row_number, row in enumerate(reader, start=2):
            image_value = (row.get("image") or "").strip()
            if not image_value:
                raise ValidationError(f"positions.csv row {row_number} has an empty image field.")
            image_path = _resolve_csv_image(rgba_dir, image_value)
            name = image_path.name
            if name in seen_names:
                raise ValidationError(f"Duplicate CSV image name: {name}")
            seen_names.add(name)
            try:
                position_index = int(row["position_index"])
                phi_deg = float(row["phi_deg"])
                theta_deg = float(row["theta_deg"])
            except (TypeError, ValueError) as exc:
                raise ValidationError(f"Invalid pose value in positions.csv row {row_number}.") from exc
            if position_index in seen_positions:
                raise ValidationError(f"Duplicate position_index: {position_index}")
            seen_positions.add(position_index)
            if not math.isfinite(phi_deg) or not math.isfinite(theta_deg):
                raise ValidationError(f"Non-finite phi/theta in positions.csv row {row_number}.")
            with Image.open(image_path) as image:
                width, height = image.size
                bands = image.getbands()
                if len(bands) != 4:
                    raise ValidationError(
                        f"Image {image_path} must contain four channels (RGBA), got {bands}."
                    )
                alpha = image.getchannel(bands[3])
                alpha_min, alpha_max = alpha.getextrema()
                if alpha_max == 0:
                    raise ValidationError(f"Image {image_path} has an empty alpha channel.")
                if alpha_min == 255 and alpha_max == 255:
                    raise ValidationError(f"Image {image_path} is fully opaque; expected an object alpha mask.")
            records.append(
                FrameRecord(
                    image=image_value,
                    image_path=image_path,
                    position_index=position_index,
                    phi_deg=phi_deg,
                    theta_deg=theta_deg,
                    width=width,
                    height=height,
                    aspect_ratio=width / height,
                )
            )
    return records


def validate_inputs(cfg: Config) -> DatasetContract:
    errors: list[str] = []
    if not cfg.rgba_dir.exists() or not cfg.rgba_dir.is_dir():
        errors.append(f"RGBA_DIR does not exist or is not a directory: {cfg.rgba_dir}")
    if not cfg.positions_csv.exists() or not cfg.positions_csv.is_file():
        errors.append(f"POSITIONS_CSV does not exist: {cfg.positions_csv}")
    if not cfg.initial_mesh_path.exists() or not cfg.initial_mesh_path.is_file():
        errors.append(f"INITIAL_MESH_PATH does not exist: {cfg.initial_mesh_path}")
    if errors:
        raise ValidationError("GATE 1 — DATA CONTRACT FAILED:\n- " + "\n- ".join(errors))

    frames = _read_csv(cfg.positions_csv, cfg.rgba_dir)
    pngs = {path.resolve() for path in cfg.rgba_dir.rglob("*.png") if path.is_file()}
    csv_paths = {frame.image_path.resolve() for frame in frames}
    missing_csv_rows = sorted(path.name for path in pngs - csv_paths)
    missing_png_files = sorted(path.name for path in csv_paths - pngs)
    if missing_csv_rows:
        raise ValidationError(
            "Every PNG used for reconstruction must have a CSV row. Missing CSV rows for: "
            + ", ".join(missing_csv_rows[:10])
        )
    if missing_png_files:
        raise ValidationError("CSV references missing PNG files: " + ", ".join(missing_png_files[:10]))
    if not frames:
        raise ValidationError("positions.csv contains no frames.")
    if len(frames) < cfg.min_usable_frames:
        raise ValidationError(
            f"GATE 3 — VIEW COVERAGE FAILED: only {len(frames)} usable frames; "
            f"minimum is {cfg.min_usable_frames}."
        )
    aspect_ratio = frames[0].aspect_ratio
    if any(abs(frame.aspect_ratio - aspect_ratio) / aspect_ratio > 1e-3 for frame in frames[1:]):
        raise ValidationError("RGBA frames have incompatible aspect ratios; refusing geometric distortion.")
    theta_values = sorted(frame.theta_deg % 360.0 for frame in frames)
    gaps = [b - a for a, b in zip(theta_values, theta_values[1:])]
    if len(theta_values) > 1:
        gaps.append(theta_values[0] + 360.0 - theta_values[-1])
    validation = {
        "rgba_dir": str(cfg.rgba_dir),
        "positions_csv": str(cfg.positions_csv),
        "initial_mesh_path": str(cfg.initial_mesh_path),
        "total_frames": len(frames),
        "usable_frames": len(frames),
        "empty_alpha_frames": 0,
        "invalid_frames": 0,
        "unique_phi_values": sorted({frame.phi_deg for frame in frames}),
        "theta_min": min(frame.theta_deg for frame in frames),
        "theta_max": max(frame.theta_deg for frame in frames),
        "maximum_theta_gap": max(gaps) if gaps else 0.0,
        "coverage_warning": (
            "Angular coverage has a gap larger than 90 degrees; reconstruction may be underconstrained."
            if gaps and max(gaps) > 90.0
            else None
        ),
        "aspect_ratio": aspect_ratio,
        "source_dimensions": sorted({f"{frame.width}x{frame.height}" for frame in frames}),
        "csv_extension_aliases": sum(
            Path(frame.image).suffix.lower() != frame.image_path.suffix.lower() for frame in frames
        ),
    }
    return DatasetContract(
        rgba_dir=cfg.rgba_dir,
        positions_csv=cfg.positions_csv,
        initial_mesh_path=cfg.initial_mesh_path,
        frames=tuple(frames),
        unique_phi_values=tuple(validation["unique_phi_values"]),
        theta_min=validation["theta_min"],
        theta_max=validation["theta_max"],
        maximum_theta_gap=validation["maximum_theta_gap"],
        aspect_ratio=aspect_ratio,
        validation=validation,
    )


def save_validation(contract: DatasetContract, path: Path) -> None:
    payload = dict(contract.validation)
    payload["frames"] = [
        {
            **asdict(frame),
            "image_path": str(frame.image_path),
        }
        for frame in contract.frames
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
