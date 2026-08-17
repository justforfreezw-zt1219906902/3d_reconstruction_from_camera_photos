from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.rgba_dataset import build_layouts, load_rgba_with_layout
from src.validation import ValidationError, validate_inputs


def _write_rgba(path: Path, width: int = 4, height: int = 6) -> None:
    array = np.zeros((height, width, 4), dtype=np.uint8)
    array[..., :3] = 120
    array[1:-1, 1:-1, 3] = np.array([[0, 64], [128, 255], [255, 255], [255, 255]])[: height - 2, : width - 2]
    Image.fromarray(array, mode="RGBA").save(path)


def _write_csv(path: Path, image_name: str) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "position_index", "phi_deg", "theta_deg", "optional"])
        writer.writeheader()
        writer.writerow({"image": image_name, "position_index": 0, "phi_deg": -30, "theta_deg": 0, "optional": "ok"})


def test_validation_maps_original_jpg_name_to_processed_png(tmp_path: Path) -> None:
    rgba_dir = tmp_path / "rgba"
    rgba_dir.mkdir()
    _write_rgba(rgba_dir / "frame.png")
    csv_path = tmp_path / "positions.csv"
    _write_csv(csv_path, "frame.jpg")
    stl_path = tmp_path / "reference.stl"
    stl_path.write_text("solid empty\nendsolid empty\n")
    cfg = SimpleNamespace(rgba_dir=rgba_dir, positions_csv=csv_path, initial_mesh_path=stl_path, min_usable_frames=1)

    contract = validate_inputs(cfg)

    assert contract.frames[0].image == "frame.jpg"
    assert contract.frames[0].image_path.name == "frame.png"
    assert contract.validation["csv_extension_aliases"] == 1


def test_rgba_validation_and_soft_alpha(tmp_path: Path) -> None:
    rgba_dir = tmp_path / "rgba"
    rgba_dir.mkdir()
    image_path = rgba_dir / "frame.png"
    _write_rgba(image_path)
    csv_path = tmp_path / "positions.csv"
    _write_csv(csv_path, "frame.png")
    stl_path = tmp_path / "reference.stl"
    stl_path.write_text("solid empty\nendsolid empty\n")
    cfg = SimpleNamespace(rgba_dir=rgba_dir, positions_csv=csv_path, initial_mesh_path=stl_path, min_usable_frames=1)
    contract = validate_inputs(cfg)
    layout = build_layouts(contract, 8)[0]
    rgb, alpha = load_rgba_with_layout(image_path, layout)
    assert alpha.max() == 1.0
    assert 0.0 < alpha[2, 2] < 1.0
    assert rgb.shape == (6, 6, 3)
    assert alpha.shape == (6, 6, 1)
    assert layout.original_width == 4
    assert layout.original_height == 6
    assert layout.processed_width == 4
    assert layout.processed_height == 6


def test_validation_rejects_missing_csv_row(tmp_path: Path) -> None:
    rgba_dir = tmp_path / "rgba"
    rgba_dir.mkdir()
    _write_rgba(rgba_dir / "frame_a.png")
    _write_rgba(rgba_dir / "frame_b.png")
    csv_path = tmp_path / "positions.csv"
    _write_csv(csv_path, "frame_a.png")
    stl_path = tmp_path / "reference.stl"
    stl_path.write_text("solid empty\nendsolid empty\n")
    cfg = SimpleNamespace(rgba_dir=rgba_dir, positions_csv=csv_path, initial_mesh_path=stl_path, min_usable_frames=1)
    with pytest.raises(ValidationError, match="Missing CSV rows"):
        validate_inputs(cfg)
