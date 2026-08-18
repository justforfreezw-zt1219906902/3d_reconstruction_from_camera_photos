from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import load_config, runtime_profile_defaults, select_device
from src.fast_silhouette import FastMesh, create_camera_proxy
from src.optimizer import select_geometry_view_indices
from src.visualization import save_stage_preview


def test_runtime_profiles_and_explicit_values() -> None:
    fast = runtime_profile_defaults("apple_fast")
    quality = runtime_profile_defaults("apple_quality")
    cuda = runtime_profile_defaults("cuda")
    assert fast["device"] == "cpu"
    assert fast["geometry_max_faces"] == 5000
    assert quality["max_image_dimension"] == 256
    assert cuda["device"] == "cuda"
    assert cuda["geometry_views_per_epoch"] == 0


def test_explicit_environment_values_override_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RUNTIME_PROFILE", "apple_fast")
    monkeypatch.setenv("GEOMETRY_MAX_FACES", "77")
    monkeypatch.setenv("MAX_IMAGE_DIMENSION", "99")
    monkeypatch.setenv("NUM_EPOCHS", "3")
    monkeypatch.setenv("GEOMETRY_VIEW_BATCH_SIZE", "2")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    cfg = load_config(tmp_path / "missing.env")
    assert cfg.geometry_max_faces == 77
    assert cfg.max_image_dimension == 99
    assert cfg.num_epochs == 3
    assert cfg.geometry_view_batch_size == 2


def test_explicit_cuda_request_fails_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="DEVICE=cuda"):
        select_device("cuda")


def test_geometry_proxy_limits_faces_preserves_dimensions_and_input(tmp_path: Path) -> None:
    vertices = np.array(
        [[x, y, z] for x in (-2.0, 2.0) for y in (-1.0, 1.0) for z in (-0.5, 0.5)],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        dtype=np.int32,
    )
    mesh = FastMesh(vertices, faces)
    reference = tmp_path / "reference.stl"
    reference.write_bytes(b"reference remains unchanged")
    before = reference.read_bytes()
    proxy, _ = create_camera_proxy(mesh, max_faces=4)
    assert len(proxy.faces) <= 4
    assert np.allclose(proxy.vertices.min(axis=0), vertices.min(axis=0), atol=1e-6)
    assert np.allclose(proxy.vertices.max(axis=0), vertices.max(axis=0), atol=1e-6)
    assert reference.read_bytes() == before


def test_geometry_view_sampling_rotates_and_covers_phi_theta() -> None:
    frames = [
        SimpleNamespace(phi_deg=phi, theta_deg=float(theta), image=f"{phi}_{theta}.png")
        for phi in (-30.0, 0.0, 30.0)
        for theta in range(0, 360, 30)
    ]
    first = select_geometry_view_indices(frames, 9, epoch=1, seed=42)
    second = select_geometry_view_indices(frames, 9, epoch=2, seed=42)
    assert len(first) == len(second) == 9
    assert {frames[i].phi_deg for i in first} == {-30.0, 0.0, 30.0}
    assert len({frames[i].theta_deg for i in first}) >= 3
    assert first != second


def test_preview_creates_parent_directory(tmp_path: Path) -> None:
    image = torch.ones(1, 8, 8, 3)
    alpha = torch.ones(1, 8, 8, 1)
    output = tmp_path / "nested" / "preview.png"
    save_stage_preview(image, alpha, alpha, output)
    assert output.exists()


def test_regularization_is_outside_view_loop() -> None:
    source = Path("src/optimizer.py").read_text()
    assert source.count("regs = regularization_losses(mesh)") == 1
    assert "for step, batch in enumerate" in source
