from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.camera_fitting import _load_camera_fit_data, select_representative_indices
from src.fast_silhouette import CameraParameters, FastMesh, FastSilhouetteRenderer, silhouette_iou
from src.pose_conventions import CameraConvention, candidate_conventions
from src.validation import validate_inputs


def _dataset_stub() -> SimpleNamespace:
    frames = []
    index = 0
    for phi in (-30.0, 0.0, 30.0):
        for theta in range(0, 360, 45):
            frames.append(SimpleNamespace(phi_deg=phi, theta_deg=float(theta), position_index=index))
            index += 1
    return SimpleNamespace(frames=frames)


def test_representatives_cover_phi_rings_and_theta_range() -> None:
    dataset = _dataset_stub()
    selected = select_representative_indices(dataset, 6)
    assert {dataset.frames[i].phi_deg for i in selected} == {-30.0, 0.0, 30.0}
    selected_theta = {dataset.frames[i].theta_deg for i in selected}
    assert min(selected_theta) == 0.0
    assert max(selected_theta) >= 180.0
    assert len(selected) == 6


def test_alpha_preload_opens_each_frame_once(tmp_path: Path, monkeypatch) -> None:
    rgba_dir = tmp_path / "rgba"
    rgba_dir.mkdir()
    for index in range(2):
        image = Image.new("RGBA", (8, 8), (100, 100, 100, 0))
        image.putpixel((3, 3), (100, 100, 100, 128))
        image.save(rgba_dir / f"frame_{index}.png")
    csv_path = tmp_path / "positions.csv"
    csv_path.write_text(
        "image,position_index,phi_deg,theta_deg\n"
        "frame_0.png,0,0,0\n"
        "frame_1.png,1,0,180\n"
    )
    stl_path = tmp_path / "reference.stl"
    stl_path.write_text("solid empty\nendsolid empty\n")
    cfg = SimpleNamespace(rgba_dir=rgba_dir, positions_csv=csv_path, initial_mesh_path=stl_path, min_usable_frames=1)
    contract = validate_inputs(cfg)

    import src.camera_fitting as camera_fitting

    calls = 0
    original = camera_fitting.load_alpha_with_layout

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(camera_fitting, "load_alpha_with_layout", counted)
    data, _ = _load_camera_fit_data(contract, 4)
    assert data.alphas[0].shape == (4, 4)
    assert calls == 2


def test_fast_renderer_known_triangle_is_deterministic() -> None:
    mesh = FastMesh(
        vertices=np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    renderer = FastSilhouetteRenderer(64)
    camera = CameraParameters(distance=2.0, fov_deg=60.0)
    first = renderer.render(mesh, camera, 0.0, 0.0)
    second = renderer.render(mesh, camera, 0.0, 0.0)
    rotated = renderer.render(mesh, camera, 90.0, 0.0)
    assert np.array_equal(first, second)
    assert first.sum() > 0
    assert not np.array_equal(first, rotated)


def test_convention_candidates_cover_axes_signs_and_orders() -> None:
    conventions = candidate_conventions()
    assert len(conventions) == 48
    assert CameraConvention("Y", "X", 1, 1, "theta_then_phi") in conventions
    assert CameraConvention("Y", "X", -1, -1, "phi_then_theta") in conventions


def test_camera_fit_has_no_pytorch3d_rasterizer_dependency() -> None:
    source = Path("src/camera_fitting.py").read_text()
    assert "MeshRasterizer" not in source
    assert "SoftSilhouetteShader" not in source
