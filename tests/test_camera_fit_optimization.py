from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from src.camera_fitting import preload_camera_fit_alpha, select_representative_indices
from src.openscan_pose import OpenScanCameraModel
from src.rgba_dataset import RGBADataset
from src.renderer import ReconstructionRenderer
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
    dataset = RGBADataset(contract, 8)

    import src.camera_fitting as camera_fitting

    calls = 0
    original = camera_fitting.load_alpha_with_layout

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(camera_fitting, "load_alpha_with_layout", counted)
    cache = preload_camera_fit_alpha(dataset, 4, torch.device("cpu"))
    assert cache.alphas.shape == (2, 4, 4, 1)
    assert calls == 2
    _ = cache.batch([0, 1])
    _ = cache.batch([0, 1])
    assert calls == 2


def test_global_stage_parameter_sets_exclude_pose_parameters() -> None:
    model = OpenScanCameraModel([0.0, 90.0], [0.0, 0.0], 2.7, 60.0, 2.0, 2.0, True, torch.device("cpu"))
    model.set_global_trainable(True)
    model.set_pose_trainable(False)
    assert all(parameter.requires_grad for parameter in model.global_parameters())
    assert not any(parameter.requires_grad for parameter in model.pose_parameters())


def test_renderer_reuse_matches_single_camera_render() -> None:
    from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
    from pytorch3d.structures import Meshes

    mesh = Meshes(
        verts=[torch.tensor([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.0, 0.5, 0.0]])],
        faces=[torch.tensor([[0, 1, 2]])],
    )
    R, T = look_at_view_transform(dist=2.7, elev=0.0, azim=[0.0, 90.0])
    cameras = FoVPerspectiveCameras(R=R, T=T)
    renderer = ReconstructionRenderer(24, torch.device("cpu"), 2)
    one = renderer.render_mask(mesh, FoVPerspectiveCameras(R=R[:1], T=T[:1]))
    batch = renderer.render_mask(mesh, cameras)
    assert torch.allclose(one, batch[:1], atol=1e-6)
