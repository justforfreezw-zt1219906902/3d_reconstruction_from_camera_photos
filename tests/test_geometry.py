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


def test_local_offsets_exclude_global_similarity_modes() -> None:
    from src.optimizer import _local_offset_basis, _project_local_offsets
    vertices = torch.tensor([[[-1., -1., -1.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]], dtype=torch.float64)
    basis = _local_offset_basis(vertices)
    centered = vertices - vertices.mean(1, keepdim=True)
    global_offsets = 0.2 * centered + torch.tensor([0.1, -0.2, 0.3])
    global_offsets += torch.linalg.cross(torch.tensor([0.2, 0.3, -0.1], dtype=vertices.dtype).expand_as(centered), centered)
    assert torch.allclose(_project_local_offsets(global_offsets, basis), torch.zeros_like(vertices), atol=1e-10)
    offsets = torch.randn_like(vertices, requires_grad=True)
    local = _project_local_offsets(offsets, basis)
    assert local.norm() > 0
    assert torch.allclose(basis.T @ local.flatten(), torch.zeros(7, dtype=vertices.dtype), atol=1e-10)
    local.square().sum().backward()
    assert torch.isfinite(offsets.grad).all()


def test_alignment_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("ALIGNMENT_ENABLED", "false")
    monkeypatch.setenv("ALIGNMENT_EPOCHS", "7")
    monkeypatch.setenv("ALIGNMENT_LR", "0.02")
    cfg = load_config(tmp_path / "missing.env")
    assert not cfg.alignment_enabled
    assert cfg.alignment_epochs == 7
    assert cfg.alignment_lr == 0.02
    for invalid in ("0", "nan", "inf", "-1"):
        monkeypatch.setenv("ALIGNMENT_LR", invalid)
        with pytest.raises(ValueError, match="ALIGNMENT"):
            load_config(tmp_path / "missing.env")


@pytest.mark.parametrize("enabled", [True, False])
def test_alignment_stage_fits_and_detaches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, enabled: bool) -> None:
    import json
    from src.mesh_io import mesh_from_arrays
    from src.pipeline import _fit_global_alignment
    import src.losses as losses

    vertices = torch.tensor([[-0.3, -0.2, -0.1], [0.4, 0., 0.], [0., 0.5, 0.], [0., 0., 0.6]])
    mesh = mesh_from_arrays(vertices, torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]), torch.device("cpu"))
    target = vertices + torch.tensor([0.08, -0.05, 0.06])
    class Dataset:
        frames = [SimpleNamespace(phi_deg=0., theta_deg=0.)]
        def __getitem__(self, index):
            return SimpleNamespace(alpha=target)
    camera = SimpleNamespace(freeze=lambda: None, cameras=lambda batch: batch)
    renderer = SimpleNamespace(render_mask=lambda mesh, cameras: mesh.verts_padded())
    # Differentiable synthetic observation isolates the stage's parameter ownership.
    monkeypatch.setattr(losses, "alpha_silhouette_loss", lambda predicted, target: (predicted - target).square().mean())
    cfg = SimpleNamespace(alignment_lr=0.01, alignment_epochs=100, alignment_enabled=enabled,
                          geometry_views_per_epoch=0, geometry_view_batch_size=1, seed=42)
    aligned = _fit_global_alignment(mesh, Dataset(), camera, renderer, cfg, torch.device("cpu"), tmp_path)
    assert not aligned.verts_padded().requires_grad
    assert torch.equal(mesh.verts_packed(), vertices)
    if enabled:
        assert (aligned.verts_packed() - target).square().mean() < 1e-6
    else:
        assert torch.equal(aligned.verts_packed(), vertices)
    metadata = json.loads((tmp_path / "global_alignment.json").read_text())
    assert metadata["enabled"] == enabled
    assert metadata["scale"] > 0


def test_pipeline_alignment_precedes_local_training() -> None:
    import inspect
    from src.pipeline import run_demo
    source = inspect.getsource(run_demo)
    assert source.index("camera_result =") < source.index("geometry_mesh = _fit_global_alignment(") < source.index("geometry_result = train_geometry(")
    assert source.index("if camera_fit_only:") < source.index("geometry_mesh = _fit_global_alignment(")


def test_local_training_exports_aligned_mesh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from dataclasses import replace
    import src.optimizer as optimizer_module
    from src.mesh_io import mesh_from_arrays, _normalize_mesh, load_initial_mesh, restore_vertices
    from src.alignment import SimilarityTransform

    device = torch.device("cpu")
    vertices = torch.tensor([[-1., -1., -1.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    mesh = mesh_from_arrays(vertices, torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]), device)
    mesh, transform = _normalize_mesh(mesh, True, True)
    alignment = SimilarityTransform(torch.eye(3), torch.tensor([0.3, -0.2, 0.1]), torch.tensor(1.5))
    aligned = mesh.update_padded(alignment.apply(mesh.verts_padded()))
    class Dataset:
        frames = [SimpleNamespace(phi_deg=0., theta_deg=0., image="test.png")]
        canvas_size = (4, 3)
        def __len__(self):
            return 1
        def __getitem__(self, index):
            return SimpleNamespace(alpha=aligned.verts_packed().sigmoid(), image=torch.ones(4, 3))
        def representative_indices(self):
            return []
    camera = SimpleNamespace(freeze=lambda: None, cameras=lambda batch: batch)
    renderer = SimpleNamespace(render_mask=lambda mesh, cameras: mesh.verts_padded().sigmoid())
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    cfg = replace(load_config(tmp_path / "missing.env"), num_epochs=1, geometry_views_per_epoch=1,
                  geometry_view_batch_size=1, loss_rgb_weight=0., loss_laplacian_weight=0.,
                  loss_edge_weight=0., loss_normal_weight=0., export_every_epochs=1)
    result = optimizer_module.train_geometry(aligned, transform, Dataset(), camera, renderer, cfg, device, tmp_path)
    assert torch.allclose(result.mesh.verts_packed(), aligned.verts_packed())
    saved = torch.load(tmp_path / "reconstruction/checkpoints/checkpoint_epoch_0001.pt", weights_only=True)
    assert torch.count_nonzero(saved["offsets"]) == 0
    expected = restore_vertices(aligned.verts_packed(), transform)
    for path in (result.final_obj, result.final_stl):
        loaded, _ = load_initial_mesh(path, device, center_normalize=False, scale_normalize=False)
        # STL may reorder vertices; compare by nearest corresponding position.
        assert torch.cdist(expected, loaded.verts_packed()).min(1).values.max() < 1e-5
