"""Regression tests for the image-only reconstruction safety boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from src.config import load_config
from src.mesh_io import load_initial_mesh, mesh_from_arrays


def _coarse_tetrahedron() -> object:
    """A deliberately non-sphere topology used as the image-derived input."""
    vertices = torch.tensor([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
        [0.0, 1.0, -1.0], [0.0, 0.0, 1.0],
    ])
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    return mesh_from_arrays(vertices, faces, torch.device("cpu"))


def test_image_only_refinement_never_enters_cad_anchor_or_local_limit(
    monkeypatch, tmp_path: Path,
) -> None:
    """The arbitrary coarse mesh must bypass all CAD/sphere-only controls."""
    import src.optimizer as optimizer

    coarse_mesh = _coarse_tetrahedron()
    coarse_faces = coarse_mesh.faces_packed().clone()
    calls: list[object] = []

    def forbidden(*_args, **_kwargs):
        raise AssertionError("image_only must not invoke a CAD anchor or local-deformation bound")

    def image_only_terms(mesh, coarse):
        # The only permitted anchor reference in this mode is the coarse mesh.
        assert torch.equal(coarse.faces_packed(), coarse_faces)
        calls.append(coarse)
        zero = mesh.verts_packed().sum() * 0.0
        return {"normal": zero, "smoothness": zero, "edge": zero, "coarse_anchor": zero}

    monkeypatch.setattr(optimizer, "regularization_losses", forbidden)
    monkeypatch.setattr(optimizer, "_bounded_local_offsets", forbidden)
    monkeypatch.setattr(optimizer, "image_only_regularization_losses", image_only_terms)
    monkeypatch.setattr(optimizer, "plot_losses", lambda *_args: None)
    monkeypatch.setattr(optimizer, "save_stage_preview", lambda *_args: None)

    class Dataset:
        frames = [SimpleNamespace(theta_deg=float(index * 90), phi_deg=0.0, image=f"{index}.png") for index in range(4)]
        canvas_size = (2, 2)

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, _index):
            return SimpleNamespace(alpha=torch.ones(2, 2, 1), image=torch.ones(2, 2, 3))

        def representative_indices(self):
            return []

    def render(mesh, cameras):
        # A differentiable image-shaped result without requiring a rasterizer.
        value = mesh.verts_padded().mean().sigmoid()
        return value.reshape(1, 1, 1, 1).expand(len(cameras), 2, 2, 1)

    cfg = replace(
        load_config(tmp_path / "absent.env"),
        reconstruction_mode="image_only", num_epochs=1, opt_lr_verts=0.1,
        geometry_views_per_epoch=0, geometry_view_batch_size=2,
        loss_rgb_weight=0.0, loss_normal_weight=0.0,
        loss_laplacian_weight=0.0, loss_edge_weight=0.0,
        image_only_coarse_anchor_weight=0.0,
        # A zero CAD limit makes an accidental use immediately observable.
        max_local_deformation_ratio=0.0, export_every_epochs=1,
        save_preview_every_epochs=99, early_stopping_patience=0,
    )
    result = optimizer.train_geometry(
        coarse_mesh, SimpleNamespace(center_normalized=False, scale_normalized=False), Dataset(),
        SimpleNamespace(freeze=lambda: None, cameras=lambda indices: indices),
        SimpleNamespace(render_mask=render), cfg, torch.device("cpu"), tmp_path,
    )

    assert calls, "image-only regularization was not evaluated"
    # The selected output is a valid export even though it originated from
    # arbitrary coarse topology rather than the initial sphere.
    for path in (result.final_obj, result.final_stl):
        exported, _ = load_initial_mesh(path, torch.device("cpu"), False, False)
        assert torch.isfinite(exported.verts_packed()).all()
        assert len(exported.faces_packed()) > 0
        assert exported.faces_packed().min() >= 0
        assert exported.faces_packed().max() < len(exported.verts_packed())


def test_image_only_camera_calibration_never_reads_the_initial_mesh(monkeypatch, tmp_path: Path) -> None:
    import src.camera_fitting as camera_fitting

    def forbidden(*_args, **_kwargs):
        raise AssertionError("independent calibration must not optimize against a sphere silhouette")

    monkeypatch.setattr(camera_fitting, "load_fast_mesh", forbidden)
    monkeypatch.setattr(camera_fitting, "FastSilhouetteRenderer", forbidden)
    contract = SimpleNamespace(frames=(
        SimpleNamespace(image="a.png", position_index=0, theta_deg=0.0, phi_deg=0.0),
        SimpleNamespace(image="b.png", position_index=1, theta_deg=180.0, phi_deg=0.0),
    ))
    cfg = SimpleNamespace(
        reconstruction_mode="image_only", camera_calibration_source="positions_rig",
        camera_distance_initial=3.0, camera_fov_initial=50.0,
    )
    result = camera_fitting.fit_camera(tmp_path / "unused-sphere.stl", contract, cfg, tmp_path / "camera")

    assert result.frozen and result.metrics["shape_based_camera_fit"] is False
    assert result.theta_deltas == [0.0, 0.0] and result.phi_deltas == [0.0, 0.0]


def test_view_split_validation_no_grad_and_best_state_are_deterministic() -> None:
    import src.optimizer as optimizer

    frames = [SimpleNamespace(theta_deg=float(index * 45), phi_deg=0.0) for index in range(8)]
    train, validation = optimizer.split_geometry_view_indices(frames, seed=91, validation_fraction=0.25)
    assert set(train).isdisjoint(validation)
    assert (train, validation) == optimizer.split_geometry_view_indices(frames, seed=91, validation_fraction=0.25)

    mesh = _coarse_tetrahedron()
    offsets = torch.zeros_like(mesh.verts_padded(), requires_grad=True)
    offsets.grad = torch.ones_like(offsets)

    class Dataset:
        def __getitem__(self, _index):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(alpha=torch.ones(1, 1, 1))

    def render(current, cameras):
        assert not torch.is_grad_enabled()
        assert not current.verts_padded().requires_grad
        return torch.ones(len(cameras), 1, 1, 1)

    original_grad = offsets.grad.clone()
    optimizer.evaluate_geometry_validation(
        mesh, offsets, Dataset(), SimpleNamespace(cameras=lambda values: values),
        SimpleNamespace(render_mask=render), validation, 2, torch.device("cpu"), 1.0,
    )
    assert torch.equal(offsets.grad, original_grad)
    best = {"validation_silhouette": 0.2, "validation_iou": 0.5,
            "mean_vertex_displacement": 0.1, "p95_vertex_displacement": 0.2}
    better = {**best, "validation_silhouette": 0.1}
    assert optimizer.is_better_geometry_state(better, best, constraints_satisfied=True)
    assert not optimizer.is_better_geometry_state(best, better, constraints_satisfied=True)
