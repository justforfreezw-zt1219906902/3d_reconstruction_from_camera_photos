from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import Config, load_config, runtime_profile_defaults, select_device
from src.fast_silhouette import FastMesh, create_camera_proxy
from src.optimizer import select_geometry_view_indices, split_geometry_view_indices
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


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 7, 336])
@pytest.mark.parametrize("fraction", [0.001, 0.2, 0.99])
def test_geometry_split_small_and_large_datasets(count, fraction) -> None:
    import random
    frames = [SimpleNamespace() for _ in range(count)]
    state = random.getstate()
    train, validation = split_geometry_view_indices(frames, 42, fraction)
    assert random.getstate() == state
    assert (train, validation) == split_geometry_view_indices(frames, 42, fraction)
    assert set(train).isdisjoint(validation)
    assert sorted(train + validation) == list(range(count))
    if count >= 2:
        assert train and validation
        assert len(validation) == min(count - 1, max(1, round(count * fraction)))
    else:
        assert train == list(range(count)) and validation == []
    assert set(select_geometry_view_indices([frames[i] for i in train], 0, 1, 42)) == set(range(len(train)))


def test_geometry_split_angular_coverage_and_seed() -> None:
    frames = [SimpleNamespace(phi_deg=phi, theta_deg=theta)
              for phi in (-30, 0, 30) for theta in range(0, 360, 10)]
    train, validation = split_geometry_view_indices(frames, 42, 0.2)
    assert validation != split_geometry_view_indices(frames, 43, 0.2)[1]
    for indices in (train, validation):
        assert {frames[i].phi_deg for i in indices} == {-30, 0, 30}
        for phi in (-30, 0, 30):
            angles = sorted(frames[i].theta_deg for i in indices if frames[i].phi_deg == phi)
            assert max(b - a for a, b in zip(angles, angles[1:] + [angles[0] + 360])) <= 100


@pytest.mark.parametrize("fraction", [0, 1, -0.1, 1.1, float("nan"), float("inf")])
def test_geometry_split_rejects_invalid_fraction(fraction) -> None:
    with pytest.raises(ValueError, match="GEOMETRY_VALIDATION_FRACTION"):
        split_geometry_view_indices([SimpleNamespace()] * 3, 42, fraction)


def test_preview_creates_parent_directory(tmp_path: Path) -> None:
    image = torch.ones(1, 8, 8, 3)
    alpha = torch.ones(1, 8, 8, 1)
    output = tmp_path / "nested" / "preview.png"
    save_stage_preview(image, alpha, alpha, output)
    assert output.exists()


def test_regularization_is_outside_view_loop() -> None:
    source = Path("src/optimizer.py").read_text()
    assert source.count("regs = regularization_losses(mesh, base_mesh)") == 1
    assert "for step, batch in enumerate" in source


def test_image_only_training_uses_coarse_topology_without_cad_offset_controls(monkeypatch, tmp_path) -> None:
    """Image-only refinement must never enter the CAD regularizer/bound path."""
    from dataclasses import replace
    import src.optimizer as module
    from src.mesh_io import _normalize_mesh, mesh_from_arrays

    # Five vertices/six faces: arbitrary coarse topology, not a sphere proxy.
    mesh, transform = _normalize_mesh(mesh_from_arrays(
        torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.5, .5, 1.5]]),
        torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 4], [1, 4, 3], [2, 3, 4]]),
        torch.device("cpu"),
    ), True, True)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    cfg = replace(load_config(tmp_path / "missing.env"), reconstruction_mode="image_only",
                  num_epochs=1, geometry_views_per_epoch=0, geometry_view_batch_size=1,
                  loss_rgb_weight=0., loss_normal_weight=0., loss_laplacian_weight=0.,
                  loss_edge_weight=0., image_only_coarse_anchor_weight=0.,
                  max_local_deformation_ratio=.03, max_vertex_displacement_ratio=.03,
                  export_every_epochs=1, save_preview_every_epochs=1)

    class Dataset:
        frames = [SimpleNamespace(phi_deg=0., theta_deg=0.), SimpleNamespace(phi_deg=0., theta_deg=180.)]
        canvas_size = (4, 3)
        def __len__(self): return 2
        def __getitem__(self, index): return SimpleNamespace(alpha=torch.zeros(5, 3), image=torch.zeros(5, 3))
        def representative_indices(self): return []

    def forbidden(*args, **kwargs):
        raise AssertionError("CAD local-deformation path must not run in image_only mode")
    monkeypatch.setattr(module, "regularization_losses", forbidden)
    monkeypatch.setattr(module, "_bounded_local_offsets", forbidden)
    monkeypatch.setattr(module, "plot_losses", lambda *args: None)
    result = module.train_geometry(
        mesh, transform, Dataset(),
        SimpleNamespace(freeze=lambda: None, cameras=lambda batch: batch),
        SimpleNamespace(render_mask=lambda current, cameras: current.verts_padded().sigmoid().expand(len(cameras), -1, -1)),
        cfg, torch.device("cpu"), tmp_path,
    )
    assert torch.equal(result.mesh.faces_packed(), mesh.faces_packed())
    assert result.mesh.verts_packed().shape == mesh.verts_packed().shape


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


@pytest.mark.parametrize("legacy_env", [False, True])
def test_reference_deformation_config_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, legacy_env: bool) -> None:
    monkeypatch.setattr(os, "environ", {})
    env_file = tmp_path / "config.env"
    env_file.write_text(f"OUTPUT_DIR={tmp_path / 'outputs'}\n" + ("NUM_EPOCHS=7\n" if legacy_env else ""))
    cfg = load_config(env_file)
    expected = {
        "loss_anchor_weight": 1.0,
        "loss_local_smoothness_weight": 0.1,
        "max_local_deformation_ratio": 0.03,
        "geometry_validation_fraction": 0.20,
        "reconstruction_mode": "cad_prior",
        "ground_truth_mesh_path": None,
        "camera_calibration_source": "positions_rig",
        "camera_calibration_path": None,
        "coarse_volume_center": None,
        "coarse_volume_scale": None,
        "coarse_volume_padding_ratio": 0.25,
        "coarse_voxel_resolution": 128,
        "coarse_min_silhouette_support": 1.0,
        "coarse_occupancy_threshold": 0.5,
        "coarse_min_component_voxels": 64,
        "coarse_marching_cubes_step": 1,
        "mesh_cleanup_enabled": True,
        "mesh_min_component_faces": 32,
        "mesh_max_edge_length_ratio": 10.0,
        "mesh_spike_ratio": 8.0,
        "image_only_refinement_enabled": True,
        "image_only_coarse_anchor_weight": 0.0,
    }
    for key, value in expected.items():
        assert getattr(cfg, key) == value
        assert cfg.as_dict()[key] == value
    # Older direct constructors can omit the newly introduced fields too.
    old_fields = {key: value for key, value in vars(cfg).items() if key not in expected}
    assert Config(**old_fields) == cfg
    assert cfg.num_epochs == (7 if legacy_env else 50)
    unchanged = {
        "camera_distance_initial": 2.7, "camera_fov_initial": 60.0,
        "camera_fit_enabled": True, "camera_fit_max_dimension": 128,
        "camera_fit_max_faces": 3000, "camera_fit_max_frames": 12,
        "camera_fit_max_evaluations": 200,
        "camera_convention_min_median_iou": 0.20, "camera_gate_min_median_iou": 0.50,
        "pose_refine": True, "pose_refine_epochs": 3,
        "max_theta_delta_deg": 2.0, "max_phi_delta_deg": 2.0,
        "alignment_enabled": True, "alignment_epochs": 10, "alignment_lr": 0.01,
        "max_vertex_displacement_ratio": 0.10,
    }
    for key, value in unchanged.items():
        assert getattr(cfg, key) == value


def test_image_only_config_isolated_snapshot_and_cad_controls_are_inert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(os, "environ", {})
    calibration = tmp_path / "independent_calibration.json"
    ground_truth = tmp_path / "evaluation_only.stl"
    env_file = tmp_path / "image_only.env"
    env_file.write_text(
        "\n".join((
            f"OUTPUT_DIR={tmp_path / 'outputs'}",
            "RECONSTRUCTION_MODE=image_only",
            f"GROUND_TRUTH_MESH_PATH={ground_truth}",
            "CAMERA_CALIBRATION_SOURCE=calibration_file",
            f"CAMERA_CALIBRATION_PATH={calibration}",
            "COARSE_VOLUME_CENTER=1.5, -2, 0.25",
            "COARSE_VOLUME_SCALE=4.0",
            "COARSE_VOLUME_PADDING_RATIO=0.5",
            "COARSE_VOXEL_RESOLUTION=64",
            "COARSE_MIN_SILHOUETTE_SUPPORT=0.75",
            "COARSE_OCCUPANCY_THRESHOLD=0.6",
            "COARSE_MIN_COMPONENT_VOXELS=12",
            "COARSE_MARCHING_CUBES_STEP=2",
            "MESH_CLEANUP_ENABLED=false",
            "MESH_MIN_COMPONENT_FACES=9",
            "MESH_MAX_EDGE_LENGTH_RATIO=12.0",
            "MESH_SPIKE_RATIO=6.0",
            "IMAGE_ONLY_REFINEMENT_ENABLED=false",
            "IMAGE_ONLY_COARSE_ANCHOR_WEIGHT=0.2",
            # These CAD-prior-only values must not become image-only controls.
            "LOSS_ANCHOR_WEIGHT=3.0",
            "MAX_LOCAL_DEFORMATION_RATIO=0.02",
        ))
    )
    cfg = load_config(env_file)
    assert cfg.reconstruction_mode == "image_only"
    assert cfg.ground_truth_mesh_path == ground_truth
    assert cfg.camera_calibration_source == "calibration_file"
    assert cfg.camera_calibration_path == calibration
    assert cfg.coarse_volume_center == (1.5, -2.0, 0.25)
    assert cfg.coarse_volume_scale == 4.0
    assert cfg.coarse_volume_padding_ratio == 0.5
    assert cfg.coarse_voxel_resolution == 64
    assert cfg.coarse_min_silhouette_support == 0.75
    assert cfg.coarse_occupancy_threshold == 0.6
    assert cfg.coarse_min_component_voxels == 12
    assert cfg.coarse_marching_cubes_step == 2
    assert not cfg.mesh_cleanup_enabled
    assert cfg.mesh_min_component_faces == 9
    assert cfg.mesh_max_edge_length_ratio == 12.0
    assert cfg.mesh_spike_ratio == 6.0
    assert not cfg.image_only_refinement_enabled
    assert cfg.image_only_coarse_anchor_weight == 0.2
    assert cfg.loss_anchor_weight == 0.0
    assert cfg.max_local_deformation_ratio == 0.0
    from src.config import save_config_snapshot

    snapshot_path = tmp_path / "run_config.json"
    save_config_snapshot(cfg, snapshot_path)
    snapshot = cfg.as_dict()
    assert snapshot["ground_truth_mesh_path"] == str(ground_truth)
    assert snapshot["camera_calibration_path"] == str(calibration)
    assert snapshot["coarse_volume_center"] == (1.5, -2.0, 0.25)
    assert snapshot_path.exists()
    assert ground_truth.as_posix() in snapshot_path.read_text()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("RECONSTRUCTION_MODE", "unknown", "RECONSTRUCTION_MODE"),
        ("CAMERA_CALIBRATION_SOURCE", "sphere_fit", "CAMERA_CALIBRATION_SOURCE"),
        ("COARSE_VOXEL_RESOLUTION", "15", "COARSE_VOXEL_RESOLUTION"),
        ("COARSE_MIN_SILHOUETTE_SUPPORT", "0", "COARSE_MIN_SILHOUETTE_SUPPORT"),
    ],
)
def test_image_only_config_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, name: str, value: str, message: str,
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        load_config(tmp_path / "missing.env")


def test_reference_deformation_config_env_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(os, "environ", {})
    env_file = tmp_path / "config.env"
    overrides = {
        "LOSS_ANCHOR_WEIGHT": 2.0,
        "LOSS_LOCAL_SMOOTHNESS_WEIGHT": 0.25,
        "MAX_LOCAL_DEFORMATION_RATIO": 0.02,
        "GEOMETRY_VALIDATION_FRACTION": 0.15,
    }
    env_file.write_text(
        f"OUTPUT_DIR={tmp_path / 'outputs'}\n"
        + "\n".join(f"{key}={value}" for key, value in overrides.items())
    )
    cfg = load_config(env_file)
    for key, value in overrides.items():
        assert getattr(cfg, key.lower()) == value
        assert cfg.as_dict()[key.lower()] == value
    # Existing process environment takes precedence over dotenv values.
    for key in overrides:
        monkeypatch.setenv(key, "0.01")
    cfg = load_config(env_file)
    for key in overrides:
        assert getattr(cfg, key.lower()) == 0.01


@pytest.mark.parametrize("ratio", [0.0, 0.001, 0.03])
def test_bounded_offsets_preserve_local_modes_and_gradients(ratio: float) -> None:
    from src.optimizer import _bounded_local_offsets, _local_offset_basis, _displacement_stats
    vertices = torch.tensor([[[-1., -1., -1.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]])
    basis = _local_offset_basis(vertices)
    size = float(torch.linalg.vector_norm(vertices.amax(1) - vertices.amin(1)))
    raw = (torch.arange(12.).reshape_as(vertices).sin() * 100).requires_grad_()
    bounded = _bounded_local_offsets(raw, basis, size * ratio)
    assert _displacement_stats(bounded, size)["max_vertex_displacement_ratio"] <= ratio
    assert torch.allclose(basis.T @ bounded.flatten(), torch.zeros(basis.shape[1]), atol=1e-6)
    bounded.square().sum().backward()
    assert torch.isfinite(raw.grad).all()
    zero = torch.zeros_like(raw, requires_grad=True)
    _bounded_local_offsets(zero, basis, size * ratio).sum().backward()
    assert torch.isfinite(zero.grad).all()


@pytest.mark.parametrize("emergency_limit", [0.1, 0.0001])
@pytest.mark.parametrize("view_count,view_limit,batch_size", [(2, 0, 1), (9, 3, 2), (9, 0, 4)])
def test_training_bounds_every_mesh_and_weights_reference_losses(monkeypatch, tmp_path, emergency_limit, view_count, view_limit, batch_size) -> None:
    import csv
    import json
    from dataclasses import replace
    import src.optimizer as module
    from src.mesh_io import mesh_from_arrays, _normalize_mesh

    vertices = torch.tensor([[-1., -1., -1.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    mesh = mesh_from_arrays(vertices, torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]), torch.device("cpu"))
    mesh, transform = _normalize_mesh(mesh, True, True)
    reference = mesh.verts_packed().clone()
    size = float(torch.linalg.vector_norm(reference.amax(0) - reference.amin(0)))
    class Dataset:
        frames = [SimpleNamespace(phi_deg=0., theta_deg=i * 360 / view_count, image=f"{i}.png") for i in range(view_count)]
        canvas_size = (4, 3)
        def __len__(self):
            return view_count
        def __getitem__(self, index):
            return SimpleNamespace(alpha=torch.ones(4, 3), image=torch.ones(4, 3))
        def representative_indices(self):
            return list(range(view_count))
    observed = []
    gradient_views, backward_views, preview_views = set(), set(), set()
    def check_mesh(current):
        displacement = current.verts_packed().detach() - reference
        assert module._displacement_stats(displacement, size)["max_vertex_displacement_ratio"] <= 0.03
        observed.append(displacement.clone())
    def render(current, cameras):
        check_mesh(current)
        result = current.verts_padded().sigmoid().expand(len(cameras), -1, -1)
        if torch.is_grad_enabled():
            gradient_views.update(cameras)
            result.register_hook(lambda grad: backward_views.update(cameras))
        else:
            preview_views.update(cameras)
        return result
    exports = []
    def export(current, path, transform):
        check_mesh(current)
        exports.append(path.name)
    monkeypatch.setattr(module, "export_mesh_obj", export)
    monkeypatch.setattr(module, "export_mesh_stl", export)
    monkeypatch.setattr(module, "save_stage_preview", lambda *args: None)
    monkeypatch.setattr(module, "plot_losses", lambda *args: None)
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    cfg = replace(load_config(tmp_path / "missing.env"), num_epochs=3, geometry_views_per_epoch=view_limit,
                  geometry_view_batch_size=batch_size, opt_lr_verts=10., loss_rgb_weight=0.,
                  loss_normal_weight=0., loss_anchor_weight=2.3, loss_local_smoothness_weight=0.7,
                  max_local_deformation_ratio=0.03, max_vertex_displacement_ratio=emergency_limit,
                  export_every_epochs=1, save_preview_every_epochs=1, early_stopping_patience=0)
    def train():
        return module.train_geometry(mesh, transform, Dataset(),
            SimpleNamespace(freeze=lambda: None, cameras=lambda batch: batch),
            SimpleNamespace(render_mask=render), cfg, torch.device("cpu"), tmp_path)
    if emergency_limit < cfg.max_local_deformation_ratio:
        with pytest.raises(module.ReconstructionGateError, match="DEFORMATION SAFETY"):
            train()
        saved = torch.load(tmp_path / "reconstruction/checkpoints/last_valid_checkpoint.pt", weights_only=True)
        assert torch.count_nonzero(saved["offsets"]) == 0
        assert not exports
        return
    result = train()
    split = json.loads((tmp_path / "reconstruction/view_split.json").read_text())
    training, validation = split_geometry_view_indices(Dataset.frames, cfg.seed, cfg.geometry_validation_fraction)
    assert split["training_indices"] == training
    assert split["validation_indices"] == validation
    assert gradient_views and backward_views == gradient_views
    assert gradient_views <= set(training)
    assert gradient_views.isdisjoint(validation)
    assert set(validation) <= preview_views
    usage = list(csv.DictReader((tmp_path / "reconstruction/view_usage.csv").open()))
    assert all(int(usage[i]["usage_count"]) == 0 for i in validation)
    check_mesh(result.mesh)
    assert "final.obj" in exports and "final.stl" in exports
    assert any(torch.count_nonzero(offset) for offset in observed)
    rows = list(csv.DictReader((tmp_path / "reconstruction/losses.csv").open()))
    assert len(rows) == 3
    for row in rows:
        required = {"epoch", "total", "silhouette", "rgb", "normal", "iou",
                    "training_objective", "training_silhouette", "training_iou",
                    "validation_silhouette", "validation_iou", "anchor", "local_smoothness",
                    "mean_vertex_displacement", "median_vertex_displacement",
                    "p95_vertex_displacement", "max_vertex_displacement",
                    "max_vertex_displacement_ratio", "best_epoch", "is_best"}
        assert required <= row.keys()
        assert all(np.isfinite(float(row[key])) for key in required)
        for explicit, legacy in [("training_objective", "total"),
                                 ("training_silhouette", "silhouette"), ("training_iou", "iou")]:
            assert row[explicit] == row[legacy]
        expected = (cfg.loss_silhouette_weight * float(row["silhouette"])
                    + 2.3 * float(row["anchor"]) + 0.7 * float(row["local_smoothness"]))
        assert float(row["total"]) == pytest.approx(expected, rel=1e-6)
        assert float(row["max_vertex_displacement_ratio"]) <= 0.03
    assert float(rows[-1]["anchor"]) > 0
    assert float(rows[-1]["local_smoothness"]) > 0
    for path in (tmp_path / "reconstruction/checkpoints").glob("*.pt"):
        saved = torch.load(path, weights_only=True)
        assert module._displacement_stats(saved["offsets"], size)["max_vertex_displacement_ratio"] <= 0.03


@pytest.mark.parametrize("ratio", [0.0, 0.001, 0.017])
def test_configured_bound_survives_rendering_and_real_exports(monkeypatch, tmp_path, ratio) -> None:
    """Oversized optimizer steps must never escape through previews or mesh files."""
    import json
    from dataclasses import replace
    import src.optimizer as module
    from src.mesh_io import mesh_from_arrays, _normalize_mesh, load_initial_mesh, restore_vertices

    device = torch.device("cpu")
    vertices = torch.tensor([[-1., -1., -1.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]) * 7 + 12
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    mesh, transform = _normalize_mesh(mesh_from_arrays(vertices, faces, device), True, True)
    reference = mesh.verts_packed().clone()
    size = torch.linalg.vector_norm(reference.amax(0) - reference.amin(0)).item()
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_LOCAL_DEFORMATION_RATIO", str(ratio))
    cfg = replace(load_config(tmp_path / "missing.env"), num_epochs=2, opt_lr_verts=10.,
                  geometry_views_per_epoch=0, geometry_view_batch_size=8, seed=713,
                  loss_rgb_weight=0., loss_normal_weight=0., loss_anchor_weight=0.,
                  loss_local_smoothness_weight=0., max_vertex_displacement_ratio=1.,
                  export_every_epochs=1, save_preview_every_epochs=1, early_stopping_patience=0)
    assert cfg.max_local_deformation_ratio == ratio

    class Dataset:
        frames = [SimpleNamespace(phi_deg=0., theta_deg=i * 180., image=f"{i}.png") for i in range(2)]
        canvas_size = (4, 3)
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return SimpleNamespace(alpha=torch.ones(4, 3), image=torch.ones(4, 3))
        def representative_indices(self):
            return [0, 1]

    rendered_ratios = []
    def render(current, cameras):
        # Measure actual vertices independently of the production statistics helper.
        actual = (current.verts_packed().detach() - reference).norm(dim=-1).max().item() / size
        assert actual <= ratio + 1e-7
        rendered_ratios.append(actual)
        return current.verts_padded().sigmoid().expand(len(cameras), -1, -1)

    monkeypatch.setattr(module, "save_stage_preview", lambda *args: None)
    monkeypatch.setattr(module, "plot_losses", lambda *args: None)
    splits = []
    for run in range(2):
        # Perturb ambient randomness: cfg.seed alone must determine the split.
        import random
        state = random.getstate()
        try:
            random.seed(100 + run)
            result = module.train_geometry(mesh, transform, Dataset(),
                SimpleNamespace(freeze=lambda: None, cameras=lambda batch: batch),
                SimpleNamespace(render_mask=render), cfg, device, tmp_path / str(run))
        finally:
            random.setstate(state)
        folder = tmp_path / str(run) / "reconstruction"
        split = json.loads((folder / "view_split.json").read_text())
        train, validation = split["training_indices"], split["validation_indices"]
        assert train and validation
        assert set(train).isdisjoint(validation)
        assert sorted(train + validation) == [0, 1]
        splits.append((train, validation))
        assert (result.mesh.verts_packed() - reference).norm(dim=-1).max().item() / size <= ratio + 1e-7
        paths = list(folder.rglob("*.obj")) + list(folder.rglob("*.stl"))
        assert len(paths) == 6  # Two epoch exports and final export, in both formats.
        raw_reference = restore_vertices(reference, transform)
        raw_size = size * transform.scale
        for path in paths:
            loaded, _ = load_initial_mesh(path, device, center_normalize=False, scale_normalize=False)
            # STL can reorder/deduplicate vertices; every exported point must be bounded.
            distances = torch.cdist(loaded.verts_packed(), raw_reference)
            assert distances.min(dim=1).values.max().item() / raw_size <= ratio + 1e-6
            assert distances.min(dim=0).values.max().item() / raw_size <= ratio + 1e-6
    assert splits[0] == splits[1]
    assert rendered_ratios
    if ratio:
        assert max(rendered_ratios) > ratio * 0.9  # Ensure the hard limit was exercised.


def test_best_geometry_comparison() -> None:
    from src.optimizer import is_better_geometry_state as better
    best = dict(validation_silhouette=.2, validation_iou=.6,
                mean_vertex_displacement=.02, p95_vertex_displacement=.03)
    assert better(best, None, constraints_satisfied=True)
    assert not better(best, None, constraints_satisfied=False)
    assert not better(best, best, constraints_satisfied=True)
    for changes, expected in [
        ({'validation_silhouette': .1, 'mean_vertex_displacement': .03}, True),
        ({'validation_silhouette': .3, 'mean_vertex_displacement': .01}, False),
        ({'validation_iou': .7}, True), ({'validation_iou': .5}, False),
        ({'validation_silhouette': .2000001, 'mean_vertex_displacement': .01}, True),
        ({'p95_vertex_displacement': .02}, True),
        ({'mean_vertex_displacement': .03, 'p95_vertex_displacement': .01}, False),
        ({'validation_silhouette': float('nan')}, False),
        ({'p95_vertex_displacement': float('inf')}, False),
    ]:
        candidate = {**best, **changes}
        assert better(candidate, best, constraints_satisfied=True) == expected
        assert not better(candidate, best, constraints_satisfied=False)


def test_best_geometry_comparison_prefers_validity_when_holdout_quality_is_tied() -> None:
    from src.optimizer import is_better_geometry_state as better

    best = dict(
        validation_silhouette=.2, validation_iou=.6,
        mean_vertex_displacement=.02, p95_vertex_displacement=.03,
        validity_is_valid=0., validity_non_finite_vertex_count=0.,
        validity_degenerate_face_count=2., validity_self_intersection_pair_count=1.,
        validity_spike_vertex_count=3., validity_extreme_edge_count=4.,
        validity_disconnected_fragment_count=1., validity_small_component_count=2.,
    )
    # A valid mesh outranks an invalid one even if it moved farther from its
    # refinement base; selection must not use CAD proximity as the tie-break.
    cleaner = {**best, "validity_is_valid": 1., "mean_vertex_displacement": .5,
               "p95_vertex_displacement": .6}
    assert better(cleaner, best, constraints_satisfied=True)
    # When both are invalid, the defect counts provide a deterministic order.
    fewer_intersections = {**best, "validity_self_intersection_pair_count": 0.}
    assert better(fewer_intersections, best, constraints_satisfied=True)
    worse_degeneracy = {**best, "validity_degenerate_face_count": 3.}
    assert not better(worse_degeneracy, best, constraints_satisfied=True)
    assert not better({**best, "validity_spike_vertex_count": float("nan")}, None,
                      constraints_satisfied=True)


def test_validation_no_grad_and_view_weighting() -> None:
    from src.optimizer import evaluate_geometry_validation
    from pytorch3d.structures import Meshes
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    mesh = Meshes(verts=[vertices], faces=[torch.tensor([[0, 1, 2]])])
    offsets = torch.zeros_like(mesh.verts_padded(), requires_grad=True)
    offsets.grad = torch.ones_like(offsets)
    before, gradient = offsets.detach().clone(), offsets.grad.clone()
    calls = []
    class Dataset:
        def __getitem__(self, index):
            assert not torch.is_grad_enabled()
            return SimpleNamespace(alpha=torch.ones(1, 1, 1))
    def render(current, cameras):
        assert not torch.is_grad_enabled()
        assert not current.verts_padded().requires_grad
        calls.extend(cameras)
        return torch.tensor(cameras, dtype=torch.float32).reshape(-1, 1, 1, 1) / 4
    result = evaluate_geometry_validation(mesh, offsets, Dataset(),
        SimpleNamespace(cameras=lambda batch: batch), SimpleNamespace(render_mask=render),
        [1, 2, 3], 2, torch.device('cpu'), 1.)
    assert calls == [1, 2, 3]
    assert result['validation_silhouette'] == pytest.approx((.75**2 + .5**2 + .25**2) / 3)
    assert result['validation_iou'] == pytest.approx(.5)
    assert result['mean_vertex_displacement'] == 0
    assert torch.equal(offsets, before) and torch.equal(offsets.grad, gradient)


@pytest.mark.parametrize('patience,expected_epochs', [(1, 2), (0, 3)])
def test_training_restores_best_validation_offsets(monkeypatch, tmp_path, patience, expected_epochs) -> None:
    from dataclasses import replace
    import csv
    import src.optimizer as module
    from src.mesh_io import mesh_from_arrays, _normalize_mesh, load_initial_mesh, restore_vertices
    vertices = torch.tensor([[-1., -1., -1.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    mesh = mesh_from_arrays(vertices, torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]), torch.device('cpu'))
    mesh, transform = _normalize_mesh(mesh, True, True)
    class Dataset:
        frames = [SimpleNamespace() for _ in range(2)]
        canvas_size = (4, 3)
        def __len__(self):
            return 2
        def __getitem__(self, index):
            return SimpleNamespace(alpha=torch.ones(4, 3), image=torch.ones(4, 3))
        def representative_indices(self):
            return []
    evaluated = []
    qualities = [.1, .2, .3]
    evaluate = module.evaluate_geometry_validation
    def validation(*args):
        result = evaluate(*args)
        evaluated.append(args[1].detach().clone())
        # Controlled held-out deterioration, regardless of training improvement.
        result.update(validation_silhouette=qualities[len(evaluated) - 1], validation_iou=.5)
        return result
    monkeypatch.setattr(module, 'evaluate_geometry_validation', validation)
    monkeypatch.setattr(module, 'plot_losses', lambda *args: None)
    monkeypatch.setenv('OUTPUT_DIR', str(tmp_path))
    cfg = replace(load_config(tmp_path / 'missing.env'), num_epochs=3, geometry_views_per_epoch=0,
                  geometry_view_batch_size=1, opt_lr_verts=.001, loss_rgb_weight=0.,
                  early_stopping_patience=patience, export_every_epochs=1)
    result = module.train_geometry(mesh, transform, Dataset(),
        SimpleNamespace(freeze=lambda: None, cameras=lambda batch: batch),
        SimpleNamespace(render_mask=lambda current, cameras: current.verts_padded().sigmoid()),
        cfg, torch.device('cpu'), tmp_path)
    assert len(evaluated) == expected_epochs
    assert not torch.equal(evaluated[0], evaluated[-1])
    expected = mesh.verts_padded() + evaluated[0]
    assert torch.equal(result.mesh.verts_padded(), expected)
    def check_exports(result, expected):
        expected_triangles = restore_vertices(expected[0], transform)[mesh.faces_packed()]
        for path in (result.final_obj, result.final_stl):
            loaded, _ = load_initial_mesh(path, torch.device('cpu'), False, False)
            triangles = loaded.verts_packed()[loaded.faces_packed()]
            assert torch.allclose(triangles, expected_triangles, atol=1e-5)
        assert not result.mesh.verts_padded().requires_grad

    check_exports(result, expected)
    rows = list(csv.DictReader((tmp_path / 'reconstruction/losses.csv').open()))
    assert [int(row['is_best']) for row in rows] == [1] + [0] * (expected_epochs - 1)
    assert all(int(row['best_epoch']) == 1 for row in rows)

    # A longer run must preserve the exact files unless a later state wins.
    original_files = [path.read_bytes() for path in (result.final_obj, result.final_stl)]
    for later_wins in (False, True):
        evaluated.clear()
        qualities[:] = [.1, .2, .3, .05 if later_wins else .4]
        longer = module.train_geometry(mesh, transform, Dataset(),
            SimpleNamespace(freeze=lambda: None, cameras=lambda batch: batch),
            SimpleNamespace(render_mask=lambda current, cameras: current.verts_padded().sigmoid()),
            replace(cfg, num_epochs=4, early_stopping_patience=0), torch.device('cpu'),
            tmp_path / ('later_best' if later_wins else 'longer'))
        selected = evaluated[-1] if later_wins else evaluated[0]
        with (longer.final_obj.parent / 'losses.csv').open() as handle:
            longer_rows = list(csv.DictReader(handle))
        assert [int(row['is_best']) for row in longer_rows] == [1, 0, 0, int(later_wins)]
        assert [int(row['best_epoch']) for row in longer_rows] == [1, 1, 1, 4 if later_wins else 1]
        check_exports(longer, mesh.verts_padded() + selected)
        files = [path.read_bytes() for path in (longer.final_obj, longer.final_stl)]
        if later_wins:
            assert all(a != b for a, b in zip(files, original_files))
        else:
            assert files == original_files


@pytest.mark.parametrize('epochs', [1, 3])
@pytest.mark.parametrize('expanded', [False, True])
def test_plot_losses_schema_compatibility(tmp_path, monkeypatch, epochs, expanded) -> None:
    from src import visualization
    csv_path = tmp_path / 'losses.csv'
    for epoch in range(1, epochs + 1):
        row = dict(epoch=epoch, total=.2, silhouette=.1, iou=.6, rgb=.01, normal=.02)
        if expanded:
            row.update(training_objective=.2, training_silhouette=.1, training_iou=.6,
                       validation_silhouette=.15, validation_iou=.5, anchor=0., local_smoothness=0.,
                       mean_vertex_displacement=0., median_vertex_displacement=0.,
                       p95_vertex_displacement=0., max_vertex_displacement=0.,
                       max_vertex_displacement_ratio=0., best_epoch=1, is_best=int(epoch == 1))
        visualization.append_csv(csv_path, row)
    labels = []
    plot = visualization.plt.plot
    def record_plot(*args, **kwargs):
        labels.append(kwargs['label'])
        return plot(*args, **kwargs)
    monkeypatch.setattr(visualization.plt, 'plot', record_plot)
    output = tmp_path / 'plots' / 'losses.png'
    visualization.plot_losses(csv_path, output)
    assert output.is_file()
    with Image.open(output) as image:
        image.verify()
    assert {'rgb', 'normal'} <= set(labels)
    assert not {'epoch', 'best_epoch', 'is_best'} & set(labels)
    if expanded:
        assert {'training_objective', 'training_silhouette', 'training_iou',
                'validation_silhouette', 'validation_iou', 'anchor', 'local_smoothness'} <= set(labels)
        assert not {'total', 'silhouette', 'iou'} & set(labels)
    else:
        assert {'total', 'silhouette', 'iou'} <= set(labels)
