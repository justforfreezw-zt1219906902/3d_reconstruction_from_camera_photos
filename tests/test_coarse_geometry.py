from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.coarse_geometry import (
    carve_visual_hull,
    derive_carving_volume,
    extract_marching_cubes_surface,
    generate_coarse_geometry,
)
from src.fast_silhouette import CameraParameters, FastMesh
from src.pose_conventions import DEFAULT_CONVENTION
from src.rgba_dataset import ImageLayout


def _cfg(**overrides):
    values = dict(
        coarse_volume_center=None,
        coarse_volume_scale=None,
        coarse_volume_padding_ratio=0.5,
        max_image_dimension=64,
        coarse_voxel_resolution=24,
        coarse_min_silhouette_support=1.0,
        coarse_occupancy_threshold=0.5,
        coarse_min_component_voxels=0,
        coarse_marching_cubes_step=1,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _frames(count=2):
    return tuple(
        SimpleNamespace(image_path=Path(f"frame_{index}.png"), theta_deg=float(index * 90), phi_deg=0.0,
                        width=64, height=64)
        for index in range(count)
    )


def _layout():
    return ImageLayout(64, 64, 64, 64, 64, 0, 0, 1.0)


def test_volume_hint_is_padded_and_does_not_preserve_sphere_bounds():
    sphere_positions = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    volume = derive_carving_volume(FastMesh(sphere_positions, np.array([[0, 1, 2]])), _cfg())
    assert np.allclose(volume.center, [0.0, 0.0, 0.0])
    assert volume.side_length == 4.0
    assert np.all(volume.minimum < -1.0)
    assert np.all(volume.maximum > 1.0)


def test_carving_uses_calibrated_alpha_masks_and_extracts_new_connectivity(monkeypatch):
    import src.coarse_geometry as coarse

    mask = np.zeros((64, 64, 1), dtype=np.float32)
    mask[17:47, 17:47, 0] = 1.0
    monkeypatch.setattr(coarse, "load_alpha_with_layout", lambda *_: torch.from_numpy(mask.copy()))
    volume = derive_carving_volume(None, _cfg(coarse_volume_scale=3.0))
    occupancy, support = carve_visual_hull(
        _frames(), (_layout(), _layout()), CameraParameters(distance=3.0, fov_deg=90.0),
        DEFAULT_CONVENTION, volume, 24, 1.0, 0.5,
    )
    vertices, faces = extract_marching_cubes_surface(occupancy, volume)
    assert occupancy.any() and not occupancy.all()
    assert support.max() == 1.0
    assert len(vertices) > 4 and len(faces) > 4
    # Extracted vertices can use the configured padded region, not an implicit
    # unit-sphere clipping box.
    assert np.max(np.abs(vertices)) > 1.0


def test_generate_coarse_geometry_uses_only_position_hint(monkeypatch):
    import src.coarse_geometry as coarse

    mask = np.zeros((64, 64, 1), dtype=np.float32)
    mask[17:47, 17:47, 0] = 1.0
    monkeypatch.setattr(coarse, "load_alpha_with_layout", lambda *_: torch.from_numpy(mask.copy()))
    frames = _frames()
    contract = SimpleNamespace(frames=frames)
    # A tetrahedron stands in for the sphere initializer.  Its faces must not
    # survive in the generated topology.
    hint = FastMesh(
        np.array([[-1., -1., -1.], [1., -1., 1.], [-1., 1., 1.], [1., 1., -1.]]),
        np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=np.int32),
    )
    result = generate_coarse_geometry(
        contract, CameraParameters(distance=3.0, fov_deg=90.0), DEFAULT_CONVENTION,
        _cfg(), torch.device("cpu"), initial_mesh_hint=hint,
    )
    assert result.mesh.num_verts_per_mesh().item() == result.stats["vertex_count"]
    assert result.mesh.num_faces_per_mesh().item() == result.stats["face_count"]
    assert result.stats["face_count"] != len(hint.faces)
    assert torch.isfinite(result.mesh.verts_packed()).all()
