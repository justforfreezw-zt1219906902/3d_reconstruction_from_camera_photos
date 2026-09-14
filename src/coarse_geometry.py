"""Image-derived coarse geometry for ``RECONSTRUCTION_MODE=image_only``.

This module builds a visual hull: each voxel in a padded, normalized volume is
projected into every calibrated RGBA alpha mask and retained only when enough
masks support it.  A marching-cubes surface is then extracted from that
occupancy.  It deliberately has no ground-truth input and never compares a
voxel to the initial mesh.  The optional initial mesh is used *only* for a
rough centre and largest extent when no explicit volume is configured.

The resulting vertices are in the normalized object frame used by
``mesh_io.load_initial_mesh``.  Consequently they can be passed directly to
the image-only optimizer and exported with the same ``MeshTransform`` as the
initial mesh.

Visual hulls cannot recover concavities that do not alter a silhouette, or
internally occluded structure.  They are a conservative image-derived coarse
shape, not a replacement for multi-view photometric or depth reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .fast_silhouette import CameraParameters, FastMesh
from .pose_conventions import CameraConvention, object_rotation
from .rgba_dataset import ImageLayout, build_layouts, load_alpha_with_layout
from .validation import DatasetContract, FrameRecord


class CoarseGeometryError(RuntimeError):
    """Raised when calibrated silhouettes cannot produce a usable visual hull."""


@dataclass(frozen=True)
class CarvingVolume:
    """Axis-aligned normalized-frame carving volume."""

    minimum: np.ndarray
    maximum: np.ndarray
    center: np.ndarray
    side_length: float


@dataclass
class CoarseGeometryResult:
    """Image-derived mesh and diagnostics; occupancy is useful for previews/tests."""

    mesh: Any
    occupancy: np.ndarray
    volume: CarvingVolume
    support_fraction: np.ndarray
    stats: dict[str, float | int | list[float]]


def _vertices_from_hint(initial_mesh: Any | None) -> np.ndarray | None:
    """Extract positions only; connectivity and surface shape are intentionally ignored."""
    if initial_mesh is None:
        return None
    if isinstance(initial_mesh, FastMesh):
        vertices = initial_mesh.vertices
    elif isinstance(initial_mesh, np.ndarray):
        vertices = initial_mesh
    elif hasattr(initial_mesh, "verts_packed"):
        vertices = initial_mesh.verts_packed().detach().cpu().numpy()
    else:
        raise TypeError("initial_mesh must be a FastMesh, vertices array, PyTorch3D mesh, or None.")
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Initial-mesh volume hint must contain finite [N, 3] vertices.")
    return vertices


def derive_carving_volume(initial_mesh: Any | None, cfg: Any) -> CarvingVolume:
    """Derive a padded cube from a *rough* initialization centre/extent.

    ``coarse_volume_padding_ratio`` expands each side beyond the hint's largest
    dimension.  The hint's bounds are never used as clipping bounds, so the
    image-derived surface may extend beyond a sphere's exact bounding box.
    """
    vertices = _vertices_from_hint(initial_mesh)
    configured_center = getattr(cfg, "coarse_volume_center", None)
    configured_scale = getattr(cfg, "coarse_volume_scale", None)
    if configured_center is not None:
        center = np.asarray(configured_center, dtype=np.float64)
    elif vertices is not None and len(vertices):
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
    else:
        center = np.zeros(3, dtype=np.float64)
    if configured_scale is not None:
        rough_scale = float(configured_scale)
    elif vertices is not None and len(vertices):
        rough_scale = float(np.ptp(vertices, axis=0).max())
    else:
        raise CoarseGeometryError(
            "A coarse carving volume needs COARSE_VOLUME_SCALE when no initial-mesh volume hint is supplied."
        )
    if not np.isfinite(center).all() or not np.isfinite(rough_scale) or rough_scale <= 0:
        raise CoarseGeometryError("Coarse carving volume centre and scale must be finite and positive.")
    padding = float(getattr(cfg, "coarse_volume_padding_ratio", 0.25))
    if not np.isfinite(padding) or padding < 0:
        raise CoarseGeometryError("coarse_volume_padding_ratio must be finite and non-negative.")
    side = rough_scale * (1.0 + 2.0 * padding)
    return CarvingVolume(center - side / 2.0, center + side / 2.0, center, side)


def _remove_small_components(occupancy: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1 or not occupancy.any():
        return occupancy
    try:
        from scipy import ndimage
    except ImportError as exc:  # pragma: no cover - scipy is a project dependency
        raise RuntimeError("scipy is required for visual-hull component filtering.") from exc
    labels, count = ndimage.label(occupancy, structure=ndimage.generate_binary_structure(3, 1))
    if count == 0:
        return occupancy
    sizes = np.bincount(labels.ravel())
    keep = sizes >= min_voxels
    keep[0] = False
    return keep[labels]


def _sample_mask(mask: np.ndarray, projected: np.ndarray) -> np.ndarray:
    """Nearest-pixel alpha lookup, treating out-of-frame projections as background."""
    height, width = mask.shape
    x = np.rint(projected[:, 0]).astype(np.int64)
    y = np.rint(projected[:, 1]).astype(np.int64)
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    sampled = np.zeros(len(x), dtype=np.float32)
    sampled[valid] = mask[y[valid], x[valid]]
    return sampled


def carve_visual_hull(
    frames: Sequence[FrameRecord],
    layouts: Sequence[ImageLayout],
    camera: CameraParameters,
    convention: CameraConvention,
    volume: CarvingVolume,
    resolution: int,
    min_silhouette_support: float,
    alpha_threshold: float,
    theta_deltas: Sequence[float] | None = None,
    phi_deltas: Sequence[float] | None = None,
    min_component_voxels: int = 0,
    chunk_size: int = 262_144,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(occupancy, support_fraction)`` from frozen calibrated cameras.

    Projection intentionally duplicates :class:`FastSilhouetteRenderer` so the
    calibration's object-rotation and NDC principal-point conventions remain
    identical.  No optimization occurs here; camera values are read once.
    """
    if len(frames) == 0 or len(frames) != len(layouts):
        raise CoarseGeometryError("Visual hull needs one layout and alpha mask for every frame.")
    if resolution < 2:
        raise ValueError("Visual-hull resolution must be at least 2.")
    if not 0 < min_silhouette_support <= 1 or not 0 < alpha_threshold <= 1:
        raise ValueError("Silhouette support and alpha threshold must be in (0, 1].")
    if not np.isfinite([camera.distance, camera.fov_deg, camera.x_offset, camera.y_offset]).all():
        raise CoarseGeometryError("Frozen camera parameters must be finite.")
    if camera.distance <= 0 or not 0 < camera.fov_deg < 180:
        raise CoarseGeometryError("Frozen camera distance/FOV is invalid for visual hull carving.")

    theta_deltas = theta_deltas or [0.0] * len(frames)
    phi_deltas = phi_deltas or [0.0] * len(frames)
    if len(theta_deltas) != len(frames) or len(phi_deltas) != len(frames):
        raise ValueError("Camera pose delta lists must match the number of frames.")
    masks = [load_alpha_with_layout(frame.image_path, layout).numpy()[..., 0] for frame, layout in zip(frames, layouts)]
    rotations = [
        object_rotation(frame.theta_deg + theta, frame.phi_deg + phi, convention)
        for frame, theta, phi in zip(frames, theta_deltas, phi_deltas)
    ]
    focal = 1.0 / np.tan(np.deg2rad(camera.fov_deg) / 2.0)
    # Centres, not endpoints: the extracted isosurface can naturally reach the
    # padded volume boundary without inheriting the hint mesh's exact bounds.
    step = volume.side_length / resolution
    axis = volume.minimum[0] + (np.arange(resolution, dtype=np.float64) + 0.5) * step
    axes = [axis, volume.minimum[1] + (np.arange(resolution) + 0.5) * step,
            volume.minimum[2] + (np.arange(resolution) + 0.5) * step]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    support = np.zeros(len(grid), dtype=np.float32)
    for start in range(0, len(grid), chunk_size):
        points = grid[start : start + chunk_size]
        counts = np.zeros(len(points), dtype=np.float32)
        for mask, rotation in zip(masks, rotations):
            camera_points = points @ rotation
            z = camera_points[:, 2] + camera.distance
            valid_depth = z > 1e-6
            projected = np.empty((len(points), 2), dtype=np.float64)
            projected[:, 0] = ((focal * camera_points[:, 0] / np.maximum(z, 1e-6) + camera.x_offset) + 1.0) * 0.5 * mask.shape[1]
            projected[:, 1] = (1.0 - (focal * camera_points[:, 1] / np.maximum(z, 1e-6) + camera.y_offset)) * 0.5 * mask.shape[0]
            counts += (_sample_mask(mask, projected) >= alpha_threshold) & valid_depth
        support[start : start + len(points)] = counts / len(masks)
    support_grid = support.reshape((resolution, resolution, resolution))
    occupancy = _remove_small_components(support_grid >= min_silhouette_support, min_component_voxels)
    return occupancy, support_grid


def _fallback_voxel_surface(occupancy: np.ndarray, origin: np.ndarray, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Dependency-free block-surface fallback used when scikit-image is unavailable.

    The primary extractor below is Lewiner marching cubes.  This fallback
    preserves an image-derived, arbitrary-topology mesh in lean environments
    while retaining the same voxel-centre coordinate convention.
    """
    # Surface quads between occupied and empty voxels.  This is deliberately a
    # conservative fallback; installations with scikit-image use marching cubes.
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    index: dict[tuple[float, float, float], int] = {}
    directions = ((0, (-1, 0, 0)), (0, (1, 0, 0)), (1, (0, -1, 0)), (1, (0, 1, 0)), (2, (0, 0, -1)), (2, (0, 0, 1)))
    for cell in np.argwhere(occupancy):
        for axis, delta in directions:
            neighbour = cell + np.asarray(delta)
            if np.all((neighbour >= 0) & (neighbour < np.asarray(occupancy.shape))) and occupancy[tuple(neighbour)]:
                continue
            base = origin + cell * spacing - spacing * 0.5
            other = base + spacing
            coords = [[base[0], base[1], base[2]], [other[0], base[1], base[2]], [other[0], other[1], base[2]], [base[0], other[1], base[2]]]
            if axis == 0:
                x = other[0] if delta[0] > 0 else base[0]; coords = [[x, base[1], base[2]], [x, other[1], base[2]], [x, other[1], other[2]], [x, base[1], other[2]]]
            elif axis == 1:
                y = other[1] if delta[1] > 0 else base[1]; coords = [[base[0], y, base[2]], [other[0], y, base[2]], [other[0], y, other[2]], [base[0], y, other[2]]]
            else:
                z = other[2] if delta[2] > 0 else base[2]; coords = [[base[0], base[1], z], [other[0], base[1], z], [other[0], other[1], z], [base[0], other[1], z]]
            quad = []
            for point in coords:
                key = tuple(float(x) for x in point)
                if key not in index:
                    index[key] = len(vertices); vertices.append(np.asarray(point))
                quad.append(index[key])
            faces.extend([[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]])
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64)


def extract_marching_cubes_surface(occupancy: np.ndarray, volume: CarvingVolume, step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Extract an arbitrary-topology surface in the normalized volume frame."""
    if not occupancy.any() or occupancy.all():
        raise CoarseGeometryError("Visual hull is empty or fills the entire carving volume; adjust calibration/volume.")
    spacing = volume.side_length / occupancy.shape[0]
    origin = volume.minimum + spacing * 0.5
    try:
        from skimage.measure import marching_cubes
        vertices, faces, _, _ = marching_cubes(occupancy.astype(np.float32), level=0.5, spacing=(spacing,) * 3, step_size=step)
        return (vertices + origin).astype(np.float32), faces.astype(np.int64)
    except ImportError:
        return _fallback_voxel_surface(occupancy, origin, spacing)


def generate_coarse_geometry(
    contract: DatasetContract,
    camera: CameraParameters,
    convention: CameraConvention,
    cfg: Any,
    device: torch.device,
    initial_mesh_hint: Any | None = None,
    theta_deltas: Sequence[float] | None = None,
    phi_deltas: Sequence[float] | None = None,
) -> CoarseGeometryResult:
    """Generate an image-only coarse mesh from frozen calibration and RGBA masks.

    ``initial_mesh_hint`` is intentionally position-only and optional.  It must
    not be a ground-truth mesh and is never rendered, sampled, or used in any
    occupancy decision.
    """
    volume = derive_carving_volume(initial_mesh_hint, cfg)
    layouts = build_layouts(contract, int(cfg.max_image_dimension))
    occupancy, support = carve_visual_hull(
        contract.frames, layouts, camera, convention, volume,
        int(cfg.coarse_voxel_resolution), float(cfg.coarse_min_silhouette_support),
        float(cfg.coarse_occupancy_threshold), theta_deltas, phi_deltas,
        int(cfg.coarse_min_component_voxels),
    )
    vertices, faces = extract_marching_cubes_surface(occupancy, volume, int(cfg.coarse_marching_cubes_step))
    if len(vertices) == 0 or len(faces) == 0:
        raise CoarseGeometryError("Visual hull extraction produced no faces.")
    from .mesh_io import mesh_from_arrays
    mesh = mesh_from_arrays(torch.from_numpy(vertices), torch.from_numpy(faces), device)
    return CoarseGeometryResult(mesh, occupancy, volume, support, {
        "voxel_resolution": int(cfg.coarse_voxel_resolution),
        "occupied_voxels": int(occupancy.sum()),
        "vertex_count": int(len(vertices)), "face_count": int(len(faces)),
        "volume_min": volume.minimum.tolist(), "volume_max": volume.maximum.tolist(),
        "volume_side_length": float(volume.side_length),
    })
