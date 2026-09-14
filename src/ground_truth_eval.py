"""Post-reconstruction ground-truth mesh evaluation.

This module is deliberately outside camera fitting, carving, optimisation, and
training.  In particular, it does not import any of those modules and the
ground-truth path is consumed only by :func:`evaluate_reconstruction_against_ground_truth`.
Call that public entry point after all reconstruction outputs have been chosen
and exported.  Its primary metrics compare meshes in their supplied physical
coordinate frame; optional rigid alignment is explicitly diagnostic only.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from pytorch3d.io import load_obj
from scipy.spatial import cKDTree


class GroundTruthEvaluationError(RuntimeError):
    """Raised when a mesh cannot be evaluated reliably."""


def _load_ground_truth_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a raw, *non-normalized* OBJ or STL mesh.

    This private loader is intentionally called only by the public evaluation
    entry point, so evaluation is the sole owner of ground-truth file access.
    """
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth mesh not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".obj":
        vertices, faces, _ = load_obj(str(path), load_textures=False)
        face_indices = faces.verts_idx
        return vertices.detach().cpu().numpy(), face_indices.detach().cpu().numpy()
    if suffix == ".stl":
        data = path.read_bytes()
        is_binary = len(data) >= 84 and 84 + int.from_bytes(data[80:84], "little") * 50 == len(data)
        raw_vertices: list[tuple[float, float, float]] = []
        if is_binary:
            for offset in range(84, len(data), 50):
                values = struct.unpack_from("<12fH", data, offset)
                raw_vertices.extend((tuple(values[3:6]), tuple(values[6:9]), tuple(values[9:12])))
        else:
            for line in data.decode(errors="ignore").splitlines():
                tokens = line.strip().split()
                if len(tokens) == 4 and tokens[0].lower() == "vertex":
                    raw_vertices.append(tuple(float(value) for value in tokens[1:]))
        index_by_vertex: dict[tuple[float, float, float], int] = {}
        vertices: list[tuple[float, float, float]] = []
        faces: list[list[int]] = []
        for start in range(0, len(raw_vertices) - 2, 3):
            face: list[int] = []
            for vertex in raw_vertices[start : start + 3]:
                key = tuple(round(value, 7) for value in vertex)
                if key not in index_by_vertex:
                    index_by_vertex[key] = len(vertices)
                    vertices.append(vertex)
                face.append(index_by_vertex[key])
            if len(set(face)) == 3:
                faces.append(face)
        return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    raise ValueError(f"Unsupported ground-truth mesh format '{suffix}'. Use .obj or .stl.")


def _as_arrays(mesh: Any, transform: Any | None) -> tuple[np.ndarray, np.ndarray]:
    """Extract triangles from PyTorch3D meshes or ``(vertices, faces)`` pairs."""
    if hasattr(mesh, "verts_packed") and hasattr(mesh, "faces_packed"):
        vertices = mesh.verts_packed().detach().cpu().numpy()
        faces = mesh.faces_packed().detach().cpu().numpy()
    elif isinstance(mesh, (tuple, list)) and len(mesh) == 2:
        vertices, faces = mesh
        vertices, faces = np.asarray(vertices), np.asarray(faces)
    else:
        raise TypeError("Each reconstructed mesh must be a PyTorch3D mesh or (vertices, faces) pair.")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if transform is not None:
        if getattr(transform, "scale_normalized", False):
            vertices = vertices * float(transform.scale)
        if getattr(transform, "center_normalized", False):
            vertices = vertices + np.asarray(transform.center, dtype=np.float64)
    _validate_mesh(vertices, faces)
    return vertices, faces


def _validate_mesh(vertices: np.ndarray, faces: np.ndarray) -> None:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise GroundTruthEvaluationError("Mesh vertices must be a non-empty [N, 3] array.")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise GroundTruthEvaluationError("Mesh faces must be a non-empty [M, 3] triangle array.")
    if not np.isfinite(vertices).all() or faces.min() < 0 or faces.max() >= len(vertices):
        raise GroundTruthEvaluationError("Mesh has non-finite vertices or invalid face indices.")
    triangles = vertices[faces]
    if np.all(np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) == 0):
        raise GroundTruthEvaluationError("Mesh contains no non-degenerate triangles.")


def _sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    double_areas = np.linalg.norm(crosses, axis=1)
    valid = double_areas > np.finfo(np.float64).eps
    triangles, crosses, double_areas = triangles[valid], crosses[valid], double_areas[valid]
    if len(triangles) == 0:
        raise GroundTruthEvaluationError("Mesh contains no sampleable triangles.")
    indices = rng.choice(len(triangles), size=count, p=double_areas / double_areas.sum())
    selected = triangles[indices]
    u = rng.random(count)
    v = rng.random(count)
    reflected = u + v > 1.0
    u[reflected] = 1.0 - u[reflected]
    v[reflected] = 1.0 - v[reflected]
    points = selected[:, 0] + u[:, None] * (selected[:, 1] - selected[:, 0]) + v[:, None] * (selected[:, 2] - selected[:, 0])
    normals = crosses[indices] / double_areas[indices, None]
    return points, normals


def _nearest(source: np.ndarray, target: np.ndarray, target_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = cKDTree(target).query(source, k=1)
    return np.asarray(distances, dtype=np.float64), target_normals[indices]


def _metric_summary(source_points: np.ndarray, source_normals: np.ndarray, target_points: np.ndarray, target_normals: np.ndarray, diagonal: float) -> dict[str, float]:
    forward, nearest_target_normals = _nearest(source_points, target_points, target_normals)
    backward, nearest_source_normals = _nearest(target_points, source_points, source_normals)
    normal_forward = np.abs(np.einsum("ij,ij->i", source_normals, nearest_target_normals))
    normal_backward = np.abs(np.einsum("ij,ij->i", target_normals, nearest_source_normals))
    all_distances = np.concatenate((forward, backward))
    chamfer = float((np.mean(forward ** 2) + np.mean(backward ** 2)) / 2.0)
    mean = float((np.mean(forward) + np.mean(backward)) / 2.0)
    robust_hausdorff = float(np.percentile(all_distances, 95.0))
    primary = {
        "symmetric_chamfer_distance": chamfer,
        "mean_surface_distance": mean,
        "mean_reconstruction_to_ground_truth": float(np.mean(forward)),
        "mean_ground_truth_to_reconstruction": float(np.mean(backward)),
        "p95_surface_distance": robust_hausdorff,
        "hausdorff_distance": float(np.max(all_distances)),
        "robust_hausdorff_distance": robust_hausdorff,
        "normal_consistency": float((np.mean(normal_forward) + np.mean(normal_backward)) / 2.0),
    }
    normalized = {
        key: (value / diagonal ** 2 if key == "symmetric_chamfer_distance" else value / diagonal)
        for key, value in primary.items()
        if key != "normal_consistency"
    }
    normalized["normal_consistency"] = primary["normal_consistency"]
    return {"physical": primary, "normalized_by_gt_bbox_diagonal": normalized}


def _rigid_transform(source: np.ndarray, target: np.ndarray, iterations: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic point-to-point ICP, returning rotation and translation."""
    rotation = np.eye(3)
    translation = np.zeros(3)
    tree = cKDTree(target)
    for _ in range(iterations):
        moved = source @ rotation.T + translation
        _, indices = tree.query(moved, k=1)
        matches = target[indices]
        source_center, target_center = moved.mean(axis=0), matches.mean(axis=0)
        covariance = (moved - source_center).T @ (matches - target_center)
        u, _, vt = np.linalg.svd(covariance)
        update = vt.T @ u.T
        if np.linalg.det(update) < 0:
            vt[-1] *= -1
            update = vt.T @ u.T
        rotation = update @ rotation
        translation = (translation - source_center) @ update.T + target_center
    return rotation, translation


def _diagnostic_rigid_metrics(source_points: np.ndarray, source_normals: np.ndarray, target_points: np.ndarray, target_normals: np.ndarray, diagonal: float) -> dict[str, Any]:
    rotation, translation = _rigid_transform(source_points, target_points)
    aligned_points = source_points @ rotation.T + translation
    aligned_normals = source_normals @ rotation.T
    return {
        "label": "diagnostic_only_rigid_aligned_not_primary",
        "metrics": _metric_summary(aligned_points, aligned_normals, target_points, target_normals, diagonal),
    }


def evaluate_reconstruction_against_ground_truth(
    ground_truth_mesh_path: str | Path,
    sphere_mesh: Any,
    coarse_mesh: Any,
    final_mesh: Any,
    *,
    transforms: Mapping[str, Any | None] | None = None,
    sample_count: int = 20_000,
    seed: int = 42,
    include_rigid_aligned_diagnostic: bool = False,
) -> dict[str, Any]:
    """Evaluate sphere, coarse, and final meshes after reconstruction finishes.

    The result is JSON-serializable. ``physical`` is the primary, unaligned
    error in the supplied mesh frame.  Values in
    ``normalized_by_gt_bbox_diagonal`` divide linear distances by the ground
    truth bounding-box diagonal and the squared Chamfer value by its square.
    ``rigid_aligned_diagnostic`` is optional and must not be used for model
    selection or reported as the primary error.
    """
    if sample_count < 1:
        raise ValueError("sample_count must be positive.")
    # This is the only ground-truth load in the module, and it happens only
    # after callers provide all completed reconstruction stage meshes.
    gt_vertices, gt_faces = _load_ground_truth_mesh(Path(ground_truth_mesh_path))
    _validate_mesh(gt_vertices, gt_faces)
    diagonal = float(np.linalg.norm(gt_vertices.max(axis=0) - gt_vertices.min(axis=0)))
    if not np.isfinite(diagonal) or diagonal <= np.finfo(np.float64).eps:
        raise GroundTruthEvaluationError("Ground-truth bounding-box diagonal must be positive.")
    transforms = transforms or {}
    stage_meshes = {"sphere": sphere_mesh, "coarse": coarse_mesh, "final": final_mesh}
    rng = np.random.default_rng(seed)
    gt_points, gt_normals = _sample_surface(gt_vertices, gt_faces, sample_count, rng)
    stages: dict[str, Any] = {}
    for name, mesh in stage_meshes.items():
        vertices, faces = _as_arrays(mesh, transforms.get(name))
        points, normals = _sample_surface(vertices, faces, sample_count, rng)
        result: dict[str, Any] = _metric_summary(points, normals, gt_points, gt_normals, diagonal)
        if include_rigid_aligned_diagnostic:
            result["rigid_aligned_diagnostic"] = _diagnostic_rigid_metrics(
                points, normals, gt_points, gt_normals, diagonal
            )
        stages[name] = result
    return {
        "ground_truth_mesh_path": str(ground_truth_mesh_path),
        "ground_truth_bbox_diagonal_physical": diagonal,
        "sample_count_per_mesh": sample_count,
        "primary_metric_frame": "raw_physical_unaligned",
        "stages": stages,
    }
