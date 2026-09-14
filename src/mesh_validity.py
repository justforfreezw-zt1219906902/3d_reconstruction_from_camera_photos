"""Topology-agnostic validity diagnostics for image-derived triangle meshes.

The checks in this module intentionally use only a mesh's own vertices and
faces; no external reference geometry is accepted.  They are useful both for
visual-hull surfaces (whose topology is newly generated) and for their
fixed-topology refinement.

Self-intersection detection is conservative: pairs sharing a vertex are
excluded, and only intersections detected by triangle edge/plane tests are
reported.  Coplanar overlap is therefore not guaranteed to be found.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MeshValidityReport:
    """Serializable diagnostics for an arbitrary triangle mesh.

    Counts are deliberately kept separate from ``is_valid`` so callers can
    choose whether, for example, multiple disconnected components are allowed
    for a particular capture.  The default ``is_valid`` is conservative.
    """

    vertex_count: int
    face_count: int
    non_finite_vertex_count: int
    invalid_face_count: int
    degenerate_face_count: int
    extreme_edge_count: int
    disconnected_fragment_count: int
    small_component_count: int
    spike_vertex_count: int
    self_intersection_pair_count: int
    self_intersection_tested: bool
    median_edge_length: float
    max_edge_length: float
    has_non_finite_vertices: bool
    has_degenerate_faces: bool
    has_extreme_edges: bool
    has_disconnected_fragments: bool
    has_spikes: bool
    has_self_intersections: bool
    is_valid: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


def _as_arrays(mesh_or_vertices: Any, faces: Any | None) -> tuple[np.ndarray, np.ndarray]:
    """Accept PyTorch3D meshes or plain arrays without requiring PyTorch3D here."""
    if faces is None and hasattr(mesh_or_vertices, "verts_packed") and hasattr(mesh_or_vertices, "faces_packed"):
        vertices = mesh_or_vertices.verts_packed().detach().cpu().numpy()
        faces = mesh_or_vertices.faces_packed().detach().cpu().numpy()
    else:
        vertices = mesh_or_vertices.detach().cpu().numpy() if hasattr(mesh_or_vertices, "detach") else mesh_or_vertices
        faces = faces.detach().cpu().numpy() if hasattr(faces, "detach") else faces
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape [V, 3].")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("faces must have shape [F, 3].")
    if not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("faces must contain integer vertex indices.")
    return vertices, faces.astype(np.int64, copy=False)


def _face_components(faces: np.ndarray, vertex_count: int) -> list[int]:
    """Return face counts for components connected through shared vertices."""
    parent = np.arange(vertex_count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for face in faces:
        union(int(face[0]), int(face[1]))
        union(int(face[0]), int(face[2]))
    roots = np.fromiter((find(int(face[0])) for face in faces), dtype=np.int64, count=len(faces))
    _, counts = np.unique(roots, return_counts=True)
    return counts.tolist()


def _segment_intersects_triangle(
    start: np.ndarray, end: np.ndarray, triangle: np.ndarray, epsilon: float,
) -> bool:
    """Möller-Trumbore segment/triangle test, excluding coplanar ambiguity."""
    direction = end - start
    edge_a, edge_b = triangle[1] - triangle[0], triangle[2] - triangle[0]
    cross = np.cross(direction, edge_b)
    determinant = float(np.dot(edge_a, cross))
    if abs(determinant) <= epsilon:
        return False
    inverse = 1.0 / determinant
    offset = start - triangle[0]
    u = inverse * float(np.dot(offset, cross))
    if u < -epsilon or u > 1.0 + epsilon:
        return False
    q = np.cross(offset, edge_a)
    v = inverse * float(np.dot(direction, q))
    if v < -epsilon or u + v > 1.0 + epsilon:
        return False
    distance = inverse * float(np.dot(edge_b, q))
    return -epsilon <= distance <= 1.0 + epsilon


def _triangles_intersect(first: np.ndarray, second: np.ndarray, epsilon: float) -> bool:
    for triangle, other in ((first, second), (second, first)):
        for index in range(3):
            if _segment_intersects_triangle(triangle[index], triangle[(index + 1) % 3], other, epsilon):
                return True
    return False


def _self_intersection_count(vertices: np.ndarray, faces: np.ndarray, limit: int) -> tuple[int, bool]:
    """Detect non-adjacent triangle intersections with a sweep over AABBs.

    The candidate limit avoids turning diagnostics into an unbounded quadratic
    operation on pathological meshes.  ``tested=False`` means the count is a
    lower bound and callers should treat it as unavailable rather than clean.
    """
    if len(faces) < 2:
        return 0, True
    triangles = vertices[faces]
    minimum, maximum = triangles.min(axis=1), triangles.max(axis=1)
    order = np.argsort(minimum[:, 0], kind="stable")
    active: list[int] = []
    pairs_checked = intersections = 0
    epsilon = max(float(np.ptp(vertices, axis=0).max()) * 1e-10, 1e-12)
    for current in order:
        active = [other for other in active if maximum[other, 0] >= minimum[current, 0]]
        for other in active:
            if pairs_checked >= limit:
                return intersections, False
            if maximum[other, 1] < minimum[current, 1] or minimum[other, 1] > maximum[current, 1]:
                continue
            if maximum[other, 2] < minimum[current, 2] or minimum[other, 2] > maximum[current, 2]:
                continue
            if np.intersect1d(faces[current], faces[other]).size:
                continue
            pairs_checked += 1
            if _triangles_intersect(triangles[current], triangles[other], epsilon):
                intersections += 1
        active.append(int(current))
    return intersections, True


def inspect_mesh_validity(
    mesh_or_vertices: Any,
    faces: Any | None = None,
    *,
    max_edge_length_ratio: float = 10.0,
    spike_ratio: float = 8.0,
    min_component_faces: int = 32,
    degenerate_area_ratio: float = 1e-12,
    self_intersection_pair_limit: int = 100_000,
) -> MeshValidityReport:
    """Inspect mesh health without any reference geometry or topology assumption.

    ``max_edge_length_ratio`` and ``spike_ratio`` are measured against the
    mesh's own median positive edge length, making the diagnostics invariant to
    global scale and usable on topology that differs from the initializer.
    """
    if max_edge_length_ratio <= 0 or spike_ratio <= 0 or min_component_faces < 0:
        raise ValueError("edge/spike ratios must be positive and min_component_faces non-negative.")
    if degenerate_area_ratio < 0 or self_intersection_pair_limit < 0:
        raise ValueError("degenerate_area_ratio and self_intersection_pair_limit must be non-negative.")
    vertices, all_faces = _as_arrays(mesh_or_vertices, faces)
    finite_vertices = np.isfinite(vertices).all(axis=1)
    non_finite_vertex_count = int((~finite_vertices).sum())
    valid_indices = (all_faces >= 0).all(axis=1) & (all_faces < len(vertices)).all(axis=1)
    invalid_face_count = int((~valid_indices).sum())
    usable_faces = all_faces[valid_indices]
    repeated = np.array([len(set(face.tolist())) != 3 for face in usable_faces], dtype=bool)
    face_uses_non_finite = finite_vertices[usable_faces].all(axis=1) if len(usable_faces) else np.empty(0, dtype=bool)
    areas = np.zeros(len(usable_faces), dtype=np.float64)
    computable = (~repeated) & face_uses_non_finite
    if computable.any():
        triangles = vertices[usable_faces[computable]]
        areas[computable] = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    finite_points = vertices[finite_vertices]
    scale = float(np.ptp(finite_points, axis=0).max()) if len(finite_points) else 0.0
    area_limit = degenerate_area_ratio * max(scale * scale, np.finfo(np.float64).tiny)
    degenerate = repeated | ~face_uses_non_finite | (areas <= area_limit)
    degenerate_face_count = int(degenerate.sum()) + invalid_face_count
    healthy_faces = usable_faces[~degenerate]

    if len(healthy_faces):
        edges = np.concatenate((healthy_faces[:, [0, 1]], healthy_faces[:, [1, 2]], healthy_faces[:, [2, 0]]))
        edges.sort(axis=1)
        edges = np.unique(edges, axis=0)
        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
        positive = lengths[np.isfinite(lengths) & (lengths > 0)]
    else:
        edges, lengths, positive = np.empty((0, 2), dtype=np.int64), np.empty(0), np.empty(0)
    median_edge_length = float(np.median(positive)) if len(positive) else 0.0
    max_edge_length = float(np.max(positive)) if len(positive) else 0.0
    extreme = lengths > max_edge_length_ratio * median_edge_length if median_edge_length > 0 else np.zeros(len(lengths), dtype=bool)
    extreme_edge_count = int(extreme.sum())
    spike_vertices: set[int] = set()
    if median_edge_length > 0:
        for edge in edges[lengths > spike_ratio * median_edge_length]:
            # Both endpoints are suspicious; the mesh itself supplies the
            # scale baseline, not an external reference shape.
            spike_vertices.update((int(edge[0]), int(edge[1])))

    component_counts = _face_components(healthy_faces, len(vertices)) if len(healthy_faces) else []
    disconnected_fragment_count = max(0, len(component_counts) - 1)
    small_component_count = sum(count < min_component_faces for count in component_counts)
    intersection_count, intersection_tested = (
        _self_intersection_count(vertices, healthy_faces, self_intersection_pair_limit)
        if non_finite_vertex_count == 0 else (0, False)
    )
    has_non_finite = non_finite_vertex_count > 0
    has_degenerate = degenerate_face_count > 0
    has_extreme = extreme_edge_count > 0
    has_disconnected = disconnected_fragment_count > 0
    has_spikes = bool(spike_vertices)
    has_intersections = intersection_count > 0
    # An interrupted self-intersection scan is not evidence that the mesh is
    # clean, so the conservative aggregate result remains false in that case.
    is_valid = intersection_tested and not any(
        (has_non_finite, has_degenerate, has_extreme, has_disconnected, has_spikes, has_intersections)
    )
    return MeshValidityReport(
        len(vertices), len(all_faces), non_finite_vertex_count, invalid_face_count,
        degenerate_face_count, extreme_edge_count, disconnected_fragment_count,
        small_component_count, len(spike_vertices), intersection_count,
        intersection_tested, median_edge_length, max_edge_length, has_non_finite,
        has_degenerate, has_extreme, has_disconnected, has_spikes,
        has_intersections, is_valid,
    )
