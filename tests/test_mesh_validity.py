from __future__ import annotations

import numpy as np
import torch

from src.mesh_io import inspect_mesh_validity as inspect_pytorch3d_mesh
from src.mesh_io import mesh_from_arrays
from src.mesh_validity import inspect_mesh_validity


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]),
        np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]),
    )


def test_clean_arbitrary_topology_mesh_is_valid_and_pytorch3d_wrapper_matches() -> None:
    vertices, faces = _tetrahedron()
    report = inspect_mesh_validity(vertices, faces, min_component_faces=1)
    mesh = mesh_from_arrays(torch.from_numpy(vertices).float(), torch.from_numpy(faces), torch.device("cpu"))
    wrapped = inspect_pytorch3d_mesh(mesh, min_component_faces=1)

    assert report.is_valid
    assert report.as_dict() == wrapped.as_dict()
    assert report.degenerate_face_count == report.extreme_edge_count == 0
    assert report.disconnected_fragment_count == report.spike_vertex_count == 0


def test_reports_nonfinite_degenerate_extreme_edges_spikes_and_disconnected_fragments() -> None:
    vertices, faces = _tetrahedron()
    vertices = np.vstack((vertices, [[100., 1., 0.], [np.nan, 0., 0.], [10., 0., 0.], [11., 0., 0.], [10., 1., 0.]]))
    faces = np.vstack((
        faces,
        [[0, 1, 4]],  # Long edge / catastrophic spike from the tetrahedron.
        [[0, 0, 1]],  # Repeated index / degenerate.
        [[6, 7, 8]],  # Disconnected triangle fragment.
    ))
    report = inspect_mesh_validity(vertices, faces, max_edge_length_ratio=4.0, spike_ratio=4.0, min_component_faces=2)

    assert report.non_finite_vertex_count == 1
    assert report.has_non_finite_vertices
    assert report.degenerate_face_count >= 1 and report.has_degenerate_faces
    assert report.extreme_edge_count >= 1 and report.has_extreme_edges
    assert report.spike_vertex_count >= 1 and report.has_spikes
    assert report.disconnected_fragment_count == 1 and report.has_disconnected_fragments
    assert report.small_component_count >= 1
    assert not report.is_valid


def test_detects_non_adjacent_triangle_self_intersection() -> None:
    vertices = np.array([
        [-1., -1., 0.], [1., -1., 0.], [0., 1., 0.],
        [0., 0., -1.], [0., 0., 1.], [0.5, 0.5, 0.],
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    report = inspect_mesh_validity(vertices, faces, min_component_faces=1)

    assert report.self_intersection_tested
    assert report.self_intersection_pair_count == 1
    assert report.has_self_intersections
    assert not report.is_valid


def test_report_has_no_external_reference_geometry_argument() -> None:
    vertices, faces = _tetrahedron()
    report = inspect_mesh_validity(vertices, faces, self_intersection_pair_limit=0, min_component_faces=1)

    assert report.self_intersection_pair_count == 0
    assert not report.self_intersection_tested
    assert not report.is_valid
    # The public API takes only mesh vertices/faces and local thresholds.
    assert report.vertex_count == len(vertices)
