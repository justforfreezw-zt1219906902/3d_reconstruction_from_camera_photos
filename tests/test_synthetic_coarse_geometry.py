"""End-to-end visual-hull regression using a topology-changing synthetic target.

The torus is deliberately used only to rasterize fixture masks and to score the
result afterwards.  ``generate_coarse_geometry`` sees just those masks, frozen
camera calibration, and a sphere's position/scale hint.
"""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree

from src.coarse_geometry import generate_coarse_geometry
from src.fast_silhouette import CameraParameters, FastMesh
from src.pose_conventions import DEFAULT_CONVENTION, object_rotation
from src.validation import FrameRecord


IMAGE_SIZE = 160
CAMERA = CameraParameters(distance=4.0, fov_deg=55.0)


def _torus_mesh(major_radius=0.68, minor_radius=0.27, major_steps=64, minor_steps=32):
    """Return a watertight genus-one torus whose hole is clear in frontal views."""
    u = np.arange(major_steps) * (2.0 * np.pi / major_steps)
    v = np.arange(minor_steps) * (2.0 * np.pi / minor_steps)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    vertices = np.column_stack((
        (major_radius + minor_radius * np.cos(vv)).ravel() * np.cos(uu).ravel(),
        (major_radius + minor_radius * np.cos(vv)).ravel() * np.sin(uu).ravel(),
        (minor_radius * np.sin(vv)).ravel(),
    ))
    faces = []
    for i in range(major_steps):
        for j in range(minor_steps):
            a = i * minor_steps + j
            b = ((i + 1) % major_steps) * minor_steps + j
            c = ((i + 1) % major_steps) * minor_steps + (j + 1) % minor_steps
            d = i * minor_steps + (j + 1) % minor_steps
            faces.extend(((a, b, c), (a, c, d)))
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.int64)


def _sphere_hint(radius=0.95, latitude_steps=16, longitude_steps=32):
    vertices = [[0.0, radius, 0.0], [0.0, -radius, 0.0]]
    for lat in range(1, latitude_steps):
        phi = np.pi * lat / latitude_steps
        for lon in range(longitude_steps):
            theta = 2.0 * np.pi * lon / longitude_steps
            vertices.append((radius * np.sin(phi) * np.cos(theta), radius * np.cos(phi), radius * np.sin(phi) * np.sin(theta)))
    faces = []
    for lon in range(longitude_steps):
        nxt = (lon + 1) % longitude_steps
        faces.append((0, 2 + lon, 2 + nxt))
        faces.append((1, 2 + (latitude_steps - 2) * longitude_steps + nxt, 2 + (latitude_steps - 2) * longitude_steps + lon))
    for lat in range(latitude_steps - 2):
        row = 2 + lat * longitude_steps
        following = row + longitude_steps
        for lon in range(longitude_steps):
            nxt = (lon + 1) % longitude_steps
            faces.extend(((row + lon, following + lon, following + nxt), (row + lon, following + nxt, row + nxt)))
    return FastMesh(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int64))


def _rasterized_alpha(vertices, faces, theta_deg, phi_deg):
    """Rasterize a target mesh solely to construct the observed alpha fixture."""
    rotated = vertices @ object_rotation(theta_deg, phi_deg, DEFAULT_CONVENTION)
    focal = 1.0 / np.tan(np.deg2rad(CAMERA.fov_deg) / 2.0)
    z = rotated[:, 2] + CAMERA.distance
    projected = np.column_stack((
        (focal * rotated[:, 0] / z + 1.0) * IMAGE_SIZE * 0.5,
        (1.0 - focal * rotated[:, 1] / z) * IMAGE_SIZE * 0.5,
    ))
    alpha = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    # Unioning all projected triangles produces the opaque-object silhouette;
    # it retains the torus aperture in the front/back views.
    for triangle in projected[faces]:
        cv2.fillConvexPoly(alpha, np.rint(triangle).astype(np.int32), 255, lineType=cv2.LINE_8)
    return alpha


def _euler_characteristic(faces):
    faces = np.asarray(faces, dtype=np.int64)
    edges = np.sort(
        np.concatenate((faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0),
        axis=1,
    )
    return int(np.unique(faces).size - len(np.unique(edges, axis=0)) + len(faces))


def _symmetric_surface_distance(points_a, points_b):
    """Topology-agnostic symmetric nearest-surface sample distance."""
    a_to_b = cKDTree(points_b).query(points_a)[0].mean()
    b_to_a = cKDTree(points_a).query(points_b)[0].mean()
    return float((a_to_b + b_to_a) / 2.0)


def test_visual_hull_reconstructs_torus_topology_without_using_target_as_input(tmp_path):
    target_vertices, target_faces = _torus_mesh()
    # Two face-on views preserve the hole; the six orthogonal/oblique views
    # constrain the hull enough to avoid a merely flat annulus.
    poses = ((0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (270.0, 0.0), (0.0, 90.0), (0.0, -90.0))
    frames = []
    for index, (theta_deg, phi_deg) in enumerate(poses):
        path = tmp_path / f"torus_{index}.png"
        alpha = _rasterized_alpha(target_vertices, target_faces, theta_deg, phi_deg)
        Image.fromarray(np.dstack((np.zeros_like(alpha), np.zeros_like(alpha), np.zeros_like(alpha), alpha)), "RGBA").save(path)
        frames.append(FrameRecord(path.name, path, index, phi_deg, theta_deg, IMAGE_SIZE, IMAGE_SIZE, 1.0))

    cfg = SimpleNamespace(
        coarse_volume_center=None,
        coarse_volume_scale=None,
        coarse_volume_padding_ratio=0.35,
        max_image_dimension=IMAGE_SIZE,
        coarse_voxel_resolution=64,
        coarse_min_silhouette_support=1.0,
        coarse_occupancy_threshold=0.5,
        coarse_min_component_voxels=12,
        coarse_marching_cubes_step=1,
    )
    sphere_hint = _sphere_hint()
    # The target mesh intentionally does not appear in this call: the coarse
    # stage gets only observed masks, calibration, and the sphere position hint.
    result = generate_coarse_geometry(
        SimpleNamespace(frames=tuple(frames)), CAMERA, DEFAULT_CONVENTION, cfg,
        torch.device("cpu"), initial_mesh_hint=sphere_hint,
    )
    coarse_vertices = result.mesh.verts_packed().detach().cpu().numpy()
    coarse_faces = result.mesh.faces_packed().detach().cpu().numpy()

    # A closed orientable sphere has Euler characteristic 2; the recovered
    # toroidal visual hull must retain a handle (Euler characteristic 0).
    assert _euler_characteristic(sphere_hint.faces) == 2
    assert _euler_characteristic(coarse_faces) == 0
    assert len(coarse_faces) != len(sphere_hint.faces)
    assert np.isfinite(coarse_vertices).all()

    # This is evaluated only after reconstruction.  The image-derived mesh
    # must improve substantially over the generic spherical initialization.
    sphere_error = _symmetric_surface_distance(sphere_hint.vertices, target_vertices)
    coarse_error = _symmetric_surface_distance(coarse_vertices, target_vertices)
    assert coarse_error < sphere_error * 0.70, (coarse_error, sphere_error)
