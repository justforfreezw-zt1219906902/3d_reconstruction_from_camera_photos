from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .pose_conventions import CameraConvention, DEFAULT_CONVENTION, object_rotation


@dataclass
class FastMesh:
    vertices: np.ndarray
    faces: np.ndarray


@dataclass
class FastMeshTransform:
    center: np.ndarray
    scale: float
    center_normalized: bool
    scale_normalized: bool


@dataclass
class CameraParameters:
    distance: float = 2.7
    fov_deg: float = 60.0
    x_offset: float = 0.0
    y_offset: float = 0.0

    def as_dict(self) -> dict:
        return {
            "distance": float(self.distance),
            "fov_deg": float(self.fov_deg),
            "principal_point_x_ndc": float(self.x_offset),
            "principal_point_y_ndc": float(self.y_offset),
        }


def _deduplicate_triangles(vertices: list[tuple[float, float, float]]) -> tuple[np.ndarray, np.ndarray]:
    index_by_vertex: dict[tuple[float, float, float], int] = {}
    unique: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for start in range(0, len(vertices), 3):
        if start + 2 >= len(vertices):
            break
        face: list[int] = []
        for vertex in vertices[start : start + 3]:
            key = tuple(round(float(value), 7) for value in vertex)
            if key not in index_by_vertex:
                index_by_vertex[key] = len(unique)
                unique.append(vertex)
            face.append(index_by_vertex[key])
        if len(set(face)) == 3:
            faces.append(face)
    if not faces:
        raise ValueError("STL contains no non-degenerate triangular faces.")
    return np.asarray(unique, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def _read_stl(path: Path) -> FastMesh:
    data = path.read_bytes()
    is_binary = len(data) >= 84
    if is_binary:
        count = struct.unpack_from("<I", data, 80)[0]
        is_binary = 84 + count * 50 == len(data)
    vertices: list[tuple[float, float, float]] = []
    if is_binary:
        count = struct.unpack_from("<I", data, 80)[0]
        offset = 84
        for _ in range(count):
            values = struct.unpack_from("<12fH", data, offset)
            vertices.extend([tuple(values[3:6]), tuple(values[6:9]), tuple(values[9:12])])
            offset += 50
    else:
        for line in data.decode(errors="ignore").splitlines():
            tokens = line.strip().split()
            if len(tokens) == 4 and tokens[0].lower() == "vertex":
                vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
    verts, faces = _deduplicate_triangles(vertices)
    return FastMesh(verts, faces)


def normalize_fast_mesh(
    mesh: FastMesh,
    center_normalize: bool = True,
    scale_normalize: bool = True,
) -> tuple[FastMesh, FastMeshTransform]:
    center = mesh.vertices.mean(axis=0) if center_normalize else np.zeros(3, dtype=np.float64)
    centered = mesh.vertices - center
    scale = float(np.max(np.abs(centered))) if scale_normalize else 1.0
    scale = scale or 1.0
    return (
        FastMesh(centered / scale, mesh.faces.copy()),
        FastMeshTransform(center, scale, center_normalize, scale_normalize),
    )


def load_fast_mesh(path: Path, center_normalize: bool = True, scale_normalize: bool = True) -> tuple[FastMesh, FastMeshTransform]:
    suffix = path.suffix.lower()
    if suffix != ".stl":
        raise ValueError("Fast Camera Fit currently requires an STL initial mesh.")
    return normalize_fast_mesh(_read_stl(path), center_normalize, scale_normalize)


def _fallback_proxy(mesh: FastMesh, max_faces: int) -> FastMesh:
    if len(mesh.faces) <= max_faces:
        return FastMesh(mesh.vertices.copy(), mesh.faces.copy())
    centroids = mesh.vertices[mesh.faces].mean(axis=1)
    extent = np.ptp(mesh.vertices, axis=0).max()
    grid_side = max(2, int(round((max_faces / max(len(mesh.faces), 1)) ** (-1.0 / 3.0) * 16)))
    bins = np.floor((centroids - mesh.vertices.min(axis=0)) / max(extent, 1e-8) * grid_side).astype(np.int64)
    _, unique_indices = np.unique(bins, axis=0, return_index=True)
    selected = np.sort(unique_indices)
    if len(selected) < max_faces:
        extra = np.linspace(0, len(mesh.faces) - 1, max_faces, dtype=np.int64)
        selected = np.unique(np.concatenate([selected, extra]))[:max_faces]
    return FastMesh(mesh.vertices.copy(), mesh.faces[selected])


def create_camera_proxy(mesh: FastMesh, max_faces: int) -> tuple[FastMesh, str]:
    if len(mesh.faces) <= max_faces:
        return FastMesh(mesh.vertices.copy(), mesh.faces.copy()), "original_mesh"
    try:
        import open3d as o3d

        source = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(mesh.vertices),
            o3d.utility.Vector3iVector(mesh.faces),
        )
        simplified = source.simplify_quadric_decimation(target_number_of_triangles=max_faces)
        simplified.remove_degenerate_triangles()
        simplified.remove_duplicated_triangles()
        # Open3D arrays are views into the temporary mesh. Copy them before the
        # temporary object is released, otherwise later projections read freed memory.
        proxy = FastMesh(
            np.asarray(simplified.vertices).copy(),
            np.asarray(simplified.triangles, dtype=np.int32).copy(),
        )
        if len(proxy.faces) > 0:
            return proxy, "open3d_quadric_decimation"
    except Exception:
        pass
    return _fallback_proxy(mesh, max_faces), "deterministic_face_voxel_fallback"


def save_fast_mesh_stl(mesh: FastMesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"OpenScan camera-fit proxy".ljust(80, b" "))
        handle.write(struct.pack("<I", len(mesh.faces)))
        for i, j, k in mesh.faces:
            a, b, c = mesh.vertices[[i, j, k]]
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm:
                normal = normal / norm
            handle.write(struct.pack("<3f", *normal.astype(np.float32)))
            handle.write(struct.pack("<9f", *(a.tolist() + b.tolist() + c.tolist())))
            handle.write(struct.pack("<H", 0))


class FastSilhouetteRenderer:
    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)
        self.render_count = 0
        self.total_render_seconds = 0.0

    def render(
        self,
        mesh: FastMesh,
        camera: CameraParameters,
        theta_deg: float,
        phi_deg: float,
        convention: CameraConvention = DEFAULT_CONVENTION,
    ) -> np.ndarray:
        started = time.perf_counter()
        rotation = object_rotation(theta_deg, phi_deg, convention)
        vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            camera_vertices = vertices @ rotation
        if not np.isfinite(camera_vertices).all():
            raise ValueError("Fast silhouette projection produced non-finite camera vertices.")
        camera_vertices[:, 2] += camera.distance
        valid = camera_vertices[:, 2] > 1e-6
        focal = 1.0 / np.tan(np.deg2rad(camera.fov_deg) / 2.0)
        width = height = self.image_size
        projected = np.zeros((len(camera_vertices), 2), dtype=np.float64)
        projected[:, 0] = ((focal * camera_vertices[:, 0] / np.maximum(camera_vertices[:, 2], 1e-6) + camera.x_offset) + 1.0) * 0.5 * width
        projected[:, 1] = (1.0 - (focal * camera_vertices[:, 1] / np.maximum(camera_vertices[:, 2], 1e-6) + camera.y_offset)) * 0.5 * height
        mask = np.zeros((height, width), dtype=np.float32)
        for face in mesh.faces:
            if not valid[face].all():
                continue
            polygon = np.rint(projected[face]).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, 1.0)
        self.render_count += 1
        self.total_render_seconds += time.perf_counter() - started
        return mask


def silhouette_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    predicted = predicted > 0.5
    target = target > 0.05
    intersection = np.logical_and(predicted, target).sum()
    union = np.logical_or(predicted, target).sum()
    return float(intersection / max(union, 1))


def silhouette_summary(mask: np.ndarray) -> dict[str, float]:
    ys, xs = np.where(mask > 0.05)
    if len(xs) == 0:
        return {"area": 0.0, "centroid_x": 0.0, "centroid_y": 0.0, "width": 0.0, "height": 0.0}
    return {
        "area": float(len(xs)),
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
        "width": float(xs.max() - xs.min() + 1),
        "height": float(ys.max() - ys.min() + 1),
    }
