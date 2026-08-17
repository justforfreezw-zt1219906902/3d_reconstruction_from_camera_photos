from __future__ import annotations

import json
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from pytorch3d.io import load_objs_as_meshes, save_obj
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


@dataclass
class MeshTransform:
    center: list[float]
    scale: float
    center_normalized: bool
    scale_normalized: bool
    original_bbox_min: list[float]
    original_bbox_max: list[float]
    original_bbox_diagonal: float
    normalized_object_size: float


def _deduplicate_triangles(vertices: list[tuple[float, float, float]]) -> tuple[torch.Tensor, torch.Tensor]:
    index_by_vertex: dict[tuple[float, float, float], int] = {}
    unique: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for start in range(0, len(vertices), 3):
        if start + 2 >= len(vertices):
            break
        face = []
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
    return torch.tensor(unique, dtype=torch.float32), torch.tensor(faces, dtype=torch.int64)


def _load_binary_stl(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError("Binary STL is shorter than the required 84-byte header.")
    count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + count * 50
    if expected > len(data):
        raise ValueError("Binary STL triangle count exceeds file size.")
    vertices: list[tuple[float, float, float]] = []
    offset = 84
    for _ in range(count):
        values = struct.unpack_from("<12fH", data, offset)
        vertices.extend([tuple(values[3:6]), tuple(values[6:9]), tuple(values[9:12])])
        offset += 50
    return _deduplicate_triangles(vertices)


def _load_ascii_stl(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    vertices: list[tuple[float, float, float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        tokens = line.strip().split()
        if len(tokens) == 4 and tokens[0].lower() == "vertex":
            vertices.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
    return _deduplicate_triangles(vertices)


def _load_stl(path: Path, device: torch.device) -> Meshes:
    data = path.read_bytes()
    is_binary = len(data) >= 84
    if is_binary:
        count = struct.unpack_from("<I", data, 80)[0]
        is_binary = 84 + count * 50 == len(data)
    vertices, faces = _load_binary_stl(path) if is_binary else _load_ascii_stl(path)
    verts = vertices.to(device)
    mesh = Meshes(verts=[verts], faces=[faces.to(device)])
    mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_padded()))
    return mesh


def _normalize_mesh(mesh: Meshes, center_normalize: bool, scale_normalize: bool) -> tuple[Meshes, MeshTransform]:
    verts = mesh.verts_packed()
    bbox_min = verts.min(dim=0).values
    bbox_max = verts.max(dim=0).values
    bbox_diagonal = float(torch.linalg.vector_norm(bbox_max - bbox_min).detach().cpu())
    center = verts.mean(0) if center_normalize else torch.zeros(3, device=verts.device)
    centered = verts - center
    scale_tensor = centered.abs().max() if scale_normalize else torch.tensor(1.0, device=verts.device)
    scale = float(scale_tensor.detach().cpu()) or 1.0
    normalized = centered / scale
    normalized_mesh = mesh.update_padded(normalized[None])
    transform = MeshTransform(
        center=[float(x) for x in center.detach().cpu()],
        scale=scale,
        center_normalized=center_normalize,
        scale_normalized=scale_normalize,
        original_bbox_min=[float(x) for x in bbox_min.detach().cpu()],
        original_bbox_max=[float(x) for x in bbox_max.detach().cpu()],
        original_bbox_diagonal=bbox_diagonal,
        normalized_object_size=bbox_diagonal / scale if scale else bbox_diagonal,
    )
    return normalized_mesh, transform


def load_initial_mesh(
    path: Path,
    device: torch.device,
    center_normalize: bool = True,
    scale_normalize: bool = True,
) -> tuple[Meshes, MeshTransform]:
    if not path.exists():
        raise FileNotFoundError(f"Initial mesh not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".stl":
        mesh = _load_stl(path, device)
    elif suffix == ".obj":
        mesh = load_objs_as_meshes([str(path)], device=device)
        if mesh.textures is None:
            mesh.textures = TexturesVertex(verts_features=torch.ones_like(mesh.verts_padded()))
    else:
        raise ValueError(f"Unsupported initial mesh format '{suffix}'. Use .stl or .obj.")
    return _normalize_mesh(mesh, center_normalize, scale_normalize)


def restore_vertices(verts: torch.Tensor, transform: MeshTransform) -> torch.Tensor:
    result = verts
    if transform.scale_normalized:
        result = result * transform.scale
    if transform.center_normalized:
        result = result + torch.tensor(transform.center, dtype=result.dtype, device=result.device)
    return result


def save_transform(transform: MeshTransform, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(transform), indent=2))


def export_mesh_obj(mesh: Meshes, path: Path, transform: MeshTransform | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = mesh.verts_packed().detach()
    if transform is not None:
        verts = restore_vertices(verts, transform)
    save_obj(str(path), verts, mesh.faces_packed().detach())


def export_mesh_stl(mesh: Meshes, path: Path, transform: MeshTransform | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = mesh.verts_packed().detach()
    if transform is not None:
        verts = restore_vertices(verts, transform)
    faces = mesh.faces_packed().detach().cpu().tolist()
    vertices = verts.cpu()
    with path.open("wb") as f:
        f.write(b"OpenScan reconstruction STL".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        for face in faces:
            a, b, c = (vertices[index].tolist() for index in face)
            normal = torch.linalg.cross(torch.tensor(b) - torch.tensor(a), torch.tensor(c) - torch.tensor(a))
            norm = torch.linalg.vector_norm(normal).item()
            normal_values = [value / norm for value in normal.tolist()] if norm else [0.0, 0.0, 0.0]
            f.write(struct.pack("<3f", *normal_values))
            f.write(struct.pack("<9f", *(a + b + c)))
            f.write(struct.pack("<H", 0))
