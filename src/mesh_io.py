from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes


@dataclass
class MeshTransform:
    center: list[float]
    scale: float
    center_normalized: bool
    scale_normalized: bool


def load_original_mesh(
    obj_path: Path,
    device: torch.device,
    center_normalize: bool = True,
    scale_normalize: bool = True,
) -> tuple[Meshes, MeshTransform]:
    if not obj_path.exists():
        raise FileNotFoundError(f"Original OBJ not found: {obj_path}")
    mesh = load_objs_as_meshes([str(obj_path)], device=device)
    verts = mesh.verts_packed()
    center = verts.mean(0) if center_normalize else torch.zeros(3, device=device)
    centered = verts - center
    scale = centered.abs().max() if scale_normalize else torch.tensor(1.0, device=device)
    if float(scale.detach().cpu()) <= 0:
        scale = torch.tensor(1.0, device=device)
    new_verts = centered / scale
    mesh = mesh.update_padded(new_verts[None])
    _ensure_vertex_texture(mesh)
    transform = MeshTransform(
        center=[float(x) for x in center.detach().cpu()],
        scale=float(scale.detach().cpu()),
        center_normalized=center_normalize,
        scale_normalized=scale_normalize,
    )
    return mesh, transform


def _ensure_vertex_texture(mesh: Meshes) -> Meshes:
    if mesh.textures is None:
        verts_rgb = torch.ones_like(mesh.verts_padded())
        mesh.textures = TexturesVertex(verts_features=verts_rgb)
    return mesh


def restore_vertices(verts: torch.Tensor, transform: MeshTransform) -> torch.Tensor:
    out = verts
    if transform.scale_normalized:
        out = out * transform.scale
    if transform.center_normalized:
        out = out + torch.tensor(transform.center, dtype=out.dtype, device=out.device)
    return out


def save_transform(transform: MeshTransform, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(transform), indent=2))
