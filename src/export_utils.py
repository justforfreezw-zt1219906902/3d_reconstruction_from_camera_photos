from __future__ import annotations

from pathlib import Path

import torch
from pytorch3d.io import save_obj
from pytorch3d.structures import Meshes

from .mesh_io import MeshTransform, restore_vertices


def _write_simple_mtl(path: Path) -> None:
    path.write_text(
        "newmtl reconstructed_material\n"
        "Ka 1.000000 1.000000 1.000000\n"
        "Kd 1.000000 1.000000 1.000000\n"
        "Ks 0.000000 0.000000 0.000000\n"
        "d 1.0\n"
        "illum 2\n"
    )


def _add_mtl_reference(obj_path: Path, mtl_name: str) -> None:
    text = obj_path.read_text()
    if text.startswith("mtllib "):
        return
    lines = text.splitlines()
    out: list[str] = [f"mtllib {mtl_name}"]
    inserted_usemtl = False
    for line in lines:
        if not inserted_usemtl and line.startswith("f "):
            out.append("usemtl reconstructed_material")
            inserted_usemtl = True
        out.append(line)
    obj_path.write_text("\n".join(out) + "\n")


def export_mesh_obj(mesh: Meshes, path: Path, transform: MeshTransform | None = None, restore: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    verts = mesh.verts_packed().detach()
    if restore and transform is not None:
        verts = restore_vertices(verts, transform)
    faces = mesh.faces_packed().detach()
    verts_rgb = None
    if mesh.textures is not None and hasattr(mesh.textures, "verts_features_packed"):
        try:
            verts_rgb = mesh.textures.verts_features_packed().detach().clamp(0, 1)
        except Exception:
            verts_rgb = None
    if verts_rgb is not None:
        try:
            save_obj(str(path), verts, faces, verts_rgb=verts_rgb)
        except TypeError:
            print("Warning: this PyTorch3D save_obj does not support vertex colors; exporting geometry only.")
            save_obj(str(path), verts, faces)
    else:
        save_obj(str(path), verts, faces)
    mtl_path = path.with_suffix(".mtl")
    _write_simple_mtl(mtl_path)
    _add_mtl_reference(path, mtl_path.name)


def save_checkpoint(path: Path, verts_offsets: torch.Tensor, texture: torch.Tensor | None, iteration: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"iteration": iteration, "verts_offsets": verts_offsets.detach().cpu()}
    if texture is not None:
        payload["texture"] = texture.detach().cpu()
    torch.save(payload, path)
