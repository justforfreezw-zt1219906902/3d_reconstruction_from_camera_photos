from __future__ import annotations

import struct
from pathlib import Path

import torch

from src.mesh_io import export_mesh_stl, load_initial_mesh, restore_vertices


def _write_triangle_stl(path: Path) -> None:
    with path.open("wb") as f:
        f.write(b"triangle".ljust(80, b" "))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<3f", 0.0, 0.0, 1.0))
        f.write(struct.pack("<9f", 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0))
        f.write(struct.pack("<H", 0))


def test_stl_load_normalize_restore_and_export(tmp_path: Path) -> None:
    source = tmp_path / "source.stl"
    _write_triangle_stl(source)
    mesh, transform = load_initial_mesh(source, torch.device("cpu"))
    restored = restore_vertices(mesh.verts_packed(), transform)
    expected = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert torch.allclose(restored, expected, atol=1e-5)
    output = tmp_path / "final.stl"
    export_mesh_stl(mesh, output, transform)
    assert output.exists()
