# 3D Reconstruction Project

## Project

This repository reconstructs the geometry of a real printed object from
multi-view RGBA photographs.

The original CAD/STL is a design prior, not ground-truth geometry of the
printed object.

The current local repository state is the authoritative implementation state.

## Project constraints

Preserve the existing external interface unless a task explicitly requires
otherwise:

- `main.py`
- existing CLI flags
- `positions.csv`
- RGBA image input
- CAD/STL input
- existing output directory conventions
- `final.obj`
- `final.stl`

Reconstruction parameter ownership should remain explicit between camera/pose,
global CAD-to-real alignment, and local geometry deformation.

## Skills

Use project-scoped Skills under `.agents/skills/` when relevant.

For coding work:

- use the repository-state Skill to inspect and protect the current Git state;
- use `agent-state` for durable task lifecycle, Git SHA, engineering-decision,
  and validation metadata.

Do not duplicate Skill procedures in this file.