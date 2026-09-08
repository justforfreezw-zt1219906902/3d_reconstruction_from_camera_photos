# 3D Reconstruction From Camera Photos

## Purpose

Reconstruct the real geometry of a photographed physical object from calibrated
multi-view RGBA images, using the CAD/STL model as a design prior.

The CAD model is not ground-truth geometry for the manufactured object.

## Inputs

- Multi-view RGBA photographs
- `positions.csv`
- CAD/STL design prior

Current dataset contains approximately 336 views distributed across multiple
camera elevation rings.

## Outputs

Preserve the existing output contract, including:

- `final.obj`
- `final.stl`
- existing output directory conventions

## Current pipeline

The repository currently contains stages for:

1. Input/data validation
2. Camera fitting and pose handling
3. Pose convention selection
4. CAD proxy preparation / normalization
5. Geometry optimization
6. Mesh export

## Parameter ownership

The intended architecture separates three classes of parameters.

### Camera / pose

Owns:

- camera projection
- camera extrinsics / pose
- camera-related calibration parameters

Camera error should not silently be compensated for by geometry deformation.

### Global CAD-to-real alignment

Owns global similarity alignment between the CAD design prior and the
observed manufactured object:

- rotation
- translation
- scale

This layer must be explicit.

### Local geometry deformation

Owns only residual/local shape differences after camera and global alignment
have been accounted for.

Per-vertex offsets must not become the implicit owner of unresolved global
rotation, translation, or scale.

## Architectural direction

The desired flow is:

Camera / Pose
    ↓
Global CAD-to-real Similarity Alignment
    ↓
Coarse / Fine / Local Geometry Deformation
    ↓
Final Mesh

Avoid designs where the optimizer immediately exposes unrestricted per-vertex
offsets before global alignment has been resolved.

## Compatibility requirements

Preserve:

- `main.py` as the primary CLI entry point
- `positions.csv` format
- RGBA image inputs
- CAD/STL inputs
- output directory conventions
- `final.obj`
- `final.stl`

Existing CLI behavior must remain compatible, including:

- `--validate-only`
- `--camera-fit-only`
- `--skip-camera-fit`

Do not rewrite the CLI architecture unless a concrete requirement demands it.

## Engineering principles

- Treat Git as the source of code truth.
- Preserve existing user changes unless explicitly asked to replace them.
- Prefer explicit ownership of optimization parameters.
- Add regression tests when changing parameter ownership or pipeline stages.
- Validate architecture changes against both synthetic cases and existing
  pipeline behavior where practical.
- Keep changes incremental enough to isolate regressions.

## Runtime architecture

Forge-Orchestrator is the task and multi-agent runtime.

`.forge/` contains local orchestration/task runtime state and is intentionally
not tracked by Git.

Git remains authoritative for source code and committed implementation history.

Domain-specific reconstruction knowledge belongs in the
`reconstruction-geometry` project Skill.