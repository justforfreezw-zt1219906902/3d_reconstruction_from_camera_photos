# 3D Reconstruction From Camera Photos

## Branch Purpose

This branch evaluates image-driven 3D reconstruction from a generic initial
shape.

The experiment intentionally starts from a sphere that does not resemble the
target object.

The purpose is to measure how much target geometry can be recovered from
multi-view images rather than inherited from a CAD design prior.

This branch is therefore different from the CAD-prior refinement workflow.

## Core Experimental Question

Given:

- calibrated or independently established camera / pose information
- multi-view RGBA observations
- a generic sphere occupying approximately the correct spatial region

can the system reconstruct a 3D mesh that is measurably closer to the real
object than the initial sphere?

## Inputs

Primary reconstruction inputs:

- Multi-view RGBA photographs
- `positions.csv`
- generic initial sphere / spatial initialization
- independent camera calibration where required

Current sphere:

    /Users/zhaowei/Downloads/openscan-benchy-model_files/initial_sphere_demo_scale.stl

Optional benchmark-only input:

- ground-truth STL

Ground truth is evaluation-only and must never participate in reconstruction.

## Initial Sphere Semantics

The sphere is not:

- a CAD prior
- ground truth
- a geometry anchor
- a shape-preservation target

The sphere may provide only:

- approximate spatial center
- approximate scale / extent
- initial bounding volume
- numerical initialization

The reconstruction must be free to become substantially non-spherical.

## Desired Architecture

The intended experimental flow is:

Camera / Pose
    ↓
Multi-view Image Observations
    ↓
Image-Derived Coarse Geometry
    ↓
Topology / Surface Generation
    ↓
Geometry Refinement
    ↓
Final Mesh
    ↓
Separate Ground-Truth Evaluation

The reconstruction should not rely exclusively on fixed-topology vertex
offsets from the sphere.

## Camera Ownership

Camera and pose parameters must represent the acquisition system.

Geometry mismatch between the sphere and the photographed object must not be
silently absorbed into:

- camera FOV
- camera distance
- principal-point offsets
- pose convention
- per-frame pose corrections

Camera calibration should therefore be independent of target-shaped geometry
for this branch.

## Geometry Ownership

### Image-derived coarse geometry

Owns the large shape change from generic spatial initialization toward the
observed object.

It may change:

- vertex count
- face count
- topology
- surface placement

### Geometry refinement

Owns residual corrections after an image-derived coarse geometry exists.

It should improve agreement with the observations without reintroducing
catastrophic mesh artifacts.

The original sphere is not the refinement reference.

## Ground Truth

The known target mesh is withheld from reconstruction.

It may only be used after reconstruction for quantitative benchmark metrics.

Primary benchmark comparison:

    initial sphere -> ground truth

versus:

    final reconstruction -> ground truth

A successful image-driven reconstruction should reduce 3D error by a
meaningful margin.

## Outputs

Preserve the existing output contract where practical, including:

- `final.obj`
- `final.stl`
- existing output directory conventions

Also preserve enough intermediate artifacts to evaluate:

- initial sphere
- image-derived coarse geometry
- final geometry

## Compatibility

Preserve:

- `main.py` as the primary CLI entry point
- `positions.csv`
- RGBA image inputs
- output directory conventions
- `final.obj`
- `final.stl`

Existing CLI behavior should remain compatible where its semantics remain
valid:

- `--validate-only`
- `--camera-fit-only`
- `--skip-camera-fit`

Do not preserve a historical behavior when doing so would violate the
image-only experiment, especially camera fitting against sphere shape or
CAD-reference anchoring.

## Engineering Principles

- Git is the source of code truth.
- Ground truth must be isolated from reconstruction code.
- Camera error and geometry error must have explicit ownership.
- Do not use the generic sphere as hidden shape supervision.
- Prefer image-derived geometry before local mesh refinement.
- Maintain deterministic benchmark outputs.
- Add regression tests when architectural ownership changes.
- Evaluate both held-out 2D views and withheld 3D ground truth.
- Keep CAD-prior behavior isolated if retained as a separate mode.

## Runtime Architecture

Forge-Orchestrator is the task and multi-agent runtime.

`.forge/` contains local orchestration/task runtime state and is intentionally
not tracked by Git.

Git remains authoritative for source code and committed implementation history.

Domain-specific reconstruction knowledge belongs in the
`reconstruction-geometry` project Skill.