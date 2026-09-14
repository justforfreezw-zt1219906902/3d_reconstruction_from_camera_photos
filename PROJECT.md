# 3D Reconstruction From Camera Photos

## Purpose

This repository reconstructs 3D geometry from calibrated multi-view RGBA
images.

This branch focuses on image-driven reconstruction from a generic initial
shape rather than refinement of a target-shaped CAD prior.

The central experimental question is:

> Given multi-view images, acquisition poses, and only a generic sphere as
> spatial initialization, how much of the target object's 3D geometry can be
> recovered from image evidence itself?

The initial sphere must not provide hidden target-shape supervision.


## Reconstruction Modes

The architecture should keep the distinction between two reconstruction
concepts explicit.

### `cad_prior`

A target-shaped CAD/STL model is available as a genuine design prior.

Conceptually:

    Camera / Pose
        ->
    Global CAD-to-real Alignment
        ->
    Local Residual Deformation
        ->
    Final Mesh

Reference-aware anchor losses and bounded local deformation belong to this
mode.

### `image_only`

No target-shaped CAD prior is available.

A generic sphere may be provided through `INITIAL_MESH_PATH`, but it is only
spatial initialization.

Conceptually:

    Independent Camera / Pose
        ->
    Multi-view RGBA Observations
        ->
    Image-derived Coarse Geometry
        ->
    Surface Refinement
        ->
    Final Mesh
        ->
    Separate Ground-Truth Evaluation

The current sphere experiment uses this mode.


## Initial Sphere Semantics

In `image_only` mode, the initial sphere may provide:

- approximate object center
- rough object scale
- initial spatial support
- reconstruction-volume initialization

It is not:

- a CAD design prior
- a geometry reference
- ground truth
- a shape-preservation target
- an anchor for final geometry

The reconstruction must be allowed to move substantially outside the sphere
shape and may generate topology unrelated to the sphere.

The reconstruction volume must not be restricted to the exact sphere bounding
box. Configurable padding or volume expansion must allow observed geometry to
extend beyond the initial sphere.


## Inputs

Primary reconstruction inputs are:

- multi-view RGBA photographs
- `positions.csv`
- camera / acquisition calibration
- optional generic initialization mesh

The concrete local path for `INITIAL_MESH_PATH` belongs in `.env`, not in this
document.

For benchmark evaluation, an optional ground-truth mesh may also be provided.


## Camera / Pose Ownership

Camera parameters describe the acquisition system.

In `image_only` mode, geometry mismatch between a generic sphere and the
photographed object must not be absorbed into camera fitting.

The pipeline must not fit camera parameters merely by forcing the sphere
silhouette to resemble the target object.

Camera / pose should instead come from information independent of target
geometry where practical, including:

- acquisition angles from `positions.csv`
- known rig geometry
- known camera intrinsics
- independently established calibration

Camera parameters should be frozen before image-driven geometry reconstruction.

If sufficient independent calibration is unavailable, the pipeline should fail
explicitly rather than silently convert geometry error into camera error.


## Image-Derived Geometry

In `image_only` mode, major target shape must come from image observations.

The preferred baseline is:

    calibrated silhouettes
        ->
    bounded 3D reconstruction volume
        ->
    multi-view occupancy / visual hull
        ->
    surface extraction
        ->
    coarse image-derived mesh

Visual hull / voxel carving with Marching Cubes is an acceptable baseline.

Equivalent methods are allowed if coarse geometry is generated from image
evidence rather than inherited from sphere topology.


## Topology Ownership

The initial sphere topology is not authoritative in `image_only` mode.

The pipeline must not rely exclusively on:

    V_final = V_sphere + vertex_offsets

The image-derived geometry stage must be allowed to create a new mesh with
different:

- vertices
- faces
- connectivity
- topology

After a new image-derived coarse mesh has been created, a refinement stage may
keep that coarse topology fixed while optimizing its surface.

The important requirement is that refinement topology originates from the
image-derived geometry stage, not from the original sphere.


## Geometry Refinement

Image-derived coarse geometry may be refined using differentiable rendering or
other justified image-based optimization.

Useful signals may include:

- silhouette agreement
- RGB agreement where meaningful
- normal consistency
- surface smoothness
- edge regularity
- local deformation regularity
- mesh validity constraints

In `image_only` mode:

- do not anchor geometry to the sphere
- do not apply the CAD-oriented 3% local deformation bound
- do not interpret distance from the sphere as reconstruction error

If a refinement anchor is used, it must refer explicitly to the image-derived
coarse mesh and remain configurable.


## Ground-Truth Isolation

Ground truth is evaluation-only.

A ground-truth mesh must never participate in:

- camera fitting
- coarse geometry generation
- geometry optimization
- training loss
- validation loss
- best-epoch selection
- automatic hyperparameter selection

The reconstruction pipeline must complete before ground truth is loaded for
benchmark evaluation.

The strict boundary is:

    Images + Camera + Validation
        ->
    Select Best Reconstruction
        ->
    final.obj / final.stl

    --------------------------------

    Ground Truth
        ->
    Post-reconstruction Evaluation


## Training and Validation

Geometry optimization must preserve deterministic held-out validation views.

Requirements:

- training and validation views are disjoint
- split is reproducible from the configured seed
- validation views never contribute gradients
- best reconstruction is selected using held-out image observations
- final export must not blindly use the last epoch

Ground-truth metrics must not influence best-state selection.


## Mesh Validity

For image-derived geometry, validity should be judged using geometry-quality
checks rather than proximity to the initial sphere.

Useful diagnostics include:

- non-finite vertices
- degenerate or zero-area faces
- extreme edge-length distributions
- catastrophic spikes
- disconnected fragments
- self-intersections where practical to detect
- invalid OBJ/STL output


## Benchmark Evaluation

The sphere experiment should compare at least:

    A. Initial sphere
    B. Image-derived coarse mesh
    C. Final refined mesh

### 2D image-space evaluation

Track at least:

- training silhouette loss
- training IoU
- held-out validation silhouette loss
- held-out validation IoU

### 3D evaluation

When an evaluation-only ground-truth mesh is available, use topology-independent
surface metrics such as:

- symmetric Chamfer distance
- reconstruction-to-ground-truth surface distance
- ground-truth-to-reconstruction surface distance
- p95 surface distance
- robust Hausdorff distance
- normal consistency where meaningful

Report physical distance and a normalized distance based on a documented object
scale.

Primary success condition:

    error(final, ground_truth)
        <
    error(initial_sphere, ground_truth)

The coarse image-derived mesh should also be evaluated independently so the
contribution of coarse reconstruction and later refinement can be distinguished.


## Outputs

Preserve the existing external output contract:

- `final.obj`
- `final.stl`
- existing output directory conventions

For `image_only` mode, also preserve enough intermediate artifacts to inspect
the reconstruction process, including the concepts of:

- initial sphere
- image-derived coarse mesh
- final mesh
- train / validation split
- optimization diagnostics
- mesh validity diagnostics
- optional ground-truth evaluation report


## Compatibility Requirements

Preserve where practical:

- `main.py` as the primary CLI entry point
- `positions.csv`
- RGBA image inputs
- environment-file configuration
- `--validate-only`
- `--camera-fit-only`
- `--skip-camera-fit`
- `final.obj`
- `final.stl`

Compatibility must not preserve historical behavior that violates
`image_only` semantics.

In particular, do not preserve:

- camera fitting against generic sphere shape
- sphere-based CAD anchor loss
- sphere-based local-deformation limits

inside `image_only` mode.


## Limitations

Silhouette-based reconstruction cannot uniquely recover all possible 3D
geometry.

In particular, geometry may remain ambiguous when it involves:

- concavities that never change the external silhouette
- internal cavities
- hidden surfaces
- structures never visible from available views

These limitations should be measured and documented rather than hidden by
ground-truth leakage or target-shaped initialization.


## Engineering Principles

- Git is the source of code truth.
- Keep camera error and geometry error explicitly separated.
- Ground truth must remain isolated from reconstruction.
- Do not use the generic sphere as hidden shape supervision.
- Prefer image-derived geometry before local mesh refinement.
- Preserve deterministic benchmark behavior.
- Add regression tests when parameter or stage ownership changes.
- Evaluate both held-out 2D observations and withheld 3D ground truth.
- Keep CAD-prior behavior isolated from image-only behavior.
- Avoid unrelated framework redesign while implementing reconstruction work.


## Runtime Architecture

Forge-Orchestrator is the task and multi-agent runtime.

Claude is used as the planning brain.

Codex CLI is the execution worker.

`.forge/` contains local orchestration/runtime state and is intentionally not
tracked by Git.

Git remains authoritative for source code and committed implementation history.

Domain-specific reconstruction workflows and reusable knowledge belong in the
`reconstruction-geometry` project Skill.