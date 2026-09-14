# Objective

Evaluate and improve the reconstruction pipeline's ability to recover 3D
geometry from multi-view RGBA images when no target-shaped CAD prior is
available.

This branch intentionally uses a generic sphere as the initial mesh:

    INITIAL_MESH_PATH=
    /Users/zhaowei/Downloads/openscan-benchy-model_files/initial_sphere_demo_scale.stl

The sphere is deliberately not shaped like the photographed object.

The purpose of this experiment is to answer:

    How much of the target object's 3D geometry can be recovered from the
    images and camera observations themselves?

The reconstruction must not succeed merely because the initial mesh already
resembles the target.

The desired conceptual flow is:

    Camera / Pose
        ->
    Multi-view image observations
        ->
    Image-derived coarse 3D geometry
        ->
    Surface / geometry refinement
        ->
    Final Mesh
        ->
    Separate Ground-Truth Evaluation

The ground-truth mesh must never participate in reconstruction.


# Experimental Principle

This branch is an image-driven reconstruction benchmark.

The following distinction is fundamental:

    Initial sphere
        = generic spatial initialization

    RGBA images
        = reconstruction evidence

    positions.csv / independent camera calibration
        = acquisition geometry

    Ground-truth STL
        = evaluation only

The sphere must not be treated as:

- a CAD design prior
- a geometric truth reference
- a shape-preservation target
- a vertex anchor
- evidence that the target object is spherical

A successful reconstruction should be allowed to move substantially away
from the sphere when the image observations support doing so.


# Repository State

The current Git working tree is authoritative.

Inspect the current implementation before designing changes.

Important current areas include:

- src/pipeline.py
- src/camera_fitting.py
- src/optimizer.py
- src/losses.py
- src/renderer.py
- src/fast_silhouette.py
- src/config.py
- src/mesh_io.py
- src/rgba_dataset.py
- tests/

Do not assume the existing CAD-prior architecture remains appropriate for
this branch merely because it exists in the current implementation.

Reuse working infrastructure where useful, but change architectural
assumptions when required by the sphere-initialization experiment.


# Existing Infrastructure to Preserve Where Practical

Preserve useful existing infrastructure unless the new architecture requires
a targeted change:

- main.py as the primary CLI entry point
- positions.csv input format
- multi-view RGBA input handling
- dataset validation
- renderer infrastructure
- device selection and fallback behavior
- deterministic seeds
- geometry train/validation split
- held-out validation evaluation
- best-state selection
- diagnostic CSV output
- OBJ export
- STL export
- output directory conventions

Existing CLI options should remain usable where their semantics still make
sense:

- --validate-only
- --camera-fit-only
- --skip-camera-fit

Do not redesign unrelated infrastructure.


# Ground-Truth Isolation

The benchmark may optionally receive a known target mesh for evaluation.

Introduce a clearly separated evaluation-only configuration concept such as:

    GROUND_TRUTH_MESH_PATH

The exact configuration name may follow existing conventions.

Ground truth must obey a strict boundary:

- reconstruction code must not use it
- camera fitting must not use it
- geometry initialization must not use it
- training losses must not use it
- best-epoch selection must not use it
- hyperparameters must not be adapted automatically from ground truth

Ground truth may only be loaded after reconstruction for benchmark metrics.

The implementation should make this separation explicit enough that tests can
verify it.


# Camera and Pose Independence

Camera estimation must not confuse sphere-to-target shape mismatch with camera
error.

The current architecture historically uses the initial mesh silhouette during
camera fitting. That assumption is unsafe when the initial mesh is a generic
sphere.

For image-only reconstruction mode, camera / pose must be determined
independently of target geometry as far as practical.

Allowed information includes:

- positions.csv acquisition angles
- known acquisition-rig geometry
- known or previously calibrated camera intrinsics
- known pose convention
- calibration produced independently of the target ground-truth mesh

The implementation must not optimize camera parameters simply until a sphere
silhouette resembles the photographed target.

If trustworthy camera parameters cannot be determined independently, the
pipeline should fail explicitly or require explicit calibration input rather
than silently absorbing geometry error into:

- FOV
- camera distance
- principal point
- pose convention
- per-frame angle corrections

Camera parameters should be frozen before image-driven geometry reconstruction.


# Initial Sphere Semantics

The initial sphere exists only to provide a generic spatial starting condition.

It may provide:

- approximate object center
- approximate object spatial extent
- an initial bounding volume
- a numerical initialization for an image-driven geometry algorithm

It must not impose a strong shape-preservation objective.

In image-only reconstruction mode:

- do not use the existing CAD reference anchor against the sphere
- do not require final vertices to remain within a small percentage of sphere
  vertices
- do not assume sphere topology is final topology
- do not interpret distance from the sphere as reconstruction error

The current CAD-oriented reference-anchor behavior must remain available only
for CAD-prior reconstruction if that mode is preserved.


# Image-Derived Coarse Geometry

The pipeline must create a coarse 3D shape primarily from the multi-view image
observations.

The implementation should investigate an image-driven occupancy or surface
initialization strategy.

A preferred baseline is conceptually:

    calibrated silhouettes
        ->
    bounded 3D volume
        ->
    multi-view occupancy / visual-hull carving
        ->
    surface extraction
        ->
    coarse mesh

Visual hull / voxel carving plus a surface-extraction method such as Marching
Cubes is a reasonable reference implementation.

Equivalent approaches are acceptable if they satisfy the same architectural
requirement:

    coarse geometry must be created from image evidence rather than inherited
    from the sphere's shape.

The exact implementation should be selected after inspecting the current
renderer, coordinate conventions, object normalization, available compute,
and dataset structure.


# Topology Freedom

The reconstruction must not be permanently constrained to the sphere's fixed
vertex and face topology.

The existing fixed-topology formulation:

    V_final = V_sphere + offsets

is insufficient as the only geometry-generation mechanism for this benchmark.

The image-driven coarse reconstruction stage must be able to create a new mesh
whose:

- vertex count
- face count
- surface layout
- topology

are not required to match the initial sphere.

Remeshing, occupancy-to-mesh extraction, or another explicit topology-changing
stage is acceptable.

The system is not required to recover geometry that cannot be inferred from
the available views.

In particular, silhouette-based reconstruction has fundamental limitations for
concavities and internal structures that never affect any observed silhouette.
These limitations must be documented rather than hidden.


# Geometry Refinement

After image-derived coarse geometry exists, differentiable mesh refinement may
be used to improve agreement with the multi-view observations.

The refinement stage may optimize signals such as:

- silhouette agreement
- RGB agreement if meaningful
- surface smoothness
- normal consistency
- edge regularity
- deformation-field regularity
- other justified geometric validity constraints

However:

- the sphere must not be the geometric anchor
- the sphere's 3% local-deformation limit must not restrict reconstruction
- generic regularization must not dominate image evidence
- regularizers must have clearly documented geometric meaning

If an anchor is useful after coarse reconstruction, it may refer to the
image-derived coarse mesh rather than the original sphere, and its role must be
explicit and configurable.


# Mesh Validity

Because large geometry changes are expected, replace CAD-proximity safety
assumptions with reconstruction-validity checks appropriate to image-derived
geometry.

Investigate and diagnose where practical:

- degenerate faces
- zero-area triangles
- extreme edge-length distributions
- self-intersections where detectable
- disconnected fragments
- non-finite vertices
- catastrophic spikes
- invalid export geometry

Do not solve these problems merely by forcing the mesh to remain close to the
sphere.


# Training and Validation Views

Preserve the deterministic held-out geometry validation concept.

Requirements:

- training and validation view indices are disjoint
- split is reproducible from cfg.seed
- validation views never contribute gradients
- validation views are not used for image-driven geometry fitting
- useful angular coverage should be maintained where practical

Held-out validation remains an important measure of whether reconstructed
geometry generalizes to unseen viewpoints.


# Best Reconstruction Selection

Do not automatically export the last optimization epoch.

Continue tracking the best valid geometry state using held-out observations.

At minimum track:

- training silhouette loss
- training IoU
- validation silhouette loss
- validation IoU
- geometry validity metrics

Where quality is effectively tied, prefer the geometrically cleaner solution.

The final.obj and final.stl outputs must correspond to the selected best state.


# Benchmark Baselines

The benchmark must preserve enough intermediate geometry to compare distinct
stages.

At minimum evaluate:

    A. Initial sphere

    B. Image-derived coarse reconstruction

    C. Final refined reconstruction

The objective is to determine whether image evidence progressively improves
geometry.

A reconstruction that merely remains close to the initial sphere is not a
successful result.


# 2D Evaluation

Report image-space quality independently from 3D ground truth.

At minimum report:

- training silhouette loss
- training IoU
- held-out validation silhouette loss
- held-out validation IoU

Where useful also report metrics per elevation ring or viewing-angle range.

This allows diagnosis of:

    good 2D agreement but poor 3D shape

and:

    poor image fit due to incorrect camera calibration.


# 3D Ground-Truth Evaluation

When GROUND_TRUTH_MESH_PATH is supplied, evaluate reconstructed geometry only
after reconstruction has completed.

At minimum implement robust surface-based comparisons suitable for meshes with
different topology.

Preferred metrics include:

- symmetric Chamfer distance
- reconstruction-to-ground-truth mean surface distance
- ground-truth-to-reconstruction mean surface distance
- p95 surface distance
- Hausdorff or robust Hausdorff distance
- normal consistency where meaningful

Normalize distance metrics by a documented physical scale such as the
ground-truth bounding-box diagonal in addition to reporting physical units.

Do not rely on direct vertex-to-vertex correspondence because the reconstructed
mesh and ground-truth mesh may have different topology.

Report metrics for at least:

    initial sphere -> ground truth
    coarse image reconstruction -> ground truth
    final reconstruction -> ground truth

The primary success condition is:

    3D_error(final, ground_truth)
        <
    3D_error(initial_sphere, ground_truth)

with a meaningful margin.

Also report whether the coarse image-derived reconstruction already improves
over the sphere before differentiable refinement.


# Evaluation Alignment

Do not hide reconstruction failures through unrestricted post-hoc alignment.

The primary benchmark should evaluate geometry in the reconstruction's intended
physical coordinate frame.

If an additional diagnostic alignment is useful, report it separately.

For example:

- raw / physical-frame error
- optional rigid-aligned diagnostic error

Do not use free scale alignment as the only reported metric when reconstruction
scale is part of the intended problem.


# Diagnostics and Artifacts

Preserve or add artifacts sufficient to inspect every major stage.

Useful outputs include concepts such as:

    reconstruction/
        initial_sphere.stl
        image_coarse_mesh.stl
        final.obj
        final.stl
        view_split.json
        losses.csv
        reconstruction_metrics.json
        ground_truth_evaluation.json
        previews/

Exact filenames may follow existing conventions.

Record enough metadata to reproduce:

- reconstruction mode
- camera source / calibration source
- volume or coarse-geometry parameters
- mesh extraction parameters
- refinement parameters
- train / validation split
- selected best state


# Configuration

Introduce an explicit mode or equivalent architectural switch separating:

    CAD-prior refinement

from:

    generic image-driven reconstruction

A reasonable concept is:

    RECONSTRUCTION_MODE=image_only

or an equivalent name consistent with the repository.

For image-only mode, configuration concepts may include:

- image-derived coarse geometry method
- reconstruction volume bounds
- voxel / occupancy resolution
- minimum silhouette support
- surface extraction parameters
- mesh cleanup parameters
- refinement enable/disable
- independent camera calibration source
- optional ground-truth evaluation path

Do not blindly reuse CAD-only parameters.

In particular:

    LOSS_ANCHOR_WEIGHT

must not anchor reconstruction to the sphere.

And:

    MAX_LOCAL_DEFORMATION_RATIO

must not prevent the sphere from becoming a substantially different shape.

Preserve environment-file configuration conventions.


# Required Investigation

Before implementation, inspect the current code and explicitly determine:

1. Which camera-fitting operations currently depend on initial-mesh shape.
2. Which parts of the global-alignment stage remain meaningful for a sphere.
3. Which local optimizer assumptions require identical topology.
4. Which CAD-oriented losses must be disabled or separated in image-only mode.
5. Which existing renderer and camera utilities can be reused for silhouette
   carving or equivalent image-derived geometry.
6. Which coordinate transforms are required so the coarse image reconstruction,
   refinement mesh, final export, and ground-truth evaluator use consistent
   units and frames.
7. Whether the current dataset provides sufficient calibrated information to
   construct rays / occupancy volumes reliably.

Do not implement around these questions without first verifying the current
code behavior.


# Testing Requirements

Add deterministic tests appropriate to the new architecture.

Tests should cover at minimum:

- image-only mode does not apply CAD reference-anchor loss to the sphere
- sphere local-deformation limits do not restrict image-derived coarse geometry
- ground-truth mesh is inaccessible to reconstruction/training code
- ground truth is loaded only by evaluation code
- camera estimation in image-only mode does not optimize against sphere shape
  as if it were the target
- image-derived geometry can change topology relative to the sphere
- synthetic multi-view silhouettes produce a non-spherical coarse shape
- synthetic reconstruction improves 3D error relative to an initial sphere
- train / validation view sets remain disjoint
- validation views produce no gradients
- best-state selection still works
- final OBJ/STL export works
- generated meshes contain finite vertices and valid faces
- existing CLI contracts remain compatible where applicable
- existing CAD-prior mode remains functional if retained

Run the complete automated test suite after implementation.


# Synthetic Verification

Before relying only on the Benchy dataset, verify the image-driven stage using
a simple synthetic object whose geometry is known.

Use a shape clearly different from a sphere.

The test should demonstrate:

    sphere
        ->
    multi-view synthetic silhouettes
        ->
    image-derived geometry
        ->
    reconstructed mesh closer to known target

This test should not use the target mesh during reconstruction.


# Real Benchy Verification

Run the new image-driven mode on the real multi-view Benchy dataset using:

    INITIAL_MESH_PATH=
    /Users/zhaowei/Downloads/openscan-benchy-model_files/initial_sphere_demo_scale.stl

The initial sphere should be visibly different from the Benchy geometry while
occupying a reasonable spatial region.

Record:

- sphere baseline held-out IoU
- coarse image reconstruction held-out IoU
- final reconstruction held-out IoU
- sphere-to-ground-truth 3D error
- coarse-to-ground-truth 3D error
- final-to-ground-truth 3D error
- final mesh validity
- runtime

Visually inspect:

- initial sphere
- image-derived coarse mesh
- final mesh
- ground truth shown only for comparison after reconstruction


# Acceptance Criteria

The branch is successful as an image-driven reconstruction benchmark when:

- the initial sphere is treated only as generic spatial initialization
- reconstruction does not depend on target-shaped CAD geometry
- camera parameters are not fitted by forcing a sphere to resemble the target
- image observations create a coarse non-spherical 3D geometry
- reconstructed topology is not permanently inherited from the sphere
- sphere-based CAD anchor loss is not active
- the old 3% CAD residual limit does not prevent large shape recovery
- ground truth is strictly isolated from reconstruction
- held-out validation remains independent
- final mesh selection uses validation rather than final epoch alone
- 3D ground-truth metrics compare sphere, coarse reconstruction, and final mesh
- the final mesh is measurably closer to ground truth than the initial sphere
- final held-out image agreement improves over the sphere baseline
- final.obj and final.stl remain valid
- tests pass
- limitations of silhouette-based geometry recovery are documented

Do not claim that image-only 3D reconstruction is solved merely because 2D
silhouette IoU improves.

The key result is whether image evidence produces a materially better 3D shape
than the generic sphere when compared against withheld ground truth.