# Objective

Fix local-deformation overfitting in the reconstruction stage.

The current reconstruction pipeline has already separated ownership into:

Camera / Pose
    ->
Global Similarity Alignment R/T/S
    ->
Frozen aligned CAD reference
    ->
Local residual deformation

This architecture must be preserved.

A real reconstruction run shows that additional geometry-optimization
epochs improve the 2D fitting objective while progressively degrading
the reconstructed 3D shape.

Observed metrics:

Epoch 1:
- total = 0.10310
- silhouette = 0.09856
- IoU = 0.28861
- mean vertex displacement = 0.01313
- p95 vertex displacement = 0.02132
- max vertex displacement ratio = 0.01519

Epoch 4:
- total = 0.08950
- silhouette = 0.08734
- IoU = 0.31396
- mean vertex displacement = 0.03030
- p95 vertex displacement = 0.05751
- max vertex displacement ratio = 0.05783

Therefore:

2D fitting quality is improving,
while local vertex deformation is increasingly moving the mesh away
from the aligned CAD prior.

The goal is to prevent the local optimizer from exploiting excessive
vertex deformation merely to reduce silhouette loss.

The solution should preserve useful residual deformation while keeping
the reconstructed geometry physically and structurally close to the
aligned CAD prior.


# Repository State

The current Git repository is authoritative.

Inspect the current implementation directly before making design
decisions.

Do not assume previous task reports accurately describe the current
working tree.

Relevant implementation is expected primarily around:

- src/optimizer.py
- src/losses.py
- src/config.py
- src/pipeline.py
- tests/

The existing global-alignment implementation and its tests should be
treated as already-established architecture unless a regression requires
otherwise.


# Shared Constraints

Preserve existing external contracts:

- preserve main.py CLI
- preserve --validate-only
- preserve --camera-fit-only
- preserve --skip-camera-fit
- preserve positions.csv input contract
- preserve final.obj
- preserve final.stl
- preserve current camera fitting behavior
- preserve current global similarity alignment behavior

CAD is a design prior.

Global rotation, translation, and scale must remain owned by the global
similarity-alignment stage.

Local deformation must remain a residual operation after global
alignment.

The local optimizer must not be allowed to reintroduce global
translation, rotation, or scale through vertex offsets.

Avoid redesigning unrelated parts of the reconstruction pipeline.


# Required Investigation

Before implementation, inspect the current local-deformation objective
and determine why increasing optimization epochs allows geometry quality
to degrade.

In particular inspect the behavior and mathematical meaning of:

- silhouette loss
- RGB loss if enabled
- mesh Laplacian regularization
- mesh edge regularization
- normal consistency
- local vertex offsets
- displacement safety gates
- early stopping
- checkpoint selection
- final mesh export

Determine which current regularizers preserve generic mesh smoothness
versus which actually preserve the aligned CAD reference geometry.

Do not assume that a decreasing regularization loss means the
reconstruction is becoming more faithful to the CAD prior.


# Reference-Aware Local Deformation

The local optimizer must explicitly know that the globally aligned CAD
mesh is its geometric reference.

Introduce a reference-aware deformation penalty.

The loss should be zero when local displacement is zero and increase as
vertices move away from their aligned reference positions.

A reasonable formulation is conceptually:

    L_anchor =
        mean(||V_current - V_reference||^2)

or an equivalent normalized or robust formulation.

The implementation should be scale-aware so the configured weight is not
unreasonably dependent on source object units.

This loss applies only to local deformation.

Do not penalize the already-completed global similarity transform.


# Local Deformation Smoothness

The optimization should discourage isolated vertex motion and high
frequency deformation.

A simple preferred formulation is deformation-field smoothness over mesh
edges:

    L_local_smoothness =
        mean(
            ||offset_i - offset_j||^2
        )

for connected vertices i and j.

Equivalent well-justified formulations are acceptable.

The objective should distinguish between:

- preserving local deformation coherence
- preserving CAD-reference proximity
- generic mesh smoothness

Do not add unnecessary geometric complexity when a simpler residual-field
regularizer is sufficient.


# Existing Edge and Laplacian Losses

Audit the current use of PyTorch3D geometry regularizers.

In particular verify whether the current edge loss encourages preservation
of reference edge lengths or merely minimizes absolute edge lengths.

Likewise verify whether the current Laplacian loss preserves the original
local geometry or simply encourages smoothing/shrinkage.

Replace, supplement, or remove misleading regularizers where necessary.

Do not retain a loss merely because its numeric value decreases during
training.

Any retained regularizer must have a clear role in the reconstruction
objective.


# Bounded Local Deformation

Introduce an explicit configurable upper bound on local residual
deformation.

The bound should be relative to object size.

A reasonable initial default is approximately:

    MAX_LOCAL_DEFORMATION_RATIO = 0.03

The exact implementation may use:

- projection
- clipping
- bounded parameterization
- another numerically stable method

but the mesh actually used for rendering and export must satisfy the
configured bound.

The existing larger displacement safety gate may remain as an emergency
failure condition, but it must not be the only mechanism controlling
deformation.

The optimizer should not spend many epochs moving vertices progressively
farther away until an emergency gate finally aborts the run.


# Training and Validation Views

Introduce deterministic held-out validation views for geometry
optimization.

Do not optimize on all available images and use the same images to choose
the best reconstruction epoch.

Requirements:

- split determined reproducibly from cfg.seed
- training and validation indices must be disjoint
- validation views must never contribute gradients
- preserve useful angular coverage where practical
- work with both real datasets and small synthetic/unit-test datasets

A reasonable default validation fraction is approximately 15–20%.

The exact sampling strategy may be designed based on the current
phi/theta dataset structure.


# Best Reconstruction Selection

The final reconstruction must not automatically equal the final optimizer
epoch.

Evaluate reconstruction quality on held-out validation views after each
epoch.

Track at least:

- validation silhouette loss
- validation IoU
- local displacement statistics

Maintain the best valid local-deformation state.

A reasonable selection strategy is:

1. candidate satisfies all geometry/deformation constraints
2. better held-out validation quality is preferred
3. when validation quality is effectively tied, prefer smaller deformation

The exact comparison logic should be explicit, deterministic, and tested.

At the end of optimization:

    final.obj
    final.stl

must be exported using the selected best offsets.

They must not blindly use the offsets from the last epoch.

If early stopping is used, the selected best state must also be restored.


# Diagnostics

Improve reconstruction diagnostics so future real runs can distinguish
2D fitting improvement from 3D geometry degradation.

losses.csv should retain existing useful metrics and include enough data
to observe:

- training objective
- training silhouette quality
- training IoU
- validation silhouette quality
- validation IoU
- CAD-reference deformation penalty
- local deformation smoothness
- displacement distribution
- selected best epoch

Useful displacement metrics include:

- mean vertex displacement
- median vertex displacement
- p95 vertex displacement
- max vertex displacement
- max vertex displacement ratio

The output should make situations like this visible:

    training IoU improves
    validation IoU stops improving
    displacement continues increasing

or:

    silhouette loss decreases
    while deformation approaches the configured limit

Record whether an epoch becomes the current best reconstruction.


# Configuration

Expose configuration for the new optimization behavior.

Reasonable configuration concepts include:

    LOSS_ANCHOR_WEIGHT
    LOSS_LOCAL_SMOOTHNESS_WEIGHT
    MAX_LOCAL_DEFORMATION_RATIO
    GEOMETRY_VALIDATION_FRACTION

Names may be adjusted to fit existing configuration conventions.

Choose conservative defaults.

Preserve backward compatibility with the current environment-file based
configuration.

Do not silently modify camera-fitting or global-alignment parameters.


# Testing Requirements

Add deterministic focused tests for the new behavior.

Tests should cover at minimum:

- zero displacement produces zero reference-anchor loss
- deformation increases reference-anchor loss
- uniform offset has near-zero local deformation-field smoothness
- isolated deformation is penalized by local smoothness
- configured local-deformation bound cannot be exceeded
- geometry train/validation split is deterministic
- train and validation view sets are disjoint
- validation views do not participate in optimization gradients
- best-state selection responds to validation quality
- tie-breaking favors smaller deformation where appropriate
- final export uses selected best offsets rather than last-epoch offsets
- existing global R/T/S ownership remains intact
- existing alignment tests remain valid
- existing CLI behavior remains compatible
- OBJ/STL export remains compatible

Run the full existing test suite after implementation.


# Empirical Verification

Unit tests alone are not sufficient to declare reconstruction quality
fixed.

After implementation, run a reduced multi-epoch reconstruction using the
existing real dataset.

Compare multiple epochs using:

- training silhouette
- training IoU
- validation silhouette
- validation IoU
- mean vertex displacement
- p95 vertex displacement
- max vertex displacement ratio

Verify that the configured deformation limit is respected throughout the
run.

Verify that the final exported mesh corresponds to the selected best
reconstruction state rather than automatically to the largest epoch
number.

The real reconstruction result should be visually inspected as a final
quality check.


# Acceptance Criteria

The refactor is successful when:

- Camera -> Global Alignment -> Local Deformation ownership remains intact
- globally aligned CAD is explicitly used as the local geometry reference
- local displacement has an explicit reference penalty
- isolated/high-frequency local deformation is regularized
- local deformation has a configurable hard bound
- held-out validation views exist
- validation views do not influence gradients
- reconstruction quality is evaluated per epoch
- a best local-deformation state is retained
- final.obj and final.stl use the selected best state
- losses.csv can expose 2D-overfit / 3D-drift behavior
- increasing NUM_EPOCHS no longer automatically means exporting the most
  deformed epoch
- existing CLI and data contracts remain compatible
- existing global alignment behavior remains compatible
- the complete automated test suite passes

Do not claim that the underlying 3D reconstruction problem is solved
solely because losses or unit tests pass.