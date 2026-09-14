# Reconstruction overfitting fix — maintainer guide

This guide documents the T-007–T-020 investigation and implementation for T-021.
The fix constrains local deformation relative to the frozen aligned CAD proxy,
evaluates held-out views, and exports the best eligible epoch. It controls the
observed drift; it does not establish that reconstructed geometry is ground truth.

Implementation sources: [losses](../src/losses.py), [optimizer](../src/optimizer.py),
[configuration](../src/config.py), [pipeline](../src/pipeline.py), and
[diagnostic plotting](../src/visualization.py). The motivating experiment is in
[SPEC.md](../SPEC.md); measured verification is in the
[T-020 empirical report](../reports/empirical_verification.md).

## Investigation: why lower loss produced worse geometry

The original run improved total loss from 0.10310 to 0.08950 and silhouette loss
from 0.09856 to 0.08734 between epochs 1 and 4. IoU increased from 0.28861 to
0.31396, while mean displacement grew from 0.01313 to 0.03030, p95 from 0.02132
to 0.05751, and maximum displacement ratio from 0.01519 to 0.05783.

Alpha silhouette MSE rewards image-space agreement, which does not uniquely
determine 3D shape, especially with incomplete angular coverage. Optional RGB
loss is target-alpha-masked squared color error, normalized by alpha mass; it
also measures image agreement rather than geometric fidelity. RGB is disabled
by default. Neither objective prevents offsets from exploiting uncertain or
unobserved surfaces.

The old PyTorch3D `mesh_edge_loss(mesh)` used its default target length of zero:
it encouraged shorter absolute edges, not preservation of CAD edge lengths.
Uniform `mesh_laplacian_smoothing` penalized absolute vertex-to-neighbor-average
displacement, encouraging smoothing and possible shrinkage instead of preserving
reference curvature. Adjacent-face normal consistency likewise rewards generic
orientation agreement and can penalize intentional sharp creases. A falling
regularizer value therefore did not imply increasing CAD fidelity.

The prior displacement gate was an epoch-end emergency abort, not a bound on
every rendered state. Early stopping monitored training total loss and retained
no best-offset snapshot. Final export used the last epoch's offsets even after
early stopping. These mechanisms could accept progressively worse geometry as
long as the training objective improved.

## Reference geometry and loss terms

Ownership remains camera/pose → global similarity alignment (rotation,
translation, uniform scale) → frozen aligned CAD proxy → local residual offsets.
`_fit_global_alignment` completes before `train_geometry`; the latter detaches
the aligned base mesh, freezes cameras, and gives Adam only the local offsets.
CAD is a design prior, allowing real manufactured residual differences.

Let `R` denote frozen aligned proxy vertices, `V = R + d` the current vertices,
`E` the unique undirected reference edges, and `D` the axis-aligned bounding-box
diagonal of `R`. Current and reference meshes must share topology and vertex
ordering. `D` is frozen and must be finite and positive.

| Term | Definition and role | Difference from prior regularizers |
| --- | --- | --- |
| `anchor` | `mean_i(||V_i - R_i||² / D²)`; zero at the reference, increasing with residual magnitude. | Explicit reference proximity rather than absolute mesh smoothness; does not penalize the completed global transform. |
| `local_smoothness` | `mean_(i,j in E)(||d_i - d_j||² / D²)`; discourages isolated or high-frequency offset changes. | Smooths the deformation field, not reference positions or curvature; zero for uniform offsets and for zero deformation. |
| `normal` | PyTorch3D adjacent-face normal consistency, retained as a generic surface regularizer. | Still not reference-aware; sharp CAD features can incur a penalty even at zero offsets. |

Dividing the two new penalties by `D²` makes them invariant to a common change
of coordinate units. Anchor controls magnitude; local smoothness controls
coherence. Uniform translation has zero smoothness loss, so smoothness alone
cannot anchor a mesh or enforce parameter ownership. Reference tensors and scale
are detached from gradients. An edgeless mesh has differentiable zero local
smoothness. Generic callers of `regularization_losses` without a reference get
only `normal`.

The local training objective is:

```text
total = LOSS_SILHOUETTE_WEIGHT * silhouette
      + LOSS_RGB_WEIGHT * rgb
      + LOSS_NORMAL_WEIGHT * normal
      + LOSS_ANCHOR_WEIGHT * anchor
      + LOSS_LOCAL_SMOOTHNESS_WEIGHT * local_smoothness
```

Absolute edge and Laplacian losses have been removed from this objective, not
renamed. Their legacy config keys remain loadable but have no effect here.

## Deformation bound and safety gates

`_local_offset_basis` builds translation, infinitesimal rotation, and scale modes
from the frozen reference using SVD. `_project_local_offsets` removes those
modes from the residual field, preserving centroid, scale moment, and rotational
moment relative to the reference. This is a local linear constraint, not a
general nonlinear rigid-motion estimator.

`_bounded_local_offsets` first projects out those modes, then contracts the
entire field by one scalar if its largest vertex displacement exceeds
`MAX_LOCAL_DEFORMATION_RATIO * D`. Shared contraction preserves orthogonality;
independent per-vertex clipping could reintroduce global modes. A margin of
eight machine epsilons keeps the result just inside the limit. A zero limit
forces zero offsets; negative or nonfinite limits are rejected.

The bound applies at initialization, in the differentiable training render
path, and in-place after every Adam update. Validation, previews, checkpoints,
and final export therefore use bounded offsets. The reference frame is the
aligned proxy in optimization coordinates. Export restores source coordinates
using `MeshTransform`; displacement ratios are dimensionless.

The independent `MAX_VERTEX_DISPLACEMENT_RATIO` Gate 4 still checks at epoch
end. Exceeding it restores the previous valid epoch's offsets, saves
`reconstruction/checkpoints/last_valid_checkpoint.pt`, and raises an error;
this is an abort, not successful best-state export. Nonfinite offsets, losses,
or gradients and empty training renders also abort. If the emergency threshold
is below the local cap, the emergency gate can still fire.

## Configuration

Set these environment keys in the existing `.env` workflow or process
environment. `load_config` loads `.env` without overriding existing process
values. Inspect `run_config.json` for effective values; it uses lowercase
`Config` field names. The four new defaults are independent of runtime profile.

| New environment key | Config field | Default | Intended effect |
| --- | --- | --- | --- |
| `LOSS_ANCHOR_WEIGHT` | `loss_anchor_weight` | `1.0` | Increase to penalize reference departure more strongly; zero disables the soft anchor, not the hard cap. |
| `LOSS_LOCAL_SMOOTHNESS_WEIGHT` | `loss_local_smoothness_weight` | `0.1` | Increase to discourage neighboring vertices moving differently; zero disables residual-field smoothing. |
| `MAX_LOCAL_DEFORMATION_RATIO` | `max_local_deformation_ratio` | `0.03` | Maximum residual norm as a fraction of aligned reference diagonal; finite and nonnegative, zero freezes local geometry. |
| `GEOMETRY_VALIDATION_FRACTION` | `geometry_validation_fraction` | `0.20` | Fraction reserved from local training; finite and strictly between zero and one. Integer rounding and small datasets affect actual fraction. |

Related existing controls:

| Environment key | Code default | Current meaning |
| --- | --- | --- |
| `SEED` | `42` | Reproducible split and rotating training-view sampling for a fixed dataset. |
| `EARLY_STOPPING_PATIENCE` | `0` | Disabled at zero; positive values stop after this many consecutive epochs fail best-state selection, provided holdouts exist. |
| `MAX_VERTEX_DISPLACEMENT_RATIO` | `0.10` | Emergency epoch-end abort threshold; distinct from the new 0.03 local cap. T-020 used an environment override of 0.25. |
| `LOSS_SILHOUETTE_WEIGHT` | `1.0` | Training alpha MSE weight. |
| `LOSS_RGB_WEIGHT` | `0.0` | Optional training masked RGB weight; validation selection still uses silhouette and IoU. |
| `LOSS_NORMAL_WEIGHT` | `0.01` | Generic normal consistency weight. |
| `LOSS_EDGE_WEIGHT`, `LOSS_LAPLACIAN_WEIGHT` | `0.1` each | Accepted for config compatibility; ignored by the local objective. |

`GEOMETRY_VIEWS_PER_EPOCH` now samples only the training partition; zero uses
all training views. `GEOMETRY_VIEW_BATCH_SIZE` applies to both training and
validation. Existing export/preview cadences govern epoch artifacts, not which
state wins. The comparison tolerance is a function default of `1e-6`, not an
environment knob. The fix does not change camera-fitting or alignment controls.

## Deterministic validation split

`split_geometry_view_indices` operates on validated dataset frames before any
local training-view sampling. For `N >= 2`, the holdout count is
`min(N - 1, max(1, round(N * fraction)))`, using Python rounding. Frames are
sorted by elevation `phi_deg` rounded to five decimals, azimuth `theta_deg`
modulo 360, then dataset index. This order is partitioned into equal-sized
index strata; `random.Random(seed)` chooses one view per stratum. Returned
training and validation lists are disjoint, sorted dataset indices.

This spreads holdouts over available angles without independently rounding up
every small ring's quota. It does not guarantee a holdout in every ring or fill
missing angular coverage. Synthetic frames without angles fall back to dataset
order. Changing dataset membership, order, or angles can change the split even
with the same seed. `reconstruction/view_split.json` records the actual indices,
seed, and requested fraction.

Training samples rotate within the training partition and prefer underused
frames. After each epoch, `evaluate_geometry_validation` runs under
`torch.no_grad()` on every fixed holdout using the same post-update mesh.
Validation MSE is weighted by batch view count, and IoU is averaged per image,
so a final short batch is not overweighted. No validation backward pass or
optimizer step occurs.

With fewer than two frames the helper returns all frames for training and no
holdout. Validation quality is then NaN, no candidate can win, early stopping
is disabled, and final export uses the zero-offset aligned reference with
`best_epoch=0`. Normal pipeline minimum-frame gates usually prevent this case.
Holdouts are excluded only from **local geometry gradients**: upstream camera
fitting/refinement and global alignment can use them. These metrics are not an
independent end-to-end generalization estimate.

## Best-state selection, stopping, and export

`is_better_geometry_state` applies this deterministic comparison after each
epoch:

1. Reject candidates with nonfinite displacement statistics, a maximum ratio
   above either deformation threshold, or nonfinite validation/selection
   metrics. These checks do not certify mesh topology or physical correctness.
2. Accept the first eligible candidate. There is no validation evaluation of
   the undeformed epoch-zero reference in the incumbent comparison.
3. Prefer lower validation silhouette MSE if the difference exceeds absolute
   tolerance `1e-6`.
4. If silhouette is within tolerance, prefer higher validation IoU if its
   difference exceeds the same absolute tolerance.
5. If both quality metrics are effectively tied, prefer the lexicographically
   smaller `(mean_vertex_displacement, p95_vertex_displacement)` tuple. There is
   no tolerance on this tuple; exact ties retain the incumbent.

Thus higher IoU cannot override meaningfully worse silhouette MSE. Each winner
clones offsets into an in-memory `best_offsets` snapshot, records `best_epoch`,
and resets the stale counter. Every nonwinner increments that counter. Positive
patience with holdouts stops when the counter reaches its limit; training total
loss no longer determines patience.

On normal completion or early stopping, `final.obj`, `final.stl`, and the
returned result mesh are built directly from `best_offsets`. If no eligible
epoch exists (including zero epochs), the fallback is the zero-offset aligned
reference. A runtime gate exception instead aborts the run.

Epoch checkpoint meshes, `.pt` offsets, and previews describe their own epoch,
not the selected best. Best snapshots are held in memory; there is no dedicated
persistent best checkpoint or automatic resume of that selection. To verify an
export against its winning checkpoint, use `EXPORT_EVERY_EPOCHS=1`. With a
sparser cadence the winning epoch might not have an epoch checkpoint. Outputs
retain the existing `reconstruction/final.obj` and `reconstruction/final.stl`
contract, and the CLI flags and input formats remain unchanged.

## Diagnostics and debugging

`reconstruction/losses.csv` contains:

| Columns | Interpretation |
| --- | --- |
| `epoch`, `total`, `silhouette`, `rgb`, `normal`, `anchor`, `local_smoothness`, `iou` | Legacy useful metrics plus the new unweighted loss components. `total` is weighted. Training values average batches before their updates; a short batch has equal weight here. Removed `edge`/`laplacian` columns are not emitted. |
| `training_objective`, `training_silhouette`, `training_iou` | Explicit aliases of `total`, `silhouette`, `iou`. |
| `validation_silhouette`, `validation_iou` | Fixed held-out, post-epoch quality; NaN if no holdouts. |
| `mean_vertex_displacement`, `median_vertex_displacement`, `p95_vertex_displacement`, `max_vertex_displacement`, `max_vertex_displacement_ratio` | Post-epoch residual norms relative to the aligned proxy; absolute values use optimization units, maximum ratio divides by reference size. Median follows `torch.median` (lower middle for even counts). |
| `best_epoch`, `is_best` | Incumbent epoch after this row and integer 1/0 indicating whether this epoch became best. Historical winners remain marked 1. |

`losses.png` plots metrics on a logarithmic axis, omitting duplicate aliases and
selection metadata. Zeros/NaNs and very different metric scales make the CSV
the authoritative debugging source. `view_usage.csv` counts training selections;
holdouts should have zero usage even though validation and previews render them.
`profile.json` records steps and timing; its `epochs` value is the configured
budget, so use CSV rows for actual completed epochs after early stopping.

For a suspicious run:

1. Read `run_config.json`, `global_alignment.json`, camera diagnostics, and
   `view_split.json`. Confirm reference alignment and available angular coverage
   before trying to correct a projection mismatch with larger local offsets.
2. Compare fixed validation curves with mean/p95/max displacement. Changing
   sampled training views and pre-update averaging mean the training curve is
   not a fixed-view evaluation of one mesh. Improving training IoU while
   validation stalls and displacement grows is evidence of possible overfit.
3. Check the ratio against the local cap and emergency gate. A capped maximum
   does not stop mean/p95 from increasing as more vertices move toward the cap.
4. Replay the comparison above against `best_epoch`/`is_best`; inspect the final
   row's incumbent rather than assuming the largest epoch number won. Compare
   final exports with that epoch's checkpoint, accounting for mesh coordinate
   restoration. Inspect the actual 3D mesh as well as silhouette previews.

## Empirical evidence and remaining limitations

The [T-020 report](../reports/empirical_verification.md) contains commands,
effective settings, all 32 epoch rows, export comparisons, visual assessment,
and artifact links. Its reduced CPU run used 30 real frames (24 training,
6 holdout), 12 training views per epoch, a 722-vertex/1,500-face proxy, 64-pixel
geometry images, and the new default loss weights and local cap.

The largest recorded ratio was **0.029999976139362575 < 0.03**; the cap became
active at epoch 6. Epoch **27** won with validation silhouette **0.0904551384**
and IoU **0.2827983623**. Epoch 32 was worse at **0.0904964991** and
**0.2823795875**, despite improved sampled training metrics and greater mean/p95
displacement. Final OBJ and STL were byte-identical to epoch 27's exports and
different from epoch 32's. Independent checkpoint checks reproduced selection
and displacement statistics. Visual inspection found no gross collapse from
the inspected viewpoints, but the result remained coarse and locally irregular.

These results demonstrate bounded displacement and restoration of an earlier
best epoch in one experiment. They do not isolate anchor/smoothness benefits:
there was no matched unbounded ablation, and the original SPEC run used different
settings. Mean/p95 drift and small validation regressions persist within the
bound. A 3% limit can also suppress genuine larger manufactured differences.
Sparse coverage (one elevation ring and a 142.5-degree angular gap), low soft
validation IoU around 0.283, and aggressive proxy/image reduction limit claims
about 3D accuracy. Neither selection nor the bound guarantees watertightness,
absence of self-intersections, dimensional accuracy, or fine-detail recovery.
Reference-based normalization is unit-aware but still depends on proxy topology
and the aligned bounding box. Camera-gate IoU uses a different evaluation path
and should not be directly compared with geometry soft IoU.

T-020 recorded **143 passed, 1 skipped** (CUDA unavailable), plus independent
artifact checks. Relevant regression coverage is in
[test_losses.py](../tests/test_losses.py),
[test_geometry.py](../tests/test_geometry.py),
[test_best_state.py](../tests/test_best_state.py),
[test_alignment.py](../tests/test_alignment.py),
[test_mesh_io.py](../tests/test_mesh_io.py), and
[test_cli.py](../tests/test_cli.py). Run the full suite with the project
interpreter using `python -m pytest -q -ra`; tests establish implementation
behavior, not reconstruction quality. No project type checker is configured.
The report's linked `outputs/t020*` evidence is ignored by Git: archive those
directories alongside the report when sharing verification results.
