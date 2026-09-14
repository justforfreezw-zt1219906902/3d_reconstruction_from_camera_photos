# Image-only reconstruction architecture

## Status and goal

This document specifies the `image_only` reconstruction path.  It is a
separate path from the existing CAD-prior pipeline: it reconstructs a mesh
from calibrated RGBA observations, while an `INITIAL_MESH_PATH` sphere supplies
only an approximate centre and scale.  In particular, it must not turn the
initial sphere into hidden shape supervision.

`cad_prior` remains the default to preserve existing behaviour.  Its ownership
chain remains:

```text
camera/pose -> global CAD-to-real R/T/uniform-S -> CAD residual deformation
```

The new chain is:

```text
independent frozen camera calibration + RGBA alpha masks
  -> visual-hull voxel occupancy -> marching-cubes coarse mesh
  -> image-only surface refinement -> validity-gated validation selection
  -> final.obj/final.stl -> optional post-completion GT evaluation
```

## Configuration contract

All keys are environment-backed `Config` fields and therefore appear in the
saved `run_config.json` snapshot.  Defaults below are deliberately conservative
and can be revised by a runtime profile; explicit environment values win.

| Environment key / `Config` field | Default | Meaning |
| --- | ---: | --- |
| `RECONSTRUCTION_MODE` / `reconstruction_mode` | `cad_prior` | Enum: `cad_prior` or `image_only`. Invalid values fail config validation. |
| `GROUND_TRUTH_MESH_PATH` / `ground_truth_mesh_path` | empty / `None` | Optional evaluation-only OBJ/STL path. It is not a reconstruction input. |
| `CAMERA_CALIBRATION_SOURCE` / `camera_calibration_source` | `positions_rig` | Enum: `positions_rig` (acquisition angles plus configured/known rig), or `calibration_file` (independent supplied calibration). Neither denotes mesh-silhouette fitting in image-only mode. |
| `CAMERA_CALIBRATION_PATH` / `camera_calibration_path` | empty / `None` | Required when source is `calibration_file`; contains intrinsics/extrinsics or rig parameters in the documented schema. |
| `COARSE_VOLUME_CENTER` / `coarse_volume_center` | initial-mesh normalized centre (`[0,0,0]`) | Explicit normalized carving-volume centre; absent value uses only the sphere-derived rough centre. |
| `COARSE_VOLUME_SCALE` / `coarse_volume_scale` | initial-mesh normalized diagonal | Explicit rough volume extent before padding; absent value uses only sphere-derived rough scale. |
| `COARSE_VOLUME_PADDING_RATIO` / `coarse_volume_padding_ratio` | `0.25` | Padding on every side relative to rough extent. The volume is never clipped to the sphere bounding box. |
| `COARSE_VOXEL_RESOLUTION` / `coarse_voxel_resolution` | `128` | Number of samples along the longest volume axis; other axes preserve volume aspect ratio. Must be at least 16. |
| `COARSE_MIN_SILHOUETTE_SUPPORT` / `coarse_min_silhouette_support` | `1.0` | Required fraction of usable views whose back-projected alpha contains a voxel. `1.0` is strict visual-hull intersection; lower values tolerate segmentation/calibration error. |
| `COARSE_OCCUPANCY_THRESHOLD` / `coarse_occupancy_threshold` | `0.5` | Iso-level used by marching cubes after support is converted to occupancy. |
| `COARSE_MIN_COMPONENT_VOXELS` / `coarse_min_component_voxels` | `64` | Remove tiny voxel components before surface extraction. |
| `COARSE_MARCHING_CUBES_STEP` / `coarse_marching_cubes_step` | `1` | Marching-cubes sampling stride; one uses the full grid. |
| `MESH_CLEANUP_ENABLED` / `mesh_cleanup_enabled` | `true` | Enable post-extraction removal of duplicate vertices, zero-area faces, tiny fragments, and unreferenced vertices. |
| `MESH_MIN_COMPONENT_FACES` / `mesh_min_component_faces` | `32` | Minimum connected-component size retained after extraction/refinement. |
| `MESH_MAX_EDGE_LENGTH_RATIO` / `mesh_max_edge_length_ratio` | `10.0` | Validity warning/failure threshold relative to median finite edge length. |
| `MESH_SPIKE_RATIO` / `mesh_spike_ratio` | `8.0` | Validity threshold for a vertex whose incident edges are extreme relative to local/median edges. |
| `IMAGE_ONLY_REFINEMENT_ENABLED` / `image_only_refinement_enabled` | `true` | If false, export the validity-checked coarse mesh as final. |
| `IMAGE_ONLY_COARSE_ANCHOR_WEIGHT` / `image_only_coarse_anchor_weight` | `0.0` | Optional explicit anchor to the coarse image-derived mesh, never to the sphere. Zero disables it. |

Existing `LOSS_ANCHOR_WEIGHT` and `MAX_LOCAL_DEFORMATION_RATIO` retain their
current defaults (`1.0` and `0.03`) but are **CAD-prior-only controls**.  Their
effective values in `image_only` are zero/not applied; configuration reporting
must say that they are ignored for this mode. `MAX_VERTEX_DISPLACEMENT_RATIO`
is likewise not used as a sphere/CAD-proximity signal in image-only mode; mesh
validity gates take its place.

## Module boundaries and APIs

Each module has one ownership role.  `pipeline.py` chooses the mode and wires
stage outputs; it must not reimplement their algorithms.

| Boundary | Proposed public responsibility | Inputs it may use | Inputs it must not use |
| --- | --- | --- | --- |
| `camera_fitting.py` / `openscan_pose.py` | Produce `FrozenCameraCalibration`: projection parameters, per-frame poses/convention, source metadata, and frame-consistency diagnostics. | `positions.csv`, independent rig/calibration file, configured initial intrinsics, dataset metadata. | Alpha/RGB agreement with the initial mesh, any GT mesh, coarse/final geometry. |
| `coarse_geometry.py` | Back-project alpha masks through frozen cameras, carve a bounded occupancy grid, clean it, and run marching cubes. Return a newly created `Meshes` plus voxel/volume diagnostics. | RGBA alpha masks, frozen cameras, configured volume, and sphere rough centre/scale only. | Sphere vertices/faces/topology as carving evidence, GT, optimizer losses. |
| `optimizer.py` and `losses.py` | Refine positions of vertices on the coarse mesh topology with image evidence and generic regularizers; choose a held-out-validation best state. | Coarse mesh, RGB/alpha observations, frozen cameras, train/validation split, validity summary. | Sphere reference mesh, `reference_anchor_loss`, CAD local-deformation cap, GT path/metrics. |
| `mesh_validity.py` | Return topology-agnostic `MeshValidityReport` used as an optimizer/export gate. | Arbitrary mesh vertices and faces and cleanup thresholds. | Initial sphere, CAD distances, GT. |
| `ground_truth_eval.py` | After successful export, load GT and calculate labelled post-hoc comparisons. | Exported reconstructed meshes, mesh transform/physical-frame metadata, GT path. | Optimizer state, camera fitting, coarse carving, selection callback. |
| `mesh_io.py` | Normalize/restores meshes and exports raw physical-frame OBJ/STL; exposes conversion metadata. | Vertices/faces and `MeshTransform`. | Any evaluation or training policy. |

`CameraFitGateError` is the image-only calibration gate: if positions/rig data
cannot establish a complete independent calibration, it fails with an explicit
message. `--skip-camera-fit` is permitted only when an already complete,
independent calibration source is available; it never means "use the sphere to
guess a camera." `--camera-fit-only` writes calibration metadata and stops.

## Camera and coordinate-frame contract

There are two named object frames, and no stage may silently mix them.

| Frame | Definition | Producers / consumers |
| --- | --- | --- |
| `physical_object` | Source mesh units and origin. | Input sphere, raw GT, OBJ/STL export, primary GT metrics. |
| `normalized_object` | `p_norm = (p_physical - mesh_transform.center) / mesh_transform.scale`, applying only enabled centre/scale steps. | Frozen geometry cameras, carving volume/grid, coarse mesh, refinement, renderer, validity checks. |

`MeshTransform` is written once for the run and is the sole authority for the
conversion.  Coarse geometry receives a normalized volume, returns normalized
vertices, and refinement preserves that frame.  Export restores vertices once:
`p_physical = p_norm * scale + center` for enabled operations.  It must never
apply the transform to faces, camera angles, or twice to vertices.

Camera calibration records its declared source frame. Acquisition angles are
unitless and pass through unchanged; focal/FOV values are projection
parameters and are unchanged. Any physical camera target/translation is
converted explicitly to normalized world units before `OpenScanCameraModel`
is frozen (distance and translation scale by `1/scale`, with the centre shift
applied consistently). This conversion and the convention name are persisted
in `camera_calibration.json`. Thus a ray used to test a voxel and a ray used by
the differentiable renderer describe the same normalized object frame.

Global CAD-to-real alignment is skipped in `image_only`: there is no CAD shape
whose R/T/S can be aligned. The optional sphere is only used to initialize an
otherwise expanded volume; it cannot establish a hidden object transform.

## Coarse geometry: calibrated visual hull

1. Validate RGBA masks and obtain a calibration with all camera parameters
   frozen (`requires_grad=False`) before any voxel test.
2. Determine the normalized axis-aligned volume from explicit coarse settings
   or the sphere's rough centre/scale, then expand every side by
   `COARSE_VOLUME_PADDING_RATIO`. The generated mesh may therefore extend
   outside the initial sphere bounds.
3. At voxel centres, project into each usable alpha mask with the same camera
   convention used by rendering. Count foreground support and retain occupancy
   at or above `COARSE_MIN_SILHOUETTE_SUPPORT`.
4. Remove configured tiny components; extract the configured occupancy
   iso-surface with marching cubes; map vertex coordinates from voxel indices
   into `normalized_object` coordinates; run cleanup and validity checks.

The output's connectivity is generated by marching cubes and must not reuse
the sphere faces or vertex ordering. A failure to obtain a closed/nonempty
enough surface is a coarse-geometry gate error, not a reason to fall back to
sphere topology.

## Refinement and selection

The coarse mesh is the sole refinement reference/topology. Fixed topology
during this stage is acceptable because its topology originates from images,
not the sphere. Vertex positions are optimized with frozen cameras and:

- silhouette loss: match observed alpha masks;
- optional masked RGB loss: use photo evidence where appearance modelling is
  meaningful;
- normal consistency: discourage adjacent faces from folding;
- edge regularity: discourage extreme local edge distortion/collapse;
- optional coarse-image anchor: an explicitly weighted proximity term to the
  **coarse mesh**, if enabled;
- validity penalties/gates: reject non-finite, degenerate, fragmented, or
  catastrophically spiky candidates.

`reference_anchor_loss`, `local_deformation_smoothness_loss` as a CAD-offset
field, and the 3% CAD local-displacement projection are not called in this
path. Image-only smoothness may use current/coarse edges but must be named
separately so it cannot be mistaken for CAD preservation.

Use the existing deterministic, disjoint seed-based train/validation split.
Validation renders under `torch.no_grad()` and never back-propagates. Best
state comparison first requires a passing `MeshValidityReport`, then prefers
better held-out silhouette loss/IoU; within an explicit quality tolerance it
prefers fewer validity warnings (and then deterministic epoch ordering).
It has no GT argument, import, field, or metric. The selected state—not the
last epoch—is restored and exported.

`MeshValidityReport` includes: non-finite vertex count, zero/near-zero-area
face count, extreme-edge count/distribution, disconnected component count and
discarded faces, spike count, and detectable self-intersection count/unknown
status. These are geometry properties and never distances to the sphere.

## Ground-truth isolation contract

`GROUND_TRUTH_MESH_PATH` has an intentionally one-way boundary:

```text
reconstruction inputs (RGBA + independent calibration + optional rough sphere)
  -> camera -> coarse mesh -> refinement/validation selection -> final export

GROUND_TRUTH_MESH_PATH
  -> ground_truth_eval.evaluate_after_reconstruction(final, coarse, sphere)
  -> ground_truth_evaluation.json
```

No reconstruction module imports `ground_truth_eval`, reads the path, accepts
GT vertices, or has a GT-dependent branch. The path is loaded only inside the
evaluation module's public entry point, after final OBJ/STL export. Evaluation
receives already-produced sphere/coarse/final meshes in physical frame (or
restores them using `MeshTransform`) and reports, for each stage, symmetric
Chamfer, both directional mean surface distances, p95, Hausdorff/robust
Hausdorff, and normal consistency. Distances are reported in physical units
and divided by the GT bounding-box diagonal. Optional rigidly aligned results
are clearly diagnostic only; physical-frame results are primary.

Tests enforce the boundary by mocking/guarding GT mesh loading and asserting
that camera fitting, `coarse_geometry`, `optimizer`, and selection can run
without—and cannot receive—a GT path. Import tests ensure those modules do
not import evaluation; a call-order test proves `evaluate_after_reconstruction`
runs only after exports. Further deterministic tests verify image-only camera
calibration never renders the sphere, CAD anchor/cap calls are absent, coarse
connectivity differs topologically from a synthetic sphere case, validation is
disjoint/no-gradient, and final OBJ/STL contain finite vertices and valid
faces.

## Diagnostics and artifacts

All image-only runs preserve the ordinary validation/config snapshots and add:

```text
reconstruction/
  initial_sphere.stl
  image_coarse_mesh.stl
  final.obj
  final.stl
  view_split.json
  losses.csv
  reconstruction_metrics.json
  camera_calibration.json
  mesh_transform.json
  previews/{sphere,coarse,final}_*.png
  ground_truth_evaluation.json        # only when GT is configured, after export
```

`reconstruction_metrics.json` records the mode, camera source and freeze/gate
status, volume/carving and cleanup parameters, coarse and final validity
reports, deterministic train/validation indices, refinement weights, selected
best epoch/state, and per-stage held-out image metrics. It must not contain GT
metrics used for selection. `losses.csv` labels train versus validation metrics
and validity fields. CAD-prior artifacts remain available and retain their
existing meanings.

## Silhouette limitations

A visual hull contains every point that is consistent with all foreground
silhouettes; it is not a full observation of surface geometry. Concavities not
visible in any silhouette, cavities, through-holes hidden by view coverage,
internal structure, and occluded back-side detail cannot in general be
recovered from alpha masks. Limited angles, calibration error, segmentation
error, and coarse voxels can additionally create inflated volumes, missing thin
parts, aliasing, or spurious components. RGB refinement may improve visible
surface agreement but does not make unseen geometry observable. These limits
must be reported with benchmark results rather than interpreted as evidence
that the generic sphere or GT should be used as a reconstruction prior.
