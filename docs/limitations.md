# Silhouette reconstruction limitations

`RECONSTRUCTION_MODE=image_only` creates an image-derived visual hull from
calibrated RGBA alpha masks. It is an evidence-driven baseline, not a complete
observation of an object. Results must be interpreted with the limitations
below in mind.

## What silhouettes cannot determine

- A silhouette retains an outline, not the depth of every visible surface.
  Different 3D shapes can have identical masks from the available cameras.
- Concavities that do not alter any observed outline cannot be recovered.
  This includes recesses, dimples, undercuts, and cavities hidden by the
  external contour.
- Internal structure is unobservable from alpha masks: material thickness,
  internal ribs, enclosed voids, and other occluded geometry have no direct
  silhouette evidence.
- Through-holes or openings are recoverable only when enough views reveal
  them. A hole hidden by the available view directions can be filled by the
  visual hull.
- Back-side and self-occluded detail cannot be inferred reliably without
  views that expose it. RGB refinement can improve agreement for observed
  surfaces, but it cannot establish geometry that no image observes.

## Practical sources of error

The visual hull is also sensitive to view coverage, camera calibration,
foreground segmentation, and grid resolution. Sparse angles can broaden or
miss features; calibration or mask errors can carve away real material or add
inflated regions; coarse voxels can alias thin features and small holes.
Marching cubes and cleanup may remove tiny disconnected components or smooth
features below the configured resolution. Mesh-validity diagnostics identify
degenerate, non-finite, fragmented, extreme-edge, and spike-like outputs, but
they do not prove that an apparently valid mesh is the object's true geometry.

Report held-out silhouette agreement, validity diagnostics, camera source,
view coverage, carving parameters, and the limits above with every benchmark.
Do not interpret a close visual-hull result as evidence that unobserved
concavities or internal geometry were recovered.

## Ground-truth isolation

An optional `GROUND_TRUTH_MESH_PATH` is not a reconstruction input. Camera
calibration, coarse carving, refinement, validity diagnostics, and best-state
selection operate without it. The pipeline passes a configuration copy with
the path removed into reconstruction stages, exports `final.obj`/`final.stl`,
and only then calls the evaluation module to load the ground-truth mesh and
write `ground_truth_evaluation.json`.

Automated isolation tests guard this contract: they reject evaluation imports
or GT-path use in reconstruction components, ensure GT loading occurs only in
the evaluation entry point, and assert that evaluation is ordered after stage
exports. Ground-truth measurements are therefore post-hoc benchmark metrics,
never losses, camera-fitting signals, validation inputs, or best-epoch
selection criteria.
