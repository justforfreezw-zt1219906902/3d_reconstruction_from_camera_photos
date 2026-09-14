# T-020: Empirical verification on the real dataset

Verified 2026-09-09. **PASS for the requested empirical checks**, with reconstruction-quality limitations below. The reduced 32-epoch CPU run completed all 384 local optimization steps without triggering the emergency displacement gate. Every epoch stayed below `MAX_LOCAL_DEFORMATION_RATIO=0.03`. The final exports restore **epoch 27, not epoch 32**, verified using checkpoint offsets, selection metrics, and both mesh formats.

## Scope and reproduction

Verified the existing working tree based on commit `2b434c9c732e0808803e2b5f142d20ec1c572544`, including pre-existing modifications to config, losses, optimizer, visualization and tests. No production code or existing tests were changed. `.forge/tasks/T-020.md` assigns `reports/empirical_verification.md` to Codex; orchestration state remains managed by Forge.

Interpreter: `/Users/zhaowei/PyCharmMiscProject/xolo_3d_reconstrcuction/segmentation_compare/.venv/bin/python` (Python 3.10.10, CPU PyTorch3D). Run from the repository root:

```sh
PY=/Users/zhaowei/PyCharmMiscProject/xolo_3d_reconstrcuction/segmentation_compare/.venv/bin/python
OUTPUT_DIR=outputs/t020-extended GEOMETRY_MAX_FACES=1500 MAX_IMAGE_DIMENSION=64 GEOMETRY_VIEWS_PER_EPOCH=12 NUM_EPOCHS=32 EXPORT_EVERY_EPOCHS=1 SAVE_PREVIEW_EVERY_EPOCHS=32 OMP_NUM_THREADS=1 "$PY" main.py
"$PY" -m pytest -q -ra
```

Other settings came from the existing `.env` and Config defaults; `.env` was unchanged. The complete effective configuration is [run_config.json](../outputs/t020-extended/run_config.json). Inputs:

- CAD: `/Users/zhaowei/Downloads/openscan-benchy-model_files/openscanbenchy-45mm.stl`.
- RGBA: `/Users/zhaowei/PyCharmMiscProject/xolo_3d_reconstrcuction/demo_preparing_and_evaluation/output/processed/rgba`.
- Positions: `/Users/zhaowei/Downloads/2026-08-25_12.33.45-default/positions.csv`.

All 30 usable real frames were retained: 24 local-training and 6 deterministic validation views, seed 42, validation fraction 0.20. Each epoch samples 12 training views. Geometry uses a 722-vertex / 1,500-face proxy (source: 93,849 vertices / 187,754 faces), 64-pixel square canvas, batch size 1, five faces per pixel, vertex learning rate 0.0005, anchor weight 1.0, local smoothness weight 0.1, normal weight 0.01, RGB weight 0. No early stopping. Local limit 0.03; emergency gate 0.25.

Camera fitting retained its original 128-pixel / 3,000-face / 12-frame / 200-evaluation settings and pose refinement. Camera gate median IoU was 0.635503 against threshold 0.50. Global alignment retained 10 epochs and learning rate 0.01, followed by frozen cameras/alignment and local offsets. Geometry wall time was 62.01 seconds. Camera-gate IoU uses a different rendering/evaluation path and should not be directly compared with the soft geometry IoU below.

An initial eight-epoch run under `outputs/t020/` used the same settings except `NUM_EPOCHS=8` and `SAVE_PREVIEW_EVERY_EPOCHS=8`; it completed in 16.08 seconds of geometry time and selected epoch 8. The 32-epoch rerun was used to observe rejected late epochs and demonstrate restoration of an earlier best state.

## Per-epoch evidence

Training silhouette/IoU are averages over changing sampled training views **before each optimization update**. Validation metrics evaluate all six fixed holdouts **after each epoch**. Training fluctuations therefore are not a fixed-view learning curve. Displacements below are in normalized aligned-object coordinates, relative to the frozen aligned CAD proxy. Multiply by 30.8714065552 for source mesh units. The aligned reference bounding-box diagonal is approximately 2.2607194033 normalized units. Ratios are dimensionless; displayed 0.030000 values are rounded, not violations.

| Epoch | Train silhouette | Train IoU | Val silhouette | Val IoU | Mean disp. | p95 disp. | Max disp. | Max ratio | Best epoch | Selected now |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.118837 | 0.192846 | 0.099099 | 0.264435 | 0.006697 | 0.010572 | 0.016757 | 0.007412 | 1 | yes |
| 2 | 0.079040 | 0.337148 | 0.097774 | 0.266204 | 0.011375 | 0.018549 | 0.029221 | 0.012925 | 2 | yes |
| 3 | 0.100814 | 0.235171 | 0.096238 | 0.268660 | 0.014985 | 0.024369 | 0.042008 | 0.018582 | 3 | yes |
| 4 | 0.090455 | 0.305707 | 0.094940 | 0.270433 | 0.018054 | 0.029886 | 0.054567 | 0.024137 | 4 | yes |
| 5 | 0.074907 | 0.348102 | 0.093587 | 0.273271 | 0.020709 | 0.033750 | 0.065934 | 0.029165 | 5 | yes |
| 6 | 0.111138 | 0.202954 | 0.093037 | 0.274000 | 0.021724 | 0.035578 | 0.067822 | 0.030000 | 6 | yes |
| 7 | 0.078808 | 0.340033 | 0.092564 | 0.275110 | 0.023105 | 0.038705 | 0.067822 | 0.030000 | 7 | yes |
| 8 | 0.105006 | 0.215576 | 0.092422 | 0.274892 | 0.024014 | 0.041512 | 0.067822 | 0.030000 | 8 | yes |
| 9 | 0.103529 | 0.236962 | 0.092503 | 0.273642 | 0.024696 | 0.044119 | 0.067629 | 0.029915 | 8 | no |
| 10 | 0.079483 | 0.317946 | 0.092226 | 0.274579 | 0.025484 | 0.045705 | 0.067729 | 0.029959 | 10 | yes |
| 11 | 0.109151 | 0.206940 | 0.091926 | 0.275483 | 0.026102 | 0.047576 | 0.067822 | 0.030000 | 11 | yes |
| 12 | 0.072840 | 0.351192 | 0.091739 | 0.276179 | 0.026574 | 0.048686 | 0.067822 | 0.030000 | 12 | yes |
| 13 | 0.090969 | 0.263458 | 0.091681 | 0.276913 | 0.026947 | 0.050696 | 0.067694 | 0.029943 | 13 | yes |
| 14 | 0.090315 | 0.297936 | 0.091611 | 0.277678 | 0.027002 | 0.051146 | 0.067822 | 0.030000 | 14 | yes |
| 15 | 0.072092 | 0.355935 | 0.091269 | 0.279916 | 0.027162 | 0.052165 | 0.067822 | 0.030000 | 15 | yes |
| 16 | 0.108539 | 0.209512 | 0.091051 | 0.280717 | 0.027578 | 0.052080 | 0.067822 | 0.030000 | 16 | yes |
| 17 | 0.079446 | 0.338560 | 0.090924 | 0.281213 | 0.027986 | 0.053908 | 0.067822 | 0.030000 | 17 | yes |
| 18 | 0.100715 | 0.229122 | 0.091050 | 0.280789 | 0.028154 | 0.053202 | 0.067822 | 0.030000 | 17 | no |
| 19 | 0.107945 | 0.212229 | 0.091111 | 0.279511 | 0.028616 | 0.054803 | 0.067822 | 0.030000 | 17 | no |
| 20 | 0.072084 | 0.355367 | 0.091034 | 0.279892 | 0.028853 | 0.055638 | 0.067683 | 0.029939 | 17 | no |
| 21 | 0.107538 | 0.213752 | 0.090937 | 0.279899 | 0.029317 | 0.056383 | 0.067822 | 0.030000 | 17 | no |
| 22 | 0.071959 | 0.355771 | 0.090806 | 0.280599 | 0.029318 | 0.056933 | 0.067822 | 0.030000 | 22 | yes |
| 23 | 0.086044 | 0.286951 | 0.090904 | 0.280612 | 0.029058 | 0.056372 | 0.067822 | 0.030000 | 22 | no |
| 24 | 0.093564 | 0.282938 | 0.090986 | 0.280076 | 0.029215 | 0.056798 | 0.067822 | 0.030000 | 22 | no |
| 25 | 0.071719 | 0.357650 | 0.090566 | 0.282599 | 0.029683 | 0.057838 | 0.067822 | 0.030000 | 25 | yes |
| 26 | 0.107485 | 0.214523 | 0.090698 | 0.281664 | 0.029925 | 0.058245 | 0.067726 | 0.029958 | 25 | no |
| 27 | 0.081706 | 0.332654 | 0.090455 | 0.282798 | 0.030159 | 0.058463 | 0.067813 | 0.029996 | 27 | yes |
| 28 | 0.097155 | 0.241198 | 0.090518 | 0.282429 | 0.030168 | 0.057007 | 0.067822 | 0.030000 | 27 | no |
| 29 | 0.107075 | 0.216140 | 0.090646 | 0.281289 | 0.029950 | 0.057818 | 0.067822 | 0.030000 | 27 | no |
| 30 | 0.071934 | 0.355569 | 0.090518 | 0.282031 | 0.030463 | 0.059081 | 0.067764 | 0.029975 | 27 | no |
| 31 | 0.105859 | 0.218420 | 0.090494 | 0.281974 | 0.030438 | 0.058914 | 0.067822 | 0.030000 | 27 | no |
| 32 | 0.072895 | 0.354164 | 0.090496 | 0.282380 | 0.030353 | 0.058899 | 0.067822 | 0.030000 | 27 | no |

Full precision: [losses.csv](../outputs/t020-extended/reconstruction/losses.csv). The largest ratio across all 32 epochs is **0.029999976139362575 < 0.03**, also far below the 0.25 emergency gate. The bound first becomes active at epoch 6. Independent checks recomputed mean, p95, max and ratio from every saved offset tensor and matched the CSV within seven decimal places. The eight-epoch run also stayed below the bound. This is per-epoch empirical evidence; the existing bound tests and optimizer projection cover intermediate render/update behavior.

## Best-state and mesh export verification

The selection policy prefers lower held-out silhouette loss, then higher IoU within an absolute quality tolerance of 1e-6, then smaller mean/p95 displacement for effectively tied quality. Replaying this policy against all rows reproduces every `is_best` and `best_epoch` entry. Epoch 27 has validation silhouette **0.0904551384**, IoU **0.2827983623**; epoch 32 is worse at **0.0904964991**, **0.2823795875**. Their silhouette difference exceeds the selection tolerance.

An independent OBJ parser recovered the common frozen reference from epoch 1's exported vertices minus epoch 1's saved offsets times normalization scale. All 32 epoch exports minus their respective offsets agreed with that reference within 2e-5 source units. Subtracting this reference from the final OBJ and dividing by normalization scale gave:

| Comparison | Maximum absolute component error, normalized units |
| --- | --- |
| Final recovered offsets versus epoch 27 checkpoint | 4.71944614e-7 |
| Final recovered offsets versus epoch 32 checkpoint | 0.0227058779 |

The maximum vertex-vector distance between epoch 27 and epoch 32 offsets is 0.0229192413 normalized units. Both `final.obj` and `final.stl` are **byte-identical** to their epoch-27 checkpoint exports and **different** from epoch-32 exports. Binary STL triangle count, file length and coordinates were independently checked against the parsed OBJ, with coordinate tolerance 1e-5 source units. Exports contain 722 OBJ vertices and 1,500 triangles; all coordinates are finite.

Final export SHA-256:

- OBJ: `9363b91cdb70c498103507240f7ce0033cce97c4ada4f857b7cdbeee7b4ebca3`
- STL: `14a697b7d6b80c8b15aceb5dcd5d5996bddbe58ddcf6c41f4c2b94cc2e8763ea`

Machine-readable comparison results: [verification.json](../outputs/t020-extended/verification.json).

## Does training improvement still hide validation/deformation degradation?

**Small validation regressions and increasing mean/p95 displacement still occur, but displacement growth is bounded and worse validation states no longer automatically become the final mesh.** From epoch 27 to 32, training silhouette improves from 0.081706 to 0.072895 and training IoU increases from 0.332654 to 0.354164, while validation silhouette/IoU worsen slightly and mean/p95 displacement increase from 0.030159/0.058463 to 0.030353/0.058899. Those later states are rejected; epoch 27 is exported. The training comparison is descriptive because its sampled views differ.

Across the run, validation silhouette improves substantially from epoch 1 to the selected epoch, but mean displacement continues increasing even after maximum displacement reaches its cap. A maximum bound does not freeze the displacement distribution. Thus the prior failure mode is controlled in this experiment, not eliminated as a general optimization tendency. These results do not prove the anchor/smoothness weights are optimal or isolate their causal benefit: no matched unbounded ablation was run, and the earlier SPEC metrics used different settings.

Validation is held out from **local geometry gradients**, not from the entire pipeline: camera evaluation/refinement and global alignment use the available dataset upstream. The result is not an independent end-to-end generalization estimate.

## Qualitative visual assessment

The actual parsed final OBJ was plotted from two oblique viewpoints alongside the recovered frozen aligned reference and the last epoch, using common coordinates and aspect ratios. The inspection sheet was opened and visually reviewed:

![Aligned reference, selected epoch 27, and last epoch 32](../outputs/t020-extended/reconstruction/mesh_inspection.png)

The upright figure, circular base, loop/openings, and thin pointed protrusions remain recognizable. No gross collapse or large separation from the aligned prior is visible from these viewpoints. Much of the faceting and uneven triangulation is already present in the aggressively decimated reference; local optimization alters surface contours around the body/base and appendages. The final mesh still looks coarse and locally irregular. Differences between epochs 27 and 32 are subtle at this display scale, so the numeric comparisons, rather than appearance alone, establish which state was exported.

This is a limited visual assessment, not proof of watertightness, absence of self-intersections, dimensional accuracy, or faithful fine-detail recovery. Only one elevation ring (phi -30 degrees) and a 142.5-degree angular gap are available. Low geometry validation IoU (~0.283), coarse resolution and incomplete coverage limit any claim of high-quality 3D reconstruction.

## Tests and artifacts

- Full existing suite: **143 passed, 1 skipped, 14 warnings**, 3.86 seconds. CUDA alignment test skipped because CUDA is unavailable. [Test log](../outputs/t020/tests.log).
- Four inline `unittest` evidence checks passed: completion/split/gate; all-epoch bounds and recomputed statistics; selection replay and recovered export offsets; OBJ/STL consistency and common reference across checkpoints.
- Report-specific inline unit checks verify all 32 table rows against the source CSV, required conclusions and thresholds, artifact links, and the inspection image. No new executable source module or test file was introduced. This report is the only new non-ignored repository file; generated run artifacts are under ignored `outputs/` directories.
- No project type-checker configuration, dependency or command was found; type checker: **not applicable**. No corresponding `.test` file or mocks were modified. `git diff --check` passes.

Primary artifacts: [run log](../outputs/t020-extended/run.log), [configuration](../outputs/t020-extended/run_config.json), [summary](../outputs/t020-extended/summary.json), [split](../outputs/t020-extended/reconstruction/view_split.json), [profile](../outputs/t020-extended/reconstruction/profile.json), [final OBJ](../outputs/t020-extended/reconstruction/final.obj), [final STL](../outputs/t020-extended/reconstruction/final.stl), and all 32 offset/mesh checkpoints in `outputs/t020-extended/reconstruction/checkpoints/`. These local artifacts are ignored by Git; retain that directory with this report when sharing or archiving the evidence.
