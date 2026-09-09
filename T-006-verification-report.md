# T-006: Independent alignment integration verification

Verified 2026-09-09. Result: PASS for CPU compatibility and export checks; CUDA coverage skipped because CUDA is unavailable. No production regression found within the checks below. This is not a reconstruction-quality certification.

Verified the working tree based on `091e006fa086498f02a8c6af4f99bbe58d16bb0e`, including the pre-existing changes to `src/config.py`, `src/optimizer.py`, `src/pipeline.py`, `tests/test_geometry.py`, and the untracked alignment module and tests. `.forge/tasks/T-006.md` assigned this verification to Codex; the aggregate state still reported six pending tasks. Orchestration state was left untouched.

| Check | Status | Evidence |
| --- | --- | --- |
| `test_openscan_pose.py` | PASS | 2 tests |
| `test_camera_fit_optimization.py` | PASS | 5 tests |
| `test_mesh_io.py` | PASS | 1 test |
| `test_validation_and_rgba.py` | PASS | 3 tests |
| `test_geometry.py` | PASS | 13 tests, including alignment integration and restored exports |
| `test_alignment.py` | PASS / CUDA SKIP | 40 passed, 1 skipped |
| Full existing suite | PASS | `python -m pytest -q -ra`: 64 passed, 1 skipped, 2 warnings in 3.76 seconds |
| CLI help | PASS | Exit 0; all three existing flags remain exposed |
| Mutually exclusive flags | PASS | `--camera-fit-only --skip-camera-fit` exits 2 with the expected parser error |
| `main.py --validate-only` | PASS | Exit 0, 30 real RGBA frames validated |
| `main.py --camera-fit-only` | PASS | Exit 0; standalone camera-fit path completed |
| `main.py --skip-camera-fit` | PASS | Exit 0; alignment, local optimization and exports completed |
| `main.py` without arguments | PASS | Exit 0; camera fit, alignment, local optimization and exports completed |
| `positions.csv` compatibility | PASS | Six-column source accepted, 30 rows; SHA-256 unchanged across reconstruction runs |
| `final.obj` / `final.stl` | PASS | Both reconstruction runs independently parsed and checked as described below |
| Shared ownership constraints | PASS | Frozen cameras → explicit global similarity → detached aligned mesh → local residual offsets |
| Type checker | N/A | No project type-checker configuration, command, or dependency found |
| Production files preserved | PASS | SHA-256 comparison of `src/*.py`, `main.py`, and existing `tests/*.py` unchanged |
| Whitespace | PASS | `git diff --check` |

## Environment and reproduction

Interpreter: `/Users/zhaowei/PyCharmMiscProject/xolo_3d_reconstrcuction/segmentation_compare/.venv/bin/python` (Python 3.10.10, Torch 2.13.0, CPU with working PyTorch3D).

Inputs were the existing `.env` dataset: `/Users/zhaowei/Downloads/openscan-benchy-model_files/openscanbenchy-45mm.stl`, RGBA images under `/Users/zhaowei/PyCharmMiscProject/xolo_3d_reconstrcuction/demo_preparing_and_evaluation/output/processed/rgba`, and `/Users/zhaowei/Downloads/2026-08-25_12.33.45-default/positions.csv`.

Commands below use `PY` for the interpreter above. All other environment values were inherited from the existing `.env`; its contents were not changed.

```sh
"$PY" -m pytest -q -ra
"$PY" main.py --help
"$PY" main.py --camera-fit-only --skip-camera-fit
OUTPUT_DIR=/tmp/t006-verification/validate "$PY" main.py --validate-only
OUTPUT_DIR=/tmp/t006-verification/camera-only CAMERA_FIT_MAX_DIMENSION=32 CAMERA_FIT_MAX_FACES=500 CAMERA_FIT_MAX_EVALUATIONS=20 "$PY" main.py --camera-fit-only
OUTPUT_DIR=/tmp/t006-verification/reconstruct GEOMETRY_MAX_FACES=500 MAX_IMAGE_DIMENSION=32 GEOMETRY_VIEWS_PER_EPOCH=2 NUM_EPOCHS=1 "$PY" main.py --skip-camera-fit
OUTPUT_DIR=/tmp/t006-verification/default GEOMETRY_MAX_FACES=500 MAX_IMAGE_DIMENSION=32 GEOMETRY_VIEWS_PER_EPOCH=2 NUM_EPOCHS=1 CAMERA_FIT_MAX_DIMENSION=32 CAMERA_FIT_MAX_FACES=500 CAMERA_FIT_MAX_EVALUATIONS=20 "$PY" main.py
```

These are reduced-cost smoke runs using existing environment settings and unchanged CLI arguments. Alignment remained enabled with its default 10 epochs and learning rate 0.01. Safety thresholds were not lowered. The no-argument run passed the camera gate over all 30 frames, with median IoU 0.6397986 against threshold 0.50. The full default-resolution, 20-epoch experiment was not run.

## Export and CSV validation

`positions.csv` is an INPUT contract, not an exported pipeline artifact. Its header remains `image,position_index,phi_deg,theta_deg,focus_index,focus_value`. Validation accepted all 30 rows and matched their JPG names to processed PNGs by filename stem. The source checksum after both reconstruction runs matched the checksum captured before the no-argument run and before the skip-camera-fit run finished: `948a355436050376a2813561b345da8a35dccea63c7b4a7a81940a3f82c62705`. Source inspection also found no positions.csv writer; camera fitting writes `frame_poses.csv` separately.

An inline `unittest` verification pass ran three checks successfully: mesh formats/coordinates for both runs, CSV preservation, and unchanged source/test hashes. OBJ vertices and triangular face indices were parsed independently; STL was decoded using its binary header and 50-byte triangle records. Checks covered finite coordinates, valid indices, exact file length, matching triangle counts and winding, nondegenerate triangles, normalized face normals and zero attribute fields.

| Run | Vertices / triangles | Maximum OBJ–STL coordinate error | Maximum distance to aligned proxy, source units |
| --- | --- | --- | --- |
| Skip camera fit | 222 / 500 | 4.9952e-7 | 0.08946082 |
| No arguments | 222 / 500 | 4.9866e-7 | 0.07034126 |

Both global rotations were orthonormal with determinant +1 within 1e-6, with positive uniform scale. Exported positions were compared to the independently composed transform `((raw_proxy - center) / normalization_scale @ rotation.T * alignment_scale + translation) * normalization_scale + center`. Maximum nearest-point deviations were within the recorded local displacement times normalization scale, plus 2e-5 numerical tolerance. The existing synthetic integration test additionally validates known aligned coordinates through both exporters to 1e-5 with zero local displacement. Together these checks support preserved restoration and a single application of global alignment.

Artifacts and logs are under `/tmp/t006-verification/`: `validate.log`, `camera-only.log`, `reconstruct.log`, `default.log`, `export-check.json`, and the corresponding output directories. Successful reconstruction outputs retain `reconstruction/final.obj`, `reconstruction/final.stl`, `mesh_transform.json`, and `summary.json` paths. These temporary artifacts may be cleaned by the operating system.

## Findings and limits

- No integration regression detected. Camera parameters are frozen during alignment; only global rotation/translation/log-scale parameters are optimized there. The aligned mesh is detached before local training, whose offset projection excludes infinitesimal global similarity modes. Local safety checks use the aligned object size. CAD remains a prior; residual deformation is permitted.
- CUDA execution remains unverified. The skipped test is `test_cuda_fit_and_module`.
- Existing warnings: PyTorch sparse invariant checks disabled, OBJ without an MTL file, and dataset angular coverage gap of 142.5 degrees with only one elevation ring. These do not fail the compatibility checks or establish an integration regression.
- An initial ad hoc export check incorrectly treated `load_fast_mesh`'s tuple return as a mesh. Correcting the verification expression to unpack the tuple and disable normalization produced the successful checks above; production code needed no change.
- This report is the only new repository file. No executable modules or test files were created or changed; report assertions were checked with inline unittest checks. No corresponding `.test` mocks required updating.
