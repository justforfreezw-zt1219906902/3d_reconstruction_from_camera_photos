# OpenScan Reconstruction Demo

This project reconstructs 3D geometry from one processed OpenScan RGBA image folder and one OpenScan pose CSV. The alpha channel is the silhouette source. The default experiment fits camera nuisance parameters first, gates that fit, then performs controlled geometry refinement from the initial STL.

Two reconstruction modes are available:

- `cad_prior` (the default) treats `INITIAL_MESH_PATH` as a CAD design prior and performs global alignment followed by bounded residual deformation.
- `image_only` reconstructs a new mesh from independently calibrated multi-view RGBA silhouettes. The initial mesh is only a rough centre/scale hint for the carving volume; its faces, vertices, and shape are not reconstruction evidence.

## Inputs

```text
dataset/
├── rgba/
│   ├── default_0_1.png
│   ├── default_0_2.png
│   └── ...
└── positions.csv
```

`positions.csv` must contain at least:

```text
image,position_index,phi_deg,theta_deg
default_0_1.png,0,-30,0
default_0_2.png,1,-30,10
```

Images must be RGBA PNGs. RGB is target appearance and alpha is the object mask. Separate mask folders, white-background inference, and thresholding are not used.

If a preprocessing step converted the images from JPG to PNG but left the old `.jpg` suffixes in `positions.csv`, the loader matches the corresponding processed PNG by filename stem and records the mapping in `validation.json`.

The initial mesh is supplied separately and may be `.stl` or `.obj`, with STL as the primary workflow:

```env
INITIAL_MESH_PATH=/absolute/path/to/reference.stl
RGBA_DIR=/absolute/path/to/dataset/rgba
POSITIONS_CSV=/absolute/path/to/dataset/positions.csv
OUTPUT_DIR=./outputs/demo
```

## Image-only reconstruction

Set `RECONSTRUCTION_MODE=image_only` to run this sequence:

```text
independent, frozen camera calibration
→ calibrated alpha-mask voxel carving / visual hull
→ marching-cubes image-derived coarse mesh
→ optional silhouette/RGB refinement with generic mesh regularizers
→ validity-gated held-out-validation selection
→ final OBJ/STL export
→ optional post-export ground-truth evaluation
```

The input mesh may be a generic sphere in this mode. It supplies a rough normalized centre and extent only; the carving volume includes configurable padding and can extend beyond the sphere bounds. The coarse mesh is newly created by marching cubes, so its connectivity does not inherit the sphere topology.

An image-only `.env` needs the usual input paths plus independent calibration. `positions_rig` uses the acquisition angles/known rig contract; `calibration_file` requires a separately measured calibration file. Image-only camera fitting never compares the input sphere to an alpha mask. If an independent calibration is incomplete, the camera gate fails rather than fitting projection parameters to absorb shape mismatch.

```env
RECONSTRUCTION_MODE=image_only
INITIAL_MESH_PATH=/absolute/path/to/generic_sphere.stl
RGBA_DIR=/absolute/path/to/dataset/rgba
POSITIONS_CSV=/absolute/path/to/dataset/positions.csv
OUTPUT_DIR=./outputs/image_only

CAMERA_CALIBRATION_SOURCE=positions_rig
# Or: CAMERA_CALIBRATION_SOURCE=calibration_file
#     CAMERA_CALIBRATION_PATH=/absolute/path/to/calibration.json

# Optional coarse-volume overrides; omitted values use the sphere only as a rough hint.
COARSE_VOLUME_CENTER=0,0,0
COARSE_VOLUME_SCALE=2.0
COARSE_VOLUME_PADDING_RATIO=0.25
COARSE_VOXEL_RESOLUTION=128
COARSE_MIN_SILHOUETTE_SUPPORT=1.0
COARSE_OCCUPANCY_THRESHOLD=0.5
COARSE_MIN_COMPONENT_VOXELS=64
COARSE_MARCHING_CUBES_STEP=1

MESH_CLEANUP_ENABLED=true
MESH_MIN_COMPONENT_FACES=32
MESH_MAX_EDGE_LENGTH_RATIO=10.0
MESH_SPIKE_RATIO=8.0
IMAGE_ONLY_REFINEMENT_ENABLED=true
IMAGE_ONLY_COARSE_ANCHOR_WEIGHT=0.0

# Optional and evaluation-only; never an input to reconstruction.
# GROUND_TRUTH_MESH_PATH=/absolute/path/to/benchmark_ground_truth.stl
```

`LOSS_ANCHOR_WEIGHT` and `MAX_LOCAL_DEFORMATION_RATIO` are CAD-prior-only controls. Their effective image-only values are zero/not applied. `IMAGE_ONLY_COARSE_ANCHOR_WEIGHT`, when nonzero, anchors only to the image-derived coarse mesh.

The same CLI remains available in both modes:

```bash
python main.py                         # run the configured image_only pipeline
python main.py --validate-only          # validate RGBA/CSV inputs only
python main.py --camera-fit-only        # establish and gate independent calibration only
python main.py --skip-camera-fit        # use an already complete independent calibration
```

`--skip-camera-fit` does not permit a sphere-based camera estimate. `--camera-fit-only` and `--skip-camera-fit` remain mutually exclusive.

## Setup

PyTorch3D on Apple Silicon often needs to be built from source for the selected Python/PyTorch combination. Install it in the same environment used to run the project.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For the existing project environment, use its absolute interpreter consistently:

```bash
/Users/zhaowei/PyCharmMiscProject/xolo_3d_reconstrcuction/segmentation_compare/.venv/bin/python main.py
```

PyTorch3D rasterization is kept off MPS by default. Geometry uses CUDA when explicitly configured and available, otherwise CPU. `DEVICE=cuda` fails before reconstruction if CUDA or the PyTorch3D probe is unavailable; `DEVICE=auto` may fall back to CPU.

Runtime profiles provide reproducible geometry defaults:

```env
RUNTIME_PROFILE=apple_fast    # CPU, 5k faces, 128 px, 32 views/epoch, 20 epochs
# RUNTIME_PROFILE=apple_quality  # CPU, 10k faces, 256 px, 64 views/epoch, 30 epochs
# RUNTIME_PROFILE=cuda           # CUDA, 30k faces, 512 px, all views, 50 epochs
```

Every explicit environment value overrides its profile value. `GEOMETRY_VIEWS_PER_EPOCH=0` means all validated frames. The geometry stage creates `reconstruction/geometry_proxy.stl` for optimization and leaves the original reference STL untouched.

## Commands

Copy `.env.example` to `.env` and fill in the three input paths. The default end-to-end command is:

```bash
python main.py
```

Useful gates and diagnostics:

```bash
python main.py --validate-only
python main.py --camera-fit-only
python main.py --skip-camera-fit
```

`--validate-only` stops after Gate 1. The normal path is:

```text
Gate 1 input validation
→ initial pose previews
→ global camera fitting and bounded pose refinement
→ Gate 2 camera fit
→ stratified view-batch geometry reconstruction
→ deformation and optimization health gates
→ OBJ/STL export
```

## Development notes

Run `python main.py --help` to list the supported CLI flags. Use the following flags to run individual stages during development:

- `--validate-only`: run Gate 1 input validation and stop before camera fitting or geometry reconstruction.
- `--camera-fit-only`: run input validation, camera fitting, and Gate 2, then stop before geometry reconstruction.
- `--skip-camera-fit`: skip camera fitting and reconstruct geometry using the commanded CSV poses and initial camera parameters.

`--camera-fit-only` and `--skip-camera-fit` are mutually exclusive. `--validate-only` takes precedence over either flag when used in an otherwise valid combination.

## Safety gates

- Gate 1 rejects missing CSV rows, duplicate image names or positions, non-RGBA files, empty/fully opaque alpha, unreadable images, and incompatible aspect ratios.
- The convention sanity gate stops before fitting when the best candidate is below `CAMERA_CONVENTION_MIN_MEDIAN_IOU`.
- Gate 2 stops geometry optimization when median silhouette IoU is below `CAMERA_GATE_MIN_MEDIAN_IOU`.
- Gate 3 requires at least `MIN_USABLE_FRAMES` and reports angular coverage.
- Gate 4 stops if vertex displacement exceeds `MAX_VERTEX_DISPLACEMENT_RATIO` of the normalized object size.
- Gate 5 aborts on NaN/Inf losses or gradients, empty rendered silhouettes, or invalid optimization state.

The STL is frozen during camera fitting and camera parameters are frozen during the standard geometry baseline. RGB loss and texture optimization are disabled by default.

Camera Fit is a standalone CPU stage and does not import PyTorch3D. It preloads low-resolution alpha masks, creates a camera-only proxy mesh, searches plausible OpenScan conventions, numerically fits global projection parameters, and performs optional bounded pose refinement. The final Camera Fit Gate still evaluates every validated frame. PyTorch3D is initialized only after this gate passes for geometry reconstruction.

## Aspect ratio and poses

Images are uniformly resized only when the relevant resolution limit requires it, then symmetrically padded to a fixed renderer canvas. They are never stretched to a square. The run records original/processed dimensions, scale, and padding in `validation.json`.

Pose conventions are isolated in `src/pose_conventions.py`. The camera stage searches axis, sign, and rotation-order hypotheses before fitting, then writes the selected result to `camera_fit/selected_convention.json`. The CSV is always the pose source; image filesystem order is never used to invent poses.

## Outputs

```text
outputs/demo/
├── validation.json
├── run_config.json
├── camera_fit/
│   ├── convention_search.csv
│   ├── frame_poses.csv
│   ├── global_camera_parameters.json
│   ├── metrics.json
│   ├── profile.json
│   ├── proxy_mesh.stl
│   ├── proxy_mesh_stats.json
│   ├── selected_convention.json
│   ├── selected_frames.csv
│   ├── profile.json
│   ├── pose_convention.json
│   ├── selected_frames.csv
│   └── previews/
├── reconstruction/
│   ├── geometry_proxy.stl
│   ├── geometry_proxy_stats.json
│   ├── profile.json
│   ├── view_usage.csv
│   ├── final.obj
│   ├── final.stl
│   ├── mesh_transform.json
│   ├── losses.csv
│   ├── losses.png
│   └── previews/
├── checkpoints/
└── summary.json
```

`final.stl` and `final.obj` are restored to the original reference mesh coordinate system. `final.stl` is the required artifact for downstream comparison.

In `image_only` mode, the reconstruction directory also records the stages that replaced the initial topology:

```text
reconstruction/
├── initial_sphere.stl                 # supplied generic initialization
├── image_coarse_mesh.stl              # visual-hull / marching-cubes result
├── final.obj
├── final.stl
├── view_split.json                    # deterministic disjoint train/validation views
├── losses.csv
├── reconstruction_metrics.json        # mode, camera source, carving/refinement settings, selected state
├── mesh_transform.json
├── previews/                          # sphere, coarse, and final stage previews
└── ground_truth_evaluation.json       # only when GROUND_TRUTH_MESH_PATH is configured
```

`reconstruction_metrics.json` contains camera/calibration metadata, coarse-volume settings, refinement settings, the train/validation split, and the held-out-validation-selected best state. It does not use ground-truth metrics for selection.

## Ground-truth isolation and silhouette limits

`GROUND_TRUTH_MESH_PATH` is strictly evaluation-only. The pipeline removes it from the configuration passed to camera fitting, visual-hull generation, refinement, validity checks, and best-state selection. Only after `final.obj` and `final.stl` have been written does `ground_truth_eval` load it and write `ground_truth_evaluation.json`, comparing sphere, coarse, and final meshes. The primary measurements are in the physical mesh frame; optionally rigid-aligned measurements are diagnostic only.

Tests enforce this boundary by checking that reconstruction modules do not import evaluation, guarding GT loading, and verifying evaluation occurs after exports. They also verify that image-only camera fitting does not render the sphere, CAD anchoring and local-deformation caps are inactive, validation is held out/no-gradient, and selected—not final-epoch—geometry is exported.

Silhouettes constrain only what is visible in their foreground outlines. A visual hull cannot generally recover concavities absent from all silhouettes, internally occluded structure, hidden cavities, or unseen back-side detail. Limited angular coverage, calibration or segmentation error, and voxel resolution can inflate shapes, erase thin parts/holes, or create artifacts. RGB refinement may improve visible-surface agreement but cannot make unobserved geometry identifiable. See [Silhouette reconstruction limitations](docs/limitations.md) for the complete interpretation and reporting guidance.

## Tests

Run tests with the same interpreter used for the project:

```bash
/absolute/path/to/python -m pytest -q
```

Tests cover CSV/RGBA validation, soft alpha preservation, aspect-safe padding, OpenScan rotation/inverse consistency, STL loading/restoration, geometry profiles and sampling, proxy limits, output directory creation, and safety gate behavior.
