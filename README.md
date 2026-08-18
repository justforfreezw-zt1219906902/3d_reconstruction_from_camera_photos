# OpenScan CAD-Guided Reconstruction Demo

This project reconstructs a known reference mesh from one processed OpenScan RGBA image folder and one OpenScan pose CSV. The alpha channel is the only silhouette source. The default experiment fits camera nuisance parameters first, gates that fit, then performs controlled geometry refinement from the initial STL.

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

PyTorch3D rasterization may not support every MPS operation. The pipeline retries the complete run on CPU if an MPS renderer operation fails, so partial MPS tensors are never mixed with CPU tensors.

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
→ shuffled epoch-based geometry reconstruction
→ deformation and optimization health gates
→ OBJ/STL export
```

## Safety gates

- Gate 1 rejects missing CSV rows, duplicate image names or positions, non-RGBA files, empty/fully opaque alpha, unreadable images, and incompatible aspect ratios.
- Gate 2 stops geometry optimization when median silhouette IoU is below `CAMERA_GATE_MIN_MEDIAN_IOU`.
- Gate 3 requires at least `MIN_USABLE_FRAMES` and reports angular coverage.
- Gate 4 stops if vertex displacement exceeds `MAX_VERTEX_DISPLACEMENT_RATIO` of the normalized object size.
- Gate 5 aborts on NaN/Inf losses or gradients, empty rendered silhouettes, or invalid optimization state.

The STL is frozen during camera fitting and camera parameters are frozen during the standard geometry baseline. RGB loss and texture optimization are disabled by default.

Camera fitting uses a separate low-resolution alpha cache. Stage A uses up to `CAMERA_FIT_MAX_FRAMES` deterministic representatives, then the Camera Fit Gate evaluates every usable frame. Optional pose refinement runs afterward with bounded corrections and a small number of epochs. The reconstruction resolution in `MAX_IMAGE_DIMENSION` is not used for camera fitting.

## Aspect ratio and poses

Images are uniformly resized only when `MAX_IMAGE_DIMENSION` requires it, then symmetrically padded to a fixed renderer canvas. They are never stretched to a square. The run records original/processed dimensions, scale, and padding in `validation.json`.

The pose convention is isolated in `src/openscan_pose.py`: theta is the +Y turntable rotation, phi is the +X tilt rotation, and the object-to-equivalent-camera transformation is documented there. The CSV is always the pose source; image filesystem order is never used to invent poses.

## Outputs

```text
outputs/demo/
├── validation.json
├── run_config.json
├── camera_fit/
│   ├── camera_parameters.json
│   ├── frame_poses.csv
│   ├── metrics.json
│   ├── profile.json
│   ├── pose_convention.json
│   ├── selected_frames.csv
│   └── previews/
├── reconstruction/
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

## Tests

Run tests with the same interpreter used for the project:

```bash
/absolute/path/to/python -m pytest -q
```

Tests cover CSV/RGBA validation, soft alpha preservation, aspect-safe padding, OpenScan rotation/inverse consistency, STL loading/restoration, epoch order, and safety gate behavior.
