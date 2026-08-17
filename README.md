# OBJ Reconstruction with PyTorch3D

This project follows the PyTorch3D textured mesh fitting tutorial and turns it into a reusable pipeline for one original OBJ and three datasets:

- `synthetic`: generated benchmark views rendered from the original OBJ
- `real1`: processed real capture set with white background and masks
- `real2`: second processed real capture set with white background and masks

The same original OBJ is used as the initial mesh for every run. Training optimizes vertex offsets and optional vertex colors with silhouette, RGB, Laplacian, edge-length, and normal-consistency regularization.

## Setup

Create an environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch3D on Apple Silicon can be difficult to install because official wheels may lag behind current Python/PyTorch releases. If `pip install pytorch3d` fails, use a Python/PyTorch combination supported by PyTorch3D, build PyTorch3D from source, or run CPU-only in a compatible environment.

For CPU-only operation, set:

```env
DEVICE=cpu
```

For automatic Apple MPS selection with CPU fallback:

```env
DEVICE=auto
```

Some PyTorch3D rasterization operations may still fail on MPS. The scripts catch those failures and retry on CPU.

## Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```env
ORIGINAL_OBJ_PATH=/absolute/path/to/original.obj
DATASET_REAL_1_DIR=/absolute/path/to/real_dataset_1
DATASET_REAL_2_DIR=/absolute/path/to/real_dataset_2
IMAGE_SIZE=512
DEVICE=auto
```

For faster tests on a Mac, use:

```env
IMAGE_SIZE=256
OPT_NUM_ITERS=200
```

## Dataset Layout

Real datasets should use:

```text
real_dataset/
├── images/
│   ├── image_000.png
│   ├── image_001.png
│   └── ...
└── masks/
    ├── image_000_mask.png
    ├── image_001_mask.png
    └── ...
```

If explicit masks are missing, the loader can infer masks from a white background, but explicit masks are preferred.

For real turntable data, camera poses are estimated from image order using:

```env
REAL_AZIMUTH_START_DEG=0
REAL_AZIMUTH_END_DEG=90
REAL_ELEVATION_DEG=20
REAL_CAMERA_DISTANCE=2.7
```

You can override estimated cameras by placing `camera_metadata.json` or `metadata.json` in the dataset folder. The file should contain a `cameras` list with `R`, `T`, `azimuth`, `elevation`, `distance`, and `image_size`.

## Commands

Generate the synthetic benchmark:

```bash
python main.py --mode generate_synthetic
```

Train one dataset:

```bash
python main.py --mode train --dataset synthetic
python main.py --mode train --dataset real1
python main.py --mode train --dataset real2
```

Train all datasets:

```bash
python main.py --mode train_all
```

The direct script entry points also work:

```bash
python scripts/generate_synthetic_dataset.py
python scripts/train_reconstruction.py --dataset synthetic
```

## Outputs

Synthetic benchmark:

```text
outputs/synthetic_benchmark/
├── images/
├── masks/
├── metadata.json
├── mesh_transform.json
└── preview.jpg
```

Training runs:

```text
outputs/runs/
├── synthetic/
│   ├── final.obj
│   ├── previews/
│   ├── checkpoints/
│   ├── losses.csv
│   └── losses.png
├── real1/
└── real2/
```

## Inspecting in Blender

Open `outputs/runs/<dataset>/final.obj` in Blender. The mesh is exported back in the original OBJ coordinate frame using the saved normalization transform, so it should be easier to compare against the source asset.

## Notes

- Input datasets are never overwritten.
- Intermediate mesh checkpoints are exported every `EXPORT_EVERY` iterations.
- Preview overlays are saved every `SAVE_PREVIEW_EVERY` iterations.
- Losses are logged every iteration to `losses.csv`.
- Regularization is enabled by default so the original OBJ remains a strong prior.
- Texture optimization uses vertex colors by default. If you only want geometry fitting, set `OPT_OPTIMIZE_TEXTURE=false`.
