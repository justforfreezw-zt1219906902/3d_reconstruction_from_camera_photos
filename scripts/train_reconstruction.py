from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.config import ensure_pytorch3d, load_config, select_device, set_seed
from src.dataset import load_dataset
from src.mesh_io import load_original_mesh, save_transform
from src.optimizer import train_reconstruction


def train_dataset(dataset_name: str) -> None:
    ensure_pytorch3d()
    cfg = load_config()
    set_seed(cfg.seed)
    device = select_device(cfg.device)

    def run_on(run_device: torch.device):
        mesh, transform = load_original_mesh(
            cfg.original_obj_path,
            run_device,
            center_normalize=cfg.mesh_center_normalize,
            scale_normalize=cfg.mesh_scale_normalize,
        )
        dataset = load_dataset(dataset_name, cfg, run_device)
        save_transform(transform, cfg.run_dir(dataset_name) / "mesh_transform.json")
        return train_reconstruction(dataset_name, dataset, mesh, transform, cfg, run_device)

    try:
        result = run_on(device)
    except Exception as exc:
        if device.type != "mps":
            raise
        print(f"Warning: training failed on MPS ({exc}). Retrying full run on CPU.")
        result = run_on(torch.device("cpu"))
    print(f"Finished {dataset_name}: {result.final_obj}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "real1", "real2"], required=True)
    args = parser.parse_args()
    train_dataset(args.dataset)


if __name__ == "__main__":
    main()
