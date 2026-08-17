from __future__ import annotations

import argparse

from scripts.generate_synthetic_dataset import generate_synthetic_dataset
from scripts.train_reconstruction import train_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch3D OBJ reconstruction pipeline")
    parser.add_argument(
        "--mode",
        choices=["generate_synthetic", "train", "train_all"],
        required=True,
    )
    parser.add_argument("--dataset", choices=["synthetic", "real1", "real2"])
    args = parser.parse_args()

    if args.mode == "generate_synthetic":
        generate_synthetic_dataset()
    elif args.mode == "train":
        if not args.dataset:
            raise SystemExit("--dataset is required for --mode train")
        train_dataset(args.dataset)
    elif args.mode == "train_all":
        for name in ["synthetic", "real1", "real2"]:
            train_dataset(name)


if __name__ == "__main__":
    main()
