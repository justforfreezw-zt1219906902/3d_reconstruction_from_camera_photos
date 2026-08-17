from __future__ import annotations

import argparse
import json

from src.config import load_config
from src.pipeline import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenScan CAD-guided PyTorch3D reconstruction demo")
    parser.add_argument("--validate-only", action="store_true", help="Run Gate 1 and stop before PyTorch3D.")
    parser.add_argument("--camera-fit-only", action="store_true", help="Run camera fitting and Gate 2 only.")
    parser.add_argument("--skip-camera-fit", action="store_true", help="Skip Stage A and use commanded poses.")
    args = parser.parse_args()
    if args.camera_fit_only and args.skip_camera_fit:
        parser.error("--camera-fit-only and --skip-camera-fit are mutually exclusive.")
    result = run_from_config(
        load_config(),
        validate_only=args.validate_only,
        camera_fit_only=args.camera_fit_only,
        skip_camera_fit=args.skip_camera_fit,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
