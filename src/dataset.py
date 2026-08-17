from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras
from torch.utils.data import Dataset

from .camera_utils import cameras_from_records, camera_records, create_turntable_cameras, linspace_angles, read_camera_metadata
from .config import Config


@dataclass
class ReconstructionSample:
    image: torch.Tensor
    mask: torch.Tensor
    camera: FoVPerspectiveCameras
    index: int
    image_path: Path


def _load_rgb(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_mask(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
    arr = (np.asarray(img).astype(np.float32) / 255.0) > 0.5
    return torch.from_numpy(arr.astype(np.float32))[..., None]


def infer_mask_from_white(image: torch.Tensor, threshold: float = 0.98) -> torch.Tensor:
    return (image.min(dim=-1, keepdim=True).values < threshold).float()


class ReconstructionDataset(Dataset):
    def __init__(self, root: Path, cfg: Config, device: torch.device, name: str) -> None:
        self.root = root
        self.cfg = cfg
        self.device = device
        self.name = name
        if not root.exists():
            raise FileNotFoundError(f"Dataset folder not found: {root}")
        image_pattern = cfg.real_image_pattern if name.startswith("real") else "*.png"
        image_dirs = [root / "images", root / "white_bg", root / "white_bg_images", root]
        self.images_dir = root
        self.image_paths = sorted(root.rglob(image_pattern))
        for image_dir in image_dirs:
            if not image_dir.exists():
                continue
            paths = sorted(image_dir.rglob(image_pattern))
            if paths:
                self.images_dir = image_dir
                self.image_paths = paths
                break
        mask_dirs = [root / "masks", root / "binary_masks", root / "mask", root]
        self.masks_dir = next((mask_dir for mask_dir in mask_dirs if mask_dir.exists()), root)
        self.mask_paths_by_name = {
            path.name: path for path in self.masks_dir.rglob("*.png")
        }
        self.mask_paths_by_stem = {
            path.stem: path for path in self.masks_dir.rglob("*.png")
        }
        if self.images_dir == self.masks_dir:
            self.image_paths = [p for p in self.image_paths if not p.stem.endswith("_mask")]
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        self.camera_records = self._load_or_create_camera_records()
        self.cameras = cameras_from_records(self.camera_records, device)

    def _load_or_create_camera_records(self) -> list[dict]:
        metadata_candidates = [self.root / "camera_metadata.json", self.root / "metadata.json"]
        for path in metadata_candidates:
            if path.exists():
                return read_camera_metadata(path)
        if self.name == "synthetic":
            azimuth_start = self.cfg.synth_azimuth_start_deg
            azimuth_end = self.cfg.synth_azimuth_end_deg
            elevation = self.cfg.synth_elevation_deg
            distance = self.cfg.synth_camera_distance
        else:
            azimuth_start = self.cfg.real_azimuth_start_deg
            azimuth_end = self.cfg.real_azimuth_end_deg
            elevation = self.cfg.real_elevation_deg
            distance = self.cfg.real_camera_distance
        azimuths = linspace_angles(azimuth_start, azimuth_end, len(self.image_paths))
        cameras = create_turntable_cameras(
            azimuths=azimuths,
            elevation=elevation,
            distance=distance,
            device=torch.device("cpu"),
        )
        records = camera_records(cameras, azimuths, elevation, distance, self.cfg.image_size)
        print(f"Using generated turntable camera poses for {self.root}. Input data was not modified.")
        return records

    def _mask_path(self, image_path: Path) -> Path:
        stem = image_path.stem
        candidates = [
            self.masks_dir / self.cfg.real_mask_pattern.format(stem=stem),
            self.masks_dir / f"{stem}_mask.png",
        ]
        if stem.endswith("_white_bg"):
            base = stem.removesuffix("_white_bg")
            candidates.extend(
                [
                    self.masks_dir / self.cfg.real_mask_pattern.format(stem=base),
                    self.masks_dir / f"{base}_mask.png",
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
            if candidate.name in self.mask_paths_by_name:
                return self.mask_paths_by_name[candidate.name]
            if candidate.stem in self.mask_paths_by_stem:
                return self.mask_paths_by_stem[candidate.stem]
        return candidates[0]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> ReconstructionSample:
        image_path = self.image_paths[index]
        image = _load_rgb(image_path, self.cfg.image_size)
        mask_path = self._mask_path(image_path)
        if mask_path.exists():
            mask = _load_mask(mask_path, self.cfg.image_size)
        elif self.name.startswith("real") and self.cfg.real_use_masks:
            print(f"Warning: mask missing for {image_path.name}; inferring from white background.")
            mask = infer_mask_from_white(image)
        elif (self.root / "masks" / image_path.name).exists():
            mask = _load_mask(self.root / "masks" / image_path.name, self.cfg.image_size)
        else:
            mask = infer_mask_from_white(image)
        return ReconstructionSample(
            image=image.to(self.device),
            mask=mask.to(self.device),
            camera=self.cameras[index],
            index=index,
            image_path=image_path,
        )


def load_dataset(name: str, cfg: Config, device: torch.device) -> ReconstructionDataset:
    return ReconstructionDataset(cfg.dataset_dir(name), cfg, device, name)
