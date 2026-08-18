from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .validation import DatasetContract, FrameRecord


@dataclass(frozen=True)
class ImageLayout:
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    canvas_size: int
    padding_left: int
    padding_top: int
    resize_scale: float


def build_layouts(contract: DatasetContract, max_image_dimension: int) -> tuple[ImageLayout, ...]:
    processed_sizes: list[tuple[int, int, float]] = []
    for frame in contract.frames:
        scale = min(1.0, max_image_dimension / max(frame.width, frame.height))
        processed_sizes.append(
            (
                max(1, round(frame.width * scale)),
                max(1, round(frame.height * scale)),
                scale,
            )
        )
    canvas_size = max(max(width, height) for width, height, _ in processed_sizes)
    return tuple(
        ImageLayout(
            original_width=frame.width,
            original_height=frame.height,
            processed_width=processed_width,
            processed_height=processed_height,
            canvas_size=canvas_size,
            padding_left=(canvas_size - processed_width) // 2,
            padding_top=(canvas_size - processed_height) // 2,
            resize_scale=scale,
        )
        for frame, (processed_width, processed_height, scale) in zip(contract.frames, processed_sizes)
    )


def load_rgba_with_layout(path: Path, layout: ImageLayout) -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        resized = rgba.resize((layout.processed_width, layout.processed_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (layout.canvas_size, layout.canvas_size), (0, 0, 0, 0))
        canvas.paste(resized, (layout.padding_left, layout.padding_top))
    array = np.asarray(canvas).astype(np.float32) / 255.0
    rgb = torch.from_numpy(array[..., :3].copy())
    alpha = torch.from_numpy(array[..., 3:4].copy())
    return rgb, alpha


def load_alpha_with_layout(path: Path, layout: ImageLayout) -> torch.Tensor:
    """Load only the soft alpha channel and apply the same aspect-safe layout."""
    with Image.open(path) as source:
        alpha = source.convert("RGBA").getchannel("A")
        resized = alpha.resize((layout.processed_width, layout.processed_height), Image.Resampling.LANCZOS)
        canvas = Image.new("L", (layout.canvas_size, layout.canvas_size), 0)
        canvas.paste(resized, (layout.padding_left, layout.padding_top))
    array = np.asarray(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(array[..., None].copy())


def load_rgb_with_layout(path: Path, layout: ImageLayout) -> torch.Tensor:
    """Load RGB only for an explicitly requested diagnostic preview."""
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        resized = rgb.resize((layout.processed_width, layout.processed_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (layout.canvas_size, layout.canvas_size), (255, 255, 255))
        canvas.paste(resized, (layout.padding_left, layout.padding_top))
    array = np.asarray(canvas).astype(np.float32) / 255.0
    return torch.from_numpy(array[..., :3].copy())


@dataclass
class ReconstructionSample:
    image: torch.Tensor
    alpha: torch.Tensor
    frame: FrameRecord
    layout: ImageLayout
    index: int
    image_path: Path


class RGBADataset:
    def __init__(self, contract: DatasetContract, max_image_dimension: int) -> None:
        self.contract = contract
        self.frames = contract.frames
        self.layouts = build_layouts(contract, max_image_dimension)
        self.canvas_size = self.layouts[0].canvas_size

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> ReconstructionSample:
        frame = self.frames[index]
        rgb, alpha = load_rgba_with_layout(frame.image_path, self.layouts[index])
        return ReconstructionSample(
            image=rgb,
            alpha=alpha,
            frame=frame,
            layout=self.layouts[index],
            index=index,
            image_path=frame.image_path,
        )

    def representative_indices(self) -> list[int]:
        if len(self.frames) <= 8:
            return list(range(len(self.frames)))
        by_phi = sorted(range(len(self.frames)), key=lambda i: self.frames[i].phi_deg)
        selected = {by_phi[0], by_phi[len(by_phi) // 2], by_phi[-1]}
        for phi_group in (min(self.frames, key=lambda x: x.theta_deg).theta_deg, 90.0, 180.0, 270.0):
            selected.add(min(range(len(self.frames)), key=lambda i: abs((self.frames[i].theta_deg % 360) - phi_group)))
        return sorted(selected)
