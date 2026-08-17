from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def tensor_image_to_pil(image: torch.Tensor) -> Image.Image:
    arr = image.detach().cpu().clamp(0, 1).numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def save_image(image: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_image_to_pil(image).save(path)


def save_contact_sheet(images: list[Path], out_path: Path, thumb_size: int = 160) -> None:
    if not images:
        return
    thumbs = [Image.open(p).convert("RGB").resize((thumb_size, thumb_size), Image.Resampling.LANCZOS) for p in images]
    cols = min(6, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), "white")
    for i, img in enumerate(thumbs):
        sheet.paste(img, ((i % cols) * thumb_size, (i // cols) * thumb_size))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def save_overlay(rendered_rgb: torch.Tensor, target_rgb: torch.Tensor, mask: torch.Tensor, path: Path) -> None:
    r = tensor_image_to_pil(rendered_rgb)
    t = tensor_image_to_pil(target_rgb)
    m = tensor_image_to_pil(mask)
    w, h = r.size
    sheet = Image.new("RGB", (w * 3, h), "white")
    sheet.paste(t, (0, 0))
    sheet.paste(r, (w, 0))
    sheet.paste(m, (w * 2, 0))
    draw = ImageDraw.Draw(sheet)
    for x, label in [(8, "target"), (w + 8, "render"), (2 * w + 8, "mask")]:
        draw.text((x, 8), label, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def append_losses_csv(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_losses(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        return
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    if data.size == 0:
        return
    plt.figure(figsize=(9, 5))
    for name in data.dtype.names:
        if name != "iteration":
            plt.plot(data["iteration"], data[name], label=name)
    plt.yscale("log")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
