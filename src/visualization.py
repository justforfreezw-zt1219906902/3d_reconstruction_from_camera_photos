from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu().clamp(0, 1).numpy()
    if array.ndim == 4:
        array = array[0]
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray((array * 255).astype(np.uint8))


def save_stage_preview(
    target_rgb: torch.Tensor,
    target_alpha: torch.Tensor,
    rendered_alpha: torch.Tensor,
    path: Path,
    rendered_rgb: torch.Tensor | None = None,
) -> None:
    target = tensor_to_pil(target_rgb)
    alpha = tensor_to_pil(target_alpha)
    rendered = tensor_to_pil(rendered_rgb if rendered_rgb is not None else rendered_alpha)
    overlay = Image.blend(alpha.convert("RGB"), rendered.convert("RGB"), 0.5)
    images = [target, alpha.convert("RGB"), rendered, overlay]
    labels = ["target RGB", "target alpha", "render", "overlay"]
    w, h = target.size
    sheet = Image.new("RGB", (w * 4, h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        sheet.paste(image, (index * w, 0))
        draw.text((index * w + 8, 8), label, fill=(255, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


@torch.no_grad()
def save_mesh_stage_previews(
    mesh,
    dataset,
    camera_model,
    renderer,
    indices: list[int],
    output_dir: Path,
    stage: str,
) -> list[Path]:
    """Render representative views for one named reconstruction stage.

    This is deliberately diagnostic-only: it does not create gradients or
    change the mesh/camera state.  Keeping stage names in the filenames makes
    sphere, image-derived coarse, and selected-final previews directly
    comparable in an artifact directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = mesh.verts_packed().device
    paths: list[Path] = []
    for index in sorted(set(indices)):
        sample = dataset[index]
        rendered_alpha = renderer.render_mask(mesh, camera_model.cameras([index]))
        path = output_dir / f"{stage}_{index:04d}.png"
        save_stage_preview(
            sample.image[None].to(device), sample.alpha[None].to(device), rendered_alpha, path,
        )
        paths.append(path)
    return paths


def append_csv(path: Path, row: dict[str, float | int | str]) -> None:
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
    names = data.dtype.names or ()
    aliases = {"total": "training_objective", "silhouette": "training_silhouette", "iou": "training_iou"}
    for name in names:
        if name in {"epoch", "best_epoch", "is_best"}:
            continue
        if aliases.get(name) in names:
            continue
        plt.plot(data["epoch"], data[name], label=name)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
