from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import Config
from .losses import soft_iou
from .openscan_pose import OpenScanCameraModel, save_pose_convention
from .rgba_dataset import RGBADataset
from .renderer import ReconstructionRenderer
from .visualization import save_stage_preview


class CameraFitGateError(RuntimeError):
    pass


@dataclass
class CameraFitResult:
    model: OpenScanCameraModel
    metrics: dict


def _frame_target(dataset: RGBADataset, index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sample = dataset[index]
    return sample.image.to(device)[None], sample.alpha.to(device)[None]


def _render_metrics(
    model: OpenScanCameraModel,
    mesh,
    dataset: RGBADataset,
    renderer: ReconstructionRenderer,
    device: torch.device,
    output_dir: Path,
    stage: str,
) -> dict:
    ious: list[float] = []
    for index in tqdm(range(len(dataset)), desc=f"{stage} camera metrics"):
        target_rgb, target_alpha = _frame_target(dataset, index, device)
        rendered_alpha = renderer.render_mask(mesh, model.cameras([index]))
        ious.append(float(soft_iou(rendered_alpha, target_alpha).detach().cpu()))
        if index in dataset.representative_indices():
            save_stage_preview(
                target_rgb,
                target_alpha,
                rendered_alpha,
                output_dir / "previews" / f"{stage}_{index:04d}.png",
            )
    metrics = {
        "iou_per_frame": ious,
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "worst_frame_iou": float(np.min(ious)),
    }
    return metrics


def fit_camera(
    mesh,
    dataset: RGBADataset,
    model: OpenScanCameraModel,
    renderer: ReconstructionRenderer,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
) -> CameraFitResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pose_convention(output_dir / "pose_convention.json")
    initial_metrics = _render_metrics(model, mesh, dataset, renderer, device, output_dir, "initial")
    if not cfg.camera_fit_enabled:
        model.freeze()
        metrics = {"enabled": False, "initial": initial_metrics, "final": initial_metrics}
        (output_dir / "camera_parameters.json").write_text(json.dumps(model.parameters_snapshot(), indent=2))
        with (output_dir / "frame_poses.csv").open("w", newline="") as f:
            records = model.pose_records()
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return CameraFitResult(model=model, metrics=metrics)

    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01)
    for _epoch in tqdm(range(1, cfg.camera_fit_epochs + 1), desc="camera fit"):
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(cfg.seed + _epoch))
        for index in order.tolist():
            target_rgb, target_alpha = _frame_target(dataset, index, device)
            rendered_alpha = renderer.render_mask(mesh, model.cameras([index]))
            loss = 1.0 - soft_iou(rendered_alpha, target_alpha)
            if not torch.isfinite(loss):
                raise RuntimeError("GATE 5 — OPTIMIZATION HEALTH FAILED: camera fit loss is NaN/Inf.")
            optimizer.zero_grad()
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in model.parameters()):
                raise RuntimeError("GATE 5 — OPTIMIZATION HEALTH FAILED: camera fit gradient is NaN/Inf.")
            optimizer.step()

    final_metrics = _render_metrics(model, mesh, dataset, renderer, device, output_dir, "camera_fitted")
    model.parameters_snapshot()
    (output_dir / "camera_parameters.json").write_text(json.dumps(model.parameters_snapshot(), indent=2))
    with (output_dir / "frame_poses.csv").open("w", newline="") as f:
        records = model.pose_records()
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "metrics.json").write_text(
        json.dumps({"initial": initial_metrics, "final": final_metrics}, indent=2)
    )
    if final_metrics["median_iou"] < cfg.camera_gate_min_median_iou:
        raise CameraFitGateError(
            "Camera fit gate failed. Do not deform the STL because camera alignment is insufficient. "
            f"Median IoU={final_metrics['median_iou']:.4f}, "
            f"required={cfg.camera_gate_min_median_iou:.4f}."
        )
    model.freeze()
    return CameraFitResult(model=model, metrics={"enabled": True, "initial": initial_metrics, "final": final_metrics})
