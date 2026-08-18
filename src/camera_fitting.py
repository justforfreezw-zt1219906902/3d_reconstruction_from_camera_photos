from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import Config
from .losses import soft_iou_per_image
from .openscan_pose import OpenScanCameraModel, save_pose_convention
from .rgba_dataset import RGBADataset, ImageLayout, load_alpha_with_layout, load_rgb_with_layout
from .renderer import ReconstructionRenderer
from .visualization import save_stage_preview


class CameraFitGateError(RuntimeError):
    pass


@dataclass
class CameraFitResult:
    model: OpenScanCameraModel
    metrics: dict


@dataclass
class CameraFitAlphaCache:
    """Low-resolution alpha tensors kept alive for the entire camera fit."""

    alphas: torch.Tensor
    layouts: tuple[ImageLayout, ...]
    preload_seconds: float
    batch_size: int = 1

    @property
    def image_size(self) -> int:
        return int(self.alphas.shape[1])

    def batch(self, indices: list[int]) -> torch.Tensor:
        return self.alphas[indices]


def preload_camera_fit_alpha(
    dataset: RGBADataset,
    max_dimension: int,
    device: torch.device,
) -> CameraFitAlphaCache:
    started = time.perf_counter()
    # Layout creation is shared with reconstruction and never stretches images.
    from .rgba_dataset import build_layouts

    layouts = build_layouts(dataset.contract, max_dimension)
    alphas = [
        load_alpha_with_layout(frame.image_path, layout)
        for frame, layout in zip(dataset.frames, layouts)
    ]
    return CameraFitAlphaCache(
        alphas=torch.stack(alphas).to(device),
        layouts=layouts,
        preload_seconds=time.perf_counter() - started,
        batch_size=1,
    )


def _theta_distance(a: float, b: float) -> float:
    difference = abs((a - b) % 360.0)
    return min(difference, 360.0 - difference)


def select_representative_indices(dataset: RGBADataset, max_frames: int) -> list[int]:
    """Select deterministic phi-ring and theta-stratified representatives."""
    groups: dict[float, list[int]] = {}
    for index, frame in enumerate(dataset.frames):
        groups.setdefault(round(frame.phi_deg, 6), []).append(index)
    rings = sorted(groups)
    if not rings:
        return []
    # Keeping one frame per phi ring is a correctness requirement. A smaller
    # configured limit therefore expands to the number of rings.
    limit = max(max_frames, len(rings))
    base, remainder = divmod(limit, len(rings))
    selected: set[int] = set()
    for ring_index, phi in enumerate(rings):
        candidates = sorted(groups[phi], key=lambda i: (dataset.frames[i].theta_deg % 360.0, dataset.frames[i].position_index))
        slots = base + (1 if ring_index < remainder else 0)
        slots = max(1, min(slots, len(candidates)))
        for slot in range(slots):
            target_theta = 360.0 * slot / slots
            choice = min(
                (i for i in candidates if i not in selected),
                key=lambda i: (_theta_distance(dataset.frames[i].theta_deg, target_theta), dataset.frames[i].position_index),
                default=None,
            )
            if choice is not None:
                selected.add(choice)
    if len(selected) < limit:
        remaining = [i for i in range(len(dataset)) if i not in selected]
        remaining.sort(
            key=lambda i: (
                -min(_theta_distance(dataset.frames[i].theta_deg, dataset.frames[j].theta_deg) for j in selected),
                dataset.frames[i].position_index,
            )
        )
        selected.update(remaining[: limit - len(selected)])
    return sorted(selected, key=lambda i: dataset.frames[i].position_index)


def save_selected_frames(dataset: RGBADataset, indices: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "image", "position_index", "phi_deg", "theta_deg"])
        writer.writeheader()
        for index in indices:
            frame = dataset.frames[index]
            writer.writerow(
                {
                    "index": index,
                    "image": frame.image,
                    "position_index": frame.position_index,
                    "phi_deg": frame.phi_deg,
                    "theta_deg": frame.theta_deg,
                }
            )


def _preview(
    dataset: RGBADataset,
    cache: CameraFitAlphaCache,
    index: int,
    rendered_alpha: torch.Tensor,
    output_dir: Path,
    stage: str,
) -> None:
    # RGB is intentionally opened only for representative visual diagnostics.
    rgb = load_rgb_with_layout(dataset.frames[index].image_path, cache.layouts[index])
    save_stage_preview(
        rgb[None].to(rendered_alpha.device),
        cache.alphas[index][None],
        rendered_alpha,
        output_dir / "previews" / f"{stage}_{index:04d}.png",
    )


def _render_metrics(
    model: OpenScanCameraModel,
    mesh,
    dataset: RGBADataset,
    renderer: ReconstructionRenderer,
    cache: CameraFitAlphaCache,
    device: torch.device,
    output_dir: Path,
    stage: str,
    indices: list[int] | None = None,
    preview_indices: set[int] | None = None,
) -> dict:
    indices = list(range(len(dataset))) if indices is None else indices
    preview_indices = set() if preview_indices is None else preview_indices
    batch_size = max(1, getattr(cache, "batch_size", 1))
    ious: list[float] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(indices), batch_size), desc=f"{stage} camera metrics"):
            batch_indices = indices[start : start + batch_size]
            rendered_alpha = renderer.render_mask(mesh, model.cameras(batch_indices))
            target_alpha = cache.batch(batch_indices)
            batch_ious = soft_iou_per_image(rendered_alpha, target_alpha)
            ious.extend(float(value.detach().cpu()) for value in batch_ious)
            for offset, index in enumerate(batch_indices):
                if index in preview_indices:
                    _preview(dataset, cache, index, rendered_alpha[offset : offset + 1], output_dir, stage)
    return {
        "iou_per_frame": ious,
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "worst_frame_iou": float(np.min(ious)),
        "evaluated_frames": len(indices),
    }


def _write_camera_outputs(model: OpenScanCameraModel, output_dir: Path, metrics: dict) -> None:
    (output_dir / "camera_parameters.json").write_text(json.dumps(model.parameters_snapshot(), indent=2))
    records = model.pose_records()
    with (output_dir / "frame_poses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def _check_finite(loss: torch.Tensor, model: OpenScanCameraModel) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError("GATE 5 — OPTIMIZATION HEALTH FAILED: camera fit loss is NaN/Inf.")
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError("GATE 5 — OPTIMIZATION HEALTH FAILED: camera fit gradient is NaN/Inf.")


def _batched_indices(indices: list[int], batch_size: int):
    for start in range(0, len(indices), max(1, batch_size)):
        yield indices[start : start + max(1, batch_size)]


def fit_camera(
    mesh,
    dataset: RGBADataset,
    model: OpenScanCameraModel,
    renderer: ReconstructionRenderer,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
) -> CameraFitResult:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_pose_convention(output_dir / "pose_convention.json")
    cache = preload_camera_fit_alpha(dataset, cfg.camera_fit_max_dimension, device)
    cache.batch_size = cfg.camera_fit_batch_size
    selected = select_representative_indices(dataset, cfg.camera_fit_max_frames)
    save_selected_frames(dataset, selected, output_dir / "selected_frames.csv")
    representative_set = set(selected)
    preview_set = set(dataset.representative_indices()) & representative_set

    initial_started = time.perf_counter()
    initial_metrics = _render_metrics(
        model,
        mesh,
        dataset,
        renderer,
        cache,
        device,
        output_dir,
        "initial",
        indices=selected,
        preview_indices=preview_set,
    )
    full_gate_seconds = time.perf_counter() - initial_started
    if not cfg.camera_fit_enabled:
        model.freeze()
        metrics = {"enabled": False, "initial": initial_metrics, "final": initial_metrics}
        _write_camera_outputs(model, output_dir, metrics)
        _write_profile(
            output_dir,
            cfg,
            len(dataset),
            len(selected),
            cache.preload_seconds,
            0.0,
            full_gate_seconds,
            0.0,
            time.perf_counter() - started,
            0,
            0,
        )
        return CameraFitResult(model=model, metrics=metrics)

    model.set_global_trainable(True)
    model.set_pose_trainable(False)
    global_optimizer = torch.optim.Adam(model.global_parameters(), lr=0.01)
    best_score = -float("inf")
    stale_epochs = 0
    global_history: list[dict] = []
    global_started = time.perf_counter()
    global_epochs_executed = 0
    for epoch in tqdm(range(1, cfg.camera_fit_global_epochs + 1), desc="camera fit global"):
        order = sorted(selected, key=lambda i: dataset.frames[i].position_index)
        generator = torch.Generator().manual_seed(cfg.seed + epoch)
        order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
        epoch_ious: list[float] = []
        for batch_indices in _batched_indices(order, cfg.camera_fit_batch_size):
            rendered_alpha = renderer.render_mask(mesh, model.cameras(batch_indices))
            batch_iou = soft_iou_per_image(rendered_alpha, cache.batch(batch_indices))
            loss = 1.0 - batch_iou.mean()
            global_optimizer.zero_grad()
            loss.backward()
            _check_finite(loss, model)
            global_optimizer.step()
            epoch_ious.extend(float(value.detach().cpu()) for value in batch_iou)
        global_epochs_executed = epoch
        epoch_metrics = {
            "epoch": epoch,
            "mean_iou": float(np.mean(epoch_ious)),
            "median_iou": float(np.median(epoch_ious)),
        }
        global_history.append(epoch_metrics)
        score = max(epoch_metrics["mean_iou"], epoch_metrics["median_iou"])
        if score > best_score + cfg.camera_fit_min_delta:
            best_score = score
            stale_epochs = 0
        else:
            stale_epochs += 1
        if cfg.camera_fit_early_stop_patience > 0 and stale_epochs >= cfg.camera_fit_early_stop_patience:
            break
    global_seconds = time.perf_counter() - global_started

    # Gate evaluation is always full resolution for camera fitting and always
    # uses every validated frame, even when Stage A used representatives.
    model.set_global_trainable(False)
    model.set_pose_trainable(False)
    gate_started = time.perf_counter()
    pre_pose_metrics = _render_metrics(
        model, mesh, dataset, renderer, cache, device, output_dir, "global_fitted", preview_indices=preview_set
    )
    full_gate_seconds += time.perf_counter() - gate_started

    pose_seconds = 0.0
    pose_epochs_executed = 0
    if cfg.pose_refine and cfg.pose_refine_epochs > 0:
        model.set_pose_trainable(True)
        pose_optimizer = torch.optim.Adam(model.pose_parameters(), lr=0.005)
        pose_started = time.perf_counter()
        for epoch in tqdm(range(1, cfg.pose_refine_epochs + 1), desc="camera pose refine"):
            order = sorted(range(len(dataset)), key=lambda i: dataset.frames[i].position_index)
            generator = torch.Generator().manual_seed(cfg.seed + 10000 + epoch)
            order = [order[i] for i in torch.randperm(len(order), generator=generator).tolist()]
            for batch_indices in _batched_indices(order, cfg.camera_fit_batch_size):
                rendered_alpha = renderer.render_mask(mesh, model.cameras(batch_indices))
                batch_iou = soft_iou_per_image(rendered_alpha, cache.batch(batch_indices))
                loss = 1.0 - batch_iou.mean()
                pose_optimizer.zero_grad()
                loss.backward()
                _check_finite(loss, model)
                pose_optimizer.step()
            pose_epochs_executed = epoch
        pose_seconds = time.perf_counter() - pose_started
        model.set_pose_trainable(False)

    final_started = time.perf_counter()
    final_metrics = _render_metrics(
        model, mesh, dataset, renderer, cache, device, output_dir, "camera_fitted", preview_indices=preview_set
    )
    full_gate_seconds += time.perf_counter() - final_started
    metrics = {
        "enabled": True,
        "initial": initial_metrics,
        "global_history": global_history,
        "global_fitted": pre_pose_metrics,
        "final": final_metrics,
        "camera_gate_uses_all_frames": True,
    }
    _write_camera_outputs(model, output_dir, metrics)
    _write_profile(
        output_dir,
        cfg,
        len(dataset),
        len(selected),
        cache.preload_seconds,
        global_seconds,
        full_gate_seconds,
        pose_seconds,
        time.perf_counter() - started,
        global_epochs_executed,
        pose_epochs_executed,
    )
    if final_metrics["median_iou"] < cfg.camera_gate_min_median_iou:
        raise CameraFitGateError(
            "Camera fit gate failed. Do not deform the STL because camera alignment is insufficient. "
            f"Median IoU={final_metrics['median_iou']:.4f}, "
            f"required={cfg.camera_gate_min_median_iou:.4f}."
        )
    model.freeze()
    return CameraFitResult(model=model, metrics=metrics)


def _write_profile(
    output_dir: Path,
    cfg: Config,
    total_frames: int,
    selected_frames: int,
    preload_seconds: float,
    global_seconds: float,
    full_gate_seconds: float,
    pose_seconds: float,
    total_seconds: float,
    global_epochs_executed: int,
    pose_epochs_executed: int,
) -> None:
    batch_size = max(1, cfg.camera_fit_batch_size)
    profile = {
        "total_frames": total_frames,
        "selected_coarse_frames": selected_frames,
        "camera_fit_resolution": cfg.camera_fit_max_dimension,
        "faces_per_pixel": cfg.camera_fit_faces_per_pixel,
        "camera_fit_batch_size": batch_size,
        "global_fit_epochs_executed": global_epochs_executed,
        "pose_refinement_epochs": pose_epochs_executed,
        "alpha_preload_seconds": preload_seconds,
        "global_fitting_seconds": global_seconds,
        "full_gate_evaluation_seconds": full_gate_seconds,
        "pose_refinement_seconds": pose_seconds,
        "total_camera_fit_seconds": total_seconds,
        "old_theoretical_render_count": total_frames * (getattr(cfg, "camera_fit_epochs", 100) + 2),
        "new_theoretical_render_count": (
            selected_frames + total_frames * 2
            + int(np.ceil(selected_frames / batch_size)) * global_epochs_executed
            + int(np.ceil(total_frames / batch_size)) * pose_epochs_executed
        ),
    }
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2))
