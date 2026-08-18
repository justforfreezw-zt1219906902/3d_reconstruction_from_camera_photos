from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from pytorch3d.structures import Meshes
from tqdm import tqdm

from .config import Config
from .export_utils import save_checkpoint
from .losses import alpha_silhouette_loss, masked_rgb_loss, regularization_losses, soft_iou_per_image
from .mesh_io import MeshTransform, export_mesh_obj, export_mesh_stl
from .openscan_pose import OpenScanCameraModel
from .rgba_dataset import RGBADataset
from .renderer import ReconstructionRenderer
from .visualization import append_csv, plot_losses, save_stage_preview


class ReconstructionGateError(RuntimeError):
    pass


@dataclass
class ReconstructionResult:
    mesh: Meshes
    final_obj: Path
    final_stl: Path


def _mesh_with_offsets(base_mesh: Meshes, offsets: torch.Tensor) -> Meshes:
    return base_mesh.update_padded(base_mesh.verts_padded() + offsets)


def _displacement_stats(offsets: torch.Tensor, object_size: float) -> dict[str, float]:
    values = torch.linalg.vector_norm(offsets.detach().reshape(-1, 3), dim=-1)
    return {
        "mean_vertex_displacement": float(values.mean().cpu()),
        "median_vertex_displacement": float(values.median().cpu()),
        "p95_vertex_displacement": float(torch.quantile(values, 0.95).cpu()),
        "max_vertex_displacement": float(values.max().cpu()),
        "max_vertex_displacement_ratio": float(values.max().cpu()) / max(object_size, 1e-8),
    }


def select_geometry_view_indices(
    frames: Sequence,
    views_per_epoch: int,
    epoch: int,
    seed: int,
    usage_counts: Sequence[int] | None = None,
) -> list[int]:
    """Select rotating phi/theta-stratified views, preferring underused frames."""
    count = len(frames) if views_per_epoch <= 0 else min(views_per_epoch, len(frames))
    if count == 0:
        return []
    usage = list(usage_counts or [0] * len(frames))
    rng = random.Random(seed + epoch * 1009)
    groups: dict[float, list[int]] = {}
    for index, frame in enumerate(frames):
        groups.setdefault(round(float(frame.phi_deg), 5), []).append(index)
    phi_groups = sorted(groups.items())
    selected: list[int] = []
    for group_index, (_, candidates) in enumerate(phi_groups):
        quota = max(1, round(count * len(candidates) / len(frames)))
        quota = min(quota, len(candidates), count - len(selected))
        target_theta = (epoch * 37.0 + group_index * 19.0) % 360.0
        ranked = sorted(
            candidates,
            key=lambda i: (
                usage[i],
                abs(((float(frames[i].theta_deg) - target_theta + 180.0) % 360.0) - 180.0),
                rng.random(),
            ),
        )
        selected.extend(ranked[:quota])
    if len(selected) < count:
        remaining = [i for i in range(len(frames)) if i not in selected]
        rng.shuffle(remaining)
        remaining.sort(key=lambda i: usage[i])
        selected.extend(remaining[: count - len(selected)])
    rng.shuffle(selected)
    return selected[:count]


def _write_view_usage(path: Path, frames: Sequence, usage_counts: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "image", "phi_deg", "theta_deg", "usage_count"])
        for index, (frame, count) in enumerate(zip(frames, usage_counts)):
            writer.writerow([index, frame.image, frame.phi_deg, frame.theta_deg, count])


def _batch_indices(indices: list[int], batch_size: int) -> list[list[int]]:
    return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]


def train_geometry(
    base_mesh: Meshes,
    transform: MeshTransform,
    dataset: RGBADataset,
    camera_model: OpenScanCameraModel,
    renderer: ReconstructionRenderer,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
    mesh_stats: dict | None = None,
) -> ReconstructionResult:
    reconstruction_dir = output_dir / "reconstruction"
    checkpoint_dir = reconstruction_dir / "checkpoints"
    preview_dir = reconstruction_dir / "previews"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    views_per_epoch = cfg.geometry_views_per_epoch or len(dataset)
    batches_per_epoch = (min(views_per_epoch, len(dataset)) + cfg.geometry_view_batch_size - 1) // cfg.geometry_view_batch_size
    planned_steps = batches_per_epoch * cfg.num_epochs
    print(f"Runtime profile: {cfg.runtime_profile}")
    print(f"Device: {device}")
    print("Mesh:")
    print(f"  original faces: {(mesh_stats or {}).get('original_faces', 'unknown')}")
    print(f"  geometry faces: {(mesh_stats or {}).get('geometry_faces', len(base_mesh.faces_packed()))}")
    print("Dataset:")
    print(f"  validated frames: {len(dataset)}")
    print(f"  views per epoch: {views_per_epoch}")
    print("Geometry:")
    print(f"  epochs: {cfg.num_epochs}")
    print(f"  batch size: {cfg.geometry_view_batch_size}")
    print(f"  planned optimization steps: {planned_steps}")
    print(f"  resolution: {dataset.canvas_size}")
    print(f"  faces per pixel: {cfg.silhouette_faces_per_pixel}")

    offsets = torch.zeros_like(base_mesh.verts_padded(), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([offsets], lr=cfg.opt_lr_verts)
    losses_path = reconstruction_dir / "losses.csv"
    if losses_path.exists():
        losses_path.unlink()
    best_loss = float("inf")
    stale_epochs = 0
    previous_valid_offsets = offsets.detach().clone()
    object_size = transform.normalized_object_size
    representative = set(dataset.representative_indices())
    usage_counts = [0] * len(dataset)
    timing = {"render": [], "regularization": [], "backward": []}
    started_total = time.perf_counter()

    for epoch in tqdm(range(1, cfg.num_epochs + 1), desc="geometry epochs"):
        selected = select_geometry_view_indices(
            dataset.frames, cfg.geometry_views_per_epoch, epoch, cfg.seed, usage_counts
        )
        for index in selected:
            usage_counts[index] += 1
        epoch_values: dict[str, list[float]] = {key: [] for key in ("total", "silhouette", "rgb", "laplacian", "edge", "normal", "iou")}
        batches = _batch_indices(selected, cfg.geometry_view_batch_size)
        step_bar = tqdm(batches, desc=f"geometry epoch {epoch}/{cfg.num_epochs}", leave=False)
        for step, batch in enumerate(step_bar, start=1):
            samples = [dataset[index] for index in batch]
            target_rgb = torch.stack([sample.image for sample in samples]).to(device)
            target_alpha = torch.stack([sample.alpha for sample in samples]).to(device)
            cameras = camera_model.cameras(batch)
            mesh = _mesh_with_offsets(base_mesh, offsets)

            render_started = time.perf_counter()
            rendered_alpha = renderer.render_mask(mesh, cameras)
            rendered_rgb = None
            if cfg.loss_rgb_weight > 0:
                rendered_rgb = renderer.render_rgb(mesh, cameras)
            timing["render"].append(time.perf_counter() - render_started)
            if rendered_alpha.sum().detach().cpu() <= 0:
                raise ReconstructionGateError("GATE 5 — OPTIMIZATION HEALTH FAILED: rendered silhouette is empty.")
            silhouette = alpha_silhouette_loss(rendered_alpha, target_alpha)
            rgb = masked_rgb_loss(rendered_rgb, target_rgb, target_alpha) if rendered_rgb is not None else torch.zeros_like(silhouette)

            regularization_started = time.perf_counter()
            regs = regularization_losses(mesh)
            timing["regularization"].append(time.perf_counter() - regularization_started)
            total = (
                cfg.loss_silhouette_weight * silhouette
                + cfg.loss_rgb_weight * rgb
                + cfg.loss_laplacian_weight * regs["laplacian"]
                + cfg.loss_edge_weight * regs["edge"]
                + cfg.loss_normal_weight * regs["normal"]
            )
            if not torch.isfinite(total):
                raise ReconstructionGateError("GATE 5 — OPTIMIZATION HEALTH FAILED: loss is NaN/Inf.")
            optimizer.zero_grad(set_to_none=True)
            backward_started = time.perf_counter()
            total.backward()
            if offsets.grad is None or not torch.isfinite(offsets.grad).all():
                raise ReconstructionGateError("GATE 5 — OPTIMIZATION HEALTH FAILED: vertex gradient is NaN/Inf.")
            optimizer.step()
            timing["backward"].append(time.perf_counter() - backward_started)

            values = {
                "total": total, "silhouette": silhouette, "rgb": rgb,
                "laplacian": regs["laplacian"], "edge": regs["edge"], "normal": regs["normal"],
                "iou": soft_iou_per_image(rendered_alpha.detach(), target_alpha).mean(),
            }
            for key, value in values.items():
                epoch_values[key].append(float(value.detach().cpu()))
            step_bar.set_postfix(loss=f"{epoch_values['total'][-1]:.4f}", iou=f"{epoch_values['iou'][-1]:.3f}")

        stats = _displacement_stats(offsets, object_size)
        if stats["max_vertex_displacement_ratio"] > cfg.max_vertex_displacement_ratio:
            offsets.data.copy_(previous_valid_offsets)
            save_checkpoint(checkpoint_dir / "last_valid_checkpoint.pt", offsets, epoch - 1, {"gate": "deformation_safety"})
            raise ReconstructionGateError(
                "GATE 4 — DEFORMATION SAFETY FAILED: "
                f"max displacement ratio {stats['max_vertex_displacement_ratio']:.4f} exceeds {cfg.max_vertex_displacement_ratio:.4f}."
            )
        previous_valid_offsets = offsets.detach().clone()
        row = {"epoch": epoch, **{key: sum(values) / max(len(values), 1) for key, values in epoch_values.items()}, **stats}
        append_csv(losses_path, row)
        if row["total"] < best_loss - 1e-6:
            best_loss, stale_epochs = row["total"], 0
        else:
            stale_epochs += 1
        if epoch % cfg.save_preview_every_epochs == 0 or epoch == 1:
            final_mesh = _mesh_with_offsets(base_mesh, offsets)
            for index in sorted(representative):
                sample = dataset[index]
                rendered_alpha = renderer.render_mask(final_mesh, camera_model.cameras([index]))
                save_stage_preview(sample.image[None].to(device), sample.alpha[None].to(device), rendered_alpha, preview_dir / f"epoch_{epoch:04d}_{index:04d}.png")
        if epoch % cfg.export_every_epochs == 0:
            checkpoint_mesh = _mesh_with_offsets(base_mesh, offsets)
            export_mesh_obj(checkpoint_mesh, checkpoint_dir / f"mesh_epoch_{epoch:04d}.obj", transform)
            export_mesh_stl(checkpoint_mesh, checkpoint_dir / f"mesh_epoch_{epoch:04d}.stl", transform)
            save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt", offsets, epoch)
        if timing["render"]:
            avg_render = sum(timing["render"]) / len(timing["render"])
            avg_step = sum(timing[key][-1] for key in ("render", "regularization", "backward"))
            print(f"Geometry timing after epoch {epoch}: render {avg_render * 1000:.1f} ms, last optimization step {avg_step:.2f} s")
        if cfg.early_stopping_patience > 0 and stale_epochs >= cfg.early_stopping_patience:
            break

    final_mesh = _mesh_with_offsets(base_mesh, offsets)
    final_obj = reconstruction_dir / "final.obj"
    final_stl = reconstruction_dir / "final.stl"
    export_mesh_obj(final_mesh, final_obj, transform)
    export_mesh_stl(final_mesh, final_stl, transform)
    (reconstruction_dir / "mesh_transform.json").write_text(json.dumps(transform.__dict__, indent=2))
    _write_view_usage(reconstruction_dir / "view_usage.csv", dataset.frames, usage_counts)
    plot_losses(losses_path, reconstruction_dir / "losses.png")
    wall = time.perf_counter() - started_total
    profile = {
        "runtime_profile": cfg.runtime_profile,
        "requested_device": cfg.device,
        "actual_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "original_mesh_vertices": (mesh_stats or {}).get("original_vertices"),
        "original_mesh_faces": (mesh_stats or {}).get("original_faces"),
        "geometry_proxy_vertices": len(base_mesh.verts_packed()),
        "geometry_proxy_faces": len(base_mesh.faces_packed()),
        "image_resolution": dataset.canvas_size,
        "faces_per_pixel": cfg.silhouette_faces_per_pixel,
        "total_available_frames": len(dataset),
        "views_per_epoch": views_per_epoch,
        "batch_size": cfg.geometry_view_batch_size,
        "epochs": cfg.num_epochs,
        "planned_optimization_steps": planned_steps,
        "actual_optimization_steps": len(timing["render"]),
        "average_render_time_seconds": sum(timing["render"]) / max(len(timing["render"]), 1),
        "average_regularization_time_seconds": sum(timing["regularization"]) / max(len(timing["regularization"]), 1),
        "average_backward_time_seconds": sum(timing["backward"]) / max(len(timing["backward"]), 1),
        "total_geometry_wall_time_seconds": wall,
    }
    (reconstruction_dir / "profile.json").write_text(json.dumps(profile, indent=2))
    return ReconstructionResult(mesh=final_mesh, final_obj=final_obj, final_stl=final_stl)
