from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch3d.structures import Meshes
from tqdm import tqdm

from .config import Config
from .export_utils import save_checkpoint
from .losses import alpha_silhouette_loss, masked_rgb_loss, regularization_losses, soft_iou
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


def train_geometry(
    base_mesh: Meshes,
    transform: MeshTransform,
    dataset: RGBADataset,
    camera_model: OpenScanCameraModel,
    renderer: ReconstructionRenderer,
    cfg: Config,
    device: torch.device,
    output_dir: Path,
) -> ReconstructionResult:
    reconstruction_dir = output_dir / "reconstruction"
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir = reconstruction_dir / "previews"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

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

    for epoch in tqdm(range(1, cfg.num_epochs + 1), desc="geometry reconstruction"):
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(cfg.seed + epoch))
        epoch_values: dict[str, list[float]] = {key: [] for key in ("total", "silhouette", "rgb", "laplacian", "edge", "normal", "iou")}
        for index in order.tolist():
            sample = dataset[index]
            target_rgb = sample.image.to(device)[None]
            target_alpha = sample.alpha.to(device)[None]
            mesh = _mesh_with_offsets(base_mesh, offsets)
            rendered_alpha = renderer.render_mask(mesh, camera_model.cameras([index]))
            if rendered_alpha.sum().detach().cpu() <= 0:
                raise ReconstructionGateError("GATE 5 — OPTIMIZATION HEALTH FAILED: rendered silhouette is empty.")
            silhouette = alpha_silhouette_loss(rendered_alpha, target_alpha)
            rgb = torch.zeros_like(silhouette)
            if cfg.loss_rgb_weight > 0:
                rendered_rgb = renderer.render_rgb(mesh, camera_model.cameras([index]))
                rgb = masked_rgb_loss(rendered_rgb, target_rgb, target_alpha)
            regs = regularization_losses(mesh)
            total = (
                cfg.loss_silhouette_weight * silhouette
                + cfg.loss_rgb_weight * rgb
                + cfg.loss_laplacian_weight * regs["laplacian"]
                + cfg.loss_edge_weight * regs["edge"]
                + cfg.loss_normal_weight * regs["normal"]
            )
            if not torch.isfinite(total):
                raise ReconstructionGateError("GATE 5 — OPTIMIZATION HEALTH FAILED: loss is NaN/Inf.")
            optimizer.zero_grad()
            total.backward()
            if offsets.grad is None or not torch.isfinite(offsets.grad).all():
                raise ReconstructionGateError("GATE 5 — OPTIMIZATION HEALTH FAILED: vertex gradient is NaN/Inf.")
            optimizer.step()
            values = {
                "total": total,
                "silhouette": silhouette,
                "rgb": rgb,
                "laplacian": regs["laplacian"],
                "edge": regs["edge"],
                "normal": regs["normal"],
                "iou": soft_iou(rendered_alpha.detach(), target_alpha),
            }
            for key, value in values.items():
                epoch_values[key].append(float(value.detach().cpu()))

        stats = _displacement_stats(offsets, object_size)
        if stats["max_vertex_displacement_ratio"] > cfg.max_vertex_displacement_ratio:
            offsets.data.copy_(previous_valid_offsets)
            save_checkpoint(checkpoint_dir / "last_valid_checkpoint.pt", offsets, epoch - 1, {"gate": "deformation_safety"})
            raise ReconstructionGateError(
                "GATE 4 — DEFORMATION SAFETY FAILED: "
                f"max displacement ratio {stats['max_vertex_displacement_ratio']:.4f} exceeds "
                f"{cfg.max_vertex_displacement_ratio:.4f}."
            )
        previous_valid_offsets = offsets.detach().clone()
        row = {
            "epoch": epoch,
            **{key: sum(values) / max(len(values), 1) for key, values in epoch_values.items()},
            **stats,
        }
        append_csv(losses_path, row)
        if row["total"] < best_loss - 1e-6:
            best_loss = row["total"]
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch % cfg.save_preview_every_epochs == 0 or epoch == 1:
            for index in sorted(representative):
                sample = dataset[index]
                final_mesh = _mesh_with_offsets(base_mesh, offsets)
                rendered_alpha = renderer.render_mask(final_mesh, camera_model.cameras([index]))
                save_stage_preview(
                    sample.image[None].to(device),
                    sample.alpha[None].to(device),
                    rendered_alpha,
                    preview_dir / f"epoch_{epoch:04d}_{index:04d}.png",
                )
        if epoch % cfg.export_every_epochs == 0:
            checkpoint_mesh = _mesh_with_offsets(base_mesh, offsets)
            export_mesh_obj(checkpoint_mesh, checkpoint_dir / f"mesh_epoch_{epoch:04d}.obj", transform)
            export_mesh_stl(checkpoint_mesh, checkpoint_dir / f"mesh_epoch_{epoch:04d}.stl", transform)
            save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt", offsets, epoch)
        if cfg.early_stopping_patience > 0 and stale_epochs >= cfg.early_stopping_patience:
            break

    final_mesh = _mesh_with_offsets(base_mesh, offsets)
    final_obj = reconstruction_dir / "final.obj"
    final_stl = reconstruction_dir / "final.stl"
    export_mesh_obj(final_mesh, final_obj, transform)
    export_mesh_stl(final_mesh, final_stl, transform)
    (reconstruction_dir / "mesh_transform.json").write_text(json.dumps(transform.__dict__, indent=2))
    plot_losses(losses_path, reconstruction_dir / "losses.png")
    return ReconstructionResult(mesh=final_mesh, final_obj=final_obj, final_stl=final_stl)
