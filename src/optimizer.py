from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from tqdm import tqdm

from .config import Config
from .dataset import ReconstructionDataset
from .export_utils import export_mesh_obj, save_checkpoint
from .losses import regularization_losses, rgb_loss, silhouette_loss
from .mesh_io import MeshTransform
from .renderer import ReconstructionRenderer
from .visualization import append_losses_csv, plot_losses, save_overlay


@dataclass
class TrainResult:
    final_mesh: Meshes
    losses_csv: Path
    final_obj: Path


def _mesh_with_params(base_mesh: Meshes, offsets: torch.Tensor, texture_logits: torch.Tensor | None) -> Meshes:
    verts = base_mesh.verts_padded() + offsets
    mesh = base_mesh.update_padded(verts)
    if texture_logits is not None:
        colors = texture_logits.sigmoid().clamp(0, 1)
        mesh.textures = TexturesVertex(verts_features=colors)
    return mesh


def train_reconstruction(
    dataset_name: str,
    dataset: ReconstructionDataset,
    base_mesh: Meshes,
    transform: MeshTransform,
    cfg: Config,
    device: torch.device,
) -> TrainResult:
    run_dir = cfg.run_dir(dataset_name)
    (run_dir / "previews").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "camera_metadata.used.json").write_text(
        json.dumps({"cameras": dataset.camera_records, "source": str(dataset.root)}, indent=2)
    )
    renderer = ReconstructionRenderer(cfg.image_size, device, cfg.background_rgb)
    verts_offsets = torch.zeros_like(base_mesh.verts_padded(), requires_grad=True, device=device)
    texture_logits = None
    params = [{"params": [verts_offsets], "lr": cfg.opt_lr_verts}]
    if cfg.opt_optimize_texture:
        initial = torch.full_like(base_mesh.verts_padded(), 0.5).clamp(1e-4, 1 - 1e-4)
        texture_logits = torch.logit(initial).detach().clone().requires_grad_(True)
        params.append({"params": [texture_logits], "lr": cfg.opt_lr_texture})
    opt = torch.optim.Adam(params)
    losses_csv = run_dir / "losses.csv"
    if losses_csv.exists():
        losses_csv.unlink()

    for iteration in tqdm(range(1, cfg.opt_num_iters + 1), desc=f"train {dataset_name}"):
        sample = dataset[(iteration - 1) % len(dataset)]
        mesh = _mesh_with_params(base_mesh, verts_offsets, texture_logits)
        rendered_rgb = renderer.render_rgb(mesh, sample.camera)
        rendered_mask = renderer.render_mask(mesh, sample.camera)
        target_rgb = sample.image[None]
        target_mask = sample.mask[None]

        sil = silhouette_loss(rendered_mask, target_mask)
        rgb = rgb_loss(rendered_rgb, target_rgb, target_mask)
        regs = regularization_losses(mesh)
        total = (
            cfg.loss_silhouette_weight * sil
            + cfg.loss_rgb_weight * rgb
            + cfg.loss_laplacian_weight * regs["laplacian"]
            + cfg.loss_edge_weight * regs["edge"]
            + cfg.loss_normal_weight * regs["normal"]
        )
        opt.zero_grad()
        total.backward()
        opt.step()

        row = {
            "iteration": iteration,
            "total": float(total.detach().cpu()),
            "silhouette": float(sil.detach().cpu()),
            "rgb": float(rgb.detach().cpu()),
            "laplacian": float(regs["laplacian"].detach().cpu()),
            "edge": float(regs["edge"].detach().cpu()),
            "normal": float(regs["normal"].detach().cpu()),
        }
        append_losses_csv(losses_csv, row)

        if iteration % cfg.save_preview_every == 0 or iteration == 1:
            save_overlay(rendered_rgb, target_rgb, rendered_mask, run_dir / "previews" / f"iter_{iteration:06d}.jpg")
        if iteration % cfg.export_every == 0:
            ckpt_mesh = _mesh_with_params(base_mesh, verts_offsets, texture_logits)
            export_mesh_obj(ckpt_mesh, run_dir / "checkpoints" / f"mesh_{iteration:06d}.obj", transform, restore=True)
            save_checkpoint(run_dir / "checkpoints" / f"checkpoint_{iteration:06d}.pt", verts_offsets, texture_logits, iteration)

    final_mesh = _mesh_with_params(base_mesh, verts_offsets, texture_logits)
    final_obj = run_dir / "final.obj"
    export_mesh_obj(final_mesh, final_obj, transform, restore=True)
    plot_losses(losses_csv, run_dir / "losses.png")
    return TrainResult(final_mesh=final_mesh, losses_csv=losses_csv, final_obj=final_obj)
