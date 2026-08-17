from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes


def silhouette_loss(rendered_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(rendered_mask, target_mask)


def rgb_loss(rendered_rgb: torch.Tensor, target_rgb: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    denom = target_mask.sum().clamp_min(1.0)
    return (((rendered_rgb - target_rgb) ** 2) * target_mask).sum() / denom


def regularization_losses(mesh: Meshes) -> dict[str, torch.Tensor]:
    return {
        "laplacian": mesh_laplacian_smoothing(mesh, method="uniform"),
        "edge": mesh_edge_loss(mesh),
        "normal": mesh_normal_consistency(mesh),
    }
