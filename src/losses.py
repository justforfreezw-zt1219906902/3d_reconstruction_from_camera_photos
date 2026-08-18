from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch3d.loss import mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.structures import Meshes


def alpha_silhouette_loss(rendered_alpha: torch.Tensor, target_alpha: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(rendered_alpha, target_alpha)


def soft_iou(rendered_alpha: torch.Tensor, target_alpha: torch.Tensor) -> torch.Tensor:
    intersection = (rendered_alpha * target_alpha).sum()
    union = (rendered_alpha + target_alpha - rendered_alpha * target_alpha).sum().clamp_min(1e-6)
    return intersection / union


def soft_iou_per_image(rendered_alpha: torch.Tensor, target_alpha: torch.Tensor) -> torch.Tensor:
    """Soft IoU for a batch, preserving one score per rendered image."""
    intersection = (rendered_alpha * target_alpha).flatten(1).sum(dim=1)
    union = (rendered_alpha + target_alpha - rendered_alpha * target_alpha).flatten(1).sum(dim=1)
    return intersection / union.clamp_min(1e-6)


def masked_rgb_loss(rendered_rgb: torch.Tensor, target_rgb: torch.Tensor, target_alpha: torch.Tensor) -> torch.Tensor:
    denom = target_alpha.sum().clamp_min(1.0)
    return (((rendered_rgb - target_rgb) ** 2) * target_alpha).sum() / denom


def regularization_losses(mesh: Meshes) -> dict[str, torch.Tensor]:
    return {
        "laplacian": mesh_laplacian_smoothing(mesh, method="uniform"),
        "edge": mesh_edge_loss(mesh),
        "normal": mesh_normal_consistency(mesh),
    }
