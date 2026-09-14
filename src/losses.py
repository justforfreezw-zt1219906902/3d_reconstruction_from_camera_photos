from __future__ import annotations

import torch
import torch.nn.functional as F
from pytorch3d.loss import mesh_normal_consistency
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


def _reference_scale(reference_vertices: torch.Tensor) -> torch.Tensor:
    """Frozen bounding-box diagonal for a single, nondegenerate CAD object."""
    if reference_vertices.ndim != 2 or reference_vertices.shape[1] != 3 or not len(reference_vertices):
        raise ValueError("reference_vertices must have nonempty shape (V, 3)")
    reference = reference_vertices.detach()
    scale = torch.linalg.vector_norm(reference.amax(dim=0) - reference.amin(dim=0))
    # A fixed epsilon in world units would make small objects unit-dependent.
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("CAD reference must have a finite, positive bounding-box diagonal")
    return scale


def reference_anchor_loss(
    current_vertices: torch.Tensor, reference_vertices: torch.Tensor,
) -> torch.Tensor:
    """Mean squared displacement / reference bounding-box diagonal squared.

    Inputs describe one object in the same aligned frame with identical vertex
    ordering. The reference is frozen: gradients belong only to current local
    geometry, never to the completed global alignment or the normalization.
    """
    if current_vertices.shape != reference_vertices.shape:
        raise ValueError("Current and reference vertex shapes must match")
    offsets = current_vertices - reference_vertices.detach()
    return (offsets / _reference_scale(reference_vertices)).square().sum(dim=-1).mean()


def local_deformation_smoothness_loss(
    offsets: torch.Tensor, edges: torch.Tensor, reference_vertices: torch.Tensor,
) -> torch.Tensor:
    """Mean squared offset differences across unique undirected CAD edges.

    Normalize by the frozen object size, not current edge lengths. Uniform
    offsets cost zero; no absolute position, edge length, or CAD curvature is
    driven toward zero. The anchor separately constrains displacement magnitude.
    ``edges`` has shape (E, 2), indexing the single object's (V, 3) vertices.
    """
    if offsets.shape != reference_vertices.shape:
        raise ValueError("Offset and reference vertex shapes must match")
    scale = _reference_scale(reference_vertices)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
    if not len(edges):
        return offsets.sum() * 0.0  # Differentiable zero for an edgeless object.
    differences = (offsets[edges[:, 0]] - offsets[edges[:, 1]]) / scale
    return differences.square().sum(dim=-1).mean()


def regularization_losses(
    mesh: Meshes, reference_mesh: Meshes | None = None,
) -> dict[str, torch.Tensor]:
    """Expose CAD proximity, residual coherence, and generic normal consistency.

    Audit: PyTorch3D mesh_edge_loss defaults to target_length=0 (edge collapse),
    and uniform mesh_laplacian_smoothing penalizes absolute vertex-to-neighbor
    displacement (smoothing/shrinkage). Neither preserves CAD geometry, so both
    have been removed rather than relabeled as reference-aware penalties.
    Normal consistency remains an optional generic adjacent-face orientation
    penalty; it is not CAD fidelity and can penalize intentional sharp creases.

    Supplying a frozen aligned reference enables the two local terms. Omitting
    it supports generic-only callers, never treating current geometry as its own
    reference. Batches are reduced per object so scale/topology of another object
    cannot affect its penalty. Current and reference meshes must share topology.
    """
    result = {"normal": mesh_normal_consistency(mesh)}
    if reference_mesh is None:
        return result
    if len(mesh) != len(reference_mesh) or len(mesh) == 0:
        raise ValueError("Current and reference mesh batches must match and be nonempty")
    anchors, smoothness = [], []
    for current, reference in zip(mesh, reference_mesh):
        if not torch.equal(current.faces_packed(), reference.faces_packed()):
            raise ValueError("Current and reference meshes must share topology")
        vertices = current.verts_packed()
        reference_vertices = reference.verts_packed().detach()
        anchors.append(reference_anchor_loss(vertices, reference_vertices))
        smoothness.append(local_deformation_smoothness_loss(
            vertices - reference_vertices, reference.edges_packed(), reference_vertices,
        ))
    result["anchor"] = torch.stack(anchors).mean()
    result["local_smoothness"] = torch.stack(smoothness).mean()
    return result
