from __future__ import annotations

import csv
import json
import math
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
from .losses import (
    alpha_silhouette_loss,
    image_only_regularization_losses,
    masked_rgb_loss,
    regularization_losses,
    soft_iou_per_image,
)
from .mesh_io import MeshTransform, export_mesh_obj, export_mesh_stl, inspect_mesh_validity
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


def _local_offset_basis(vertices: torch.Tensor) -> torch.Tensor:
    """Orthonormal global similarity modes for the single aligned mesh.

    Orthogonality fixes centroid, scale and rotational moment relative to the
    aligned reference. The deformation bound keeps residuals in this local regime.
    """
    points = vertices.detach().reshape(-1, 3)
    centered = points - points.mean(0)
    axes = torch.eye(3, dtype=points.dtype, device=points.device)
    modes = [axis.expand_as(points).reshape(-1) for axis in axes]
    modes += [torch.linalg.cross(axis.expand_as(points), centered).reshape(-1) for axis in axes]
    modes.append(centered.reshape(-1))
    u, singular, _ = torch.linalg.svd(torch.stack(modes, dim=1), full_matrices=False)
    return u[:, singular > singular.max() * 1e-6]


def _project_local_offsets(offsets: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    flat = offsets.reshape(-1)
    return (flat - basis @ (basis.T @ flat)).reshape_as(offsets)


def _bounded_local_offsets(
    offsets: torch.Tensor, basis: torch.Tensor, max_displacement: float,
) -> torch.Tensor:
    """Remove global modes, then contract the whole residual field to its limit.

    A shared scalar preserves orthogonality to global similarity modes; clipping
    each vertex independently would reintroduce those modes. Leave a small
    floating-point margin at the boundary. Interior offsets are unchanged.
    """
    if not math.isfinite(max_displacement) or max_displacement < 0:
        raise ValueError("MAX_LOCAL_DEFORMATION_RATIO must give a finite, non-negative displacement limit")
    local = _project_local_offsets(offsets, basis)
    if not torch.isfinite(local).all():
        raise ReconstructionGateError("GATE 4 — DEFORMATION SAFETY FAILED: offsets are NaN/Inf.")
    if max_displacement == 0:
        return local * 0.0
    peak = torch.linalg.vector_norm(local, dim=-1).max()
    limit = max_displacement * (1.0 - 8.0 * torch.finfo(local.dtype).eps)
    return local * (limit / peak.clamp_min(torch.finfo(local.dtype).tiny)).clamp(max=1.0)


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


def is_better_geometry_state(
    candidate: dict[str, float], best: dict[str, float] | None,
    *, constraints_satisfied: bool, quality_tolerance: float = 1e-6,
) -> bool:
    """Compare valid states lexicographically; exact ties retain the incumbent.

    Lower held-out validation MSE wins, then higher held-out IoU, using an
    absolute tolerance for each quality metric. Effectively tied image
    quality prefers the cleaner arbitrary-topology mesh before considering
    displacement. Callers own hard geometry constraint checks. This helper
    deliberately receives only reconstruction/validation diagnostics; external
    post-reconstruction reference evaluation is outside its interface.

    Older callers that do not yet collect validity diagnostics retain the
    historical displacement tie-break by treating omitted counts as zero.
    Training always supplies the complete set below.
    """
    if not math.isfinite(quality_tolerance) or quality_tolerance < 0:
        raise ValueError("quality_tolerance must be finite and non-negative")
    keys = ("validation_silhouette", "validation_iou",
            "mean_vertex_displacement", "p95_vertex_displacement")
    if not constraints_satisfied or not all(math.isfinite(candidate[k]) for k in keys):
        return False
    # Counts come from inspect_mesh_validity, which compares the candidate
    # only with itself. A fully-tested valid mesh wins first; then rank the
    # most harmful defects before less severe cleanup indicators.
    validity_keys = (
        ("validity_is_valid", -1),
        ("validity_non_finite_vertex_count", 1),
        ("validity_degenerate_face_count", 1),
        ("validity_self_intersection_pair_count", 1),
        ("validity_spike_vertex_count", 1),
        ("validity_extreme_edge_count", 1),
        ("validity_disconnected_fragment_count", 1),
        ("validity_small_component_count", 1),
    )
    if not all(math.isfinite(candidate.get(key, 0.0)) for key, _ in validity_keys):
        return False
    if best is None:
        return True
    for key, direction in ((keys[0], 1), (keys[1], -1)):
        difference = direction * (candidate[key] - best[key])
        if abs(difference) > quality_tolerance:
            return difference < 0
    for key, direction in validity_keys:
        candidate_value = candidate.get(key, 0.0)
        best_value = best.get(key, 0.0)
        if not math.isfinite(best_value):
            return True
        difference = direction * (candidate_value - best_value)
        if difference:
            return difference < 0
    return (candidate[keys[2]], candidate[keys[3]]) < (best[keys[2]], best[keys[3]])


@torch.no_grad()
def evaluate_geometry_validation(
    base_mesh: Meshes, offsets: torch.Tensor, dataset: RGBADataset,
    camera_model: OpenScanCameraModel, renderer: ReconstructionRenderer,
    indices: list[int], batch_size: int, device: torch.device, object_size: float,
) -> dict[str, float]:
    """Evaluate a fixed state on every held-out view without gradients or steps.

    Weight batch means by view count so the final short batch is not overweighted.
    Displacement describes the shared mesh, independent of viewing direction.
    """
    stats = _displacement_stats(offsets, object_size)
    silhouette_sum = iou_sum = 0.0
    mesh = _mesh_with_offsets(base_mesh, offsets)
    for batch in _batch_indices(indices, batch_size):
        target = torch.stack([dataset[index].alpha for index in batch]).to(device)
        rendered = renderer.render_mask(mesh, camera_model.cameras(batch))
        silhouette_sum += float(alpha_silhouette_loss(rendered, target).cpu()) * len(batch)
        iou_sum += float(soft_iou_per_image(rendered, target).sum().cpu())
    return {
        **stats,
        "validation_silhouette": silhouette_sum / len(indices) if indices else float("nan"),
        "validation_iou": iou_sum / len(indices) if indices else float("nan"),
    }


@torch.no_grad()
def evaluate_geometry_validity(
    base_mesh: Meshes, offsets: torch.Tensor, cfg: Config,
) -> dict[str, float]:
    """Return reference-free mesh-health metrics for best-state selection.

    The diagnostics inspect only the current reconstructed mesh. Neither the
    initial sphere/CAD positions nor external evaluation data are inputs, so
    this works for both CAD-prior and image-only topology while preserving the
    evaluation isolation boundary.
    """
    report = inspect_mesh_validity(
        _mesh_with_offsets(base_mesh, offsets),
        max_edge_length_ratio=cfg.mesh_max_edge_length_ratio,
        spike_ratio=cfg.mesh_spike_ratio,
        min_component_faces=cfg.mesh_min_component_faces,
    )
    return {
        "validity_is_valid": float(report.is_valid),
        "validity_non_finite_vertex_count": float(report.non_finite_vertex_count),
        "validity_degenerate_face_count": float(report.degenerate_face_count),
        "validity_self_intersection_pair_count": float(report.self_intersection_pair_count),
        "validity_spike_vertex_count": float(report.spike_vertex_count),
        "validity_extreme_edge_count": float(report.extreme_edge_count),
        "validity_disconnected_fragment_count": float(report.disconnected_fragment_count),
        "validity_small_component_count": float(report.small_component_count),
    }


def split_geometry_view_indices(
    frames: Sequence, seed: int, validation_fraction: float,
) -> tuple[list[int], list[int]]:
    """Return fixed, disjoint dataset indices for local training and validation.

    Sort by elevation ring and azimuth, then hold out one seeded random view
    per equal-sized stratum. This spreads validation around the available
    angles without rounding each small ring's quota up independently. Missing
    angles in synthetic frames fall back to dataset order. For fewer than two
    frames, retain all frames for training: a disjoint holdout is impossible.
    """
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("GEOMETRY_VALIDATION_FRACTION must be finite and between 0 and 1 (exclusive)")
    count = len(frames)
    if count < 2:
        return list(range(count)), []
    validation_count = min(count - 1, max(1, round(count * validation_fraction)))
    ordered = sorted(range(count), key=lambda i: (
        round(float(getattr(frames[i], "phi_deg", 0.0)), 5),
        float(getattr(frames[i], "theta_deg", i)) % 360.0,
        i,
    ))
    rng = random.Random(seed)
    validation = sorted(
        rng.choice(ordered[k * count // validation_count:(k + 1) * count // validation_count])
        for k in range(validation_count)
    )
    held_out = set(validation)
    return [i for i in range(count) if i not in held_out], validation


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
        groups.setdefault(round(float(getattr(frame, "phi_deg", 0.0)), 5), []).append(index)
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
                abs(((float(getattr(frames[i], "theta_deg", i)) - target_theta + 180.0) % 360.0) - 180.0),
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
            writer.writerow([index, getattr(frame, "image", ""), getattr(frame, "phi_deg", ""), getattr(frame, "theta_deg", ""), count])


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
    """Refine a mesh against image observations.

    ``cad_prior`` retains the legacy local-CAD deformation objective.  In
    ``image_only`` mode ``base_mesh`` is the generated coarse visual-hull mesh:
    its arbitrary topology is fixed during this differentiable stage, but its
    vertices are unconstrained by the initial sphere/CAD and by the CAD 3%
    local-deformation limit.
    """
    reconstruction_dir = output_dir / "reconstruction"
    checkpoint_dir = reconstruction_dir / "checkpoints"
    preview_dir = reconstruction_dir / "previews"
    reconstruction_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    training_indices, validation_indices = split_geometry_view_indices(
        dataset.frames, cfg.seed, cfg.geometry_validation_fraction
    )
    training_frames = [dataset.frames[i] for i in training_indices]
    (reconstruction_dir / "view_split.json").write_text(json.dumps({
        "seed": cfg.seed, "validation_fraction": cfg.geometry_validation_fraction,
        "training_indices": training_indices, "validation_indices": validation_indices,
    }, indent=2))
    views_per_epoch = min(cfg.geometry_views_per_epoch or len(training_indices), len(training_indices))
    batches_per_epoch = (views_per_epoch + cfg.geometry_view_batch_size - 1) // cfg.geometry_view_batch_size
    planned_steps = batches_per_epoch * cfg.num_epochs
    print(f"Runtime profile: {cfg.runtime_profile}")
    print(f"Device: {device}")
    print("Mesh:")
    print(f"  original faces: {(mesh_stats or {}).get('original_faces', 'unknown')}")
    print(f"  geometry faces: {(mesh_stats or {}).get('geometry_faces', len(base_mesh.faces_packed()))}")
    print("Dataset:")
    print(f"  validated frames: {len(dataset)}")
    print(f"  training / validation frames: {len(training_indices)} / {len(validation_indices)}")
    print(f"  views per epoch: {views_per_epoch}")
    print("Geometry:")
    print(f"  epochs: {cfg.num_epochs}")
    print(f"  batch size: {cfg.geometry_view_batch_size}")
    print(f"  planned optimization steps: {planned_steps}")
    print(f"  resolution: {dataset.canvas_size}")
    print(f"  faces per pixel: {cfg.silhouette_faces_per_pixel}")

    image_only = cfg.reconstruction_mode == "image_only"
    # Cameras/alignment are frozen before either refinement mode.  CAD mode
    # uses the aligned CAD reference; image-only mode uses only coarse geometry.
    base_mesh = base_mesh.detach()
    camera_model.freeze()
    local_basis = None if image_only else _local_offset_basis(base_mesh.verts_padded())
    offsets = torch.zeros_like(base_mesh.verts_padded(), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([offsets], lr=cfg.opt_lr_verts)
    losses_path = reconstruction_dir / "losses.csv"
    if losses_path.exists():
        losses_path.unlink()
    best_metrics: dict[str, float] | None = None
    best_offsets = offsets.detach().clone()
    best_epoch = 0
    stale_epochs = 0
    previous_valid_offsets = offsets.detach().clone()
    vertices = base_mesh.verts_packed()
    object_size = float(torch.linalg.vector_norm(vertices.max(0).values - vertices.min(0).values).cpu())
    max_local_displacement = cfg.max_local_deformation_ratio * object_size
    if not image_only:
        with torch.no_grad():
            offsets.copy_(_bounded_local_offsets(offsets, local_basis, max_local_displacement))
    representative = set(dataset.representative_indices())
    usage_counts = [0] * len(dataset)
    timing = {"render": [], "regularization": [], "backward": []}
    started_total = time.perf_counter()

    for epoch in tqdm(range(1, cfg.num_epochs + 1), desc="geometry epochs"):
        selected = [training_indices[i] for i in select_geometry_view_indices(
            training_frames, cfg.geometry_views_per_epoch, epoch, cfg.seed,
            [usage_counts[i] for i in training_indices],
        )]
        for index in selected:
            usage_counts[index] += 1
        epoch_values: dict[str, list[float]] = {
            key: [] for key in (
                "total", "silhouette", "rgb", "normal", "anchor", "local_smoothness",
                "smoothness", "edge", "coarse_anchor", "iou",
            )
        }
        batches = _batch_indices(selected, cfg.geometry_view_batch_size)
        step_bar = tqdm(batches, desc=f"geometry epoch {epoch}/{cfg.num_epochs}", leave=False)
        for step, batch in enumerate(step_bar, start=1):
            samples = [dataset[index] for index in batch]
            target_rgb = torch.stack([sample.image for sample in samples]).to(device)
            target_alpha = torch.stack([sample.alpha for sample in samples]).to(device)
            cameras = camera_model.cameras(batch)
            effective_offsets = offsets if image_only else _bounded_local_offsets(
                offsets, local_basis, max_local_displacement,
            )
            mesh = _mesh_with_offsets(base_mesh, effective_offsets)

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
            if image_only:
                regs = image_only_regularization_losses(mesh, base_mesh)
                # No reference_anchor_loss against a CAD/sphere and no CAD
                # local-deformation restriction are present in this branch.
                total = (
                    cfg.loss_silhouette_weight * silhouette
                    + cfg.loss_rgb_weight * rgb
                    + cfg.loss_normal_weight * regs["normal"]
                    + cfg.loss_laplacian_weight * regs["smoothness"]
                    + cfg.loss_edge_weight * regs["edge"]
                    + cfg.image_only_coarse_anchor_weight * regs["coarse_anchor"]
                )
            else:
                regs = regularization_losses(mesh, base_mesh)
                total = (
                    cfg.loss_silhouette_weight * silhouette
                    + cfg.loss_rgb_weight * rgb
                    + cfg.loss_normal_weight * regs["normal"]
                    + cfg.loss_anchor_weight * regs["anchor"]
                    + cfg.loss_local_smoothness_weight * regs["local_smoothness"]
                )
            timing["regularization"].append(time.perf_counter() - regularization_started)
            if not torch.isfinite(total):
                raise ReconstructionGateError("GATE 5 — OPTIMIZATION HEALTH FAILED: loss is NaN/Inf.")
            optimizer.zero_grad(set_to_none=True)
            backward_started = time.perf_counter()
            total.backward()
            if offsets.grad is None or not torch.isfinite(offsets.grad).all():
                raise ReconstructionGateError("GATE 5 — OPTIMIZATION HEALTH FAILED: vertex gradient is NaN/Inf.")
            optimizer.step()
            if not image_only:
                with torch.no_grad():
                    offsets.copy_(_bounded_local_offsets(offsets, local_basis, max_local_displacement))
            timing["backward"].append(time.perf_counter() - backward_started)

            values = {
                "total": total, "silhouette": silhouette, "rgb": rgb,
                "normal": regs["normal"],
                "anchor": regs.get("anchor", torch.zeros_like(silhouette)),
                "local_smoothness": regs.get("local_smoothness", torch.zeros_like(silhouette)),
                "smoothness": regs.get("smoothness", torch.zeros_like(silhouette)),
                "edge": regs.get("edge", torch.zeros_like(silhouette)),
                "coarse_anchor": regs.get("coarse_anchor", torch.zeros_like(silhouette)),
                "iou": soft_iou_per_image(rendered_alpha.detach(), target_alpha).mean(),
            }
            for key, value in values.items():
                epoch_values[key].append(float(value.detach().cpu()))
            step_bar.set_postfix(loss=f"{epoch_values['total'][-1]:.4f}", iou=f"{epoch_values['iou'][-1]:.3f}")

        stats = evaluate_geometry_validation(
            base_mesh, offsets, dataset, camera_model, renderer, validation_indices,
            cfg.geometry_view_batch_size, device, object_size,
        )
        validity = evaluate_geometry_validity(base_mesh, offsets, cfg)
        if not image_only and stats["max_vertex_displacement_ratio"] > cfg.max_vertex_displacement_ratio:
            offsets.data.copy_(previous_valid_offsets)
            save_checkpoint(checkpoint_dir / "last_valid_checkpoint.pt", offsets, epoch - 1, {"gate": "deformation_safety"})
            raise ReconstructionGateError(
                "GATE 4 — DEFORMATION SAFETY FAILED: "
                f"max displacement ratio {stats['max_vertex_displacement_ratio']:.4f} exceeds {cfg.max_vertex_displacement_ratio:.4f}."
            )
        previous_valid_offsets = offsets.detach().clone()
        row = {
            "epoch": epoch,
            **{key: sum(values) / max(len(values), 1) for key, values in epoch_values.items()},
            **stats,
            **validity,
        }
        # Training metrics average optimization batches before their updates;
        # validation and displacement describe the state after the epoch.
        # Keep the original names for existing CSV consumers.
        row.update(training_objective=row["total"], training_silhouette=row["silhouette"],
                   training_iou=row["iou"])
        constraints_satisfied = all(
            math.isfinite(value) for key, value in stats.items() if not key.startswith("validation_")
        )
        if not image_only:
            constraints_satisfied = constraints_satisfied and (
                stats["max_vertex_displacement_ratio"] <= min(
                    cfg.max_local_deformation_ratio, cfg.max_vertex_displacement_ratio)
            )
        is_best = is_better_geometry_state(row, best_metrics, constraints_satisfied=constraints_satisfied)
        if is_best:
            best_metrics = row.copy()
            best_offsets = offsets.detach().clone()
            best_epoch, stale_epochs = epoch, 0
        else:
            stale_epochs += 1
        row.update(best_epoch=best_epoch, is_best=int(is_best))
        append_csv(losses_path, row)
        if epoch % cfg.save_preview_every_epochs == 0 or epoch == 1:
            with torch.no_grad():
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
        if validation_indices and cfg.early_stopping_patience > 0 and stale_epochs >= cfg.early_stopping_patience:
            break

    # Export the selected snapshot directly, including after early stopping.
    # Epoch checkpoints above intentionally retain their own epoch's offsets.
    final_mesh = _mesh_with_offsets(base_mesh, best_offsets)
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
