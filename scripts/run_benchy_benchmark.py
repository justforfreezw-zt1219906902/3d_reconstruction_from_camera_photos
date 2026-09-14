#!/usr/bin/env python3
"""Run the image-only Benchy benchmark and write post-hoc diagnostics.

The reconstruction itself is deliberately owned by :func:`src.pipeline.run_from_config`.
This script only configures that entry point, times it, and summarizes artifacts it
produces.  In particular, the ground-truth mesh remains evaluation-only: its metrics
come from the pipeline's post-export ``ground_truth_evaluation.json`` artifact.

Example::

    python scripts/run_benchy_benchmark.py --env-file .env.benchy

The env file follows the normal ``load_config`` convention and must provide
``INITIAL_MESH_PATH``, ``GROUND_TRUTH_MESH_PATH``, ``RGBA_DIR``, and
``POSITIONS_CSV``.  The runner always forces ``image_only`` mode.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

# Support the documented ``python scripts/run_benchy_benchmark.py`` invocation
# without requiring callers to set PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, load_config, select_device
from src.pipeline import _build_geometry_model, run_from_config


def benchmark_config(env_file: str | Path | None = None) -> Config:
    """Load normal env-file configuration and force safe image-only controls."""
    cfg = load_config(env_file)
    if cfg.ground_truth_mesh_path is None:
        raise ValueError(
            "GROUND_TRUTH_MESH_PATH is required for the Benchy benchmark; "
            "it is used only for post-reconstruction evaluation."
        )
    return replace(
        cfg,
        reconstruction_mode="image_only",
        # A config file may have been authored for CAD mode.  Do not retain its
        # CAD-only effective values when this runner changes the mode.
        loss_anchor_weight=0.0,
        max_local_deformation_ratio=0.0,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected benchmark artifact was not written: {path}")
    return json.loads(path.read_text())


def _load_stage_mesh(path: Path, initial_transform: Any, device: Any):
    """Load an exported physical mesh back into the run's normalized frame."""
    from src.mesh_io import load_initial_mesh, mesh_from_arrays, normalize_vertices, restore_vertices

    # ``load_initial_mesh`` supports both OBJ and STL.  Undo its temporary
    # normalization and apply the original sphere transform so all three
    # stages share the calibrated renderer's normalized object frame.
    temporary_mesh, temporary_transform = load_initial_mesh(
        path, device, center_normalize=True, scale_normalize=True,
    )
    physical_vertices = restore_vertices(temporary_mesh.verts_packed(), temporary_transform)
    return mesh_from_arrays(
        normalize_vertices(physical_vertices, initial_transform),
        temporary_mesh.faces_packed(),
        device,
    )


def _calibrated_camera_from_artifacts(cfg: Config, contract: Any, device: Any):
    """Recreate frozen image-only cameras from the pipeline's calibration files."""
    from src.fast_silhouette import CameraParameters
    from src.pose_conventions import CameraConvention

    camera_dir = cfg.output_dir / "camera_fit"
    camera_values = _read_json(camera_dir / "global_camera_parameters.json")
    camera_metrics = _read_json(camera_dir / "metrics.json")
    convention = CameraConvention(**camera_metrics["selected_convention"])
    camera = CameraParameters(
        distance=float(camera_values["distance"]),
        fov_deg=float(camera_values["fov_deg"]),
        x_offset=float(camera_values.get("principal_point_x_ndc", 0.0)),
        y_offset=float(camera_values.get("principal_point_y_ndc", 0.0)),
    )
    result = type("FrozenCameraResult", (), {
        "camera": camera,
        "convention": convention,
        "theta_deltas": [0.0] * len(contract.frames),
        "phi_deltas": [0.0] * len(contract.frames),
    })()
    return _build_geometry_model(contract, cfg, device, result)


def collect_held_out_iou(cfg: Config) -> dict[str, float]:
    """Render sphere/coarse/final meshes on the pipeline's held-out views.

    This is post-hoc reporting only.  It reuses the dataset, camera-model and
    renderer utilities used by the pipeline; it neither optimizes geometry nor
    changes the selected state.
    """
    import torch

    from src.losses import soft_iou_per_image
    from src.mesh_io import load_initial_mesh
    from src.renderer import ReconstructionRenderer
    from src.rgba_dataset import RGBADataset
    from src.validation import validate_inputs

    reconstruction = cfg.output_dir / "reconstruction"
    split = _read_json(reconstruction / "view_split.json")
    indices = [int(index) for index in split.get("validation_indices", [])]
    if not indices:
        raise ValueError("Benchy benchmark requires at least one held-out validation view.")
    device = select_device(cfg.device)
    contract = validate_inputs(cfg)
    dataset = RGBADataset(contract, cfg.max_image_dimension)
    sphere_mesh, transform = load_initial_mesh(
        cfg.initial_mesh_path, device,
        center_normalize=cfg.center_normalize, scale_normalize=cfg.scale_normalize,
    )
    meshes = {
        "sphere": sphere_mesh,
        "coarse": _load_stage_mesh(reconstruction / "image_coarse_mesh.stl", transform, device),
        "final": _load_stage_mesh(reconstruction / "final.obj", transform, device),
    }
    camera_model = _calibrated_camera_from_artifacts(cfg, contract, device)
    renderer = ReconstructionRenderer(dataset.canvas_size, device, cfg.silhouette_faces_per_pixel)
    scores: dict[str, float] = {}
    with torch.no_grad():
        for stage, mesh in meshes.items():
            total = 0.0
            for index in indices:
                target = dataset[index].alpha[None].to(device)
                rendered = renderer.render_mask(mesh, camera_model.cameras([index]))
                total += float(soft_iou_per_image(rendered, target).item())
            scores[stage] = total / len(indices)
    return scores


def collect_final_validity(cfg: Config) -> dict[str, Any]:
    """Return arbitrary-topology validity diagnostics for exported ``final.stl``."""
    from src.mesh_io import inspect_mesh_validity, load_initial_mesh

    mesh, _ = load_initial_mesh(cfg.output_dir / "reconstruction" / "final.stl", select_device(cfg.device))
    return inspect_mesh_validity(
        mesh,
        max_edge_length_ratio=cfg.mesh_max_edge_length_ratio,
        spike_ratio=cfg.mesh_spike_ratio,
        min_component_faces=cfg.mesh_min_component_faces,
    ).as_dict()


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Benchy image-only benchmark",
        "",
        f"- Runtime: {report['runtime_seconds']:.3f} seconds",
        f"- Output directory: `{report['output_dir']}`",
        "",
        "## Held-out silhouette IoU",
        "",
        "| Stage | IoU |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {stage} | {score:.6f} |" for stage, score in report["held_out_iou"].items())
    lines += ["", "## Ground-truth error (primary physical frame)", "", "| Stage | Mean surface distance | Chamfer distance | p95 distance |", "| --- | ---: | ---: | ---: |"]
    for stage, values in report["ground_truth_evaluation"]["stages"].items():
        physical = values["physical"]
        lines.append(
            f"| {stage} | {physical['mean_surface_distance']:.6g} | "
            f"{physical['symmetric_chamfer_distance']:.6g} | {physical['p95_surface_distance']:.6g} |"
        )
    validity = report["final_mesh_validity"]
    lines += [
        "", "## Final mesh validity", "",
        f"- Valid: `{validity['is_valid']}`",
        f"- Degenerate faces: {validity['degenerate_face_count']}",
        f"- Extreme edges: {validity['extreme_edge_count']}",
        f"- Disconnected fragments: {validity['disconnected_fragment_count']}",
        f"- Non-finite vertices: {validity['non_finite_vertex_count']}",
        f"- Spike vertices: {validity['spike_vertex_count']}",
        "",
    ]
    return "\n".join(lines)


def run_benchmark(
    cfg: Config,
    *,
    pipeline_runner: Callable[[Config], dict[str, Any]] = run_from_config,
    iou_collector: Callable[[Config], dict[str, float]] = collect_held_out_iou,
    validity_collector: Callable[[Config], dict[str, Any]] = collect_final_validity,
) -> dict[str, Any]:
    """Run the normal pipeline once and write JSON and Markdown benchmark reports."""
    if cfg.reconstruction_mode != "image_only":
        raise ValueError("The Benchy benchmark only supports RECONSTRUCTION_MODE=image_only.")
    if cfg.ground_truth_mesh_path is None:
        raise ValueError("GROUND_TRUTH_MESH_PATH is required for the Benchy benchmark.")
    started = time.perf_counter()
    pipeline_summary = pipeline_runner(cfg)
    runtime_seconds = time.perf_counter() - started
    reconstruction = cfg.output_dir / "reconstruction"
    evaluation = _read_json(reconstruction / "ground_truth_evaluation.json")
    json_path = reconstruction / "benchy_benchmark_report.json"
    markdown_path = reconstruction / "benchy_benchmark_report.md"
    report = {
        "benchmark": "benchy_image_only",
        "reconstruction_mode": cfg.reconstruction_mode,
        "initial_mesh_path": str(cfg.initial_mesh_path),
        "ground_truth_mesh_path": str(cfg.ground_truth_mesh_path),
        "output_dir": str(cfg.output_dir),
        "runtime_seconds": runtime_seconds,
        "pipeline_summary": pipeline_summary,
        "held_out_iou": iou_collector(cfg),
        "ground_truth_evaluation": evaluation,
        "final_mesh_validity": validity_collector(cfg),
        "report_json": str(json_path),
        "report_markdown": str(markdown_path),
    }
    json_path.write_text(json.dumps(report, indent=2, default=str))
    markdown_path.write_text(_markdown_report(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the image-only Benchy benchmark through the normal pipeline.")
    parser.add_argument("--env-file", default=".env", help="Configuration env file (default: .env).")
    args = parser.parse_args()
    report = run_benchmark(benchmark_config(args.env_file))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
