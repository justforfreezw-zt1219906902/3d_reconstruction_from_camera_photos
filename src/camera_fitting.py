from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .config import Config
from .fast_silhouette import (
    CameraParameters,
    FastMesh,
    FastSilhouetteRenderer,
    create_camera_proxy,
    load_fast_mesh,
    save_fast_mesh_stl,
    silhouette_iou,
    silhouette_summary,
)
from .pose_conventions import CameraConvention, DEFAULT_CONVENTION, candidate_conventions
from .rgba_dataset import build_layouts, load_alpha_with_layout, load_rgb_with_layout
from .validation import DatasetContract, FrameRecord


class CameraFitGateError(RuntimeError):
    pass


@dataclass
class FastCameraFitResult:
    camera: CameraParameters
    convention: CameraConvention
    theta_deltas: list[float]
    phi_deltas: list[float]
    metrics: dict
    # The image-only result is calibration data, not an optimisation result.
    # Pipeline code must transfer it to an OpenScanCameraModel and freeze it
    # before any geometry stage begins.
    frozen: bool = False


@dataclass
class CameraFitData:
    frames: tuple[FrameRecord, ...]
    layouts: tuple
    alphas: tuple[np.ndarray, ...]
    image_size: int


def _load_camera_fit_data(contract: DatasetContract, max_dimension: int) -> tuple[CameraFitData, float]:
    started = time.perf_counter()
    layouts = build_layouts(contract, max_dimension)
    alphas = tuple(
        load_alpha_with_layout(frame.image_path, layout).numpy()[..., 0]
        for frame, layout in zip(contract.frames, layouts)
    )
    return CameraFitData(contract.frames, layouts, alphas, layouts[0].canvas_size), time.perf_counter() - started


def select_representative_indices(data: CameraFitData | object, max_frames: int) -> list[int]:
    frames = data.frames
    groups: dict[float, list[int]] = {}
    for index, frame in enumerate(frames):
        groups.setdefault(round(frame.phi_deg, 6), []).append(index)
    rings = sorted(groups)
    if not rings:
        return []
    limit = max(max_frames, len(rings))
    base, remainder = divmod(limit, len(rings))
    selected: set[int] = set()
    for ring_index, phi in enumerate(rings):
        candidates = sorted(groups[phi], key=lambda i: (frames[i].theta_deg % 360.0, frames[i].position_index))
        slots = max(1, min(len(candidates), base + (ring_index < remainder)))
        for slot in range(slots):
            desired = 360.0 * slot / slots
            choice = min(
                (i for i in candidates if i not in selected),
                key=lambda i: (min(abs((frames[i].theta_deg - desired) % 360.0), abs((desired - frames[i].theta_deg) % 360.0)), frames[i].position_index),
                default=None,
            )
            if choice is not None:
                selected.add(choice)
    return sorted(selected, key=lambda i: frames[i].position_index)


def _save_selected_frames(data: CameraFitData, indices: list[int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "image", "position_index", "phi_deg", "theta_deg"])
        writer.writeheader()
        for index in indices:
            frame = data.frames[index]
            writer.writerow({"index": index, **{key: getattr(frame, key) for key in ("image", "position_index", "phi_deg", "theta_deg")}})


def _save_preview(data: CameraFitData, index: int, predicted: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = load_rgb_with_layout(data.frames[index].image_path, data.layouts[index]).numpy()
    target = np.repeat(data.alphas[index][..., None], 3, axis=2)
    rendered = np.repeat(predicted[..., None], 3, axis=2)
    overlay = np.zeros_like(target)
    overlay[..., 0] = data.alphas[index]
    overlay[..., 1] = predicted
    image = np.concatenate([rgb, target, rendered, overlay], axis=1)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)


def _evaluate(
    renderer: FastSilhouetteRenderer,
    mesh: FastMesh,
    data: CameraFitData,
    camera: CameraParameters,
    convention: CameraConvention,
    theta_deltas: list[float],
    phi_deltas: list[float],
    indices: list[int],
    output_dir: Path | None = None,
    preview_indices: set[int] | None = None,
    stage: str = "camera_fit",
) -> dict:
    ious: list[float] = []
    preview_indices = preview_indices or set()
    for index in indices:
        frame = data.frames[index]
        predicted = renderer.render(
            mesh,
            camera,
            frame.theta_deg + theta_deltas[index],
            frame.phi_deg + phi_deltas[index],
            convention,
        )
        ious.append(silhouette_iou(predicted, data.alphas[index]))
        if output_dir is not None and index in preview_indices:
            _save_preview(data, index, predicted, output_dir / "previews" / f"{stage}_{index:04d}.png")
    return {
        "iou_per_frame": ious,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "median_iou": float(np.median(ious)) if ious else 0.0,
        "worst_frame_iou": float(np.min(ious)) if ious else 0.0,
        "evaluated_frames": len(indices),
    }


def _candidate_camera_fit(
    renderer: FastSilhouetteRenderer,
    mesh: FastMesh,
    data: CameraFitData,
    indices: list[int],
    convention: CameraConvention,
    initial: CameraParameters,
) -> CameraParameters:
    camera = CameraParameters(initial.distance, initial.fov_deg, initial.x_offset, initial.y_offset)
    predicted_summaries = []
    target_summaries = []
    for index in indices:
        frame = data.frames[index]
        predicted = renderer.render(mesh, camera, frame.theta_deg, frame.phi_deg, convention)
        predicted_summaries.append(silhouette_summary(predicted))
        target_summaries.append(silhouette_summary(data.alphas[index]))
    ratios = [
        target["width"] / max(predicted["width"], 1.0)
        for predicted, target in zip(predicted_summaries, target_summaries)
        if predicted["width"] > 0 and target["width"] > 0
    ]
    if ratios:
        effective_focal_ratio = float(np.median(ratios))
        camera.fov_deg = float(np.clip(2.0 * np.rad2deg(np.arctan(np.tan(np.deg2rad(camera.fov_deg) / 2.0) / effective_focal_ratio)), 15.0, 120.0))
    dx = [target["centroid_x"] - predicted["centroid_x"] for target, predicted in zip(target_summaries, predicted_summaries)]
    dy = [target["centroid_y"] - predicted["centroid_y"] for target, predicted in zip(target_summaries, predicted_summaries)]
    if dx:
        camera.x_offset = float(np.clip(camera.x_offset + 2.0 * np.median(dx) / data.image_size, -1.0, 1.0))
        camera.y_offset = float(np.clip(camera.y_offset - 2.0 * np.median(dy) / data.image_size, -1.0, 1.0))
    return camera


def _global_fit(
    renderer: FastSilhouetteRenderer,
    mesh: FastMesh,
    data: CameraFitData,
    indices: list[int],
    convention: CameraConvention,
    initial: CameraParameters,
    max_evaluations: int,
) -> tuple[CameraParameters, int]:
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise RuntimeError("scipy is required for the lightweight numerical camera fit.") from exc

    camera = _candidate_camera_fit(renderer, mesh, data, indices, convention, initial)
    evaluations = 0

    def bbox_loss(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        candidate = CameraParameters(*[float(value) for value in values])
        losses = []
        for index in indices:
            frame = data.frames[index]
            predicted = silhouette_summary(renderer.render(mesh, candidate, frame.theta_deg, frame.phi_deg, convention))
            target = silhouette_summary(data.alphas[index])
            losses.extend(
                [
                    abs(predicted["centroid_x"] - target["centroid_x"]) / data.image_size,
                    abs(predicted["centroid_y"] - target["centroid_y"]) / data.image_size,
                    abs(predicted["width"] - target["width"]) / data.image_size,
                    abs(predicted["height"] - target["height"]) / data.image_size,
                ]
            )
        return float(np.mean(losses))

    def iou_loss(values: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        candidate = CameraParameters(*[float(value) for value in values])
        return 1.0 - float(np.mean([
            silhouette_iou(
                renderer.render(mesh, candidate, data.frames[index].theta_deg, data.frames[index].phi_deg, convention),
                data.alphas[index],
            )
            for index in indices
        ]))

    x0 = np.asarray([camera.distance, camera.fov_deg, camera.x_offset, camera.y_offset], dtype=np.float64)
    bounds = [(0.5, 10.0), (15.0, 120.0), (-1.0, 1.0), (-1.0, 1.0)]
    first_budget = max(10, max_evaluations // 2)
    result = minimize(bbox_loss, x0, method="Powell", bounds=bounds, options={"maxfev": first_budget, "xtol": 1e-3, "ftol": 1e-3})
    result2 = minimize(iou_loss, result.x, method="Powell", bounds=bounds, options={"maxfev": max(10, max_evaluations - evaluations), "xtol": 1e-3, "ftol": 1e-3})
    return CameraParameters(*[float(value) for value in result2.x]), evaluations


def _convention_search(
    renderer: FastSilhouetteRenderer,
    proxy: FastMesh,
    data: CameraFitData,
    indices: list[int],
    initial_camera: CameraParameters,
    output_dir: Path,
) -> tuple[CameraConvention, list[dict]]:
    rows: list[dict] = []
    for convention in tqdm(candidate_conventions(), desc="camera convention search"):
        camera = _candidate_camera_fit(renderer, proxy, data, indices, convention, initial_camera)
        theta = [0.0] * len(data.frames)
        phi = [0.0] * len(data.frames)
        metrics = _evaluate(renderer, proxy, data, camera, convention, theta, phi, indices)
        rows.append({**convention.as_dict(), "mean_iou": metrics["mean_iou"], "median_iou": metrics["median_iou"]})
    rows.sort(key=lambda row: (row["median_iou"], row["mean_iou"]), reverse=True)
    with (output_dir / "convention_search.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best = CameraConvention(**{key: rows[0][key] for key in ("theta_axis", "phi_axis", "theta_sign", "phi_sign", "rotation_order")})
    (output_dir / "selected_convention.json").write_text(json.dumps(best.as_dict(), indent=2))
    for rank, row in enumerate(rows[:3], start=1):
        candidate = CameraConvention(**{key: row[key] for key in ("theta_axis", "phi_axis", "theta_sign", "phi_sign", "rotation_order")})
        candidate_camera = _candidate_camera_fit(renderer, proxy, data, indices, candidate, initial_camera)
        index = indices[0]
        frame = data.frames[index]
        predicted = renderer.render(proxy, candidate_camera, frame.theta_deg, frame.phi_deg, candidate)
        _save_preview(data, index, predicted, output_dir / "previews" / f"convention_rank_{rank}_{index:04d}.png")
    return best, rows


def _pose_refine(
    renderer: FastSilhouetteRenderer,
    proxy: FastMesh,
    data: CameraFitData,
    camera: CameraParameters,
    convention: CameraConvention,
    cfg: Config,
    output_dir: Path,
) -> tuple[list[float], list[float]]:
    theta_deltas = [0.0] * len(data.frames)
    phi_deltas = [0.0] * len(data.frames)
    for epoch in tqdm(range(max(0, cfg.pose_refine_epochs)), desc="fast pose refine"):
        radius = max(cfg.max_theta_delta_deg, cfg.max_phi_delta_deg) / (2.0 ** epoch)
        for index, frame in enumerate(data.frames):
            candidates = []
            for theta_offset in np.linspace(-min(cfg.max_theta_delta_deg, radius), min(cfg.max_theta_delta_deg, radius), 3):
                for phi_offset in np.linspace(-min(cfg.max_phi_delta_deg, radius), min(cfg.max_phi_delta_deg, radius), 3):
                    theta = float(np.clip(theta_deltas[index] + theta_offset, -cfg.max_theta_delta_deg, cfg.max_theta_delta_deg))
                    phi = float(np.clip(phi_deltas[index] + phi_offset, -cfg.max_phi_delta_deg, cfg.max_phi_delta_deg))
                    predicted = renderer.render(mesh=proxy, camera=camera, theta_deg=frame.theta_deg + theta, phi_deg=frame.phi_deg + phi, convention=convention)
                    candidates.append((silhouette_iou(predicted, data.alphas[index]), theta, phi))
            _, theta_deltas[index], phi_deltas[index] = max(candidates, key=lambda value: value[0])
    return theta_deltas, phi_deltas


def _write_frame_poses(data: CameraFitData, theta_deltas: list[float], phi_deltas: list[float], path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["index", "image", "position_index", "commanded_theta_deg", "commanded_phi_deg", "optimized_theta_delta_deg", "optimized_phi_delta_deg", "final_theta_deg", "final_phi_deg"])
        writer.writeheader()
        for index, frame in enumerate(data.frames):
            writer.writerow({
                "index": index,
                "image": frame.image,
                "position_index": frame.position_index,
                "commanded_theta_deg": frame.theta_deg,
                "commanded_phi_deg": frame.phi_deg,
                "optimized_theta_delta_deg": theta_deltas[index],
                "optimized_phi_delta_deg": phi_deltas[index],
                "final_theta_deg": frame.theta_deg + theta_deltas[index],
                "final_phi_deg": frame.phi_deg + phi_deltas[index],
            })


def _calibration_convention(value: object) -> CameraConvention:
    """Decode an independently supplied convention without consulting imagery."""
    if value is None:
        return DEFAULT_CONVENTION
    if not isinstance(value, dict):
        raise CameraFitGateError("Independent calibration field 'convention' must be an object.")
    required = {"theta_axis", "phi_axis", "theta_sign", "phi_sign", "rotation_order"}
    missing = required - set(value)
    if missing:
        raise CameraFitGateError(
            "Independent calibration convention is incomplete; missing "
            + ", ".join(sorted(missing)) + "."
        )
    try:
        convention = CameraConvention(**{key: value[key] for key in required})
    except (TypeError, ValueError) as exc:
        raise CameraFitGateError("Independent calibration contains an invalid pose convention.") from exc
    if convention not in candidate_conventions():
        raise CameraFitGateError("Independent calibration contains an unsupported pose convention.")
    return convention


def _independent_camera_from_file(path: Path) -> tuple[CameraParameters, CameraConvention]:
    if not path.exists() or not path.is_file():
        raise CameraFitGateError(f"Independent calibration file is unavailable: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CameraFitGateError(f"Independent calibration file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise CameraFitGateError("Independent calibration file must contain a JSON object.")
    values = payload.get("camera", payload)
    if not isinstance(values, dict):
        raise CameraFitGateError("Independent calibration field 'camera' must be an object.")
    required = {"distance", "fov_deg"}
    missing = required - set(values)
    if missing:
        raise CameraFitGateError(
            "Independent calibration is incomplete; required camera fields: distance, fov_deg. "
            "Missing " + ", ".join(sorted(missing)) + "."
        )
    try:
        camera = CameraParameters(
            distance=float(values["distance"]),
            fov_deg=float(values["fov_deg"]),
            x_offset=float(values.get("principal_point_x_ndc", values.get("x_offset", 0.0))),
            y_offset=float(values.get("principal_point_y_ndc", values.get("y_offset", 0.0))),
        )
    except (TypeError, ValueError) as exc:
        raise CameraFitGateError("Independent calibration camera values must be numeric.") from exc
    convention = _calibration_convention(payload.get("convention"))
    return camera, convention


def _validate_independent_camera(camera: CameraParameters, contract: DatasetContract) -> None:
    values = (camera.distance, camera.fov_deg, camera.x_offset, camera.y_offset)
    if not all(math.isfinite(value) for value in values):
        raise CameraFitGateError("Independent calibration contains non-finite camera values.")
    if not 0.5 <= camera.distance <= 10.0 or not 15.0 <= camera.fov_deg <= 120.0:
        raise CameraFitGateError(
            "Independent calibration distance/FOV is outside supported bounds "
            "(distance 0.5..10, fov_deg 15..120)."
        )
    if abs(camera.x_offset) > 1.0 or abs(camera.y_offset) > 1.0:
        raise CameraFitGateError("Independent calibration principal-point offsets must be in [-1, 1].")
    poses = {(round(frame.theta_deg % 360.0, 6), round(frame.phi_deg, 6)) for frame in contract.frames}
    if len(poses) < 2:
        raise CameraFitGateError(
            "Independent image-only calibration requires at least two distinct acquisition poses in positions.csv."
        )


def _fit_image_only_camera(contract: DatasetContract, cfg: Config, output_dir: Path) -> FastCameraFitResult:
    """Build a frozen calibration from rig metadata, never mesh/image agreement.

    ``positions_rig`` treats the commanded angles in ``positions.csv`` and
    CAMERA_DISTANCE_INITIAL/CAMERA_FOV_INITIAL as known acquisition-rig
    metadata.  ``calibration_file`` accepts an independently measured JSON
    camera record.  Neither route loads the initial mesh, alpha masks, or a
    renderer, so object shape cannot leak into camera parameters.
    """
    source = getattr(cfg, "camera_calibration_source", "positions_rig")
    if source == "positions_rig":
        camera = CameraParameters(cfg.camera_distance_initial, cfg.camera_fov_initial, 0.0, 0.0)
        convention = DEFAULT_CONVENTION
    elif source == "calibration_file":
        path = getattr(cfg, "camera_calibration_path", None)
        if path is None:
            raise CameraFitGateError(
                "CAMERA_CALIBRATION_PATH is required for image_only mode when "
                "CAMERA_CALIBRATION_SOURCE=calibration_file."
            )
        camera, convention = _independent_camera_from_file(Path(path))
    else:
        raise CameraFitGateError(
            "image_only mode requires CAMERA_CALIBRATION_SOURCE=positions_rig or calibration_file."
        )
    _validate_independent_camera(camera, contract)
    theta_deltas = [0.0] * len(contract.frames)
    phi_deltas = [0.0] * len(contract.frames)
    data = CameraFitData(contract.frames, (), (), 0)
    _write_frame_poses(data, theta_deltas, phi_deltas, output_dir / "frame_poses.csv")
    (output_dir / "global_camera_parameters.json").write_text(json.dumps(camera.as_dict(), indent=2))
    metrics = {
        "camera_source": source,
        "calibration_frozen": True,
        "pose_deltas_frozen": True,
        "selected_convention": convention.as_dict(),
        "evaluated_frames": len(contract.frames),
        "shape_based_camera_fit": False,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return FastCameraFitResult(camera, convention, theta_deltas, phi_deltas, metrics, frozen=True)


def fit_camera(mesh_path: Path, contract: DatasetContract, cfg: Config, output_dir: Path) -> FastCameraFitResult:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(cfg, "reconstruction_mode", "cad_prior") == "image_only":
        # This branch intentionally occurs before loading mesh pixels or alpha
        # masks.  It is the image-only camera/geometry ownership boundary.
        result = _fit_image_only_camera(contract, cfg, output_dir)
        profile = {
            "total_frames": len(contract.frames),
            "total_camera_fit_seconds": time.perf_counter() - started,
            "silhouette_render_count": 0,
            "calibration_source": result.metrics["camera_source"],
            "calibration_frozen": True,
        }
        (output_dir / "profile.json").write_text(json.dumps(profile, indent=2))
        return result
    data, preload_seconds = _load_camera_fit_data(contract, cfg.camera_fit_max_dimension)
    selected = select_representative_indices(data, cfg.camera_fit_max_frames)
    _save_selected_frames(data, selected, output_dir / "selected_frames.csv")
    original_mesh, _ = load_fast_mesh(mesh_path, cfg.center_normalize, cfg.scale_normalize)
    proxy_started = time.perf_counter()
    proxy, proxy_method = create_camera_proxy(original_mesh, cfg.camera_fit_max_faces)
    proxy_seconds = time.perf_counter() - proxy_started
    save_fast_mesh_stl(proxy, output_dir / "proxy_mesh.stl")
    proxy_stats = {
        "original_vertex_count": int(len(original_mesh.vertices)),
        "original_face_count": int(len(original_mesh.faces)),
        "proxy_vertex_count": int(len(proxy.vertices)),
        "proxy_face_count": int(len(proxy.faces)),
        "max_faces": cfg.camera_fit_max_faces,
        "method": proxy_method,
    }
    (output_dir / "proxy_mesh_stats.json").write_text(json.dumps(proxy_stats, indent=2))
    renderer = FastSilhouetteRenderer(data.image_size)
    initial_camera = CameraParameters(cfg.camera_distance_initial, cfg.camera_fov_initial, 0.0, 0.0)
    convention_started = time.perf_counter()
    convention, convention_rows = _convention_search(renderer, proxy, data, selected, initial_camera, output_dir)
    convention_seconds = time.perf_counter() - convention_started
    if convention_rows[0]["median_iou"] < cfg.camera_convention_min_median_iou:
        raise CameraFitGateError(
            "Camera convention sanity gate failed. The OpenScan pose convention, STL orientation, "
            f"or projection model is inconsistent with observations; best median IoU={convention_rows[0]['median_iou']:.4f}."
        )
    initial_metrics = _evaluate(
        renderer,
        proxy,
        data,
        initial_camera,
        convention,
        [0.0] * len(data.frames),
        [0.0] * len(data.frames),
        selected,
        output_dir,
        set(selected),
        "initial",
    )
    global_started = time.perf_counter()
    camera, global_evaluations = _global_fit(renderer, proxy, data, selected, convention, initial_camera, cfg.camera_fit_max_evaluations)
    global_seconds = time.perf_counter() - global_started
    (output_dir / "global_camera_parameters.json").write_text(json.dumps(camera.as_dict(), indent=2))
    pose_started = time.perf_counter()
    theta_deltas, phi_deltas = _pose_refine(renderer, proxy, data, camera, convention, cfg, output_dir) if cfg.pose_refine else ([0.0] * len(data.frames), [0.0] * len(data.frames))
    pose_seconds = time.perf_counter() - pose_started
    _write_frame_poses(data, theta_deltas, phi_deltas, output_dir / "frame_poses.csv")
    preview_indices = set(selected[: min(8, len(selected))])
    gate_started = time.perf_counter()
    metrics = _evaluate(renderer, proxy, data, camera, convention, theta_deltas, phi_deltas, list(range(len(data.frames))), output_dir, preview_indices, "camera_fitted")
    gate_seconds = time.perf_counter() - gate_started
    metrics.update({
        "initial": initial_metrics,
        "camera_gate_uses_all_frames": True,
        "selected_convention": convention.as_dict(),
        "proxy_mesh_stats": proxy_stats,
    })
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    profile = {
        "total_frames": len(data.frames),
        "selected_coarse_frames": len(selected),
        "camera_fit_resolution": cfg.camera_fit_max_dimension,
        "proxy_creation_seconds": proxy_seconds,
        "convention_search_seconds": convention_seconds,
        "global_fit_seconds": global_seconds,
        "pose_refinement_seconds": pose_seconds,
        "full_gate_evaluation_seconds": gate_seconds,
        "alpha_preload_seconds": preload_seconds,
        "total_camera_fit_seconds": time.perf_counter() - started,
        "silhouette_render_count": renderer.render_count,
        "average_silhouette_render_ms": 1000.0 * renderer.total_render_seconds / max(renderer.render_count, 1),
        "global_fit_evaluations": global_evaluations,
        "original_face_count": len(original_mesh.faces),
        "proxy_face_count": len(proxy.faces),
    }
    (output_dir / "profile.json").write_text(json.dumps(profile, indent=2))
    if metrics["median_iou"] < cfg.camera_gate_min_median_iou:
        raise CameraFitGateError(
            "Camera fit gate failed. Do not deform the STL because camera alignment is insufficient. "
            f"Median IoU={metrics['median_iou']:.4f}, required={cfg.camera_gate_min_median_iou:.4f}."
        )
    return FastCameraFitResult(camera, convention, theta_deltas, phi_deltas, metrics)
