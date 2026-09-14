from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import src.ground_truth_eval as ground_truth_eval
from src.ground_truth_eval import evaluate_reconstruction_against_ground_truth


def _tetrahedron(offset: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return vertices + np.asarray(offset), np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])


def _write_obj(path: Path, mesh) -> None:
    vertices, faces = mesh
    lines = [*(f"v {x} {y} {z}" for x, y, z in vertices), *(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces)]
    path.write_text("\n".join(lines))


def test_evaluation_reports_all_stages_physical_and_normalized_metrics(tmp_path: Path) -> None:
    target = _tetrahedron()
    path = tmp_path / "ground_truth.obj"
    _write_obj(path, target)
    result = evaluate_reconstruction_against_ground_truth(path, target, target, _tetrahedron((2, 0, 0)), sample_count=3000, seed=7)
    assert result["primary_metric_frame"] == "raw_physical_unaligned"
    assert set(result["stages"]) == {"sphere", "coarse", "final"}
    assert result["stages"]["sphere"]["physical"]["mean_surface_distance"] < 0.03
    assert result["stages"]["final"]["physical"]["mean_surface_distance"] > 0.5
    physical = result["stages"]["coarse"]["physical"]
    normalized = result["stages"]["coarse"]["normalized_by_gt_bbox_diagonal"]
    assert normalized["mean_surface_distance"] == pytest.approx(physical["mean_surface_distance"] / np.sqrt(3))
    assert normalized["symmetric_chamfer_distance"] == pytest.approx(physical["symmetric_chamfer_distance"] / 3.0)
    assert {"p95_surface_distance", "hausdorff_distance", "robust_hausdorff_distance", "normal_consistency"} <= set(physical)
    json.dumps(result)


def test_rigid_alignment_is_optional_and_explicitly_diagnostic(tmp_path: Path) -> None:
    target = _tetrahedron()
    path = tmp_path / "ground_truth.obj"
    _write_obj(path, target)
    translated = _tetrahedron((3, -2, 1))
    result = evaluate_reconstruction_against_ground_truth(
        path, translated, translated, translated, sample_count=2000, seed=8, include_rigid_aligned_diagnostic=True
    )
    stage = result["stages"]["final"]
    assert stage["physical"]["mean_surface_distance"] > 1.0
    diagnostic = stage["rigid_aligned_diagnostic"]
    assert diagnostic["label"] == "diagnostic_only_rigid_aligned_not_primary"
    assert diagnostic["metrics"]["physical"]["mean_surface_distance"] < 0.06


def test_evaluator_has_no_reconstruction_module_dependency() -> None:
    source = Path(ground_truth_eval.__file__).read_text()
    for forbidden in ("camera_fitting", "coarse_geometry", "optimizer", "pipeline", "losses", "renderer"):
        assert forbidden not in source
