from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.config import load_config


def _benchmark_module():
    path = Path(__file__).parents[1] / "scripts" / "run_benchy_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchy_benchmark", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_runs_pipeline_once_and_writes_required_summary(tmp_path: Path) -> None:
    runner = _benchmark_module()
    cfg = replace(
        load_config(tmp_path / "missing.env"), reconstruction_mode="image_only",
        initial_mesh_path=tmp_path / "sphere.stl", ground_truth_mesh_path=tmp_path / "truth.obj",
        output_dir=tmp_path / "output",
    )
    calls = []

    def pipeline_runner(actual_cfg):
        calls.append(actual_cfg)
        reconstruction = actual_cfg.output_dir / "reconstruction"
        reconstruction.mkdir(parents=True)
        (reconstruction / "ground_truth_evaluation.json").write_text(json.dumps({
            "stages": {
                stage: {"physical": {"mean_surface_distance": 0.1, "symmetric_chamfer_distance": 0.01, "p95_surface_distance": 0.2}}
                for stage in ("sphere", "coarse", "final")
            }
        }))
        return {"status": "completed", "final_obj": "final.obj"}

    report = runner.run_benchmark(
        cfg,
        pipeline_runner=pipeline_runner,
        iou_collector=lambda _cfg: {"sphere": 0.2, "coarse": 0.5, "final": 0.7},
        validity_collector=lambda _cfg: {
            "is_valid": True, "degenerate_face_count": 0, "extreme_edge_count": 0,
            "disconnected_fragment_count": 0, "non_finite_vertex_count": 0, "spike_vertex_count": 0,
        },
    )

    assert calls == [cfg]
    assert report["held_out_iou"] == {"sphere": 0.2, "coarse": 0.5, "final": 0.7}
    assert set(report["ground_truth_evaluation"]["stages"]) == {"sphere", "coarse", "final"}
    payload = json.loads(Path(report["report_json"]).read_text())
    assert payload["final_mesh_validity"]["is_valid"] is True
    markdown = Path(report["report_markdown"]).read_text()
    assert "Held-out silhouette IoU" in markdown
    assert "Ground-truth error" in markdown
    assert "Final mesh validity" in markdown


def test_benchmark_requires_image_only_mode_and_ground_truth(tmp_path: Path) -> None:
    runner = _benchmark_module()
    cfg = replace(load_config(tmp_path / "missing.env"), output_dir=tmp_path / "output")
    with pytest.raises(ValueError, match="image_only"):
        runner.run_benchmark(cfg)
    image_only_without_gt = replace(cfg, reconstruction_mode="image_only")
    with pytest.raises(ValueError, match="GROUND_TRUTH_MESH_PATH"):
        runner.run_benchmark(image_only_without_gt)


def test_benchmark_config_forces_image_only_cad_controls(monkeypatch, tmp_path: Path) -> None:
    runner = _benchmark_module()
    original = replace(
        load_config(tmp_path / "missing.env"), reconstruction_mode="cad_prior",
        ground_truth_mesh_path=tmp_path / "truth.obj", loss_anchor_weight=1.0,
        max_local_deformation_ratio=0.03,
    )
    monkeypatch.setattr(runner, "load_config", lambda env_file: original)

    cfg = runner.benchmark_config(tmp_path / "benchy.env")

    assert cfg.reconstruction_mode == "image_only"
    assert cfg.ground_truth_mesh_path == tmp_path / "truth.obj"
    assert cfg.loss_anchor_weight == 0.0
    assert cfg.max_local_deformation_ratio == 0.0
