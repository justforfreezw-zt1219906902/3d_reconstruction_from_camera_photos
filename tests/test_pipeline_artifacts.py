from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

import src.pipeline as pipeline
from src.config import load_config
from src.visualization import save_mesh_stage_previews


def test_reconstruction_metrics_records_stage_configuration_and_selected_state(tmp_path: Path) -> None:
    cfg = replace(load_config(tmp_path / "missing.env"), reconstruction_mode="image_only")
    reconstruction = tmp_path / "reconstruction"
    reconstruction.mkdir()
    (reconstruction / "view_split.json").write_text(json.dumps({"training_indices": [0, 2], "validation_indices": [1]}))
    (reconstruction / "losses.csv").write_text(
        "epoch,validation_silhouette,validation_iou,validity_is_valid,is_best\n"
        "1,0.2,0.6,1,1\n2,0.1,0.7,1,0\n"
    )
    camera = SimpleNamespace(metrics={"camera_source": "positions_rig", "calibration_frozen": True})

    result = pipeline._write_reconstruction_metrics(
        reconstruction / "reconstruction_metrics.json", cfg, camera, reconstruction,
        coarse_stats={"vertex_count": 17, "face_count": 24},
    )

    assert result["reconstruction_mode"] == "image_only"
    assert result["camera_source"] == "positions_rig"
    assert result["coarse_geometry"]["result"]["face_count"] == 24
    assert result["refinement"]["enabled"] is True
    assert result["train_validation_split"]["validation_indices"] == [1]
    assert result["selected_best_state"]["epoch"] == "1"


def test_stage_preview_helper_names_each_rendered_view(monkeypatch, tmp_path: Path) -> None:
    import src.visualization as visualization

    saved: list[Path] = []
    monkeypatch.setattr(visualization, "save_stage_preview", lambda *args: saved.append(args[-1]))

    class Mesh:
        def verts_packed(self): return torch.zeros((3, 3))
    class Dataset:
        def __getitem__(self, index):
            return SimpleNamespace(image=torch.ones((2, 2, 3)), alpha=torch.ones((2, 2, 1)))
    renderer = SimpleNamespace(render_mask=lambda *args: torch.ones((1, 2, 2, 1)))
    camera = SimpleNamespace(cameras=lambda indices: indices)

    paths = save_mesh_stage_previews(Mesh(), Dataset(), camera, renderer, [2, 0, 2], tmp_path, "coarse")
    assert [path.name for path in paths] == ["coarse_0000.png", "coarse_0002.png"]
    assert saved == paths


def test_image_only_pipeline_writes_stage_artifacts_before_optional_ground_truth(monkeypatch, tmp_path: Path) -> None:
    """Exercise artifact sequencing without a renderer or a real mesh fixture."""
    import src.camera_fitting as camera_fitting
    import src.coarse_geometry as coarse_geometry
    import src.config as config
    import src.ground_truth_eval as ground_truth_eval
    import src.mesh_io as mesh_io
    import src.optimizer as optimizer
    import src.renderer as renderer_module
    import src.rgba_dataset as rgba_dataset
    import src.visualization as visualization

    gt_path = tmp_path / "ground_truth.obj"
    gt_path.write_text("placeholder")
    cfg = replace(
        load_config(tmp_path / "missing.env"), reconstruction_mode="image_only",
        output_dir=tmp_path / "output", initial_mesh_path=tmp_path / "sphere.stl",
        ground_truth_mesh_path=gt_path, device="cpu", max_image_dimension=8,
    )
    contract = SimpleNamespace(frames=[SimpleNamespace(image="a.png")], validation={})
    order: list[str] = []

    class Mesh:
        def detach(self):
            return self

    initial, coarse, final = Mesh(), Mesh(), Mesh()
    transform = SimpleNamespace()
    camera_result = SimpleNamespace(
        camera=SimpleNamespace(), convention=object(), theta_deltas=[0.0], phi_deltas=[0.0],
        metrics={"camera_source": "positions_rig", "calibration_frozen": True},
    )
    monkeypatch.setattr(config, "ensure_pytorch3d", lambda: None)
    monkeypatch.setattr(pipeline, "save_validation", lambda *args: None)
    monkeypatch.setattr(camera_fitting, "fit_camera", lambda *args: camera_result)
    monkeypatch.setattr(pipeline, "_build_geometry_model", lambda *args: SimpleNamespace(cameras=lambda indices: indices))
    monkeypatch.setattr(mesh_io, "load_initial_mesh", lambda *args, **kwargs: (initial, transform))
    monkeypatch.setattr(mesh_io, "save_transform", lambda _transform, path: path.write_text("{}"))
    monkeypatch.setattr(mesh_io, "export_mesh_stl", lambda _mesh, path, _transform: path.write_text("stl"))
    monkeypatch.setattr(coarse_geometry, "generate_coarse_geometry", lambda *args, **kwargs: SimpleNamespace(mesh=coarse, stats={"vertex_count": 9, "face_count": 12}))
    monkeypatch.setattr(renderer_module, "ReconstructionRenderer", lambda *args: SimpleNamespace(probe=lambda *args: None))
    # Special methods are looked up on the class, so use a small real dataset class.
    class Dataset:
        canvas_size = 8
        def __len__(self): return 1
        def representative_indices(self): return [0]
    monkeypatch.setattr(rgba_dataset, "RGBADataset", lambda *args: Dataset())
    monkeypatch.setattr(visualization, "save_mesh_stage_previews", lambda _mesh, *_args: order.append(_args[-1]))

    def train(*args, **kwargs):
        reconstruction = cfg.output_dir / "reconstruction"
        (reconstruction / "view_split.json").write_text(json.dumps({"training_indices": [0], "validation_indices": []}))
        (reconstruction / "losses.csv").write_text("epoch,is_best,validation_silhouette,validation_iou\n1,1,0.1,0.8\n")
        final_obj, final_stl = reconstruction / "final.obj", reconstruction / "final.stl"
        final_obj.write_text("obj"); final_stl.write_text("stl")
        return SimpleNamespace(mesh=final, final_obj=final_obj, final_stl=final_stl)
    monkeypatch.setattr(optimizer, "train_geometry", train)

    def evaluate(*args, **kwargs):
        assert (cfg.output_dir / "reconstruction" / "final.obj").exists()
        assert (cfg.output_dir / "reconstruction" / "reconstruction_metrics.json").exists()
        order.append("ground_truth")
        return {"stages": {}}
    monkeypatch.setattr(ground_truth_eval, "evaluate_reconstruction_against_ground_truth", evaluate)

    summary = pipeline.run_demo(cfg, contract)
    reconstruction = cfg.output_dir / "reconstruction"
    for name in ("initial_sphere.stl", "image_coarse_mesh.stl", "view_split.json", "losses.csv", "reconstruction_metrics.json", "ground_truth_evaluation.json"):
        assert (reconstruction / name).is_file()
    assert order == ["sphere", "coarse", "final", "ground_truth"]
    assert summary["status"] == "completed"
