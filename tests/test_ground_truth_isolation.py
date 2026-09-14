"""Tests that prevent evaluation data from entering reconstruction stages."""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace
from pathlib import Path

from src.config import load_config
from src.pipeline import _reconstruction_config


RECONSTRUCTION_MODULES = ("src.camera_fitting", "src.coarse_geometry", "src.optimizer")


def _calls_in_run_demo() -> dict[str, ast.Call]:
    import src.pipeline as pipeline

    tree = ast.parse(Path(pipeline.__file__).read_text())
    run_demo = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_demo")
    calls: dict[str, ast.Call] = {}
    for node in ast.walk(run_demo):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            calls.setdefault(node.func.id, node)
    return calls


def _contains_ground_truth(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute) and child.attr == "ground_truth_mesh_path"
        or isinstance(child, ast.Name) and "ground_truth" in child.id
        for child in ast.walk(node)
    )


def test_reconstruction_receives_a_config_with_no_ground_truth_path(tmp_path: Path) -> None:
    original = replace(load_config(tmp_path / "absent.env"), ground_truth_mesh_path=tmp_path / "secret.obj")
    reconstruction_cfg = _reconstruction_config(original)

    assert original.ground_truth_mesh_path == tmp_path / "secret.obj"
    assert reconstruction_cfg.ground_truth_mesh_path is None
    assert reconstruction_cfg is not original


def test_ground_truth_cannot_be_passed_to_camera_carving_or_optimizer() -> None:
    """Guard the pipeline call boundary, including future positional arguments."""
    calls = _calls_in_run_demo()
    for name in ("fit_camera", "generate_coarse_geometry", "train_geometry"):
        call = calls[name]
        assert not _contains_ground_truth(call), f"{name} must not receive a ground-truth value"

    evaluation = calls["evaluate_reconstruction_against_ground_truth"]
    assert _contains_ground_truth(evaluation), "only post-export evaluation may receive ground truth"


def test_ground_truth_loader_is_owned_only_by_evaluation_module() -> None:
    """Reconstruction code has neither a GT path argument nor a GT loader import."""
    for module_name in RECONSTRUCTION_MODULES:
        module = __import__(module_name, fromlist=["*"])
        source = Path(inspect.getsourcefile(module)).read_text()
        tree = ast.parse(source)
        assert "ground_truth_eval" not in source
        assert all("ground_truth" not in parameter.arg for node in ast.walk(tree)
                   if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for parameter in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs))

    import src.ground_truth_eval as evaluator

    source = Path(evaluator.__file__).read_text()
    assert source.count("_load_ground_truth_mesh(") == 2  # definition + public evaluation call
