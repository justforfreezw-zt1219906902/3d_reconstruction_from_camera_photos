"""CLI smoke tests with real input validation and isolated expensive stages."""
import json
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

import main as cli
import src.camera_fitting as camera_fitting
import src.optimizer as optimizer
import src.pipeline as pipeline
import src.renderer as renderer
from src.config import load_config


@pytest.fixture
def cli_config(monkeypatch, tmp_path):
    monkeypatch.setattr('os.environ', {})
    rgba = tmp_path / 'rgba'
    rgba.mkdir()
    image = Image.new('RGBA', (8, 8), (0, 0, 0, 0))
    image.paste((120, 120, 120, 255), (2, 2, 6, 6))
    image.save(rgba / 'frame.png')
    positions = tmp_path / 'positions.csv'
    positions.write_text('image,position_index,phi_deg,theta_deg\nframe.png,0,0,0\n')
    mesh = tmp_path / 'reference.obj'
    mesh.write_text('v 0 0 0\nv 2 0 0\nv 0 2 0\nv 0 0 2\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n')
    cfg = replace(load_config(tmp_path / 'missing.env'), rgba_dir=rgba,
                  positions_csv=positions, initial_mesh_path=mesh,
                  output_dir=tmp_path / 'output', min_usable_frames=1,
                  device='cpu', max_image_dimension=8)
    monkeypatch.setattr(cli, 'load_config', lambda: cfg)
    return cfg


def test_validate_only_stops_before_camera_and_geometry(monkeypatch, cli_config, capsys):
    demo = Mock(side_effect=AssertionError('validation must stop before reconstruction'))
    monkeypatch.setattr(pipeline, 'run_demo', demo)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--validate-only'])
    cli.main()
    demo.assert_not_called()
    assert '"status": "validated"' in capsys.readouterr().out
    assert (cli_config.output_dir / 'validation.json').is_file()
    assert not (cli_config.output_dir / 'reconstruction').exists()


def test_camera_fit_only_stops_before_geometry(monkeypatch, cli_config, capsys):
    fit = Mock(return_value=SimpleNamespace(metrics={'passed': True}))
    geometry_device = Mock(side_effect=AssertionError('camera-only must not initialize geometry'))
    monkeypatch.setattr(camera_fitting, 'fit_camera', fit)
    monkeypatch.setattr(pipeline, 'select_device', geometry_device)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--camera-fit-only'])
    cli.main()
    fit.assert_called_once()
    assert fit.call_args.args[0] == cli_config.initial_mesh_path
    geometry_device.assert_not_called()
    summary = json.loads((cli_config.output_dir / 'summary.json').read_text())
    assert summary == {'status': 'camera_fit_only', 'camera_fit': {'passed': True}}
    assert '"status": "camera_fit_only"' in capsys.readouterr().out
    assert not (cli_config.output_dir / 'reconstruction').exists()


def test_skip_camera_fit_reaches_geometry_with_commanded_poses(monkeypatch, cli_config, capsys):
    fit = Mock(side_effect=AssertionError('camera fitting must be skipped'))
    monkeypatch.setattr(camera_fitting, 'fit_camera', fit)
    # Exercise CLI, validation, mesh/proxy preparation and orchestration. Rasterization
    # and optimization have their own tests; replace only those expensive stages.
    render = SimpleNamespace(probe=Mock())
    monkeypatch.setattr(renderer, 'ReconstructionRenderer', lambda *args: render)
    real_build = pipeline._build_geometry_model
    build = Mock(wraps=real_build)
    monkeypatch.setattr(pipeline, '_build_geometry_model', build)
    align = Mock(side_effect=lambda mesh, *args: mesh)
    monkeypatch.setattr(pipeline, '_fit_global_alignment', align)
    train = Mock(return_value=SimpleNamespace(final_obj=cli_config.output_dir / 'final.obj',
                                             final_stl=cli_config.output_dir / 'final.stl'))
    monkeypatch.setattr(optimizer, 'train_geometry', train)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--skip-camera-fit'])
    cli.main()
    fit.assert_not_called()
    build.assert_called_once()
    assert build.call_args.args[3] is None
    render.probe.assert_called_once()
    align.assert_called_once()
    train.assert_called_once()
    assert train.call_args.args[0] is align.call_args.args[0]
    summary = json.loads((cli_config.output_dir / 'summary.json').read_text())
    assert summary['status'] == 'completed'
    assert summary['camera_fit'] == {'enabled': False, 'skipped': True}
    assert '"status": "completed"' in capsys.readouterr().out


def test_camera_flags_remain_mutually_exclusive(monkeypatch, capsys):
    run = Mock()
    monkeypatch.setattr(cli, 'run_from_config', run)
    monkeypatch.setattr(sys, 'argv', ['main.py', '--camera-fit-only', '--skip-camera-fit'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert 'mutually exclusive' in capsys.readouterr().err
    run.assert_not_called()


def test_help_preserves_stage_flags(monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['main.py', '--help'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ('--validate-only', '--camera-fit-only', '--skip-camera-fit'):
        assert flag in help_text
