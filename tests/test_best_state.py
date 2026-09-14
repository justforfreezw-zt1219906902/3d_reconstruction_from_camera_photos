"""Regression tests for held-out gradients and selected-state mesh exports."""
import csv
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import src.optimizer as optimizer
from src.config import load_config
from src.mesh_io import _normalize_mesh, load_initial_mesh, mesh_from_arrays, restore_vertices


def _reference_mesh():
    vertices = torch.tensor([[-1., -1., -1.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    faces = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]])
    return _normalize_mesh(mesh_from_arrays(vertices * 7 + 12, faces, torch.device('cpu')), True, True)


@pytest.mark.parametrize('existing_gradients', [False, True])
def test_validation_preserves_all_parameter_gradients(existing_gradients):
    mesh, _ = _reference_mesh()
    offsets = torch.nn.Parameter(torch.zeros_like(mesh.verts_padded()))
    camera_parameter = torch.nn.Parameter(torch.tensor(.1))
    renderer_parameter = torch.nn.Parameter(torch.tensor(.2))
    parameters = [offsets, camera_parameter, renderer_parameter]
    for parameter in parameters:
        if existing_gradients:
            parameter.grad = torch.full_like(parameter, 3.)
    before = [None if p.grad is None else p.grad.clone() for p in parameters]
    outputs = []

    def render(current, cameras):
        output = (current.verts_padded() + cameras + renderer_parameter).sigmoid()
        outputs.append(output)
        return output

    dataset = [SimpleNamespace(alpha=torch.ones(4, 3)) for _ in range(3)]
    camera = SimpleNamespace(cameras=lambda batch: camera_parameter.expand(len(batch), 1, 1))
    renderer = SimpleNamespace(render_mask=render)
    # Positive control: this synthetic forward connects all three parameter groups.
    render(optimizer._mesh_with_offsets(mesh, offsets), camera.cameras([0])).sum().backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) for p in parameters)
    for parameter, gradient in zip(parameters, before):
        parameter.grad = None if gradient is None else gradient.clone()
    outputs.clear()
    optimizer.evaluate_geometry_validation(mesh, offsets, dataset, camera, renderer,
                                           [0, 1, 2], 2, torch.device('cpu'), 1.)
    assert len(outputs) == 2
    assert all(not output.requires_grad and output.grad_fn is None for output in outputs)
    for parameter, gradient in zip(parameters, before):
        if gradient is None:
            assert parameter.grad is None
        else:
            assert torch.equal(parameter.grad, gradient)


@pytest.mark.parametrize('qualities,amplitudes,best_epoch', [
    ([.1, .2, .3], [.01, .02, .03], 1),
    ([.3, .1, .2], [.01, .02, .03], 2),
    ([.1, .1, .1], [.02, .01, .03], 2),
    ([.1, .1000001, .1], [.02, .01, .03], 2),
])
def test_selected_offsets_drive_real_exports(monkeypatch, tmp_path, qualities, amplitudes, best_epoch):
    mesh, transform = _reference_mesh()
    basis = optimizer._local_offset_basis(mesh.verts_padded())
    direction = optimizer._project_local_offsets(
        torch.arange(12, dtype=torch.float32).reshape(1, 4, 3).square(), basis)
    direction /= direction.norm(dim=-1).max()
    states = [direction * amplitude for amplitude in amplitudes]
    steps = []

    def controlled_step(self, closure=None):
        # Control only the update; training still renders, computes losses and backpropagates.
        parameter = self.param_groups[0]['params'][0]
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        with torch.no_grad():
            parameter.copy_(states[len(steps)])
        steps.append(parameter.detach().clone())

    monkeypatch.setattr(torch.optim.Adam, 'step', controlled_step)
    evaluated = []
    real_validation = optimizer.evaluate_geometry_validation

    def validation(*args):
        metrics = real_validation(*args)
        evaluated.append(args[1].detach().clone())
        metrics.update(validation_silhouette=qualities[len(evaluated) - 1], validation_iou=.5)
        return metrics

    monkeypatch.setattr(optimizer, 'evaluate_geometry_validation', validation)
    monkeypatch.setattr(optimizer, 'plot_losses', lambda *args: None)
    monkeypatch.setattr('os.environ', {})
    cfg = replace(load_config(tmp_path / 'missing.env'), num_epochs=3,
                  geometry_views_per_epoch=0, geometry_view_batch_size=2,
                  loss_rgb_weight=0., early_stopping_patience=0, export_every_epochs=1)

    class Dataset:
        frames = [SimpleNamespace() for _ in range(2)]
        canvas_size = (4, 3)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return SimpleNamespace(alpha=torch.ones(4, 3), image=torch.ones(4, 3))

        def representative_indices(self):
            return []

    result = optimizer.train_geometry(mesh, transform, Dataset(),
        SimpleNamespace(freeze=lambda: None, cameras=lambda batch: batch),
        SimpleNamespace(render_mask=lambda current, cameras: current.verts_padded().sigmoid()),
        cfg, torch.device('cpu'), tmp_path)
    assert len(steps) == len(evaluated) == 3
    selected = evaluated[best_epoch - 1]
    assert not torch.allclose(selected, evaluated[-1])
    assert torch.equal(result.mesh.verts_padded(), mesh.verts_padded() + selected)
    faces = mesh.faces_packed()
    expected = restore_vertices(mesh.verts_packed() + selected[0], transform)[faces]
    last = restore_vertices(mesh.verts_packed() + evaluated[-1][0], transform)[faces]
    for path in (result.final_obj, result.final_stl):
        loaded, _ = load_initial_mesh(path, torch.device('cpu'), False, False)
        triangles = loaded.verts_packed()[loaded.faces_packed()]
        torch.testing.assert_close(triangles, expected, atol=1e-5, rtol=0)
        assert not torch.allclose(triangles, last, atol=1e-5, rtol=0)
    checkpoint = torch.load(tmp_path / 'reconstruction/checkpoints/checkpoint_epoch_0003.pt', weights_only=True)
    assert torch.equal(checkpoint['offsets'], evaluated[-1])

    with (tmp_path / 'reconstruction/losses.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        required = {
            'epoch', 'total', 'silhouette', 'rgb', 'normal', 'iou',
            'training_objective', 'training_silhouette', 'training_iou',
            'validation_silhouette', 'validation_iou', 'anchor', 'local_smoothness',
            'mean_vertex_displacement', 'median_vertex_displacement',
            'p95_vertex_displacement', 'max_vertex_displacement',
            'max_vertex_displacement_ratio', 'best_epoch', 'is_best',
        }
        assert required <= set(reader.fieldnames)
        rows = list(reader)
    assert [int(row['epoch']) for row in rows] == [1, 2, 3]
    current_best = 0
    for epoch, row in enumerate(rows, 1):
        assert None not in row  # Every record matches the header.
        assert all(math.isfinite(float(row[key])) for key in required)
        for alias, original in [('training_objective', 'total'),
                                ('training_silhouette', 'silhouette'), ('training_iou', 'iou')]:
            assert row[alias] == row[original]
        assert float(row['validation_silhouette']) == pytest.approx(qualities[epoch - 1])
        assert row['is_best'] in {'0', '1'}
        if int(row['is_best']):
            current_best = epoch
        assert int(row['best_epoch']) == current_best
        stats = optimizer._displacement_stats(evaluated[epoch - 1], transform.normalized_object_size)
        for key, value in stats.items():
            assert float(row[key]) == pytest.approx(value)
    assert current_best == best_epoch
