from __future__ import annotations

import pytest
import torch
from pytorch3d.loss import (
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.structures import Meshes

from src.losses import (
    coarse_edge_regularization_loss,
    image_only_regularization_losses,
    local_deformation_smoothness_loss,
    reference_anchor_loss,
    regularization_losses,
)


def reference() -> torch.Tensor:
    return torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]], dtype=torch.float64)


def test_anchor_zero_monotonic_and_gradients() -> None:
    ref = reference().requires_grad_()
    direction = torch.ones_like(ref)
    values = [reference_anchor_loss(ref.detach() + amount * direction, ref) for amount in (0., .1, .2, .4)]
    assert values[0] == 0
    assert all(a < b for a, b in zip(values, values[1:]))
    current = (ref.detach() + .2 * direction).requires_grad_()
    loss = reference_anchor_loss(current, ref)
    assert loss.item() == pytest.approx(.12 / 5.)
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert (current.grad > 0).all()
    assert ref.grad is None


@pytest.mark.parametrize('scale', [1e-9, .001, 1., 1000., 1e9])
def test_losses_are_unit_independent(scale: float) -> None:
    ref = reference()
    offsets = torch.tensor([[.1, 0., 0.], [0., .2, 0.], [0., 0., .3]], dtype=ref.dtype)
    edges = torch.tensor([[0, 1], [1, 2], [0, 2]])
    assert torch.allclose(reference_anchor_loss((ref + offsets) * scale, ref * scale),
                          reference_anchor_loss(ref + offsets, ref))
    assert torch.allclose(local_deformation_smoothness_loss(offsets * scale, edges, ref * scale),
                          local_deformation_smoothness_loss(offsets, edges, ref))


def test_smoothness_uniform_is_zero_and_isolated_motion_has_gradient() -> None:
    ref = reference().requires_grad_()
    edges = torch.tensor([[0, 1], [1, 2], [0, 2]])
    uniform = torch.ones_like(ref)
    uniform_loss = local_deformation_smoothness_loss(uniform, edges, ref)
    assert uniform_loss == 0
    isolated = torch.zeros_like(ref)
    isolated[0, 0] = 1.
    isolated.requires_grad_()
    loss = local_deformation_smoothness_loss(isolated, edges, ref)
    assert loss.item() == pytest.approx(2. / 15.)
    assert loss > uniform_loss
    loss.backward()
    assert isolated.grad[0, 0] > 0
    assert (isolated.grad[1:, 0] < 0).all()
    assert ref.grad is None
    alternating = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [1., 0., 0.]], dtype=ref.dtype)
    assert local_deformation_smoothness_loss(alternating, edges, ref) > loss


def test_empty_edges_produce_differentiable_zero() -> None:
    offsets = torch.zeros_like(reference(), requires_grad=True)
    loss = local_deformation_smoothness_loss(offsets, torch.empty((0, 2), dtype=torch.long), reference())
    loss.backward()
    assert loss == 0
    assert torch.equal(offsets.grad, torch.zeros_like(offsets))


def test_regularization_names_batch_reduction_and_reference_identity() -> None:
    ref = reference().float()
    faces = torch.tensor([[0, 1, 2]])
    base = Meshes(verts=[ref, ref * 100], faces=[faces, faces])
    identity = regularization_losses(base, base)
    assert set(identity) == {'anchor', 'local_smoothness', 'normal'}
    assert identity['anchor'] == identity['local_smoothness'] == 0
    assert set(regularization_losses(base)) == {'normal'}
    deformed = Meshes(verts=[ref + .1, ref * 100 + 10], faces=[faces, faces])
    terms = regularization_losses(deformed, base)
    assert terms['anchor'].item() == pytest.approx(.03 / 5.)
    assert terms['local_smoothness'].item() == pytest.approx(0., abs=1e-12)


def test_invalid_reference_and_topology_are_rejected() -> None:
    with pytest.raises(ValueError, match='positive'):
        reference_anchor_loss(torch.ones(3, 3), torch.zeros(3, 3))
    with pytest.raises(ValueError, match='shapes'):
        reference_anchor_loss(torch.ones(4, 3), reference())
    faces = torch.tensor([[0, 1, 2]])
    base = Meshes(verts=[reference().float()], faces=[faces])
    other = Meshes(verts=[reference().float()], faces=[faces.flip(1)])
    with pytest.raises(ValueError, match='topology'):
        regularization_losses(other, base)


@pytest.mark.parametrize('isolated', [False, True], ids=['uniform', 'single_vertex'])
def test_public_regularizers_track_displacement_magnitude(isolated: bool) -> None:
    vertices = reference().float()
    faces = torch.tensor([[0, 1, 2]])
    base = Meshes(verts=[vertices], faces=[faces])
    direction = torch.tensor([[1., -2., 3.]]).expand_as(vertices).clone()
    if isolated:
        direction[1:] = 0

    terms = [regularization_losses(
        Meshes(verts=[vertices + amount * direction], faces=[faces]), base,
    ) for amount in (0., .125, .25, .5)]

    assert terms[0]['anchor'].item() == 0.
    assert terms[0]['local_smoothness'].item() == 0.
    anchors = [term['anchor'].item() for term in terms]
    assert all(smaller < larger for smaller, larger in zip(anchors, anchors[1:]))
    assert anchors[2] == pytest.approx(4 * anchors[1])
    assert anchors[3] == pytest.approx(4 * anchors[2])
    for term in terms[1:]:
        if isolated:
            assert term['local_smoothness'].item() > 1e-6
        else:
            assert term['local_smoothness'].item() == pytest.approx(0., abs=1e-12)


def test_audited_regularizers_reject_shrinkage_and_retain_normal_consistency() -> None:
    # A closed, creased tetrahedron makes every generic penalty nonzero.
    vertices = torch.tensor([
        [0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 3.],
    ])
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    base = Meshes(verts=[vertices], faces=[faces])
    center = vertices.mean(dim=0)
    shrunken = Meshes(verts=[center + .5 * (vertices - center)], faces=[faces])

    # Audit the actual PyTorch3D defaults: both reward shrinking the CAD.
    edge = mesh_edge_loss(base)
    laplacian = mesh_laplacian_smoothing(base)
    assert edge.item() > 0
    assert laplacian.item() > 0
    assert mesh_edge_loss(shrunken).item() == pytest.approx(.25 * edge.item())
    assert mesh_laplacian_smoothing(shrunken).item() == pytest.approx(.5 * laplacian.item())

    identity = regularization_losses(base, base)
    deformed = regularization_losses(shrunken, base)
    for mesh, terms in ((base, identity), (shrunken, deformed)):
        # Exact keys prevent either absolute-geometry penalty being reintroduced.
        assert set(terms) == {'anchor', 'local_smoothness', 'normal'}
        assert terms['normal'].item() > 0
        assert terms['normal'].item() == pytest.approx(mesh_normal_consistency(mesh).item())
        generic = regularization_losses(mesh)
        assert set(generic) == {'normal'}
        assert generic['normal'].item() == pytest.approx(terms['normal'].item())

    assert identity['anchor'].item() == identity['local_smoothness'].item() == 0.
    assert deformed['anchor'].item() > identity['anchor'].item()
    assert deformed['local_smoothness'].item() > identity['local_smoothness'].item()


def test_image_only_regularizers_reference_only_the_coarse_mesh_and_have_gradients() -> None:
    # This topology deliberately does not resemble or require a sphere.
    vertices = torch.tensor([
        [0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 1.5], [1., .5, 2.],
    ])
    faces = torch.tensor([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 4], [1, 4, 3], [2, 3, 4]])
    coarse = Meshes(verts=[vertices], faces=[faces])
    current_vertices = (vertices + torch.tensor([[.1, 0., 0.], [0., 0., 0.], [0., .2, 0.], [0., 0., 0.], [0., 0., -.1]])).requires_grad_()
    current = Meshes(verts=[current_vertices], faces=[faces])

    terms = image_only_regularization_losses(current, coarse)
    assert set(terms) == {"normal", "smoothness", "edge", "coarse_anchor"}
    assert terms["smoothness"] > 0
    assert terms["edge"] > 0
    assert terms["coarse_anchor"] > 0
    sum(terms.values()).backward()
    assert torch.isfinite(current_vertices.grad).all()

    identity = coarse_edge_regularization_loss(coarse, coarse)
    assert identity.item() == pytest.approx(0., abs=1e-12)
    other_faces = faces.clone()
    other_faces[0] = other_faces[0].flip(0)
    with pytest.raises(ValueError, match="topology"):
        image_only_regularization_losses(current, Meshes(verts=[vertices], faces=[other_faces]))
