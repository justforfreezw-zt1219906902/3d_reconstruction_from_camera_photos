"""Synthetic validation for the independent global similarity boundary.

T-003's report is absent from the checkout. Provisional explicit thresholds:
float64 exact recovery <=1e-8; float32 point/parameter error <=1e-5;
rotation error <=0.05 degrees; noisy recovery <=0.1 degrees and 1e-3 for
translation/scale; optimized vertex RMS <=1e-5. No real-data quality claim.
"""
from __future__ import annotations

import math

import pytest
import torch

from src.alignment import GlobalSimilarityAlignment, SimilarityTransform, fit_similarity


def points(dtype=torch.float64, device="cpu"):
    # Asymmetric, non-coplanar geometry avoids symmetry/correspondence ambiguity.
    return torch.tensor([[0, 0, 0], [2, 0, 0], [0, 1, 0], [0, 0, 3],
                         [-1, 0.3, 0.7], [0.2, -0.7, 1.2]], dtype=dtype, device=device)


def known(dtype=torch.float64, angle=37.0, scale=1.7, translation=(0.2, -0.4, 0.7), device="cpu"):
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    # Independently constructed active rotation around Z.
    return SimilarityTransform(torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=dtype, device=device),
                               torch.tensor(translation, dtype=dtype, device=device),
                               torch.tensor(scale, dtype=dtype, device=device))


def assert_recovery(fitted, expected, atol):
    torch.testing.assert_close(fitted.rotation, expected.rotation, atol=atol, rtol=0)
    torch.testing.assert_close(fitted.translation, expected.translation, atol=atol, rtol=0)
    torch.testing.assert_close(fitted.scale, expected.scale, atol=atol, rtol=0)
    cosine = ((torch.trace(fitted.rotation @ expected.rotation.T) - 1) / 2).clamp(-1, 1)
    assert math.degrees(math.acos(float(cosine.detach()))) <= 0.05
    assert torch.linalg.det(fitted.rotation).item() == pytest.approx(1, abs=atol)


@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float64, 1e-8)])
@pytest.mark.parametrize("angle,scale,translation", [
    (0, 1, (0, 0, 0)), (90, 1, (0, 0, 0)), (0, 1, (2, -3, 1)),
    (0, 2.5, (0, 0, 0)), (37, 1.7, (0.2, -0.4, 0.7)),
    (180, 0.3, (-2, 1, 3)), (-125, 3, (4, -2, 0.8)),
])
def test_exact_recovery(dtype, atol, angle, scale, translation):
    source = points(dtype)
    expected = known(dtype, angle, scale, translation)
    target = expected.scale * torch.einsum("ij,nj->ni", expected.rotation, source) + expected.translation
    fitted = fit_similarity(source, target)
    assert_recovery(fitted, expected, atol)
    torch.testing.assert_close(fitted.apply(source), target, atol=atol, rtol=0)


def test_identity_order_batched_inverse_and_composition():
    source = points()[None].repeat(2, 1, 1)
    original = source.clone()
    identity = SimilarityTransform.identity(dtype=source.dtype)
    torch.testing.assert_close(identity.apply(source), source, atol=0, rtol=0)
    first, second = known(), known(angle=-82, scale=0.6, translation=(-1, 2, 0))
    # 90 degrees takes +X to +Y; translation follows scaling/rotation.
    torch.testing.assert_close(known(angle=90, scale=2, translation=(1, 3, 4)).apply(torch.tensor([[1., 0, 0]], dtype=source.dtype)), torch.tensor([[1., 5, 4]], dtype=source.dtype))
    torch.testing.assert_close(first.inverse().apply(first.apply(source)), source, atol=1e-12, rtol=0)
    torch.testing.assert_close(second.compose(first).apply(source), second.apply(first.apply(source)), atol=1e-12, rtol=0)
    torch.testing.assert_close(source, original, atol=0, rtol=0)


def test_planar_correspondences_are_identifiable():
    source = points()[:3]
    expected = known()
    assert_recovery(fit_similarity(source, expected.apply(source)), expected, 1e-8)


def test_noisy_recovery():
    generator = torch.Generator().manual_seed(31)
    source = torch.randn(500, 3, dtype=torch.float64, generator=generator)
    expected = known()
    clean = expected.apply(source)
    target = clean + 0.001 * torch.randn(clean.shape, dtype=clean.dtype, generator=generator)
    fitted = fit_similarity(source, target)
    assert_recovery(fitted, expected, 1e-3)
    assert (fitted.apply(source) - clean).square().mean().sqrt() < 1e-3


def test_reflection_cannot_be_absorbed_as_rotation_or_negative_scale():
    source = points()
    target = source * torch.tensor([-1, 1, 1])
    fitted = fit_similarity(source, target)
    assert torch.linalg.det(fitted.rotation) > 0
    assert fitted.scale > 0
    assert (fitted.apply(source) - target).square().mean().sqrt() > 0.1


@pytest.mark.parametrize("case", ["few", "shape", "collinear", "collapsed", "target_collapsed", "nan", "inf", "integer", "dtype"])
def test_invalid_fitting_inputs(case):
    source, target = points(), points()
    if case == "few":
        source, target = source[:2], target[:2]
    elif case == "shape":
        target = target[None]
    elif case == "collinear":
        source = torch.arange(6, dtype=source.dtype)[:, None].repeat(1, 3)
    elif case == "collapsed":
        source.zero_()
    elif case == "target_collapsed":
        target.zero_()
    elif case in ("nan", "inf"):
        target[0, 0] = float(case)
    elif case == "integer":
        source = source.long()
    elif case == "dtype":
        target = target.float()
    with pytest.raises(ValueError):
        fit_similarity(source, target)


@pytest.mark.parametrize("case", ["reflection", "shear", "zero_scale", "negative_scale", "nan", "shape", "dtype"])
def test_invalid_transforms(case):
    rotation, translation, scale = torch.eye(3), torch.zeros(3), torch.ones(())
    if case == "reflection":
        rotation[0, 0] = -1
    elif case == "shear":
        rotation[0, 1] = 0.2
    elif case == "zero_scale":
        scale.zero_()
    elif case == "negative_scale":
        scale.neg_()
    elif case == "nan":
        translation[0] = float("nan")
    elif case == "shape":
        translation = translation[None]
    elif case == "dtype":
        scale = scale.double()
    with pytest.raises(ValueError):
        SimilarityTransform(rotation, translation, scale)


def test_apply_rejects_invalid_points():
    transform = SimilarityTransform.identity()
    for invalid in (torch.zeros(3), torch.zeros(2, 4), points(), torch.full((2, 3), float("inf"))):
        with pytest.raises(ValueError):
            transform.apply(invalid)


def test_global_parameters_gradients_freezing_and_serialization():
    model = GlobalSimilarityAlignment(known(angle=180))
    assert set(dict(model.named_parameters())) == {"rotation_vector", "translation", "log_scale"}
    assert sum(p.numel() for p in model.parameters()) == 7
    source = points().requires_grad_()
    torch.testing.assert_close(model(source), known(angle=180).apply(source))
    assert torch.autograd.gradcheck(model, (source,))
    model(source).square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    restored = GlobalSimilarityAlignment().double()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(source), model(source))
    model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    assert model(source).requires_grad  # caller-owned vertex gradients survive
    fitted = fit_similarity(source, model(source))
    assert not any(t.requires_grad for t in (fitted.rotation, fitted.translation, fitted.scale))


def test_gradient_optimizer_recovers_combined_alignment_without_offsets():
    source = points()
    expected = known(angle=24, scale=1.2, translation=(0.2, -0.1, 0.3))
    target = expected.apply(source)
    model = GlobalSimilarityAlignment().double()
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=100, tolerance_grad=1e-12, tolerance_change=1e-15, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (model(source) - target).square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    assert_recovery(model.as_transform(), expected, 1e-5)
    assert (model(source) - target).square().mean().sqrt() < 1e-5


def test_normalization_alignment_restore_order():
    # Independent reproduction of mesh_io's frame contract, without PyTorch3D.
    raw = points() * 8 + torch.tensor([10., -5, 2])
    center, normalization_scale = raw.mean(0), 8.0
    normalized = (raw - center) / normalization_scale
    transform = known()
    exported = transform.apply(normalized) * normalization_scale + center
    expected = transform.scale * ((raw - center) @ transform.rotation.T) + normalization_scale * transform.translation + center
    torch.testing.assert_close(exported, expected, atol=1e-12, rtol=0)
    torch.testing.assert_close(transform.inverse().apply((exported - center) / normalization_scale), normalized, atol=1e-12, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_fit_and_module():
    source = points(torch.float32, "cuda")
    expected = known(torch.float32, device="cuda")
    fitted = fit_similarity(source, expected.apply(source))
    assert_recovery(fitted, expected, 1e-5)
    torch.testing.assert_close(GlobalSimilarityAlignment(fitted)(source), expected.apply(source))


def test_recovery_of_rotation_about_multiple_axes():
    # Quaternion (w,x,y,z) = (1,2,3,4)/sqrt(30), expanded independently.
    rotation = torch.tensor([[-20., 4, 22], [20, -10, 20], [10, 28, 4]], dtype=torch.float64) / 30
    expected = SimilarityTransform(rotation, torch.tensor([0.3, -0.8, 1.2], dtype=torch.float64), torch.tensor(0.7, dtype=torch.float64))
    source = points()
    assert_recovery(fit_similarity(source, expected.apply(source)), expected, 1e-8)


def test_parameter_gradients_match_finite_differences_at_identity():
    model = GlobalSimilarityAlignment().double()
    source = points()
    weights = torch.arange(source.numel(), dtype=source.dtype).reshape_as(source)
    loss = (model(source) * weights).sum()
    loss.backward()
    epsilon = 1e-6
    for parameter in model.parameters():
        for index in range(parameter.numel()):
            with torch.no_grad():
                flat = parameter.reshape(-1)
                original = flat[index].clone()
                flat[index] = original + epsilon
                plus = (model(source) * weights).sum()
                flat[index] = original - epsilon
                minus = (model(source) * weights).sum()
                flat[index] = original
            torch.testing.assert_close(parameter.grad.reshape(-1)[index], (plus - minus) / (2 * epsilon), atol=1e-7, rtol=1e-7)
