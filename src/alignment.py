"""Global CAD-to-real similarity transforms, independent of cameras and offsets.

Row-vector vertices use ``y = s * (x @ R.T) + t``: positive uniform scale,
then an active right-handed rotation, then translation in the target frame.
In the reconstruction pipeline x and y are normalized object coordinates;
mesh normalization/restoration remains owned by mesh_io. Restore the aligned
vertices at export, not the original vertices. This is not a camera extrinsic.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


def _check_tensor(value: Tensor, name: str) -> None:
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be float32 or float64")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _check_points(points: Tensor) -> None:
    _check_tensor(points, "points")
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points must have shape (..., N, 3)")


@dataclass(frozen=True)
class SimilarityTransform:
    """One proper rotation (3,3), translation (3,), and positive scalar scale.

    Tensors must share dtype/device. Applying, inverting and composing preserve
    autograd; fitting is intentionally detached. No input is modified in place.
    """

    rotation: Tensor
    translation: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        for name in ("rotation", "translation", "scale"):
            value = getattr(self, name)
            _check_tensor(value, name)
            if value.dtype != self.rotation.dtype or value.device != self.rotation.device:
                raise ValueError("transform tensors must share dtype and device")
        if self.rotation.shape != (3, 3) or self.translation.shape != (3,) or self.scale.ndim != 0:
            raise ValueError("expected rotation (3,3), translation (3,), scalar scale")
        eye = torch.eye(3, dtype=self.rotation.dtype, device=self.rotation.device)
        if not torch.allclose(self.rotation.T @ self.rotation, eye, atol=1e-5, rtol=1e-5) or not torch.allclose(torch.linalg.det(self.rotation), eye[0, 0], atol=1e-5, rtol=1e-5):
            raise ValueError("rotation must be orthonormal with determinant +1")
        if self.scale <= 0:
            raise ValueError("scale must be positive")

    @classmethod
    def identity(cls, *, dtype: torch.dtype = torch.float32, device: torch.device | str = "cpu") -> SimilarityTransform:
        """Create a neutral alignment on the requested dtype/device."""
        return cls(torch.eye(3, dtype=dtype, device=device), torch.zeros(3, dtype=dtype, device=device), torch.ones((), dtype=dtype, device=device))

    def apply(self, points: Tensor) -> Tensor:
        """Transform (..., N, 3) vertices without changing topology or inputs."""
        _check_points(points)
        if points.dtype != self.rotation.dtype or points.device != self.rotation.device:
            raise ValueError("points and transform must share dtype and device")
        return self.scale * (points @ self.rotation.T) + self.translation

    def inverse(self) -> SimilarityTransform:
        """Return the target-to-source transform."""
        return SimilarityTransform(self.rotation.T, -(self.translation @ self.rotation) / self.scale, self.scale.reciprocal())

    def compose(self, before: SimilarityTransform) -> SimilarityTransform:
        """Return self after before: result(x) == self(before(x))."""
        return SimilarityTransform(self.rotation @ before.rotation, self.apply(before.translation[None])[0], self.scale * before.scale)


@torch.no_grad()
def fit_similarity(source: Tensor, target: Tensor) -> SimilarityTransform:
    """Least-squares proper R/T/S from corresponding (N,3) point pairs.

    Minimizes sum ||s R source_i + t - target_i||² using centered SVD.
    Requires >=3 non-collinear pairs; planar data are supported. Rejects
    rank-deficient cross-covariance and collapsed data. Reflections are never
    returned (reflected targets retain residual error). No correspondence
    search, camera fitting, robust outlier rejection or local deformation is
    performed. The returned tensors are detached and retain dtype/device.
    """
    _check_points(source)
    _check_points(target)
    if source.ndim != 2 or target.shape != source.shape or len(source) < 3:
        raise ValueError("source and target must have matching (N,3) shapes with N >= 3")
    if source.dtype != target.dtype or source.device != target.device:
        raise ValueError("source and target must share dtype and device")
    source_mean, target_mean = source.mean(0), target.mean(0)
    x, y = source - source_mean, target - target_mean
    covariance = y.T @ x / len(x)
    u, singular, vh = torch.linalg.svd(covariance)
    tolerance = 10 * torch.finfo(source.dtype).eps * singular[0]
    if singular[0] <= 0 or singular[1] <= tolerance:
        raise ValueError("alignment is underdetermined: need non-collinear corresponding points")
    signs = torch.ones(3, dtype=source.dtype, device=source.device)
    signs[-1] = torch.linalg.det(u @ vh).sign()
    rotation = (u * signs) @ vh
    scale = (singular * signs).sum() / x.square().sum(1).mean()
    translation = target_mean - scale * (source_mean @ rotation.T)
    return SimilarityTransform(rotation, translation, scale)


class GlobalSimilarityAlignment(nn.Module):
    """Differentiable seven-DOF alignment for a global-only optimizer stage.

    Owns rotation_vector (3 radians), translation (3), and log_scale (1).
    A rotation-vector exponential keeps R proper and exp(log_scale) keeps s
    positive. A fixed initial rotation permits initialization from a fitted
    transform, including 180-degree rotations. Use module.parameters() as a
    separate optimizer group and requires_grad_(False) to freeze after fitting.
    The caller owns rendering losses and any subsequent local residual stage.
    """

    def __init__(self, initial: SimilarityTransform | None = None) -> None:
        super().__init__()
        initial = initial if initial is not None else SimilarityTransform.identity()
        self.register_buffer("initial_rotation", initial.rotation.detach().clone())
        self.rotation_vector = nn.Parameter(torch.zeros_like(initial.translation))
        self.translation = nn.Parameter(initial.translation.detach().clone())
        self.log_scale = nn.Parameter(initial.scale.detach().log().clone())

    def as_transform(self) -> SimilarityTransform:
        """Return the current differentiable global transform."""
        x, y, z = self.rotation_vector.unbind()
        zero = x * 0
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
        rotation = torch.matrix_exp(skew) @ self.initial_rotation
        return SimilarityTransform(rotation, self.translation, self.log_scale.exp())

    def forward(self, points: Tensor) -> Tensor:
        """Apply global alignment to vertices, preserving vertex gradients."""
        return self.as_transform().apply(points)
