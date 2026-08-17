from __future__ import annotations

import torch

from src.openscan_pose import inverse_object_rotation, object_rotation


def test_rotation_is_orthonormal_for_cardinal_theta() -> None:
    theta = torch.tensor([0.0, 90.0, 180.0, 270.0])
    phi = torch.zeros(4)
    rotation = object_rotation(theta, phi)
    identity = torch.eye(3).expand(4, -1, -1)
    assert torch.allclose(rotation @ rotation.transpose(-1, -2), identity, atol=1e-5)


def test_object_rotation_inverse_for_positive_and_negative_phi() -> None:
    theta = torch.tensor([0.0, 90.0, 180.0, 270.0])
    phi = torch.tensor([-30.0, 0.0, 30.0, 10.0])
    rotation = object_rotation(theta, phi)
    inverse = inverse_object_rotation(theta, phi)
    identity = torch.eye(3).expand(4, -1, -1)
    assert torch.allclose(rotation @ inverse, identity, atol=1e-5)
