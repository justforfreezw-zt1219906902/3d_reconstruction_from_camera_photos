from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


AXES = ("X", "Y", "Z")


@dataclass(frozen=True)
class CameraConvention:
    theta_axis: str = "Y"
    phi_axis: str = "X"
    theta_sign: int = 1
    phi_sign: int = 1
    rotation_order: str = "theta_then_phi"

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def label(self) -> str:
        return (
            f"theta_{self.theta_axis}{self.theta_sign:+d}_"
            f"phi_{self.phi_axis}{self.phi_sign:+d}_{self.rotation_order}"
        )


DEFAULT_CONVENTION = CameraConvention()


def candidate_conventions() -> list[CameraConvention]:
    candidates: list[CameraConvention] = []
    for theta_axis in AXES:
        for phi_axis in AXES:
            if phi_axis == theta_axis:
                continue
            for theta_sign in (1, -1):
                for phi_sign in (1, -1):
                    for rotation_order in ("theta_then_phi", "phi_then_theta"):
                        candidates.append(
                            CameraConvention(
                                theta_axis=theta_axis,
                                phi_axis=phi_axis,
                                theta_sign=theta_sign,
                                phi_sign=phi_sign,
                                rotation_order=rotation_order,
                            )
                        )
    return candidates


def axis_rotation(axis: str, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "Y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "Z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unknown rotation axis: {axis}")


def object_rotation(theta_deg: float, phi_deg: float, convention: CameraConvention) -> np.ndarray:
    theta_rotation = axis_rotation(convention.theta_axis, convention.theta_sign * theta_deg)
    phi_rotation = axis_rotation(convention.phi_axis, convention.phi_sign * phi_deg)
    if convention.rotation_order == "theta_then_phi":
        return theta_rotation @ phi_rotation
    if convention.rotation_order == "phi_then_theta":
        return phi_rotation @ theta_rotation
    raise ValueError(f"Unknown rotation order: {convention.rotation_order}")
