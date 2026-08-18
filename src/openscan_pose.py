from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from pytorch3d.renderer import FoVPerspectiveCameras

from .pose_conventions import CameraConvention, DEFAULT_CONVENTION


@dataclass(frozen=True)
class OpenScanConvention:
    """OpenScan convention used by this demo.

    The object rotates around +Y by theta, then tilts around +X by phi:
    R_object = R_y(theta) @ R_x(phi). PyTorch3D's R is world-to-view, so the
    equivalent fixed-object camera uses R_camera = R_base @ R_object. T is the
    world-to-view translation for a camera looking at the origin. Angles are
    degrees in CSV files and radians inside torch operations.
    """

    theta_axis: str = "+Y"
    phi_axis: str = "+X"
    rotation_order: str = "R_y(theta) @ R_x(phi)"
    handedness: str = "right-handed"
    camera_look_direction: str = "camera looks toward object origin"
    equivalent_transform: str = "R_camera = R_base @ R_object; object stays fixed"


CONVENTION = OpenScanConvention()


def _axis_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    zeros = torch.zeros_like(angle)
    ones = torch.ones_like(angle)
    if axis == "X":
        rows = ([ones, zeros, zeros], [zeros, c, -s], [zeros, s, c])
    elif axis == "Y":
        rows = ([c, zeros, s], [zeros, ones, zeros], [-s, zeros, c])
    elif axis == "Z":
        rows = ([c, -s, zeros], [s, c, zeros], [zeros, zeros, ones])
    else:
        raise ValueError(f"Unknown rotation axis: {axis}")
    return torch.stack([torch.stack(row, -1) for row in rows], -2)


def object_rotation(
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
    convention: CameraConvention = DEFAULT_CONVENTION,
) -> torch.Tensor:
    theta = torch.deg2rad(theta_deg)
    phi = torch.deg2rad(phi_deg)
    theta_rotation = _axis_rotation(convention.theta_axis, convention.theta_sign * theta)
    phi_rotation = _axis_rotation(convention.phi_axis, convention.phi_sign * phi)
    return theta_rotation @ phi_rotation if convention.rotation_order == "theta_then_phi" else phi_rotation @ theta_rotation


def inverse_object_rotation(
    theta_deg: torch.Tensor,
    phi_deg: torch.Tensor,
    convention: CameraConvention = DEFAULT_CONVENTION,
) -> torch.Tensor:
    return object_rotation(theta_deg, phi_deg, convention).transpose(-1, -2)


def _inverse_sigmoid(value: float) -> float:
    value = min(max(value, 1e-5), 1 - 1e-5)
    return math.log(value / (1.0 - value))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(max(value, 1e-5)))


def _inverse_bounded(value: float, bound: float) -> float:
    return math.atanh(max(-0.99999, min(0.99999, value / max(bound, 1e-8))))


class OpenScanCameraModel(torch.nn.Module):
    def __init__(
        self,
        theta_deg: list[float],
        phi_deg: list[float],
        distance_initial: float,
        fov_initial: float,
        max_theta_delta_deg: float,
        max_phi_delta_deg: float,
        pose_refine: bool,
        device: torch.device,
        convention: CameraConvention = DEFAULT_CONVENTION,
    ) -> None:
        super().__init__()
        self.register_buffer("theta_commanded", torch.tensor(theta_deg, dtype=torch.float32, device=device))
        self.register_buffer("phi_commanded", torch.tensor(phi_deg, dtype=torch.float32, device=device))
        self.register_buffer("base_r", torch.eye(3, dtype=torch.float32, device=device))
        self.raw_distance = torch.nn.Parameter(torch.tensor(_inverse_softplus(distance_initial), device=device))
        normalized_fov = (fov_initial - 10.0) / 110.0
        self.raw_fov = torch.nn.Parameter(torch.tensor(_inverse_sigmoid(normalized_fov), device=device))
        self.raw_x_offset = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.raw_y_offset = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.pose_refine = pose_refine
        self.convention = convention
        self.max_theta_delta_deg = max_theta_delta_deg
        self.max_phi_delta_deg = max_phi_delta_deg
        self.raw_theta_delta = torch.nn.Parameter(torch.zeros(len(theta_deg), device=device), requires_grad=pose_refine)
        self.raw_phi_delta = torch.nn.Parameter(torch.zeros(len(phi_deg), device=device), requires_grad=pose_refine)

    @property
    def distance(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_distance) + 1e-3

    @property
    def fov(self) -> torch.Tensor:
        return 10.0 + 110.0 * torch.sigmoid(self.raw_fov)

    @property
    def x_offset(self) -> torch.Tensor:
        return torch.tanh(self.raw_x_offset)

    @property
    def y_offset(self) -> torch.Tensor:
        return torch.tanh(self.raw_y_offset)

    @property
    def theta_delta(self) -> torch.Tensor:
        return self.max_theta_delta_deg * torch.tanh(self.raw_theta_delta)

    @property
    def phi_delta(self) -> torch.Tensor:
        return self.max_phi_delta_deg * torch.tanh(self.raw_phi_delta)

    def cameras(self, indices: torch.Tensor | list[int] | None = None) -> FoVPerspectiveCameras:
        if indices is None:
            indices = torch.arange(len(self.theta_commanded), device=self.theta_commanded.device)
        if not torch.is_tensor(indices):
            indices = torch.tensor(indices, dtype=torch.long, device=self.theta_commanded.device)
        theta = self.theta_commanded[indices] + self.theta_delta[indices]
        phi = self.phi_commanded[indices] + self.phi_delta[indices]
        rotations = object_rotation(theta, phi, self.convention)
        R = self.base_r.unsqueeze(0) @ rotations
        zeros = torch.zeros(len(indices), dtype=R.dtype, device=R.device)
        T = torch.stack([zeros, zeros, self.distance.expand(len(indices))], dim=-1)
        f = 1.0 / torch.tan(torch.deg2rad(self.fov) / 2.0)
        fx = f.expand(len(indices))
        fy = f.expand(len(indices))
        px = self.x_offset.expand(len(indices))
        py = self.y_offset.expand(len(indices))
        znear = torch.ones_like(fx)
        zfar = torch.full_like(fx, 100.0)
        projection = torch.stack(
            [
                torch.stack([fx, zeros, px, zeros], dim=-1),
                torch.stack([zeros, fy, py, zeros], dim=-1),
                torch.stack(
                    [
                        zeros,
                        zeros,
                        zfar / (zfar - znear),
                        -(zfar * znear) / (zfar - znear),
                    ],
                    dim=-1,
                ),
                torch.stack([zeros, zeros, torch.ones_like(fx), zeros], dim=-1),
            ],
            dim=1,
        )
        return FoVPerspectiveCameras(
            device=R.device,
            R=R,
            T=T,
            K=projection,
        )

    def pose_records(self) -> list[dict]:
        theta_delta = self.theta_delta.detach().cpu().tolist()
        phi_delta = self.phi_delta.detach().cpu().tolist()
        theta = self.theta_commanded.detach().cpu().tolist()
        phi = self.phi_commanded.detach().cpu().tolist()
        return [
            {
                "index": i,
                "commanded_theta_deg": theta[i],
                "commanded_phi_deg": phi[i],
                "optimized_theta_delta_deg": theta_delta[i],
                "optimized_phi_delta_deg": phi_delta[i],
                "final_theta_deg": theta[i] + theta_delta[i],
                "final_phi_deg": phi[i] + phi_delta[i],
            }
            for i in range(len(theta))
        ]

    def parameters_snapshot(self) -> dict:
        return {
            "distance": float(self.distance.detach().cpu()),
            "fov_deg": float(self.fov.detach().cpu()),
            "principal_point_x_ndc": float(self.x_offset.detach().cpu()),
            "principal_point_y_ndc": float(self.y_offset.detach().cpu()),
            "pose_refine": self.pose_refine,
            "max_theta_delta_deg": self.max_theta_delta_deg,
            "max_phi_delta_deg": self.max_phi_delta_deg,
            "convention": self.convention.as_dict(),
        }

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def global_parameters(self) -> list[torch.nn.Parameter]:
        return [self.raw_distance, self.raw_fov, self.raw_x_offset, self.raw_y_offset]

    def pose_parameters(self) -> list[torch.nn.Parameter]:
        return [self.raw_theta_delta, self.raw_phi_delta]

    def set_global_trainable(self, enabled: bool) -> None:
        for parameter in self.global_parameters():
            parameter.requires_grad_(enabled)

    def set_pose_trainable(self, enabled: bool) -> None:
        for parameter in self.pose_parameters():
            parameter.requires_grad_(enabled and self.pose_refine)

    def set_fitted_parameters(
        self,
        camera: dict,
        theta_deltas: list[float],
        phi_deltas: list[float],
    ) -> None:
        with torch.no_grad():
            self.raw_distance.copy_(torch.tensor(_inverse_softplus(float(camera["distance"])), device=self.raw_distance.device))
            normalized_fov = (float(camera["fov_deg"]) - 10.0) / 110.0
            self.raw_fov.copy_(torch.tensor(_inverse_sigmoid(normalized_fov), device=self.raw_fov.device))
            self.raw_x_offset.copy_(torch.tensor(_inverse_bounded(float(camera["principal_point_x_ndc"]), 1.0), device=self.raw_x_offset.device))
            self.raw_y_offset.copy_(torch.tensor(_inverse_bounded(float(camera["principal_point_y_ndc"]), 1.0), device=self.raw_y_offset.device))
            theta_values = [_inverse_bounded(float(value), self.max_theta_delta_deg) for value in theta_deltas]
            phi_values = [_inverse_bounded(float(value), self.max_phi_delta_deg) for value in phi_deltas]
            self.raw_theta_delta.copy_(torch.tensor(theta_values, dtype=self.raw_theta_delta.dtype, device=self.raw_theta_delta.device))
            self.raw_phi_delta.copy_(torch.tensor(phi_values, dtype=self.raw_phi_delta.dtype, device=self.raw_phi_delta.device))


def save_pose_convention(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(DEFAULT_CONVENTION.as_dict(), indent=2))
