from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform


def linspace_angles(start: float, end: float, n: int) -> list[float]:
    if n <= 1:
        return [float(start)]
    return [float(x) for x in np.linspace(start, end, n)]


def create_turntable_cameras(
    azimuths: Iterable[float],
    elevation: float,
    distance: float,
    device: torch.device,
) -> FoVPerspectiveCameras:
    azimuths = list(azimuths)
    elevations = [elevation] * len(azimuths)
    distances = [distance] * len(azimuths)
    R, T = look_at_view_transform(dist=distances, elev=elevations, azim=azimuths, device=device)
    return FoVPerspectiveCameras(device=device, R=R, T=T)


def camera_records(
    cameras: FoVPerspectiveCameras,
    azimuths: list[float],
    elevation: float,
    distance: float,
    image_size: int,
) -> list[dict]:
    R = cameras.R.detach().cpu().tolist()
    T = cameras.T.detach().cpu().tolist()
    return [
        {
            "index": i,
            "R": R[i],
            "T": T[i],
            "azimuth": float(azimuths[i]),
            "elevation": float(elevation),
            "distance": float(distance),
            "image_size": int(image_size),
        }
        for i in range(len(azimuths))
    ]


def cameras_from_records(records: list[dict], device: torch.device) -> FoVPerspectiveCameras:
    R = torch.tensor([r["R"] for r in records], dtype=torch.float32, device=device)
    T = torch.tensor([r["T"] for r in records], dtype=torch.float32, device=device)
    return FoVPerspectiveCameras(device=device, R=R, T=T)


def write_metadata(path: Path, records: list[dict], extra: dict | None = None) -> None:
    payload = {"cameras": records}
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def read_camera_metadata(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "cameras" in data:
        return data["cameras"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Camera metadata at {path} must contain a 'cameras' list.")
