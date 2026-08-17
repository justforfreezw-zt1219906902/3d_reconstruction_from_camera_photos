from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path: Path, offsets: torch.Tensor, epoch: int, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epoch": epoch, "offsets": offsets.detach().cpu()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
