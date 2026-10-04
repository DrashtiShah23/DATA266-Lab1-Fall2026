"""Checkpoint files for the sentiment models.

Weight files are created with an exclusive name so a later run cannot replace
an earlier checkpoint.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Copy parameters to CPU so the saved file is not tied to one device."""
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def exclusive_checkpoint_path(directory: Path, prefix: str) -> Path:
    """Reserve a new ``.pt`` path. Existing checkpoint files are not opened."""
    directory.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        token = uuid.uuid4().hex[:8]
        path = directory / f"{prefix}_{stamp}_{token}.pt"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return path
    raise RuntimeError("Could not reserve a unique checkpoint path.")


def save_checkpoint(directory: Path, prefix: str, payload: dict[str, Any]) -> Path:
    """Write one new checkpoint. The caller supplies the payload."""
    path = exclusive_checkpoint_path(directory, prefix)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint onto CPU. The caller moves the model if needed."""
    return torch.load(path, map_location="cpu", weights_only=False)


def load_model_weights(model: nn.Module, path: Path) -> dict[str, Any]:
    """Load model weights from a checkpoint. Optimizer state is left untouched."""
    payload = load_checkpoint(path)
    model.load_state_dict(payload["model_state"])
    return payload
