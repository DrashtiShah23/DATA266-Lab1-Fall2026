"""Seed Python random generators without pretending unavailable libraries exist."""

from __future__ import annotations

import os
import random


def seed_everything(seed: int) -> dict[str, object]:
    """Seed the generators that are installed.

    ``PYTHONHASHSEED`` is recorded for child processes. CPython does not
    apply a hash-seed change to the already running interpreter.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    seeded = ["random"]
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        np.random.seed(seed)
        seeded.append("numpy")
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        seeded.append("torch")
    return {"seed": seed, "seeded_libraries": seeded}
