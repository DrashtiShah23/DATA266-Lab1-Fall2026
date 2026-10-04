"""Collect a live hardware report from this process.

The report contains only values returned by the interpreter or an imported
library. Missing libraries are reported as unavailable rather than guessed.
"""

from __future__ import annotations

import os
import platform
from typing import Any


def collect_hardware_report() -> dict[str, Any]:
    """Return the current machine and interpreter details."""
    processor = platform.processor() or platform.machine()
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": processor,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "cpu_count": os.cpu_count(),
        "torch_available": False,
        "torch_version": None,
        "cuda_available": None,
        "mps_available": None,
    }
    try:
        import torch
    except ImportError:
        return report
    report["torch_available"] = True
    report["torch_version"] = torch.__version__
    report["cuda_available"] = bool(torch.cuda.is_available())
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None or not hasattr(mps, "is_available"):
        report["mps_available"] = False
    else:
        report["mps_available"] = bool(mps.is_available())
    return report
