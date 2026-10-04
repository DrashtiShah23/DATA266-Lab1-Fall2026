"""Resolve the configured device and read live hardware details.

Serial numbers and user-specific paths are not collected.
"""

from __future__ import annotations

import subprocess
from typing import Any

import torch

from lab1.hardware import collect_hardware_report
from lab1.memory import read_peak_rss_bytes


def resolve_device(requested: str) -> torch.device:
    """Return the configured device or raise if it is unavailable."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("The config requests mps, and MPS is not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("The config requests cuda, and CUDA is not available.")
        return torch.device("cuda")
    raise RuntimeError("Unsupported device setting.")


def synchronize(device: torch.device) -> None:
    """Wait for queued work so a timing measurement includes it."""
    if device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def apple_hardware() -> dict[str, Any]:
    """Read the Apple chip name and unified memory size when sysctl is present."""
    details: dict[str, Any] = {
        "cpu_model": None,
        "gpu_model": None,
        "gpu_memory_bytes": None,
        "unified_memory_bytes": None,
        "memory_note": None,
    }
    try:
        brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        memsize = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return details
    if brand:
        details["cpu_model"] = brand
        details["gpu_model"] = brand
    if memsize.isdigit():
        details["unified_memory_bytes"] = int(memsize)
        details["memory_note"] = (
            "Apple Silicon uses unified memory. No separate GPU memory figure is exposed, "
            "so gpu_memory_bytes is null."
        )
    return details


def training_hardware(device: torch.device) -> dict[str, Any]:
    """Hardware report for the process that is about to train."""
    report = collect_hardware_report()
    report.update(apple_hardware())
    report["selected_device"] = device.type
    report["pytorch_version"] = torch.__version__
    return report


def memory_snapshot(device: torch.device) -> dict[str, Any]:
    """Peak process RSS and, on MPS, the current allocator readings."""
    snapshot: dict[str, Any] = {"peak_rss_bytes": read_peak_rss_bytes()}
    if device.type == "mps":
        current = getattr(torch.mps, "current_allocated_memory", None)
        driver = getattr(torch.mps, "driver_allocated_memory", None)
        snapshot["mps_current_allocated_bytes"] = int(current()) if current is not None else None
        snapshot["mps_driver_allocated_bytes"] = int(driver()) if driver is not None else None
    elif device.type == "cuda":
        snapshot["cuda_max_memory_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
    return snapshot
