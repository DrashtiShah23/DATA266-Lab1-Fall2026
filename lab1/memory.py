"""Peak memory helpers.

RSS conversion is implemented for Darwin and Linux only. Other platforms
return ``None`` so a caller cannot mistake an unknown unit for a measurement.
"""

from __future__ import annotations

import sys
import tracemalloc
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not provide resource
    resource = None


def ru_maxrss_to_bytes(ru_maxrss: int, system: str) -> int | None:
    """Convert ``ru_maxrss`` to bytes for systems whose unit is known."""
    if isinstance(ru_maxrss, bool) or not isinstance(ru_maxrss, int):
        raise TypeError("ru_maxrss must be an integer.")
    if ru_maxrss < 0:
        raise ValueError("ru_maxrss must be >= 0.")
    if system == "darwin":
        return ru_maxrss
    if system == "linux":
        return ru_maxrss * 1024
    return None


def read_peak_rss_bytes() -> int | None:
    """Return peak resident set size in bytes, or ``None`` when unsupported."""
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return ru_maxrss_to_bytes(int(usage.ru_maxrss), sys.platform)


@contextmanager
def measure_python_peak_bytes() -> Iterator[dict[str, Any]]:
    """Measure peak Python-allocated bytes inside the block via tracemalloc."""
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.reset_peak()
    result: dict[str, Any] = {"peak_python_bytes": None}
    try:
        yield result
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        result["peak_python_bytes"] = peak
        if not was_tracing:
            tracemalloc.stop()
