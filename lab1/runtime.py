"""Measure elapsed time for a block of code."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any


@contextmanager
def measure_runtime() -> Iterator[dict[str, Any]]:
    """Record wall-clock start, end, and elapsed seconds for the block."""
    result: dict[str, Any] = {
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "elapsed_seconds": None,
    }
    started = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        result["end_time"] = datetime.now(timezone.utc).isoformat()
