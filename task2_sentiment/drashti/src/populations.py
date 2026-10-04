"""Load the Phase 5 cache and keep the official test split out of training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from lab1.paths import resolve_repo_path
from task2_sentiment.drashti.src.interface import example_id

TRAINING_SPLITS = ("development", "validation")
SLICE_NAMES = {1: "short", 2: "medium", 3: "long"}

EXPECTED_COUNTS = {
    "development_source_rows": 504000,
    "development_model_ready_rows": 503972,
    "validation_source_rows": 56000,
    "validation_model_ready_rows": 55999,
    "test_rows": 38000,
    "excluded_empty_development_rows": 28,
    "excluded_empty_validation_rows": 1,
    "excluded_empty_rows": 29,
}


def reject_test_split(split_name: str) -> None:
    """Training and model selection cannot read the official test split."""
    if split_name == "test":
        raise RuntimeError("The official test split is not available to training or model selection.")
    if split_name not in TRAINING_SPLITS:
        raise RuntimeError(f"Unknown training split {split_name}.")


def load_model_ready(cache_path: str, repo: Path) -> dict[str, Any]:
    """Load the Phase 5 tensor cache."""
    path = resolve_repo_path(cache_path, repo=repo)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for name in ("development", "validation", "test"):
        if name not in payload:
            raise RuntimeError(f"The model-ready cache has no {name} split.")
    return payload


def assert_phase5_populations(cache: dict[str, Any]) -> dict[str, int]:
    """Check the documented model-ready counts. Do not rebuild them."""
    counts = {
        "development_model_ready_rows": int(cache["development"]["rows"]),
        "validation_model_ready_rows": int(cache["validation"]["rows"]),
        "test_rows": int(cache["test"]["rows"]),
    }
    if counts["development_model_ready_rows"] != EXPECTED_COUNTS["development_model_ready_rows"]:
        raise RuntimeError("Development model-ready count does not match Phase 5.")
    if counts["validation_model_ready_rows"] != EXPECTED_COUNTS["validation_model_ready_rows"]:
        raise RuntimeError("Validation model-ready count does not match Phase 5.")
    if counts["test_rows"] != EXPECTED_COUNTS["test_rows"]:
        raise RuntimeError("Test model-ready count does not match Phase 5.")
    counts.update(
        {
            "development_source_rows": EXPECTED_COUNTS["development_source_rows"],
            "validation_source_rows": EXPECTED_COUNTS["validation_source_rows"],
            "excluded_empty_development_rows": EXPECTED_COUNTS["excluded_empty_development_rows"],
            "excluded_empty_validation_rows": EXPECTED_COUNTS["excluded_empty_validation_rows"],
            "excluded_empty_rows": EXPECTED_COUNTS["excluded_empty_rows"],
        }
    )
    return counts


def example_ids_for(split_cache: dict[str, Any], prefix: str) -> list[str]:
    """Stable ids in the cache row order."""
    return [example_id(prefix, int(index)) for index in split_cache["source_index"].tolist()]


def slice_name(code: int) -> str:
    return SLICE_NAMES.get(int(code), f"code_{int(code)}")
