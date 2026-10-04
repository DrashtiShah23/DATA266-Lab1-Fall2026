"""Load the official TinyStories dataset.

The Hugging Face dataset ``roneneldan/TinyStories`` is the programmatic
distribution maintained by the TinyStories authors. This module does not
synthesize stories.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lab1.paths import resolve_repo_path

DATASET_ID = "roneneldan/TinyStories"
DATASET_SOURCE = "https://huggingface.co/datasets/roneneldan/TinyStories"
TEXT_FIELD = "text"


class DatasetAcquisitionError(RuntimeError):
    """Raised when the real TinyStories dataset cannot be loaded."""


def configure_cache(cache_dir: str, repo: Path) -> Path:
    """Point Hugging Face caches at the configured repository-relative directory."""
    cache = resolve_repo_path(cache_dir, repo=repo)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return cache


def resolve_dataset_revision(dataset_id: str, requested: str | None) -> str:
    """Return the requested revision, or the current Hub commit when it is null."""
    from huggingface_hub import HfApi

    if requested:
        return requested
    info = HfApi().dataset_info(dataset_id)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or revision.strip() == "":
        raise DatasetAcquisitionError("The Hugging Face Hub did not return a dataset revision.")
    return revision


def load_tinystories(config: dict[str, Any], repo: Path) -> tuple[Any, str]:
    """Download or reuse the cached official train split.

    Returns the loaded split and the revision that was requested.
    """
    configure_cache(config["cache_dir"], repo)
    try:
        revision = resolve_dataset_revision(config["dataset_id"], config.get("dataset_revision"))
        from datasets import load_dataset

        dataset = load_dataset(
            config["dataset_id"],
            split=config["source_split"],
            revision=revision,
        )
    except DatasetAcquisitionError:
        raise
    except Exception as exc:
        raise DatasetAcquisitionError(
            "TinyStories could not be downloaded or loaded. "
            "No synthetic dataset was substituted."
        ) from exc
    column_names = list(getattr(dataset, "column_names", []))
    if config["text_field"] not in column_names:
        raise DatasetAcquisitionError(
            "Loaded TinyStories does not contain the configured text field. "
            f"Columns observed: {column_names}."
        )
    return dataset, revision
