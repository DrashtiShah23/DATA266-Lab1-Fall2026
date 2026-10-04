"""Load the official Yelp Polarity dataset.

The Hugging Face dataset ``yelp_polarity`` is the programmatic copy of the
Xiang Zhang polarity split. This module does not synthesize reviews.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lab1.paths import resolve_repo_path

DATASET_ID = "yelp_polarity"
DATASET_SOURCE = "https://huggingface.co/datasets/yelp_polarity"
LABEL_NAMES = ("1", "2")
LABEL_MEANING = {
    0: "negative",
    1: "positive",
}
LABEL_SOURCE = (
    "The dataset card says negative polarity is class 1 and positive polarity is class 2. "
    "The loaded ClassLabel names are ['1', '2'], so integer 0 is class 1 (negative) and integer 1 is class 2 (positive)."
)
EMBEDDING_POLICY = "EMBEDDINGS WILL BE LEARNED FROM SCRATCH DURING MODEL TRAINING."


class DatasetAcquisitionError(RuntimeError):
    """Raised when the real Yelp Polarity dataset cannot be loaded."""


def configure_cache(cache_dir: str, repo: Path) -> Path:
    """Point Hugging Face caches at the configured repository-relative directory."""
    cache = resolve_repo_path(cache_dir, repo=repo)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    return cache


def load_yelp_polarity(config: dict[str, Any], repo: Path) -> Any:
    """Download or reuse the cached official train and test splits."""
    configure_cache(config["cache_dir"], repo)
    revision = config["dataset_revision"]
    if not isinstance(revision, str) or revision.strip() == "":
        raise DatasetAcquisitionError("dataset_revision must be pinned.")
    try:
        from datasets import load_dataset

        dataset = load_dataset(config["dataset_id"], revision=revision)
    except Exception as exc:
        raise DatasetAcquisitionError(
            "Yelp Polarity could not be downloaded or loaded. No substitute sentiment dataset was used."
        ) from exc
    if set(dataset.keys()) != {"train", "test"}:
        raise DatasetAcquisitionError(f"Unexpected splits: {list(dataset.keys())}.")
    for split_name in ("train", "test"):
        columns = list(dataset[split_name].column_names)
        if config["text_field"] not in columns or config["label_field"] not in columns:
            raise DatasetAcquisitionError(f"{split_name} columns were {columns}.")
        names = tuple(dataset[split_name].features[config["label_field"]].names)
        if names != LABEL_NAMES:
            raise DatasetAcquisitionError(f"{split_name} label names were {names}.")
    return dataset
