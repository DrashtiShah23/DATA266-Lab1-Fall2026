"""Task 1 data settings on top of the shared config loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lab1.config import ConfigError, load_config
from lab1.paths import repo_root, resolve_repo_path

REQUIRED_TRAIN_SIZE = 100_000
REQUIRED_VALIDATION_SIZE = 10_000

_EXTRA_FIELDS = (
    "dataset_id",
    "dataset_source",
    "dataset_revision",
    "source_split",
    "text_field",
    "train_size",
    "validation_size",
    "cache_dir",
    "vocab_metadata",
    "split_metadata",
    "validation_report",
)
_PATH_FIELDS = ("cache_dir", "vocab_metadata", "split_metadata", "validation_report")


def load_task1_settings(config_path: str = "configs/task1_drashti_data.json", repo: Path | None = None) -> dict[str, Any]:
    """Load the Task 1 data config and require the assignment split sizes."""
    config = load_config(config_path, repo=repo)
    missing = [field for field in _EXTRA_FIELDS if field not in config]
    if missing:
        raise ConfigError("Task 1 config is missing fields: " + ", ".join(missing))
    if config["member"] != "drashti":
        raise ConfigError("This Task 1 config is for member drashti.")
    if config["train_size"] != REQUIRED_TRAIN_SIZE or config["validation_size"] != REQUIRED_VALIDATION_SIZE:
        raise ConfigError("Task 1 split sizes must be 100000 training and 10000 validation examples.")
    if not isinstance(config["sequence_length"], int) or isinstance(config["sequence_length"], bool):
        raise ConfigError("sequence_length must be an integer.")
    if config["sequence_length"] < 1:
        raise ConfigError("sequence_length must be positive.")
    if not isinstance(config["batch_size"], int) or isinstance(config["batch_size"], bool) or config["batch_size"] < 1:
        raise ConfigError("batch_size must be a positive integer.")
    if not isinstance(config["split_seed"], int) or isinstance(config["split_seed"], bool):
        raise ConfigError("split_seed must be an integer.")
    for field in ("dataset_id", "dataset_source", "source_split", "text_field"):
        if not isinstance(config[field], str) or config[field].strip() == "":
            raise ConfigError(f"{field} must be a non-empty string.")
    revision = config["dataset_revision"]
    if revision is not None and (not isinstance(revision, str) or revision.strip() == ""):
        raise ConfigError("dataset_revision must be a non-empty string or null.")
    root = repo or repo_root()
    for field in _PATH_FIELDS:
        resolve_repo_path(config[field], repo=root)
    return config
