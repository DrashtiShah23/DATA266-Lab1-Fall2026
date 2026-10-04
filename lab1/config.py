"""Load experiment configs. Important settings stay in config files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lab1.paths import PathSafetyError, assert_repo_relative, repo_root, resolve_repo_path

class ConfigError(ValueError):
    """Raised when a config is missing a field or has the wrong type."""


REQUIRED_FIELDS: tuple[str, ...] = (
    "experiment_name",
    "member",
    "task",
    "seed",
    "dataset_paths",
    "split_seed",
    "batch_size",
    "epochs",
    "learning_rate",
    "optimizer",
    "scheduler",
    "model_dimensions",
    "dropout",
    "sequence_length",
    "embedding_dimension",
    "checkpoint_directory",
    "output_directory",
    "log_directory",
    "manifest_directory",
    "device",
    "task_specific_loss_weights",
    "environment_file",
)

_NULLABLE_INTS = (
    "batch_size",
    "epochs",
    "sequence_length",
    "embedding_dimension",
)
_NULLABLE_NUMBERS = ("learning_rate", "dropout")
_NULLABLE_STRINGS = ("optimizer", "scheduler", "device")
_PATH_FIELDS = (
    "checkpoint_directory",
    "output_directory",
    "log_directory",
    "manifest_directory",
    "environment_file",
)
_NAME_FIELDS = ("experiment_name", "member", "task")


def load_config(config_path: str, repo: Path | None = None) -> dict[str, Any]:
    """Load and validate a JSON config stored at a repository-relative path."""
    relative_config = assert_repo_relative(config_path)
    path = resolve_repo_path(relative_config, repo=repo)
    if not path.is_file():
        raise ConfigError("Config file does not exist.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError("Config file is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a JSON object.")
    missing = [field for field in REQUIRED_FIELDS if field not in data]
    if missing:
        raise ConfigError("Config is missing required fields: " + ", ".join(missing))
    root = (repo or repo_root()).resolve()
    _validate(data, root)
    return data


def _validate(data: dict[str, Any], repo: Path) -> None:
    for field in _NAME_FIELDS:
        value = data[field]
        if not isinstance(value, str) or value.strip() == "":
            raise ConfigError(f"{field} must be a non-empty string.")
        if any(token in value for token in ("/", "\\", "..")):
            raise ConfigError(f"{field} must be a name, not a path.")
    if not isinstance(data["seed"], int) or isinstance(data["seed"], bool):
        raise ConfigError("seed must be an integer.")
    if not isinstance(data["split_seed"], int) or isinstance(data["split_seed"], bool):
        raise ConfigError("split_seed must be an integer.")
    for field in _NULLABLE_INTS:
        _require_optional_int(data[field], field)
    for field in _NULLABLE_NUMBERS:
        _require_optional_number(data[field], field)
    for field in _NULLABLE_STRINGS:
        _require_optional_string(data[field], field)
    if not isinstance(data["model_dimensions"], dict):
        raise ConfigError("model_dimensions must be a JSON object.")
    if not isinstance(data["task_specific_loss_weights"], dict):
        raise ConfigError("task_specific_loss_weights must be a JSON object.")
    _validate_dataset_paths(data["dataset_paths"], repo)
    for field in _PATH_FIELDS:
        _require_inside_repo(data[field], field, repo)


def _validate_dataset_paths(value: Any, repo: Path) -> None:
    if not isinstance(value, dict) or not value:
        raise ConfigError("dataset_paths must be a non-empty JSON object.")
    for key, path in value.items():
        if not isinstance(key, str) or key.strip() == "":
            raise ConfigError("dataset_paths keys must be non-empty strings.")
        _require_inside_repo(path, "dataset_paths", repo)


def _require_inside_repo(value: Any, field: str, repo: Path) -> None:
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a repository-relative string.")
    try:
        resolve_repo_path(value, repo=repo)
    except PathSafetyError as exc:
        raise ConfigError(f"{field} must stay inside the repository.") from exc


def _require_optional_int(value: Any, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer or null.")


def _require_optional_number(value: Any, field: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a number or null.")


def _require_optional_string(value: Any, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a string or null.")
