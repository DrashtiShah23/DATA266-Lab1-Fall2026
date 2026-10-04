"""Load the Task 2 preprocessing config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lab1.config import ConfigError, load_config
from lab1.paths import repo_root, resolve_repo_path

_EXTRA_FIELDS = (
    "dataset_id",
    "dataset_source",
    "dataset_revision",
    "text_field",
    "label_field",
    "cache_dir",
    "processed_cache",
    "validation_fraction",
    "validation_split_seed",
    "minimum_token_frequency",
    "maximum_vocabulary_size",
    "reserved_tokens",
    "padding_id",
    "unknown_id",
    "sequence_length_percentile",
    "negation_tokens_preserved",
    "stemming_or_lemmatization",
    "embedding_policy",
)
_PATH_FIELDS = ("cache_dir", "processed_cache")


def load_task2_settings(config_path: str = "configs/task2_drashti_preprocess.json", repo: Path | None = None) -> dict[str, Any]:
    """Load the preprocessing config and check the fields this phase uses."""
    config = load_config(config_path, repo=repo)
    missing = [field for field in _EXTRA_FIELDS if field not in config]
    if missing:
        raise ConfigError("Task 2 config is missing fields: " + ", ".join(missing))
    if config["member"] != "drashti":
        raise ConfigError("This Task 2 config is for member drashti.")
    if config["stemming_or_lemmatization"] != "not_used":
        raise ConfigError("This phase does not apply stemming or lemmatization.")
    if not isinstance(config["minimum_token_frequency"], int) or config["minimum_token_frequency"] < 1:
        raise ConfigError("minimum_token_frequency must be a positive integer.")
    if not isinstance(config["maximum_vocabulary_size"], int) or config["maximum_vocabulary_size"] < 1:
        raise ConfigError("maximum_vocabulary_size must be a positive integer.")
    if config["padding_id"] != 0 or config["unknown_id"] != 1:
        raise ConfigError("padding_id must be 0 and unknown_id must be 1.")
    if config["reserved_tokens"] != ["<pad>", "<unk>"]:
        raise ConfigError("Reserved tokens must be pad then unk.")
    fraction = config["validation_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise ConfigError("validation_fraction must be a number strictly between 0 and 1.")
    percentile = config["sequence_length_percentile"]
    if isinstance(percentile, bool) or not isinstance(percentile, int) or not 1 <= percentile <= 99:
        raise ConfigError("sequence_length_percentile must be an integer from 1 to 99.")
    if not isinstance(config["negation_tokens_preserved"], list) or not config["negation_tokens_preserved"]:
        raise ConfigError("negation_tokens_preserved must be a non-empty list.")
    root = repo or repo_root()
    for field in _PATH_FIELDS:
        resolve_repo_path(config[field], repo=root)
    return config
