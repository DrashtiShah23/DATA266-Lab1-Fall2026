"""Checkpoint metadata convention.

A later training run stores weights in a separate file such as
``<experiment>.pt``. This module writes only the sidecar:

    <experiment>_<timestamp>_<id>.metadata.json

Phase 1 does not create weight files and does not claim a checkpoint exists.
``checkpoint_file`` stays null until a real weight file is written.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab1.exclusive import write_unique_text
from lab1.paths import assert_repo_relative, resolve_repo_path

METADATA_FIELDS: tuple[str, ...] = (
    "member",
    "task",
    "experiment_name",
    "git_commit",
    "config_file",
    "seed",
    "created_at",
    "epoch",
    "checkpoint_file",
)


def build_checkpoint_metadata(
    *,
    member: str,
    task: str,
    experiment_name: str,
    git_commit: str | None,
    config_file: str,
    seed: int,
    epoch: int | None = None,
    checkpoint_file: str | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build sidecar metadata. ``checkpoint_file`` is null when no weights exist."""
    assert_repo_relative(config_file)
    if checkpoint_file is not None:
        assert_repo_relative(checkpoint_file)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    if epoch is not None and (isinstance(epoch, bool) or not isinstance(epoch, int)):
        raise TypeError("epoch must be an integer or null.")
    return {
        "member": member,
        "task": task,
        "experiment_name": experiment_name,
        "git_commit": git_commit,
        "config_file": config_file,
        "seed": seed,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "epoch": epoch,
        "checkpoint_file": checkpoint_file,
    }


def write_checkpoint_metadata(
    checkpoint_directory: str,
    metadata: dict[str, Any],
    repo: Path | None = None,
) -> Path:
    """Write a new metadata sidecar. Existing files are not replaced."""
    missing = [field for field in METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError("Checkpoint metadata is missing fields: " + ", ".join(missing))
    directory = resolve_repo_path(checkpoint_directory, repo=repo)
    content = json.dumps(metadata, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    return write_unique_text(directory, metadata["experiment_name"], ".metadata.json", content)
