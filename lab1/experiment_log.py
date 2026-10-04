"""Append-only raw experiment logs.

Each log is a new JSON Lines file. Writers never open an existing log with
truncation, and a second call always allocates a different filename.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lab1.exclusive import append_text, write_unique_text
from lab1.paths import resolve_repo_path, to_repo_relative


def create_raw_log(
    log_directory: str,
    experiment_name: str,
    config: dict[str, Any],
    git_commit: str | None,
    hardware: dict[str, Any],
    repo: Path | None = None,
) -> Path:
    """Create a new raw log and write its header record."""
    directory = resolve_repo_path(log_directory, repo=repo)
    header = {
        "record_type": "header",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment_name": experiment_name,
        "config": config,
        "git_commit": git_commit,
        "hardware": hardware,
    }
    return write_unique_text(
        directory,
        experiment_name,
        ".log",
        _dumps(header),
    )


def append_log_record(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object. The previous contents stay unchanged."""
    if "record_type" not in record:
        raise ValueError("Log records require record_type.")
    append_text(path, _dumps(record))


def relative_log_path(path: Path, repo: Path | None = None) -> str:
    """Return the log path relative to the repository."""
    return to_repo_relative(path, repo=repo)


def _dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n"
