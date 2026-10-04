"""Experiment manifests.

A manifest links one experiment to its config, log, checkpoint, and metrics.
Fields that do not exist yet are stored as null. They are not invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lab1.exclusive import write_unique_text
from lab1.paths import PathSafetyError, assert_repo_relative, resolve_repo_path, to_repo_relative

MANIFEST_FIELDS: tuple[str, ...] = (
    "member",
    "task",
    "experiment_name",
    "git_commit",
    "config_file",
    "environment_file",
    "raw_log",
    "checkpoint",
    "metrics_file",
    "outputs_directory",
    "hardware",
    "start_time",
    "end_time",
    "total_training_time",
)

_PATH_FIELDS = (
    "config_file",
    "environment_file",
    "raw_log",
    "checkpoint",
    "metrics_file",
    "outputs_directory",
)


def build_manifest(
    *,
    member: str,
    task: str,
    experiment_name: str,
    git_commit: str | None,
    config_file: str,
    environment_file: str,
    raw_log: str,
    checkpoint: str | None,
    metrics_file: str | None,
    outputs_directory: str,
    hardware: dict[str, Any],
    start_time: str,
    end_time: str,
    total_training_time: float | None,
) -> dict[str, Any]:
    """Build a manifest dict. Path fields must already be repository-relative."""
    manifest: dict[str, Any] = {
        "member": member,
        "task": task,
        "experiment_name": experiment_name,
        "git_commit": git_commit,
        "config_file": config_file,
        "environment_file": environment_file,
        "raw_log": raw_log,
        "checkpoint": checkpoint,
        "metrics_file": metrics_file,
        "outputs_directory": outputs_directory,
        "hardware": hardware,
        "start_time": start_time,
        "end_time": end_time,
        "total_training_time": total_training_time,
    }
    missing = [field for field in MANIFEST_FIELDS if field not in manifest]
    if missing:
        raise ValueError("Manifest is missing fields: " + ", ".join(missing))
    _validate_paths(manifest)
    if total_training_time is not None and (
        isinstance(total_training_time, bool) or not isinstance(total_training_time, (int, float))
    ):
        raise ValueError("total_training_time must be a number or null.")
    if not isinstance(hardware, dict):
        raise ValueError("hardware must be an object.")
    return manifest


def write_manifest(manifest_directory: str, manifest: dict[str, Any], repo: Path | None = None) -> Path:
    """Write a new manifest file. Existing manifests are not replaced."""
    directory = resolve_repo_path(manifest_directory, repo=repo)
    prefix = f"{manifest['member']}_{manifest['task']}_{manifest['experiment_name']}"
    content = json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path = write_unique_text(directory, prefix, ".manifest.json", content)
    return path


def relative_manifest_path(path: Path, repo: Path | None = None) -> str:
    """Return the manifest path relative to the repository."""
    return to_repo_relative(path, repo=repo)


def _validate_paths(manifest: dict[str, Any]) -> None:
    for field in _PATH_FIELDS:
        value = manifest[field]
        if value is None:
            if field in ("checkpoint", "metrics_file"):
                continue
            raise ValueError(f"{field} is required.")
        try:
            assert_repo_relative(value)
        except PathSafetyError as exc:
            raise ValueError(f"{field} must be repository-relative.") from exc
