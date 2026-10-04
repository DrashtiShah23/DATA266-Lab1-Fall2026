"""Create configured output directories inside the repository."""

from __future__ import annotations

from pathlib import Path

from lab1.paths import resolve_repo_path


def ensure_output_dir(relative_path: str, repo: Path | None = None) -> Path:
    """Create a repository-relative directory if it does not already exist."""
    path = resolve_repo_path(relative_path, repo=repo)
    path.mkdir(parents=True, exist_ok=True)
    return path
