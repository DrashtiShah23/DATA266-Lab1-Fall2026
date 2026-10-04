"""Repository-relative path handling.

Experiment configs must use paths relative to the repository root.
Personal machine paths are rejected.
"""

from __future__ import annotations

import re
from pathlib import Path

_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


class PathSafetyError(ValueError):
    """Raised when a path is absolute, personal, or escapes the repository."""


def repo_root() -> Path:
    """Return the repository root that contains this package."""
    return Path(__file__).resolve().parent.parent


def assert_repo_relative(value: str) -> str:
    """Reject absolute paths, home shortcuts, and URI-style paths."""
    if not isinstance(value, str) or value.strip() == "":
        raise PathSafetyError("A repository-relative path must be a non-empty string.")
    text = value.strip()
    if text.startswith("~"):
        raise PathSafetyError("Home-relative paths are not allowed. Use a repository-relative path.")
    if text.startswith("\\\\") or text.startswith("//"):
        raise PathSafetyError("Network paths are not allowed. Use a repository-relative path.")
    if text.startswith("/") or text.startswith("\\"):
        raise PathSafetyError("Absolute paths are not allowed. Use a repository-relative path.")
    if _DRIVE_PREFIX.match(text):
        raise PathSafetyError("Drive-letter paths are not allowed. Use a repository-relative path.")
    if "://" in text:
        raise PathSafetyError("URI paths are not allowed. Use a repository-relative path.")
    if "\x00" in text:
        raise PathSafetyError("Paths must not contain null bytes.")
    return text


def resolve_repo_path(value: str, repo: Path | None = None) -> Path:
    """Resolve a repository-relative path and require it to stay inside the repo."""
    relative = assert_repo_relative(value)
    root = (repo or repo_root()).resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise PathSafetyError(
            "Path escapes the repository. Use a repository-relative path."
        ) from exc
    return candidate


def to_repo_relative(path: Path, repo: Path | None = None) -> str:
    """Convert an existing path inside the repository to a POSIX relative path."""
    root = (repo or repo_root()).resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise PathSafetyError(
            "Path is outside the repository and cannot be stored."
        ) from exc
    return relative.as_posix()
