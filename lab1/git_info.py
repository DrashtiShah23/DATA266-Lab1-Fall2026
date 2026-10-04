"""Read the current Git commit without inventing one."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_commit_hash(repo: Path) -> str | None:
    """Return the full HEAD hash, or ``None`` when HEAD does not exist."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        return None
    return value
