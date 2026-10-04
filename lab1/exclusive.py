"""Create files without overwriting anything that already exists."""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(value: str) -> str:
    """Return a filename fragment that cannot introduce another directory."""
    cleaned = _UNSAFE.sub("_", value.strip())
    cleaned = cleaned.strip("._")
    if cleaned == "":
        raise ValueError("Name is empty after removing unsafe characters.")
    return cleaned


def write_unique_text(directory: Path, prefix: str, suffix: str, content: str) -> Path:
    """Write ``content`` to a new file. A name collision retries with a new id.

    Existing files are never opened for truncation.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stem = safe_name(prefix)
    ending = suffix if suffix.startswith(".") else f".{suffix}"
    for _ in range(20):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        token = uuid.uuid4().hex[:8]
        path = directory / f"{stem}_{stamp}_{token}{ending}"
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path
    raise RuntimeError("Could not allocate a unique file name.")


def append_text(path: Path, content: str) -> None:
    """Append text. This never truncates an existing log."""
    if not path.is_file():
        raise FileNotFoundError("Refusing to append because the file does not already exist.")
    text = content if content.endswith("\n") else content + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
