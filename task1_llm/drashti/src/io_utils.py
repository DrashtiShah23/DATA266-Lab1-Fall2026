"""Write Task 1 metadata without storing personal absolute paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lab1.paths import resolve_repo_path


def dumps_json(payload: dict[str, Any]) -> str:
    """Serialize JSON and refuse personal machine paths."""
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    home = str(Path.home())
    if home and home in text:
        raise ValueError("Refusing to write a personal home path into metadata.")
    if ("/" + "Users" + "/") in text or ("/" + "home" + "/") in text:
        raise ValueError("Refusing to write an absolute personal path into metadata.")
    return text


def write_repo_json(relative_path: str, payload: dict[str, Any], repo: Path) -> Path:
    """Write JSON at a repository-relative path."""
    path = resolve_repo_path(relative_path, repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps_json(payload), encoding="utf-8")
    return path
