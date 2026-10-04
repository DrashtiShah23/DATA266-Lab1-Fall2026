"""Committed directory layout for the Phase 1 repository."""

from __future__ import annotations

from pathlib import Path

MEMBER_NAME = "drashti"

REQUIRED_DIRECTORIES: tuple[str, ...] = (
    "task1_llm/data",
    "task1_llm/drashti/src",
    "task1_llm/drashti/data_processed",
    "task1_llm/drashti/checkpoints",
    "task1_llm/drashti/outputs",
    "task2_sentiment/data",
    "task2_sentiment/drashti/src",
    "task2_sentiment/drashti/data_processed",
    "task2_sentiment/drashti/checkpoints",
    "task2_sentiment/drashti/outputs",
    "task3_gan/data/monet_jpg",
    "task3_gan/data/photo_jpg",
    "task3_gan/drashti/src",
    "task3_gan/drashti/checkpoints",
    "task3_gan/drashti/outputs/pred_A2B",
    "task3_gan/drashti/outputs/pred_B2A",
    "reproducibility/manifests",
    "reproducibility/raw_logs",
    "report",
)


def missing_directories(repo: Path) -> list[str]:
    """Return required directories that are absent from ``repo``."""
    missing: list[str] = []
    for relative in REQUIRED_DIRECTORIES:
        if not (repo / relative).is_dir():
            missing.append(relative)
    return missing
