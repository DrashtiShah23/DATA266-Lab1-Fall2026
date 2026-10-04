"""Validation metrics used to choose a checkpoint. This is not the Phase 7 report."""

from __future__ import annotations

import torch


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Return class probabilities. Column 1 is the positive class."""
    return torch.softmax(logits, dim=-1)


def predict_label(positive_probability: torch.Tensor, threshold: float) -> torch.Tensor:
    """Predict positive when the positive probability is at least the threshold."""
    return (positive_probability >= threshold).to(dtype=torch.long)


def binary_macro_f1(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    """Mean of the per-class F1 scores for labels 0 and 1."""
    scores = []
    for class_id in (0, 1):
        true_positive = int(((predictions == class_id) & (labels == class_id)).sum())
        false_positive = int(((predictions == class_id) & (labels != class_id)).sum())
        false_negative = int(((predictions != class_id) & (labels == class_id)).sum())
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def accuracy(labels: torch.Tensor, predictions: torch.Tensor) -> float:
    if labels.numel() == 0:
        return 0.0
    return float((labels == predictions).sum()) / float(labels.numel())


def select_checkpoint(rows: list[dict]) -> dict:
    """Choose the row with the highest validation macro F1. Ties keep the earlier epoch."""
    if not rows:
        raise ValueError("Checkpoint selection needs at least one epoch.")
    selected = rows[0]
    for row in rows[1:]:
        if float(row["validation_macro_f1"]) > float(selected["validation_macro_f1"]):
            selected = row
    return selected


def should_stop(epoch: int, epochs_without_improvement: int, min_epochs: int, patience: int) -> bool:
    """Stop only after the minimum epoch count and the patience window."""
    return epoch >= min_epochs and epochs_without_improvement >= patience
