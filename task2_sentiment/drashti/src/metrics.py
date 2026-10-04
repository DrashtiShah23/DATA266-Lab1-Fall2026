"""Task 2 test metrics.

These functions score stored predictions. They do not train a model and they
do not choose a checkpoint.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
DECISION_THRESHOLD = 0.5
ECE_BINS = 15
BOOTSTRAP_SEED = 266
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_LOWER = 2.5
BOOTSTRAP_UPPER = 97.5


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Count a binary matrix.

    Rows are the true label and columns are the predicted label.
    Order is negative (0), then positive (1):

    [[true_negative, false_positive], [false_negative, true_positive]]
    """
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    return {
        "true_negative": int(np.sum((true == 0) & (pred == 0))),
        "false_positive": int(np.sum((true == 0) & (pred == 1))),
        "false_negative": int(np.sum((true == 1) & (pred == 0))),
        "true_positive": int(np.sum((true == 1) & (pred == 1))),
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _class_scores(true_positive: float, false_positive: float, false_negative: float) -> dict[str, float]:
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Accuracy and macro, micro, and weighted precision, recall, and F1."""
    counts = confusion_counts(y_true, y_pred)
    negative = _class_scores(
        counts["true_negative"],
        counts["false_negative"],
        counts["false_positive"],
    )
    positive = _class_scores(
        counts["true_positive"],
        counts["false_positive"],
        counts["false_negative"],
    )
    support_negative = counts["true_negative"] + counts["false_positive"]
    support_positive = counts["true_positive"] + counts["false_negative"]
    total = support_negative + support_positive
    correct = counts["true_negative"] + counts["true_positive"]
    accuracy = _safe_divide(correct, total)
    return {
        "accuracy": accuracy,
        "precision_macro": (negative["precision"] + positive["precision"]) / 2,
        "precision_micro": accuracy,
        "precision_weighted": _safe_divide(
            negative["precision"] * support_negative + positive["precision"] * support_positive,
            total,
        ),
        "recall_macro": (negative["recall"] + positive["recall"]) / 2,
        "recall_micro": accuracy,
        "recall_weighted": _safe_divide(
            negative["recall"] * support_negative + positive["recall"] * support_positive,
            total,
        ),
        "f1_macro": (negative["f1"] + positive["f1"]) / 2,
        "f1_micro": accuracy,
        "f1_weighted": _safe_divide(negative["f1"] * support_negative + positive["f1"] * support_positive, total),
        "support_negative": float(support_negative),
        "support_positive": float(support_positive),
    }


def matthews_correlation(counts: dict[str, int]) -> float:
    """Matthews correlation from the four confusion counts."""
    true_positive = counts["true_positive"]
    true_negative = counts["true_negative"]
    false_positive = counts["false_positive"]
    false_negative = counts["false_negative"]
    numerator = true_positive * true_negative - false_positive * false_negative
    denominator = math.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    if denominator == 0:
        return 0.0
    return numerator / denominator


def brier_score(y_true: np.ndarray, positive_probability: np.ndarray) -> float:
    """Mean squared error of the positive-class probability."""
    truth = np.asarray(y_true, dtype=np.float64)
    probability = np.asarray(positive_probability, dtype=np.float64)
    return float(np.mean((probability - truth) ** 2))


def roc_auc(y_true: np.ndarray, positive_probability: np.ndarray) -> float:
    """Area under the ROC curve from positive-class probabilities.

    Ties use average ranks. This is the Mann-Whitney form, not a hard-label score.
    """
    truth = np.asarray(y_true)
    scores = np.asarray(positive_probability, dtype=np.float64)
    positives = int(np.sum(truth == 1))
    negatives = int(truth.shape[0] - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    ranks = np.empty(sorted_scores.shape[0], dtype=np.float64)
    start = 0
    while start < sorted_scores.shape[0]:
        stop = start
        while stop + 1 < sorted_scores.shape[0] and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[start : stop + 1] = (start + 1 + stop + 1) / 2.0
        start = stop + 1
    positive_rank_sum = float(ranks[sorted_truth == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(y_true: np.ndarray, positive_probability: np.ndarray) -> float:
    """Step-function area under the precision-recall curve.

    This is average precision: each new true positive contributes its precision
    times the recall step. It is not trapezoidal integration of the curve.
    """
    truth = np.asarray(y_true)
    scores = np.asarray(positive_probability, dtype=np.float64)
    positives = int(np.sum(truth == 1))
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ranked = truth[order]
    hits = 0
    total = 0.0
    for index, label in enumerate(ranked, start=1):
        if int(label) == 1:
            hits += 1
            total += hits / index
    return total / positives


def roc_curve(y_true: np.ndarray, positive_probability: np.ndarray) -> dict[str, np.ndarray]:
    """False-positive and true-positive rates at each distinct positive probability."""
    truth = np.asarray(y_true)
    scores = np.asarray(positive_probability, dtype=np.float64)
    positives = max(int(np.sum(truth == 1)), 1)
    negatives = max(int(truth.shape[0] - np.sum(truth == 1)), 1)
    order = np.argsort(-scores, kind="mergesort")
    ranked_truth = truth[order]
    ranked_scores = scores[order]
    distinct_false = [0]
    distinct_true = [0]
    thresholds = [float(ranked_scores[0]) + 1.0 if ranked_scores.size else 1.0]
    false_positive = 0
    true_positive = 0
    index = 0
    while index < ranked_truth.shape[0]:
        threshold = float(ranked_scores[index])
        while index < ranked_truth.shape[0] and float(ranked_scores[index]) == threshold:
            if int(ranked_truth[index]) == 1:
                true_positive += 1
            else:
                false_positive += 1
            index += 1
        distinct_false.append(false_positive / negatives)
        distinct_true.append(true_positive / positives)
        thresholds.append(threshold)
    distinct_false.append(1.0)
    distinct_true.append(1.0)
    thresholds.append(0.0)
    return {
        "threshold": np.asarray(thresholds, dtype=np.float64),
        "false_positive_rate": np.asarray(distinct_false, dtype=np.float64),
        "true_positive_rate": np.asarray(distinct_true, dtype=np.float64),
    }


def precision_recall_curve(y_true: np.ndarray, positive_probability: np.ndarray) -> dict[str, np.ndarray]:
    """Precision and recall traced from high positive probability to low."""
    truth = np.asarray(y_true)
    scores = np.asarray(positive_probability, dtype=np.float64)
    positives = max(int(np.sum(truth == 1)), 1)
    order = np.argsort(-scores, kind="mergesort")
    ranked_truth = truth[order]
    ranked_scores = scores[order]
    hits = np.cumsum(ranked_truth == 1)
    considered = np.arange(1, ranked_truth.shape[0] + 1)
    change = np.r_[True, ranked_scores[1:] != ranked_scores[:-1]]
    change[-1] = True
    return {
        "threshold": ranked_scores[change],
        "precision": (hits / considered)[change],
        "recall": (hits / positives)[change],
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_probability: np.ndarray,
    negative_probability: np.ndarray,
    bins: int = ECE_BINS,
) -> dict[str, Any]:
    """Equal-width ECE on the probability of the predicted class.

    Bin i covers [i/bins, (i+1)/bins). The final bin also includes 1.
    """
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    positive = np.asarray(positive_probability, dtype=np.float64)
    negative = np.asarray(negative_probability, dtype=np.float64)
    confidence = np.where(pred == 1, positive, negative)
    correct = (pred == truth).astype(np.float64)
    edges = [index / bins for index in range(bins + 1)]
    assigned = np.minimum(bins - 1, np.floor(confidence * bins).astype(np.int64))
    rows = []
    total = confidence.shape[0]
    score = 0.0
    for index in range(bins):
        mask = assigned == index
        count = int(np.sum(mask))
        if count == 0:
            mean_confidence = 0.0
            empirical_accuracy = 0.0
            contribution = 0.0
        else:
            mean_confidence = float(np.mean(confidence[mask]))
            empirical_accuracy = float(np.mean(correct[mask]))
            contribution = (count / total) * abs(empirical_accuracy - mean_confidence)
            score += contribution
        rows.append(
            {
                "bin": index,
                "lower": edges[index],
                "upper": edges[index + 1],
                "upper_inclusive": index == bins - 1,
                "count": count,
                "mean_confidence": mean_confidence,
                "empirical_accuracy": empirical_accuracy,
                "ece_contribution": contribution,
            }
        )
    return {"bins": bins, "expected_calibration_error": score, "bin_rows": rows}


def linear_percentile(samples: np.ndarray, percent: float) -> float:
    """Linear percentile matching the common numpy definition."""
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    count = ordered.shape[0]
    if count == 0:
        raise ValueError("Percentile needs at least one sample.")
    if count == 1:
        return float(ordered[0])
    position = (percent / 100.0) * (count - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile intervals from resampling paired test examples with replacement."""
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    count = int(truth.shape[0])
    generator = np.random.Generator(np.random.PCG64(seed))
    accuracy_samples = np.empty(resamples, dtype=np.float64)
    f1_samples = np.empty(resamples, dtype=np.float64)
    mcc_samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        chosen = generator.integers(0, count, size=count)
        metrics = classification_metrics(truth[chosen], pred[chosen])
        counts = confusion_counts(truth[chosen], pred[chosen])
        accuracy_samples[index] = metrics["accuracy"]
        f1_samples[index] = metrics["f1_macro"]
        mcc_samples[index] = matthews_correlation(counts)
    def pack(samples: np.ndarray) -> dict[str, float]:
        return {
            "lower": linear_percentile(samples, BOOTSTRAP_LOWER),
            "upper": linear_percentile(samples, BOOTSTRAP_UPPER),
        }
    return {
        "seed": seed,
        "resamples": resamples,
        "sample_size": count,
        "method": "percentile",
        "lower_percentile": BOOTSTRAP_LOWER,
        "upper_percentile": BOOTSTRAP_UPPER,
        "generator": "numpy PCG64",
        "accuracy": pack(accuracy_samples),
        "f1_macro": pack(f1_samples),
        "mcc": pack(mcc_samples),
        "accuracy_samples": accuracy_samples,
        "f1_macro_samples": f1_samples,
        "mcc_samples": mcc_samples,
    }


def mcnemar_test(baseline_correct: np.ndarray, experimental_correct: np.ndarray) -> dict[str, Any]:
    """Asymptotic McNemar test with continuity correction.

    The statistic is (max(|b-c|-1, 0))^2 / (b+c), compared with a chi-square
    distribution with one degree of freedom. b is baseline-only correct.
    c is experimental-only correct.
    """
    base = np.asarray(baseline_correct, dtype=bool)
    other = np.asarray(experimental_correct, dtype=bool)
    both_correct = int(np.sum(base & other))
    baseline_only = int(np.sum(base & ~other))
    experimental_only = int(np.sum(~base & other))
    both_wrong = int(np.sum(~base & ~other))
    discordant = baseline_only + experimental_only
    if discordant == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = (max(abs(baseline_only - experimental_only) - 1, 0) ** 2) / discordant
        p_value = math.erfc(math.sqrt(statistic / 2.0))
    return {
        "both_correct": both_correct,
        "baseline_correct_experimental_wrong": baseline_only,
        "baseline_wrong_experimental_correct": experimental_only,
        "both_wrong": both_wrong,
        "discordant_count": discordant,
        "statistic": statistic,
        "p_value": p_value,
        "method": "asymptotic chi-square with one degree of freedom",
        "continuity_correction": "subtract 1 from the absolute discordant difference before squaring; floor the result at 0",
    }


def slice_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    """Macro F1 and error rate for one slice. Empty slices do not invent a score."""
    if int(np.asarray(y_true).shape[0]) == 0:
        return {"count": 0, "f1_macro": None, "error_rate": None}
    metrics = classification_metrics(y_true, y_pred)
    return {
        "count": int(np.asarray(y_true).shape[0]),
        "f1_macro": metrics["f1_macro"],
        "error_rate": 1.0 - metrics["accuracy"],
    }
