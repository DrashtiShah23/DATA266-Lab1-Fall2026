"""Score the Phase 6 official-test prediction files.

This module reads those files and does not train, and it does not replace a
selected checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from lab1.paths import repo_root, resolve_repo_path
from task2_sentiment.drashti.src.acquire import load_yelp_polarity
from task2_sentiment.drashti.src.eval_plots import write_confusion_svg, write_curve_svg, write_reliability_svg
from task2_sentiment.drashti.src.interface import original_text, parse_example_id
from task2_sentiment.drashti.src.metrics import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DECISION_THRESHOLD,
    ECE_BINS,
    average_precision,
    brier_score,
    bootstrap_intervals,
    classification_metrics,
    confusion_counts,
    expected_calibration_error,
    matthews_correlation,
    mcnemar_test,
    precision_recall_curve,
    roc_auc,
    roc_curve,
    slice_metrics,
)
from task2_sentiment.drashti.src.predictions import PREDICTION_FIELDS
from task2_sentiment.drashti.src.preprocess import preprocess_text
from task2_sentiment.drashti.src.review_cases import select_review_cases, select_review_model
from task2_sentiment.drashti.src.settings import load_task2_settings

PREDICTION_FILES = (
    ("baseline", "task2_sentiment/drashti/outputs/predictions/task2_drashti_baseline_test_predictions.csv"),
    ("experimental_1", "task2_sentiment/drashti/outputs/predictions/task2_drashti_cnn_test_predictions.csv"),
    ("experimental_2", "task2_sentiment/drashti/outputs/predictions/task2_drashti_gru_test_predictions.csv"),
)
EXPECTED_SLICE_COUNTS = {"short": 9911, "medium": 18751, "long": 9338, "contains_negation": 30968}
OUTPUT_DIR = "task2_sentiment/drashti/outputs/evaluation"
COLORS = {"baseline": "#1f4e79", "experimental_1": "#c47b2b", "experimental_2": "#2f6f4e"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_table(path: Path) -> dict[str, Any]:
    ids: list[str] = []
    source: list[int] = []
    true_label: list[int] = []
    predicted: list[int] = []
    positive: list[float] = []
    negative: list[float] = []
    token_length: list[int] = []
    content_length: list[int] = []
    length_slice: list[str] = []
    negation: list[int] = []
    checkpoints: list[str] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != PREDICTION_FIELDS:
            raise RuntimeError(f"{path.name} does not have the Phase 6 prediction columns.")
        for row in reader:
            ids.append(row["example_id"])
            source.append(int(row["source_index"]))
            true_label.append(int(row["true_label"]))
            predicted.append(int(row["predicted_label"]))
            positive.append(float(row["positive_probability"]))
            negative.append(float(row["negative_probability"]))
            token_length.append(int(row["processed_token_length"]))
            content_length.append(int(row["content_length"]))
            length_slice.append(row["length_slice"])
            negation.append(int(row["contains_negation"]))
            checkpoints.append(row["checkpoint"])
    return {
        "example_id": ids,
        "source_index": np.asarray(source, dtype=np.int64),
        "true_label": np.asarray(true_label, dtype=np.int64),
        "predicted_label": np.asarray(predicted, dtype=np.int64),
        "positive_probability": np.asarray(positive, dtype=np.float64),
        "negative_probability": np.asarray(negative, dtype=np.float64),
        "processed_token_length": np.asarray(token_length, dtype=np.int64),
        "content_length": np.asarray(content_length, dtype=np.int64),
        "length_slice": length_slice,
        "contains_negation": np.asarray(negation, dtype=np.int64),
        "checkpoint": checkpoints,
    }


def _audit(
    tables: dict[str, dict[str, Any]],
    expected_rows: int = 38000,
    expected_slices: dict[str, int] | None = None,
) -> dict[str, Any]:
    names = list(tables)
    first = tables[names[0]]
    if len(first["example_id"]) != expected_rows:
        raise RuntimeError(f"The first prediction file does not have {expected_rows} rows.")
    if len(set(first["example_id"])) != expected_rows:
        raise RuntimeError("Prediction ids are not unique.")
    report: dict[str, Any] = {"models": {}}
    for name, table in tables.items():
        if table["example_id"] != first["example_id"]:
            raise RuntimeError(f"{name} ids or ordering do not match the baseline file.")
        if not np.array_equal(table["true_label"], first["true_label"]):
            raise RuntimeError(f"{name} ground-truth labels do not match.")
        if not np.array_equal(table["source_index"], first["source_index"]):
            raise RuntimeError(f"{name} source indexes do not match.")
        labels_ok = set(np.unique(table["predicted_label"])).issubset({0, 1})
        truth_ok = set(np.unique(table["true_label"])).issubset({0, 1})
        positive = table["positive_probability"]
        negative = table["negative_probability"]
        finite = bool(np.isfinite(positive).all() and np.isfinite(negative).all())
        bounded = bool(((positive >= 0) & (positive <= 1) & (negative >= 0) & (negative <= 1)).all())
        paired = bool(np.max(np.abs(positive + negative - 1.0)) <= 1e-6)
        threshold_match = bool(
            np.array_equal(table["predicted_label"], (positive >= DECISION_THRESHOLD).astype(np.int64))
        )
        checkpoint_ok = all(item.strip() for item in table["checkpoint"])
        slices = table["length_slice"]
        if any(item not in {"short", "medium", "long"} for item in slices):
            raise RuntimeError(f"{name} has a length slice outside short, medium, and long.")
        counts = {item: slices.count(item) for item in ("short", "medium", "long")}
        counts["contains_negation"] = int(np.sum(table["contains_negation"] == 1))
        required_slices = EXPECTED_SLICE_COUNTS if expected_slices is None else expected_slices
        if counts != required_slices:
            raise RuntimeError(f"{name} slice counts {counts} do not match the required counts.")
        if not all([labels_ok, truth_ok, finite, bounded, paired, threshold_match, checkpoint_ok]):
            raise RuntimeError(f"{name} failed a prediction integrity check.")
        report["models"][name] = {
            "rows": expected_rows,
            "unique_ids": expected_rows,
            "checkpoint": table["checkpoint"][0],
            "slice_counts": counts,
        }
    report["aligned"] = True
    report["same_ground_truth"] = True
    report["first_example_id"] = first["example_id"][0]
    report["last_example_id"] = first["example_id"][-1]
    return report


def _efficiency(repo: Path, prediction_file: str) -> dict[str, Any]:
    matches = []
    output_dir = repo / "task2_sentiment" / "drashti" / "outputs"
    for path in sorted(output_dir.glob("*_summary_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("prediction_file") == prediction_file:
            matches.append((path, payload))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Phase 6 summary for {prediction_file}.")
    path, payload = matches[0]
    return {
        "summary_file": path.relative_to(repo).as_posix(),
        "parameter_count": int(payload["parameter_count"]),
        "training_seconds": float(payload["training_seconds"]),
        "examples_per_second": float(payload["examples_per_second"]),
        "peak_rss_bytes": int(payload["peak_rss_bytes"]),
        "selected_epoch": int(payload["selected_epoch"]),
        "checkpoint": payload["checkpoint"],
        "hardware": payload["hardware"],
        "device_used": payload["device_used"],
    }


def _records(table: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, example_id in enumerate(table["example_id"]):
        rows.append(
            {
                "example_id": example_id,
                "true_label": int(table["true_label"][index]),
                "predicted_label": int(table["predicted_label"][index]),
                "positive_probability": float(table["positive_probability"][index]),
                "negative_probability": float(table["negative_probability"][index]),
                "processed_token_length": int(table["processed_token_length"][index]),
                "content_length": int(table["content_length"][index]),
                "length_slice": table["length_slice"][index],
                "contains_negation": int(table["contains_negation"][index]),
            }
        )
    return rows


def _text_is_storable(text: str) -> bool:
    personal = ("/" + "Users" + "/", "/" + "home" + "/")
    return not any(marker in text for marker in personal)


def _attach_text(cases: list[dict[str, Any]], dataset: Any) -> list[str]:
    skipped: list[str] = []
    for case in cases:
        split, _index = parse_example_id(case["example_id"])
        if split != "test":
            raise RuntimeError("Error review cases must come from the official test split.")
        text = original_text(dataset, case["example_id"], "text")
        if not _text_is_storable(text):
            skipped.append(case["example_id"])
            continue
        case["original_review"] = text
        case["processed_text"] = " ".join(preprocess_text(text))
    return skipped


def _write_curve_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    keys = list(columns)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(keys)
        for index in range(columns[keys[0]].shape[0]):
            writer.writerow([f"{float(columns[key][index]):.17g}" for key in keys])


def _score_model(name: str, table: dict[str, Any], efficiency: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    truth = table["true_label"]
    pred = table["predicted_label"]
    positive = table["positive_probability"]
    negative = table["negative_probability"]
    metrics = classification_metrics(truth, pred)
    counts = confusion_counts(truth, pred)
    roc = roc_auc(truth, positive)
    pr_area = average_precision(truth, positive)
    brier = brier_score(truth, positive)
    mcc = matthews_correlation(counts)
    calibration = expected_calibration_error(truth, pred, positive, negative, ECE_BINS)
    print(f"BOOTSTRAP {name}", flush=True)
    bootstrap = bootstrap_intervals(truth, pred, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    slices = {}
    for slice_name in ("short", "medium", "long"):
        mask = np.asarray([item == slice_name for item in table["length_slice"]])
        slices[slice_name] = slice_metrics(truth[mask], pred[mask])
    negation_mask = table["contains_negation"] == 1
    slices["contains_negation"] = slice_metrics(truth[negation_mask], pred[negation_mask])
    slices["empty_after_preprocessing"] = slice_metrics(truth[:0], pred[:0])
    curve = roc_curve(truth, positive)
    pr_curve = precision_recall_curve(truth, positive)
    _write_curve_csv(output_dir / f"roc_{name}.csv", curve)
    _write_curve_csv(output_dir / f"pr_{name}.csv", pr_curve)
    write_confusion_svg(output_dir / f"confusion_{name}.svg", counts, f"{name} confusion matrix")
    write_reliability_svg(output_dir / f"reliability_{name}.svg", calibration["bin_rows"], f"{name} reliability")
    (output_dir / f"calibration_{name}.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / f"confusion_{name}.json").write_text(
        json.dumps(
            {
                "row_meaning": "true label",
                "column_meaning": "predicted label",
                "label_order": ["negative_0", "positive_1"],
                "matrix": [
                    [counts["true_negative"], counts["false_positive"]],
                    [counts["false_negative"], counts["true_positive"]],
                ],
                **counts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with (output_dir / f"bootstrap_{name}.csv").open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["resample", "accuracy", "f1_macro", "mcc"])
        for index in range(bootstrap["resamples"]):
            writer.writerow(
                [
                    index,
                    f"{float(bootstrap['accuracy_samples'][index]):.17g}",
                    f"{float(bootstrap['f1_macro_samples'][index]):.17g}",
                    f"{float(bootstrap['mcc_samples'][index]):.17g}",
                ]
            )
    summary = {key: value for key, value in bootstrap.items() if not key.endswith("_samples")}
    (output_dir / f"bootstrap_{name}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "model": name,
        "metrics": metrics,
        "counts": counts,
        "roc_auc": roc,
        "average_precision": pr_area,
        "brier": brier,
        "mcc": mcc,
        "ece": calibration["expected_calibration_error"],
        "bootstrap": summary,
        "slices": slices,
        "efficiency": efficiency,
        "roc_curve": curve,
        "pr_curve": pr_curve,
    }


def _write_metrics_csv(path: Path, scored: list[dict[str, Any]]) -> None:
    fields = [
        "model",
        "checkpoint",
        "accuracy",
        "precision_macro",
        "precision_micro",
        "precision_weighted",
        "recall_macro",
        "recall_micro",
        "recall_weighted",
        "f1_macro",
        "f1_micro",
        "f1_weighted",
        "roc_auc",
        "pr_average_precision",
        "mcc",
        "brier",
        "ece",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "parameter_count",
        "training_seconds",
        "examples_per_second",
        "peak_rss_bytes",
        "accuracy_ci_low",
        "accuracy_ci_high",
        "f1_macro_ci_low",
        "f1_macro_ci_high",
        "mcc_ci_low",
        "mcc_ci_high",
        "bootstrap_seed",
        "bootstrap_resamples",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for item in scored:
            metrics = item["metrics"]
            counts = item["counts"]
            efficiency = item["efficiency"]
            bootstrap = item["bootstrap"]
            writer.writerow(
                {
                    "model": item["model"],
                    "checkpoint": efficiency["checkpoint"],
                    "accuracy": f"{metrics['accuracy']:.17g}",
                    "precision_macro": f"{metrics['precision_macro']:.17g}",
                    "precision_micro": f"{metrics['precision_micro']:.17g}",
                    "precision_weighted": f"{metrics['precision_weighted']:.17g}",
                    "recall_macro": f"{metrics['recall_macro']:.17g}",
                    "recall_micro": f"{metrics['recall_micro']:.17g}",
                    "recall_weighted": f"{metrics['recall_weighted']:.17g}",
                    "f1_macro": f"{metrics['f1_macro']:.17g}",
                    "f1_micro": f"{metrics['f1_micro']:.17g}",
                    "f1_weighted": f"{metrics['f1_weighted']:.17g}",
                    "roc_auc": f"{item['roc_auc']:.17g}",
                    "pr_average_precision": f"{item['average_precision']:.17g}",
                    "mcc": f"{item['mcc']:.17g}",
                    "brier": f"{item['brier']:.17g}",
                    "ece": f"{item['ece']:.17g}",
                    "true_negative": counts["true_negative"],
                    "false_positive": counts["false_positive"],
                    "false_negative": counts["false_negative"],
                    "true_positive": counts["true_positive"],
                    "parameter_count": efficiency["parameter_count"],
                    "training_seconds": f"{efficiency['training_seconds']:.17g}",
                    "examples_per_second": f"{efficiency['examples_per_second']:.17g}",
                    "peak_rss_bytes": efficiency["peak_rss_bytes"],
                    "accuracy_ci_low": f"{bootstrap['accuracy']['lower']:.17g}",
                    "accuracy_ci_high": f"{bootstrap['accuracy']['upper']:.17g}",
                    "f1_macro_ci_low": f"{bootstrap['f1_macro']['lower']:.17g}",
                    "f1_macro_ci_high": f"{bootstrap['f1_macro']['upper']:.17g}",
                    "mcc_ci_low": f"{bootstrap['mcc']['lower']:.17g}",
                    "mcc_ci_high": f"{bootstrap['mcc']['upper']:.17g}",
                    "bootstrap_seed": bootstrap["seed"],
                    "bootstrap_resamples": bootstrap["resamples"],
                }
            )


def _write_slice_csv(path: Path, scored: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["model", "slice", "count", "f1_macro", "error_rate", "overlaps_length_slices"])
        for item in scored:
            for slice_name, values in item["slices"].items():
                writer.writerow(
                    [
                        item["model"],
                        slice_name,
                        values["count"],
                        "" if values["f1_macro"] is None else f"{values['f1_macro']:.17g}",
                        "" if values["error_rate"] is None else f"{values['error_rate']:.17g}",
                        "yes" if slice_name == "contains_negation" else "no",
                    ]
                )


def _write_error_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = [
        "review_group",
        "example_id",
        "original_review",
        "processed_text",
        "true_label",
        "predicted_label",
        "positive_probability",
        "confidence",
        "distance_from_threshold",
        "processed_token_length",
        "content_length",
        "length_slice",
        "contains_negation",
        "slice_memberships",
        "selection_reason",
        "error_type",
        "observation",
        "proposed_fix",
    ]
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for case in cases:
            membership = [case["length_slice"]]
            if int(case["contains_negation"]) == 1:
                membership.append("contains_negation")
            confidence = case["positive_probability"] if int(case["predicted_label"]) == 1 else case["negative_probability"]
            writer.writerow(
                {
                    "review_group": case["review_group"],
                    "example_id": case["example_id"],
                    "original_review": case["original_review"],
                    "processed_text": case["processed_text"],
                    "true_label": case["true_label"],
                    "predicted_label": case["predicted_label"],
                    "positive_probability": f"{case['positive_probability']:.8f}",
                    "confidence": f"{confidence:.8f}",
                    "distance_from_threshold": f"{case['distance_from_threshold']:.8f}",
                    "processed_token_length": case["processed_token_length"],
                    "content_length": case["content_length"],
                    "length_slice": case["length_slice"],
                    "contains_negation": case["contains_negation"],
                    "slice_memberships": "|".join(membership),
                    "selection_reason": case["selection_reason"],
                    "error_type": "",
                    "observation": "",
                    "proposed_fix": "",
                }
            )


def _indent(text: str) -> str:
    return "\n".join(f"    {line}" for line in text.splitlines())


def _write_failure_analysis(path: Path, model_name: str, rule: str, cases: list[dict[str, Any]]) -> None:
    lines = [
        "# Task 2 error review",
        "",
        f"Review model: `{model_name}`.",
        "",
        "Selection rule: the review model is the one with the highest official-test macro F1. "
        "Ties would keep the earlier model in the order baseline, experimental_1, experimental_2. "
        "Checkpoint selection is unchanged.",
        "",
        rule,
        "",
        "The four groups are disjoint. Confident false positives are the five highest positive probabilities "
        "among negative reviews predicted positive. Confident false negatives are the five lowest positive "
        "probabilities among positive reviews predicted negative. Near-threshold cases are the five remaining "
        "misclassifications closest to 0.5. Slice cases then take the highest-confidence remaining error from "
        "short, medium, long, and contains_negation, plus one more from the length slice with the highest error rate.",
        "",
        "Positive class: integer 1. Negative class: integer 0. Threshold: 0.5 on the positive-class probability.",
        "",
    ]
    for number, case in enumerate(cases, start=1):
        membership = case["length_slice"]
        if int(case["contains_negation"]) == 1:
            membership += ", contains_negation"
        confidence = case["positive_probability"] if int(case["predicted_label"]) == 1 else case["negative_probability"]
        lines.extend(
            [
                f"## Case {number}: {case['review_group']}",
                "",
                f"Example id: `{case['example_id']}`.",
                f"True label: {case['true_label']}. Predicted label: {case['predicted_label']}.",
                f"Positive probability: {case['positive_probability']:.8f}.",
                f"Confidence of the predicted class: {confidence:.8f}.",
                f"Distance from threshold 0.5: {case['distance_from_threshold']:.8f}.",
                f"Processed token length: {case['processed_token_length']}. Content length seen by the model: {case['content_length']}.",
                f"Slices: {membership}.",
                f"Why this case was selected: {case['selection_reason']}",
                "",
                "Original review:",
                "",
                _indent(case["original_review"]),
                "",
                "Processed text:",
                "",
                _indent(case["processed_text"]),
                "",
                "YOUR INPUT REQUIRED: Error type",
                "",
                "YOUR INPUT REQUIRED: Observation",
                "",
                "YOUR INPUT REQUIRED: One testable fix",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def _update_results(repo: Path, scored: list[dict[str, Any]], comparisons: list[dict[str, Any]], review_model: str) -> None:
    path = repo / "task2_sentiment" / "drashti" / "results.md"
    existing = path.read_text(encoding="utf-8")
    marker = "## Official test evaluation"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    else:
        existing = existing.split("## Strengths", 1)[0].rstrip() + "\n\n"
    lines = [
        marker,
        "",
        "Population: 38,000 official test rows. Positive class is integer 1. Negative class is integer 0. "
        "Hard labels use positive probability >= 0.5. These figures did not change the selected checkpoints.",
        "",
        "PR area is average precision, the step-function area under the precision-recall curve. "
        "It is not trapezoidal integration. ECE uses 15 equal-width bins on the probability of the predicted class. "
        "Bootstrap intervals are percentile intervals from 1,000 resamples with numpy PCG64 seed 266.",
        "",
        "| Metric | baseline | experimental_1 | experimental_2 |",
        "| --- | --- | --- | --- |",
    ]
    rows = [
        ("Accuracy", "accuracy"),
        ("Precision macro", "precision_macro"),
        ("Precision micro", "precision_micro"),
        ("Precision weighted", "precision_weighted"),
        ("Recall macro", "recall_macro"),
        ("Recall micro", "recall_micro"),
        ("Recall weighted", "recall_weighted"),
        ("F1 macro", "f1_macro"),
        ("F1 micro", "f1_micro"),
        ("F1 weighted", "f1_weighted"),
    ]
    for label, key in rows:
        lines.append("| " + label + " | " + " | ".join(_fmt(item["metrics"][key]) for item in scored) + " |")
    lines.append("| ROC AUC | " + " | ".join(_fmt(item["roc_auc"]) for item in scored) + " |")
    lines.append("| PR average precision | " + " | ".join(_fmt(item["average_precision"]) for item in scored) + " |")
    lines.append("| MCC | " + " | ".join(_fmt(item["mcc"]) for item in scored) + " |")
    lines.append("| Brier | " + " | ".join(_fmt(item["brier"]) for item in scored) + " |")
    lines.append("| ECE | " + " | ".join(_fmt(item["ece"]) for item in scored) + " |")
    lines.append(
        "| Parameters | " + " | ".join(str(item["efficiency"]["parameter_count"]) for item in scored) + " |"
    )
    lines.append(
        "| Training seconds | " + " | ".join(f"{item['efficiency']['training_seconds']:.3f}" for item in scored) + " |"
    )
    lines.append(
        "| Examples per second | "
        + " | ".join(f"{item['efficiency']['examples_per_second']:.3f}" for item in scored)
        + " |"
    )
    lines.append("| Peak RSS bytes | " + " | ".join(str(item["efficiency"]["peak_rss_bytes"]) for item in scored) + " |")
    lines.extend(["", "### Confusion counts", ""])
    for item in scored:
        counts = item["counts"]
        lines.append(
            f"{item['model']}: TN {counts['true_negative']}, FP {counts['false_positive']}, "
            f"FN {counts['false_negative']}, TP {counts['true_positive']}."
        )
    lines.extend(["", "### Bootstrap 95 percent intervals", ""])
    for item in scored:
        boot = item["bootstrap"]
        lines.append(
            f"{item['model']}: accuracy [{_fmt(boot['accuracy']['lower'])}, {_fmt(boot['accuracy']['upper'])}], "
            f"macro F1 [{_fmt(boot['f1_macro']['lower'])}, {_fmt(boot['f1_macro']['upper'])}], "
            f"MCC [{_fmt(boot['mcc']['lower'])}, {_fmt(boot['mcc']['upper'])}]."
        )
    lines.extend(["", "### McNemar", ""])
    for item in comparisons:
        lines.append(
            f"{item['comparison']}: both correct {item['both_correct']}, "
            f"baseline only {item['baseline_correct_experimental_wrong']}, "
            f"experimental only {item['baseline_wrong_experimental_correct']}, "
            f"both wrong {item['both_wrong']}, statistic {item['statistic']:.6f}, p-value {item['p_value']:.6g}."
        )
        lines.append("")
        lines.append("YOUR INPUT REQUIRED: Interpret this McNemar result in the context of the model comparison.")
        lines.append("")
    lines.extend(["### Robustness slices", ""])
    lines.append("| Model | Slice | Count | Macro F1 | Error rate |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in scored:
        for slice_name, values in item["slices"].items():
            f1 = "" if values["f1_macro"] is None else _fmt(values["f1_macro"])
            error = "" if values["error_rate"] is None else _fmt(values["error_rate"])
            lines.append(f"| {item['model']} | {slice_name} | {values['count']} | {f1} | {error} |")
    lines.extend(
        [
            "",
            "contains_negation overlaps the length slices. short, medium, and long partition the 38,000 test rows. "
            "empty_after_preprocessing has 0 official test rows, so it has no F1 or error rate.",
            "",
            f"Error-review model: `{review_model}`. The 20 cases are in `task2_sentiment/drashti/failure_analysis.md` "
            "and `task2_sentiment/drashti/outputs/error_review_20.csv`.",
            "",
            "Evidence directory: `task2_sentiment/drashti/outputs/evaluation/`. "
            "The metric table is `task2_sentiment/drashti/metrics_report.csv`.",
            "",
            "## Strengths",
            "",
            "YOUR INPUT REQUIRED",
            "",
            "## Weaknesses",
            "",
            "YOUR INPUT REQUIRED",
            "",
            "## Limitations",
            "",
            "YOUR INPUT REQUIRED",
            "",
            "## Comparative conclusions",
            "",
            "YOUR INPUT REQUIRED",
            "",
        ]
    )
    path.write_text(existing + "\n".join(lines), encoding="utf-8")


def _update_viva(repo: Path, scored: list[dict[str, Any]]) -> None:
    path = repo / "task2_sentiment" / "drashti" / "VIVA_PREP.md"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "Final precision, recall, confusion matrices, ROC, calibration, bootstrap intervals, and paired tests are not computed in this note.",
        "The official-test metrics below were computed after checkpoint selection.",
    )
    marker = "## Official test metrics"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip() + "\n\n"
    lines = [
        marker,
        "",
        "Accuracy is the fraction of test reviews whose predicted label matches the true label.",
        "Precision for a class is the fraction of predictions of that class that are correct. Recall is the fraction of true members of that class that were found. F1 is the harmonic mean of precision and recall.",
        "Macro averaging takes the unweighted mean of the negative-class and positive-class scores. Micro averaging pools the class counts; for this single-label problem it equals accuracy. Weighted averaging weights each class by its number of true reviews.",
        "The confusion matrix rows are true labels and the columns are predicted labels, ordered negative then positive. True negatives and true positives are the diagonal.",
        "ROC AUC is the Mann-Whitney probability that a random positive review receives a higher positive-class probability than a random negative review. Ties use average ranks. The curve uses those probabilities, not the hard labels.",
        "The reported PR area is average precision: the step-function area under precision versus recall. It is not a trapezoidal integral.",
        "MCC is (TP*TN - FP*FN) divided by the square root of the four confusion-margin products.",
        "Brier score is the mean squared difference between the positive-class probability and the binary true label.",
        "ECE puts the probability of the predicted class into 15 equal-width bins. Each bin contributes its example share times the absolute gap between empirical accuracy and mean confidence.",
        "A bootstrap interval resamples the 38,000 paired examples with replacement 1,000 times, seed 266, numpy PCG64, and reads the 2.5 and 97.5 linear percentiles.",
        "McNemar compares paired correctness. The reported statistic uses a continuity correction and a chi-square reference distribution with one degree of freedom. The interpretation of the p-value is left for the student.",
        "Length slices partition the test set. contains_negation overlaps them. Slice macro F1 and error rate use the same definitions as the full test set. An empty slice is reported with count 0 and no invented score.",
        "Parameter count, training time, examples per second, and peak memory are copied from the Phase 6 summary that names the same prediction file.",
        "",
        "Measured official-test macro F1: "
        + ", ".join(f"{item['model']} {_fmt(item['metrics']['f1_macro'])}" for item in scored)
        + ".",
        "",
        "## Qualitative comparison",
        "",
        "YOUR INPUT REQUIRED",
        "",
    ]
    if "## Qualitative comparison" in text:
        text = text.split("## Qualitative comparison", 1)[0].rstrip() + "\n\n"
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def run_evaluation(repo: Path | None = None) -> dict[str, Any]:
    """Evaluate the three aligned prediction files and write the evidence."""
    root = repo or repo_root()
    tracked_inputs = [resolve_repo_path(path, repo=root) for _name, path in PREDICTION_FILES]
    tables = {name: _load_table(resolve_repo_path(path, repo=root)) for name, path in PREDICTION_FILES}
    for name, table in tables.items():
        tracked_inputs.append(resolve_repo_path(table["checkpoint"][0], repo=root))
    before = {path.as_posix(): _sha256(path) for path in tracked_inputs}
    audit = _audit(tables)
    output_dir = resolve_repo_path(OUTPUT_DIR, repo=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored = []
    for name, relative in PREDICTION_FILES:
        print(f"SCORE {name}", flush=True)
        efficiency = _efficiency(root, relative)
        if efficiency["checkpoint"] != tables[name]["checkpoint"][0]:
            raise RuntimeError(f"{name} prediction checkpoint does not match its Phase 6 summary.")
        scored.append(_score_model(name, tables[name], efficiency, output_dir))
    write_curve_svg(
        output_dir / "roc_comparison.svg",
        [
            (item["model"], item["roc_curve"]["false_positive_rate"], item["roc_curve"]["true_positive_rate"], COLORS[item["model"]])
            for item in scored
        ],
        "ROC curves",
        "False positive rate",
        "True positive rate",
        True,
    )
    write_curve_svg(
        output_dir / "pr_comparison.svg",
        [
            (item["model"], item["pr_curve"]["recall"], item["pr_curve"]["precision"], COLORS[item["model"]])
            for item in scored
        ],
        "Precision recall curves",
        "Recall",
        "Precision",
        False,
    )
    for item in scored:
        write_curve_svg(
            output_dir / f"roc_{item['model']}.svg",
            [(item["model"], item["roc_curve"]["false_positive_rate"], item["roc_curve"]["true_positive_rate"], COLORS[item["model"]])],
            f"{item['model']} ROC",
            "False positive rate",
            "True positive rate",
            True,
        )
        write_curve_svg(
            output_dir / f"pr_{item['model']}.svg",
            [(item["model"], item["pr_curve"]["recall"], item["pr_curve"]["precision"], COLORS[item["model"]])],
            f"{item['model']} precision recall",
            "Recall",
            "Precision",
            False,
        )
        item.pop("roc_curve")
        item.pop("pr_curve")
    comparisons = []
    baseline_correct = tables["baseline"]["predicted_label"] == tables["baseline"]["true_label"]
    for name in ("experimental_1", "experimental_2"):
        other_correct = tables[name]["predicted_label"] == tables[name]["true_label"]
        result = mcnemar_test(baseline_correct, other_correct)
        result["comparison"] = f"baseline versus {name}"
        comparisons.append(result)
    (output_dir / "mcnemar.json").write_text(json.dumps(comparisons, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_metrics_csv(root / "task2_sentiment" / "drashti" / "metrics_report.csv", scored)
    _write_slice_csv(output_dir / "slice_metrics.csv", scored)
    review_model = select_review_model([(item["model"], item["metrics"]["f1_macro"]) for item in scored])
    review_scores = next(item for item in scored if item["model"] == review_model)
    rates = {name: float(review_scores["slices"][name]["error_rate"]) for name in ("short", "medium", "long")}
    blocked: set[str] = set()
    dataset = None
    cases: list[dict[str, Any]] = []
    for _attempt in range(3):
        cases = select_review_cases(_records(tables[review_model]), rates, blocked)
        if dataset is None:
            dataset = load_yelp_polarity(load_task2_settings(repo=root), root)
        skipped = _attach_text(cases, dataset)
        if not skipped:
            break
        blocked.update(skipped)
        cases = []
    if len(cases) != 20 or any("original_review" not in case for case in cases):
        raise RuntimeError("The 20 review texts could not be stored.")
    _write_error_csv(root / "task2_sentiment" / "drashti" / "outputs" / "error_review_20.csv", cases)
    _write_failure_analysis(
        root / "task2_sentiment" / "drashti" / "failure_analysis.md",
        review_model,
        f"This run selected `{review_model}` because its official-test macro F1 was highest.",
        cases,
    )
    procedure = {
        "positive_label": 1,
        "negative_label": 0,
        "positive_probability_column": "positive_probability",
        "decision_threshold": DECISION_THRESHOLD,
        "threshold_rule": "predict positive when positive_probability >= 0.5",
        "pr_definition": "step-function average precision, not trapezoidal integration",
        "ece_bins": ECE_BINS,
        "ece_confidence": "probability of the predicted class",
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "review_model": review_model,
        "review_model_rule": "highest official-test macro F1; earlier name wins a tie",
        "checkpoint_selection": "unchanged from Phase 6 validation macro F1",
    }
    (output_dir / "evaluation_procedure.json").write_text(
        json.dumps(procedure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "prediction_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _update_results(root, scored, comparisons, review_model)
    _update_viva(root, scored)
    after = {path.as_posix(): _sha256(path) for path in tracked_inputs}
    if after != before:
        raise RuntimeError("Evaluation changed a prediction file.")
    print(f"REVIEW_MODEL={review_model}", flush=True)
    print("REAL_EVALUATION=PASS", flush=True)
    return {"review_model": review_model, "audit": audit}


def main() -> int:
    run_evaluation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
