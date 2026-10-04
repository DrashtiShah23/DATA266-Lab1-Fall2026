"""Factual Phase 6 documents. Qualitative conclusions are left unmarked for the student."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from task2_sentiment.drashti.src.audit import audit_pretrained_resources


_DIFFERENCE = {
    "mean_pool": "The baseline averages every real token embedding and classifies that one vector. It has no convolution and no recurrent state.",
    "conv": "The convolutional model applies filters of widths 3, 4, and 5, then keeps the strongest activation of each filter. It does not average the sequence and it does not use a recurrent state.",
    "gru": "The recurrent model updates a hidden state token by token and classifies the hidden state at the last real token. It does not use convolution or a masked average.",
}


def _architecture_paragraph(summary: dict[str, Any]) -> str:
    kernels = summary["kernel_sizes"]
    kernel_text = ", ".join(str(size) for size in kernels) if kernels else "none"
    return (
        f"{summary['model_name']} is the {summary['role']}. "
        f"Architecture {summary['architecture']}. "
        f"Embedding dimension {summary['embedding_dimension']}. "
        f"Hidden dimension {summary['hidden_dimension']}. "
        f"Layers {summary['num_layers']}. "
        f"Kernel sizes {kernel_text}. "
        f"Pooling {summary['pooling']}. "
        f"Dropout {summary['dropout']}. "
        f"Activation {summary['activation']}. "
        f"Classifier {summary['classifier_head']}. "
        f"Parameter count {summary['parameter_count']}. "
        f"Optimizer {summary['optimizer']}. "
        f"Learning rate {summary['learning_rate']}. "
        f"Batch size {summary['batch_size']}. "
        f"Maximum epochs {summary['max_epochs']}. "
        f"Minimum epochs {summary['min_epochs']}. "
        f"Early stopping patience {summary['early_stopping_patience']} on {summary['selection_criterion']}. "
        f"Weight decay {summary['weight_decay']} on non-embedding parameters. "
        f"Gradient clip {summary['gradient_clip_norm']}. "
        f"Decision threshold {summary['decision_threshold']} ({summary['threshold_tuned_on']}). "
        f"Seed {summary['seed']}."
    )


def _epoch_lines(summary: dict[str, Any]) -> list[str]:
    lines = [
        "| Epoch | Training loss | Validation loss | Validation accuracy | Validation macro F1 | Runtime seconds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["epochs"]:
        lines.append(
            f"| {row['epoch']} | {row['training_loss']} | {row['validation_loss']} | "
            f"{row['validation_accuracy']} | {row['validation_macro_f1']} | {row['runtime_seconds']} |"
        )
    return lines


def write_phase6_documents(
    repo: Path,
    preflight: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    aligned: bool,
) -> None:
    """Write results, viva notes, and the measured section of the implementation notes."""
    populations = preflight[0]["populations"]
    audit = audit_pretrained_resources(repo)
    results = _results(populations, preflight, summaries, aligned, audit)
    viva = _viva(populations, summaries, aligned)
    notes_path = repo / "task2_sentiment" / "drashti" / "IMPLEMENTATION_NOTES.md"
    existing = notes_path.read_text(encoding="utf-8")
    marker = "## Measured training"
    measured = _measured(populations, preflight, summaries, aligned)
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    notes_path.write_text(existing + measured, encoding="utf-8")
    (repo / "task2_sentiment" / "drashti" / "results.md").write_text(results, encoding="utf-8")
    (repo / "task2_sentiment" / "drashti" / "VIVA_PREP.md").write_text(viva, encoding="utf-8")


def _results(
    populations: dict[str, int],
    preflight: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    aligned: bool,
    audit: dict[str, str],
) -> str:
    lines = [
        "# Task 2 results",
        "",
        "These sections record the training run. Strengths, weaknesses, limitations, and comparative conclusions are not written here.",
        "",
        "## Dataset",
        "",
        "Yelp Polarity revision `bbf1c97a1f0cf005e5aded43839fd814654a1557`.",
        f"Development source rows: {populations['development_source_rows']}.",
        f"Development model-ready rows: {populations['development_model_ready_rows']}.",
        f"Validation source rows: {populations['validation_source_rows']}.",
        f"Validation model-ready rows: {populations['validation_model_ready_rows']}.",
        f"Official test rows: {populations['test_rows']}.",
        f"Excluded empty processed training rows: {populations['excluded_empty_rows']} "
        f"({populations['excluded_empty_development_rows']} development, {populations['excluded_empty_validation_rows']} validation).",
        "",
        "## Preprocessing reference",
        "",
        "The models use `configs/task2_drashti_preprocess.json`, `task2_sentiment/drashti/data_processed/vocabulary.json`, and `task2_sentiment/drashti/data_processed/model_ready.pt`.",
        "Vocabulary size 30,002. Padding id 0. Maximum length 207. Embeddings are learned from scratch.",
        "",
        "## Architectures and hyperparameters",
        "",
    ]
    for summary in summaries:
        lines.append(_architecture_paragraph(summary))
        lines.append("")
    lines.extend(
        [
            "## Training procedure",
            "",
            "Each model trains on the development model-ready rows. Validation macro F1 selects the checkpoint. The official test labels are not used for that choice, for early stopping, or for the decision threshold. The threshold is 0.5 and was not tuned.",
            "Early stopping requires at least the configured minimum epochs and then stops after two epochs without a validation macro F1 improvement, with a maximum of four epochs.",
            "",
            "## Selected epochs",
            "",
        ]
    )
    for summary in summaries:
        lines.append(
            f"{summary['model_name']}: epoch {summary['selected_epoch']}, "
            f"validation macro F1 {summary['selected_validation_macro_f1']}, "
            f"checkpoint `{summary['checkpoint']}`."
        )
    lines.extend(["", "## Hardware", ""])
    for summary in summaries:
        hardware = summary["hardware"]
        lines.append(
            f"{summary['model_name']}: device {summary['device_used']}, "
            f"processor {hardware.get('cpu_model')}, "
            f"pytorch {hardware.get('pytorch_version')}, "
            f"unified memory bytes {hardware.get('unified_memory_bytes')}."
        )
    lines.extend(["", "## Parameter counts and training time", ""])
    for summary in summaries:
        lines.append(
            f"{summary['model_name']}: {summary['parameter_count']} parameters, "
            f"training seconds {summary['training_seconds']}, "
            f"examples per second {summary['examples_per_second']}, "
            f"peak RSS bytes {summary['peak_rss_bytes']}."
        )
    lines.extend(["", "## Epoch records", ""])
    for summary in summaries:
        lines.append(f"### {summary['model_name']}")
        lines.append("")
        lines.extend(_epoch_lines(summary))
        lines.append("")
    lines.extend(["## Prediction and checkpoint locations", ""])
    for summary in summaries:
        lines.append(f"{summary['model_name']} predictions: `{summary['prediction_file']}` ({summary['prediction_rows']} rows).")
        lines.append(f"{summary['model_name']} checkpoint: `{summary['checkpoint']}`.")
        lines.append(f"{summary['model_name']} raw log: `{summary['raw_log']}`.")
        lines.append(f"{summary['model_name']} manifest: `{summary['manifest']}`.")
    lines.append("")
    lines.append(f"Aligned example-id order: {'yes' if aligned else 'no'}.")
    if summaries:
        lines.append(f"First example id: {summaries[0]['first_example_id']}. Last example id: {summaries[0]['last_example_id']}.")
    lines.extend(["", "## External resource audit", ""])
    for label, status in audit.items():
        lines.append(f"{label}: {status}")
    lines.extend(
        [
            "",
            "## Development overfit tests",
            "",
            "These figures are labeled DEVELOPMENT OVERFIT TEST. They are not final model performance.",
            "",
        ]
    )
    for report in preflight:
        overfit = report["overfit"]
        lines.append(
            f"{report['model_name']}: samples {overfit['sample_count']}, steps {overfit['optimization_steps']}, "
            f"initial loss {overfit['initial_loss']}, final loss {overfit['final_loss']}, "
            f"decreased {overfit['loss_decreased']}."
        )
    lines.extend(
        [
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
    return "\n".join(lines)


def _viva(populations: dict[str, int], summaries: list[dict[str, Any]], aligned: bool) -> str:
    lines = [
        "# Task 2 viva preparation",
        "",
        "This note explains the three trained sentiment models. It does not state final comparison conclusions.",
        "",
        "## Shared data",
        "",
        "Each review is a length-207 integer sequence from the Phase 5 vocabulary. Id 0 is padding. The content length tells the model which positions are real tokens.",
        f"Training uses {populations['development_model_ready_rows']} development rows. "
        f"Selection uses {populations['validation_model_ready_rows']} validation rows. "
        f"The official test file has {populations['test_rows']} rows.",
        f"{populations['excluded_empty_rows']} empty processed training reviews stay out of the tensors.",
        "",
        "## Why the embeddings are learned from scratch",
        "",
        "Each model creates its embedding table with a normal initializer and then trains it with the classifier. No external word vectors are copied in, and no pretrained language model produces the features.",
        "",
        "## Why the official test split is not used for model selection",
        "",
        "The test labels are the final comparison set. Using them to pick an epoch, a threshold, or an architecture would make that comparison optimistic. Epoch choice uses validation macro F1 only. The threshold stays at 0.5.",
        "",
        "## Why the prediction files line up",
        "",
        "All three files list the same official test examples in the cache order. "
        f"The example-id sequences match: {'yes' if aligned else 'no'}. "
        "A later paired comparison can therefore line up each review across models without joining on text.",
        "",
        "Final precision, recall, confusion matrices, ROC, calibration, bootstrap intervals, and paired tests are not computed in this note.",
        "",
    ]
    for summary in summaries:
        hardware = summary["hardware"]
        lines.extend(
            [
                f"## {summary['model_name']}",
                "",
                "Input representation: integer token ids and content lengths.",
                "Embedding layer: a trainable table, padding row fixed at zero.",
                _architecture_paragraph(summary),
                f"Architecture flow: {_DIFFERENCE[summary['architecture']]}",
                f"Pooling: {summary['pooling']}.",
                f"Classifier: one linear layer with two logits. The loss is cross entropy. The positive probability is the second softmax entry.",
                f"Optimizer: {summary['optimizer']} at learning rate {summary['learning_rate']}.",
                f"Parameter count: {summary['parameter_count']}.",
                f"Technical difference: {_DIFFERENCE[summary['architecture']]}",
                f"Checkpoint selection: highest validation macro F1, earliest epoch on a tie. Selected epoch {summary['selected_epoch']} with value {summary['selected_validation_macro_f1']}.",
                f"Threshold: {summary['decision_threshold']}. It was not tuned.",
                f"Hardware: {summary['device_used']}, {hardware.get('cpu_model')}, pytorch {hardware.get('pytorch_version')}, unified memory bytes {hardware.get('unified_memory_bytes')}.",
                "",
            ]
        )
    lines.extend(
        [
            "## Qualitative comparison",
            "",
            "YOUR INPUT REQUIRED",
            "",
        ]
    )
    return "\n".join(lines)


def _measured(
    populations: dict[str, int],
    preflight: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    aligned: bool,
) -> str:
    lines = [
        "## Measured training",
        "",
        "The figures below come from the real development, validation, and test tensors. Overfit losses are development sanity checks, not final scores.",
        "",
        f"Development source rows {populations['development_source_rows']}, model-ready {populations['development_model_ready_rows']}.",
        f"Validation source rows {populations['validation_source_rows']}, model-ready {populations['validation_model_ready_rows']}.",
        f"Official test rows {populations['test_rows']}. Excluded empty rows {populations['excluded_empty_rows']}.",
        f"Aligned test prediction ids: {'yes' if aligned else 'no'}.",
        "",
    ]
    for report, summary in zip(preflight, summaries):
        benchmark = report["benchmark"]
        overfit = report["overfit"]
        lines.append(
            f"{summary['model_name']} parameters {summary['parameter_count']}. "
            f"Benchmark device {benchmark['device']}, examples per second {benchmark['examples_per_second']}, "
            f"estimated runtime seconds {benchmark['estimated_runtime_seconds']}, gate {benchmark['gate']}."
        )
        lines.append(
            f"{summary['model_name']} overfit initial {overfit['initial_loss']} final {overfit['final_loss']}."
        )
        lines.append(
            f"{summary['model_name']} selected epoch {summary['selected_epoch']} "
            f"validation macro F1 {summary['selected_validation_macro_f1']} "
            f"training seconds {summary['training_seconds']}."
        )
        lines.append("")
    return "\n".join(lines)
