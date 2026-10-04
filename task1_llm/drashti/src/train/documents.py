"""Factual Task 1 documents. Qualitative sections stay marked for the student."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from task1_llm.drashti.src.train.metrics import repeated_4gram_rate


def write_task1_documents(summary: dict[str, Any], repo: Path) -> None:
    """Write results, failure candidates, and the viva sheet from measured values."""
    member = repo / "task1_llm" / "drashti"
    (member / "results.md").write_text(_results(summary), encoding="utf-8")
    (member / "failure_analysis.md").write_text(_failures(summary), encoding="utf-8")
    (member / "VIVA_PREP.md").write_text(_viva(summary), encoding="utf-8")


def _results(summary: dict[str, Any]) -> str:
    config = summary["config"]
    metrics = summary["metrics"]
    hardware = summary["hardware"]
    lines = [
        "# Task 1 results",
        "",
        "The numbers below come from the completed training run. Personal interpretation is marked for the student.",
        "",
        "## Architecture",
        "",
        f"- Embedding dimension: {config['embedding_dimension']}",
        f"- Heads: {config['model_dimensions']['num_heads']}",
        f"- Blocks: {config['model_dimensions']['num_blocks']}",
        f"- Feed-forward dimension: {config['model_dimensions']['feedforward_dimension']}",
        f"- Dropout: {config['dropout']}",
        f"- Sequence length: {config['sequence_length']}",
        f"- Vocabulary size: {summary['vocab_size']}",
        f"- Trainable parameters: {summary['parameter_count']}",
        "- Attention is manual scaled dot-product with a causal mask. Token and position embeddings are learned. The language-model head is untied.",
        "",
        "## Hyperparameters",
        "",
        f"- Optimizer: {config['optimizer']}",
        f"- Target learning rate: {config['learning_rate']}",
        f"- Minimum learning rate: {config['minimum_learning_rate']}",
        f"- Warmup steps: {config['warmup_steps']}",
        f"- Scheduler: {config['scheduler']}",
        f"- Weight decay: {config['weight_decay']}",
        f"- AdamW betas: {config['adamw_beta1']}, {config['adamw_beta2']}",
        f"- Gradient clip norm: {config['gradient_clip_norm']}",
        f"- Batch size: {config['batch_size']}",
        f"- Epochs: {config['epochs']}",
        f"- Seed: {config['seed']}",
        f"- Dropout: {config['dropout']}",
        "",
        "## Dataset and split",
        "",
        f"- Dataset: {config['dataset_id']}",
        f"- Revision: {config['dataset_revision']}",
        f"- Split seed: {config['split_seed']}",
        f"- Selected training stories: {summary['train_selected']}",
        f"- Selected validation stories: {summary['validation_selected']}",
        f"- Eligible training windows: {summary['train_eligible']}",
        f"- Eligible validation windows: {summary['validation_eligible']}",
        f"- Training stories skipped as shorter than 129 characters: {summary['train_skipped']}",
        f"- Validation stories skipped as shorter than 129 characters: {summary['validation_skipped']}",
        "- Short stories remain in the 100,000 and 10,000 split. They are not padded.",
        "",
        "## Training procedure",
        "",
        "Each epoch shuffles the eligible training windows and runs AdamW with linear warmup and then cosine decay. Validation runs in eval mode at the end of every epoch and does not update parameters. The final metrics use epoch 10.",
        "",
        "## Hardware",
        "",
        f"- Selected device: {hardware.get('selected_device')}",
        f"- CPU model: {hardware.get('cpu_model')}",
        f"- GPU model: {hardware.get('gpu_model')}",
        f"- GPU memory bytes: {hardware.get('gpu_memory_bytes')}",
        f"- Unified memory bytes: {hardware.get('unified_memory_bytes')}",
        f"- PyTorch: {hardware.get('pytorch_version')}",
        f"- Memory note: {hardware.get('memory_note')}",
        "",
        "## Checkpoint, time, and evidence",
        "",
        f"- Final checkpoint: `{summary['checkpoint']}`",
        f"- Raw log: `{summary['raw_log']}`",
        f"- Metrics file: `{summary['metrics_file']}`",
        f"- Manifest: `{summary.get('manifest', 'written after this document')}`",
        f"- Output directory: `{summary['output_directory']}`",
        f"- Start: {summary['started_at']}",
        f"- End: {summary['ended_at']}",
        f"- Total training time seconds: {summary['training_seconds']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for name, value in metrics.items():
        lines.append(f"| {name} | {value} |")
    lines.extend(
        [
            "",
            "Perplexity is exp(validation cross entropy). Bits per character is validation cross entropy divided by ln(2). The generalization gap is validation cross entropy minus training cross entropy for epoch 10.",
            "",
            summary["repeated_4gram_definition"],
            "",
            f"Distinct-n and the repeated 4-gram rate use {summary['generation_continuation_count']} continuations of {summary['generation_characters_each']} new characters each. Prompts are excluded.",
            "",
            f"Step loss minimum: {summary['min_step_loss']}. Step loss maximum: {summary['max_step_loss']}.",
            "",
            "## Plots",
            "",
            "- `task1_llm/drashti/outputs/training_loss.svg`",
            "- `task1_llm/drashti/outputs/validation_loss.svg`",
            "- `task1_llm/drashti/outputs/combined_loss.svg`",
            "- `task1_llm/drashti/outputs/learning_rate.svg`",
            "- `task1_llm/drashti/outputs/gradient_norm.svg`",
            "",
            "## Generation samples",
            "",
            "Saved samples are in `task1_llm/drashti/outputs/generated_samples.json`. Each record has the prompt, continuation, checkpoint, method, temperature when sampling was used, and the generated character count.",
            "",
            "## Evidence paths",
            "",
            f"- Config: `{summary['config_path']}`",
            f"- Raw log: `{summary['raw_log']}`",
            f"- Checkpoint: `{summary['checkpoint']}`",
            f"- Metrics: `{summary['metrics_file']}`",
            "- Epoch table: `task1_llm/drashti/outputs/epoch_metrics.csv`",
            "- Learning-rate history: `task1_llm/drashti/outputs/learning_rate_history.csv`",
            "",
            "## Personal interpretation",
            "",
            "YOUR INPUT REQUIRED: Explain what these metrics say about the model, including strengths, weaknesses, and limitations, in your own words.",
            "",
        ]
    )
    return "\n".join(lines)


def _failures(summary: dict[str, Any]) -> str:
    ranked = sorted(summary["samples"], key=lambda sample: repeated_4gram_rate(sample["continuation"]), reverse=True)
    chosen = ranked[:3]
    lines = [
        "# Task 1 failure case candidates",
        "",
        "These three candidates are the saved continuations with the highest within-sample repeated 4-gram rates. The category below is the mechanical observation from that rate. The written explanation is left for the student.",
        "",
        f"Checkpoint for every candidate: `{summary['checkpoint']}`",
        "",
    ]
    for index, sample in enumerate(chosen, start=1):
        temperature = "not used" if sample["temperature"] is None else str(sample["temperature"])
        lines.extend(
            [
                f"## Candidate {index}",
                "",
                "Prompt:",
                "",
                "```text",
                sample["prompt"],
                "```",
                "",
                "Generated text:",
                "",
                "```text",
                sample["generated_text"],
                "```",
                "",
                f"- Checkpoint: `{sample['checkpoint']}`",
                f"- Generation method: {sample['method']}",
                f"- Temperature: {temperature}",
                f"- Maximum generated characters: {sample['max_new_characters']}",
                f"- Actual new characters: {sample['actual_new_characters']}",
                f"- Within-sample repeated 4-gram rate: {repeated_4gram_rate(sample['continuation'])}",
                "- Observable candidate category: repetition",
                "",
                "YOUR INPUT REQUIRED: Explain in your own words what failed in this generated text and why you classify it this way.",
                "",
            ]
        )
    return "\n".join(lines)


def _viva(summary: dict[str, Any]) -> str:
    config = summary["config"]
    metrics = summary["metrics"]
    return f"""# Task 1 viva preparation

This sheet describes the implementation and the completed training configuration. It does not add a qualitative judgment of the generations.

## Character tokenization

The vocabulary is every character that occurs in the 100,000 selected training stories, sorted by code point, with ids starting at 0. There are no special tokens and no lowercasing. The measured vocabulary size is 115. Encoding and decoding use that map.

## 100,000 and 10,000 split

The official TinyStories train split is shuffled with `random.Random(266)` under CPython 3.11. The first 100,000 indices are training stories. The next 10,000 are validation stories. This run kept those counts. Stories shorter than 129 characters stayed in the split and were skipped: {summary['train_skipped']} training stories and {summary['validation_skipped']} validation stories. Eligible windows were {summary['train_eligible']} and {summary['validation_eligible']}.

## Input target shifting

For an eligible story, input is character ids `[0, 128)` and target is `[1, 129)`. `target[t]` is the character immediately after `input[t]`.

## Token embeddings and positional embeddings

`token_embedding` is a learned table of shape `[115, 128]`. `position_embedding` is a learned table of shape `[128, 128]`. Their sum is the block input. Positions are absolute indexes inside the current window.

## Q, K, V and scaled dot-product attention

Each head has dimension 32 because 128 / 4 = 32. Learned projections produce query, key, and value. Scores are `query @ key.transpose / sqrt(32)`, then the causal mask, then softmax, then the weighted values. Heads are concatenated and passed through an output projection.

## Causal masking

The mask is lower triangular. Future positions are filled with the dtype minimum before softmax, so a position cannot attend to a later position.

## Multiple heads

Four heads run in parallel and are concatenated back to 128 dimensions.

## Residual connections and layer normalization

Each block is pre-norm. The attention output is added to its block input. The feed-forward output is added to the attention residual stream. Layer norm uses an explicit mean and variance plus learned scale and shift.

## Feed-forward network

Each block expands 128 to 512, applies GELU, projects back to 128, and uses dropout.

## Cross entropy

The language-model head produces 115 logits per position. Training minimizes the mean token cross entropy between those logits and the shifted targets. The reported epoch losses are token-weighted means, in nats.

## Warmup and scheduler

Warmup lasts {config['warmup_steps']} optimizer steps. The first step uses target learning rate / warmup steps, which is {float(config['learning_rate']) / int(config['warmup_steps'])}. The rate increases linearly and reaches {config['learning_rate']} on the last warmup step. Cosine decay then moves from that target to {config['minimum_learning_rate']} on the final optimizer step. The unit is the optimizer step. The saved learning-rate history is `task1_llm/drashti/outputs/learning_rate_history.csv`.

## Metrics

- Perplexity = exp(validation cross entropy) = {metrics['perplexity']}
- Bits per character = validation cross entropy / ln(2) = {metrics['bits_per_character']}
- Generalization gap = validation cross entropy minus training cross entropy = {metrics['generalization_gap']}
- Top-1 accuracy = correct argmax predictions / evaluated validation targets = {metrics['top1_accuracy']}
- Distinct-n = unique generated character n-grams / total generated character n-grams, pooled over the saved continuations.
- Repeated 4-gram rate: {summary['repeated_4gram_definition']} Measured value: {metrics['repeated_4gram_rate']}
- Gradient norm is the mean pre-clip global L2 norm over training steps = {metrics['gradient_norm']}
- Training tokens per second counts target tokens during optimizer steps only = {metrics['training_tokens_per_second']}
- Generation tokens per second is a separate greedy timing = {metrics['generation_tokens_per_second']}
- Parameter count = {metrics['parameter_count']}
- NaN count = {metrics['nan_count']}
- Loss spike count = {metrics['loss_spike_count']}
- Peak RSS bytes = {metrics['peak_memory_bytes']}
- Total training seconds = {metrics['total_training_time_seconds']}

Distinct-1, distinct-2, and distinct-3 use {summary['generation_continuation_count']} continuations of {summary['generation_characters_each']} new characters. Half of the samples are greedy and half use temperature {config['generation_temperature']}.

## Generation

Greedy decoding takes the argmax. Temperature sampling divides logits by {config['generation_temperature']} and draws with a seeded CPU generator. Prompts are the first {config['generation_prompt_characters']} characters of validation stories. The final checkpoint is `{summary['checkpoint']}`.

## Actual architecture and training configuration

- Member: {config['member']}
- Experiment: {config['experiment_name']}
- Device: {config['device']}
- Epochs completed: {summary['epochs']}
- Global steps completed: {summary['global_step']}
- Batch size: {config['batch_size']}
- Seed: {config['seed']}
- Optimizer: {config['optimizer']} with weight decay {config['weight_decay']}
- Gradient clip: {config['gradient_clip_norm']}

## What changing a value does

Increasing the embedding dimension, head count, block count, or feed-forward dimension adds capacity and compute. Decreasing them does the opposite. A longer sequence makes attention more expensive because scores grow with the square of the sequence length. A larger learning rate or a shorter warmup moves the updates faster at the start. More dropout makes the tiny overfit harder and regularizes a longer run. These are consequences of the formulas, not a claim about which setting is best.

YOUR INPUT REQUIRED: Add any personal explanation you want to give in the viva. Do not treat this sheet as that explanation.
"""


def write_continuation_documents(summary: dict[str, Any], repo: Path) -> None:
    """Write the 20-epoch results, viva sheet, and new failure candidates.

    ``failure_analysis.md`` is the epoch 10 analysis and is not modified.
    """
    member = repo / "task1_llm" / "drashti"
    (member / "results.md").write_text(_continuation_results(summary), encoding="utf-8")
    (member / "failure_analysis_best_checkpoint.md").write_text(_best_failures(summary), encoding="utf-8")
    (member / "VIVA_PREP.md").write_text(_continuation_viva(summary), encoding="utf-8")


def _continuation_results(summary: dict[str, Any]) -> str:
    config = summary["config"]
    metrics = summary["metrics"]
    hardware = summary["hardware"]
    lines = [
        "# Task 1 results",
        "",
        "The assignment minimum was 10 epochs. The same GPT was then trained through epoch 20 by resuming the epoch 10 checkpoint. The epoch 10 evidence is preserved and is not replaced by this file.",
        "",
        "## Architecture",
        "",
        f"- Embedding dimension: {config['embedding_dimension']}",
        f"- Heads: {config['model_dimensions']['num_heads']}",
        f"- Blocks: {config['model_dimensions']['num_blocks']}",
        f"- Feed-forward dimension: {config['model_dimensions']['feedforward_dimension']}",
        f"- Dropout: {config['dropout']}",
        f"- Sequence length: {config['sequence_length']}",
        f"- Vocabulary size: {summary['vocab_size']}",
        f"- Trainable parameters: {summary['parameter_count']}",
        "- Attention is manual scaled dot-product with a causal mask. Token and position embeddings are learned. The language-model head is untied. The architecture was not changed for the continuation.",
        "",
        "## Hyperparameters",
        "",
        f"- Optimizer: {config['optimizer']}",
        f"- Target learning rate of the original cosine: {config['learning_rate']}",
        f"- Minimum learning rate: {config['minimum_learning_rate']}",
        f"- Warmup steps, used only in the original run: {config['warmup_steps']}",
        f"- Original scheduler: linear warmup then cosine, completed at step 15630",
        f"- Continuation scheduler: {config['scheduler']}",
        f"- Weight decay: {config['weight_decay']}",
        f"- AdamW betas: {config['adamw_beta1']}, {config['adamw_beta2']}",
        f"- Gradient clip norm: {config['gradient_clip_norm']}",
        f"- Batch size: {config['batch_size']}",
        f"- Total epochs: {summary['epochs']}",
        f"- Seed: {config['seed']}",
        f"- Dropout: {config['dropout']}",
        "",
        "## Continuation learning rate",
        "",
        summary["schedule_note"],
        "",
        f"- Restored optimizer learning rate: {summary['restored_learning_rate']}",
        "- Warmup was not run again.",
        "- Python, NumPy, and PyTorch random state were not in the epoch 10 checkpoint. Epoch shuffle order still follows `random.Random(seed + epoch)`. Dropout after the resume is a new random stream.",
        "",
        "## Dataset and split",
        "",
        f"- Dataset: {config['dataset_id']}",
        f"- Revision: {config['dataset_revision']}",
        f"- Split seed: {config['split_seed']}",
        f"- Selected training stories: {summary['train_selected']}",
        f"- Selected validation stories: {summary['validation_selected']}",
        f"- Eligible training windows: {summary['train_eligible']}",
        f"- Eligible validation windows: {summary['validation_eligible']}",
        f"- Training stories skipped as shorter than 129 characters: {summary['train_skipped']}",
        f"- Validation stories skipped as shorter than 129 characters: {summary['validation_skipped']}",
        "- Short stories remain in the 100,000 and 10,000 split. They are not padded.",
        "",
        "## Training procedure",
        "",
        "Epochs 1 through 10 used AdamW, linear warmup, and cosine decay. Epochs 11 through 20 resumed the epoch 10 model and optimizer and held the restored learning rate. Validation ran in eval mode at the end of every epoch and did not update parameters. Final metrics use the checkpoint with the lowest validation cross entropy, not automatically epoch 20.",
        "",
        f"- Resumed from: `{summary['original_checkpoint']}`",
        f"- Lowest validation epoch: {summary['best_epoch']}",
        f"- Lowest validation cross entropy: {summary['best_validation_loss']}",
        f"- Selected checkpoint: `{summary['checkpoint']}`",
        f"- Selection rule: {summary['selection_reason']}",
        f"- Epoch 20 checkpoint, preserved either way: `{summary['epoch20_checkpoint']}`",
        "",
        "## Hardware",
        "",
        f"- Selected device: {hardware.get('selected_device')}",
        f"- CPU model: {hardware.get('cpu_model')}",
        f"- GPU model: {hardware.get('gpu_model')}",
        f"- GPU memory bytes: {hardware.get('gpu_memory_bytes')}",
        f"- Unified memory bytes: {hardware.get('unified_memory_bytes')}",
        f"- PyTorch: {hardware.get('pytorch_version')}",
        f"- Memory note: {hardware.get('memory_note')}",
        "",
        "## Checkpoint, time, and evidence",
        "",
        f"- Original raw log: `{summary['original_raw_log']}`",
        f"- Original manifest: `{summary['original_manifest']}`",
        f"- Continuation raw log: `{summary['raw_log']}`",
        f"- Continuation manifest: `{summary.get('manifest', 'written after this document')}`",
        f"- Metrics file: `{summary['metrics_file']}`",
        f"- Epoch 10 evidence copy: `task1_llm/drashti/evidence_epoch10/`",
        f"- Epochs 1 through 10 training seconds: {summary['original_training_seconds']}",
        f"- Epochs 11 through 20 training seconds: {summary['training_seconds']}",
        f"- Combined training seconds: {summary['combined_training_seconds']}",
        f"- Continuation start: {summary['started_at']}",
        f"- Continuation end: {summary['ended_at']}",
        "- Combined training time adds the two measured training intervals. It does not include the pause between the two processes or generation time.",
        "",
        "## Epochs 1 through 20",
        "",
        "| Epoch | Training cross entropy | Validation cross entropy | Generalization gap |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary["epoch_rows"]:
        train_loss = float(row["training_cross_entropy"])
        val_loss = float(row["validation_cross_entropy"])
        lines.append(
            f"| {int(row['epoch'])} | {train_loss:.12g} | {val_loss:.12g} | {val_loss - train_loss:.12g} |"
        )
    lines.extend(
        [
            "",
            summary["overfitting_note"],
            "",
            "## Metrics from the selected checkpoint",
            "",
            "| Metric | Value |",
            "| --- | --- |",
        ]
    )
    for name, value in metrics.items():
        lines.append(f"| {name} | {value} |")
    lines.extend(
        [
            "",
            "Perplexity is exp(validation cross entropy). Bits per character is validation cross entropy divided by ln(2). The generalization gap is the selected epoch's validation cross entropy minus that epoch's training cross entropy.",
            "",
            "Training cross entropy is measured in train mode, with dropout active. Validation cross entropy is measured in eval mode. The selected validation loss was recomputed from the loaded checkpoint.",
            "",
            summary["repeated_4gram_definition"],
            "",
            f"Distinct-n and the repeated 4-gram rate for the selected checkpoint use {summary['generation_continuation_count']} continuations of {summary['generation_characters_each']} new characters each. Prompts are excluded. The protocol matches the epoch 10 generation: 12 validation prompts, greedy and temperature {config['generation_temperature']}.",
            "",
            f"Gradient norm is the mean pre-clip global L2 over all {summary['global_step']} optimizer steps. Loss spikes are the sum of the per-epoch counts from epochs 1 through 20. NaN count is the sum of both runs.",
            "",
            f"Continuation step loss minimum: {summary['min_step_loss']}. Continuation step loss maximum: {summary['max_step_loss']}.",
            "",
            "## Epoch 10 compared with epoch 20",
            "",
            "| Measurement | Epoch 10 | Epoch 20 |",
            "| --- | --- | --- |",
            f"| Training cross entropy | {summary['epoch10_train_loss']} | {summary['epoch20_train_loss']} |",
            f"| Validation cross entropy | {summary['epoch10_val_loss']} | {summary['epoch20_val_loss']} |",
            f"| Perplexity | {summary['epoch10_metrics']['perplexity']} | {summary['epoch20_perplexity']} |",
            f"| Bits per character | {summary['epoch10_metrics']['bits_per_character']} | {summary['epoch20_bits_per_character']} |",
            f"| Top-1 accuracy | {summary['epoch10_metrics']['top1_accuracy']} | {summary['epoch20_top1']} |",
            f"| Distinct-1 | {summary['epoch10_metrics']['distinct_1']} | {summary['epoch20_distinct_1']} |",
            f"| Distinct-2 | {summary['epoch10_metrics']['distinct_2']} | {summary['epoch20_distinct_2']} |",
            f"| Distinct-3 | {summary['epoch10_metrics']['distinct_3']} | {summary['epoch20_distinct_3']} |",
            f"| Repeated 4-gram rate | {summary['epoch10_metrics']['repeated_4gram_rate']} | {summary['epoch20_repeated_4gram_rate']} |",
            "",
            "Epoch 10 generation metrics are the preserved values from the original run. Epoch 20 generation metrics are a new sample set from the epoch 20 checkpoint using the same protocol.",
            "",
            "## Plots",
            "",
            "Updated curves for epochs 1 through 20, with a marker at the end of epoch 10:",
            "",
            "- `task1_llm/drashti/outputs/epochs_1_to_20/training_loss.svg`",
            "- `task1_llm/drashti/outputs/epochs_1_to_20/validation_loss.svg`",
            "- `task1_llm/drashti/outputs/epochs_1_to_20/combined_loss.svg`",
            "- `task1_llm/drashti/outputs/epochs_1_to_20/learning_rate.svg`",
            "- `task1_llm/drashti/outputs/epochs_1_to_20/gradient_norm.svg`",
            "",
            "The original 10-epoch SVG files remain in `task1_llm/drashti/outputs/` and in `task1_llm/drashti/evidence_epoch10/outputs/`.",
            "",
            "## Generation samples",
            "",
            f"- Epoch 10 samples, unchanged: `task1_llm/drashti/outputs/generated_samples.json`",
            f"- Selected checkpoint samples: `{summary['best_samples_path']}`",
            f"- Epoch 20 samples: `{summary['epoch20_samples_path']}`",
            "",
            "## Evidence paths",
            "",
            f"- Continuation config: `{summary['config_path']}`",
            f"- Original config: `configs/task1_drashti_train.json`",
            f"- Original raw log: `{summary['original_raw_log']}`",
            f"- Continuation raw log: `{summary['raw_log']}`",
            f"- Selected checkpoint: `{summary['checkpoint']}`",
            f"- Metrics: `{summary['metrics_file']}`",
            "- Epoch table: `task1_llm/drashti/outputs/epochs_1_to_20/epoch_metrics.csv`",
            "- Learning-rate history: `task1_llm/drashti/outputs/epochs_1_to_20/learning_rate_history.csv`",
            "- Epoch 10 failure analysis, unchanged: `task1_llm/drashti/failure_analysis.md`",
            "- Selected-checkpoint failure candidates: `task1_llm/drashti/failure_analysis_best_checkpoint.md`",
            "",
            "## Personal interpretation",
            "",
            "YOUR INPUT REQUIRED: Explain what these metrics say about the model, including strengths, weaknesses, limitations, and whether the extra epochs were worth training, in your own words.",
            "",
        ]
    )
    return "\n".join(lines)


def _best_failures(summary: dict[str, Any]) -> str:
    ranked = sorted(summary["samples"], key=lambda sample: repeated_4gram_rate(sample["continuation"]), reverse=True)
    chosen = ranked[:3]
    lines = [
        "# Task 1 failure case candidates from the selected checkpoint",
        "",
        "The epoch 10 candidates remain in `task1_llm/drashti/failure_analysis.md`. That file was not rewritten. The three candidates below are a new set from the checkpoint selected for final evaluation.",
        "",
        "These three candidates are the new continuations with the highest within-sample repeated 4-gram rates. The written explanation is left for the student.",
        "",
        f"Checkpoint for every candidate: `{summary['checkpoint']}`",
        "",
    ]
    for index, sample in enumerate(chosen, start=1):
        temperature = "not used" if sample["temperature"] is None else str(sample["temperature"])
        lines.extend(
            [
                f"## Candidate {index}",
                "",
                "Prompt:",
                "",
                "```text",
                sample["prompt"],
                "```",
                "",
                "Generated text:",
                "",
                "```text",
                sample["generated_text"],
                "```",
                "",
                f"- Checkpoint: `{sample['checkpoint']}`",
                f"- Generation method: {sample['method']}",
                f"- Temperature: {temperature}",
                f"- Maximum generated characters: {sample['max_new_characters']}",
                f"- Actual new characters: {sample['actual_new_characters']}",
                f"- Within-sample repeated 4-gram rate: {repeated_4gram_rate(sample['continuation'])}",
                "- Observable candidate category: repetition",
                "",
                "YOUR INPUT REQUIRED: Explain in your own words what failed in this generated text and why you classify it this way.",
                "",
            ]
        )
    return "\n".join(lines)


def _continuation_viva(summary: dict[str, Any]) -> str:
    config = summary["config"]
    metrics = summary["metrics"]
    return f"""# Task 1 viva preparation

This sheet describes the implementation and the 20-epoch training configuration. It does not add a qualitative judgment of the generations.

## Character tokenization

The vocabulary is every character that occurs in the 100,000 selected training stories, sorted by code point, with ids starting at 0. There are no special tokens and no lowercasing. The measured vocabulary size is 115. Encoding and decoding use that map. The continuation did not rebuild it.

## 100,000 and 10,000 split

The official TinyStories train split is shuffled with `random.Random(266)` under CPython 3.11. The first 100,000 indices are training stories. The next 10,000 are validation stories. This run kept those counts. Stories shorter than 129 characters stayed in the split and were skipped: {summary['train_skipped']} training stories and {summary['validation_skipped']} validation stories. Eligible windows were {summary['train_eligible']} and {summary['validation_eligible']}.

## Input target shifting

For an eligible story, input is character ids `[0, 128)` and target is `[1, 129)`. `target[t]` is the character immediately after `input[t]`.

## Token embeddings and positional embeddings

`token_embedding` is a learned table of shape `[115, 128]`. `position_embedding` is a learned table of shape `[128, 128]`. Their sum is the block input. Positions are absolute indexes inside the current window.

## Q, K, V and scaled dot-product attention

Each head has dimension 32 because 128 / 4 = 32. Learned projections produce query, key, and value. Scores are `query @ key.transpose / sqrt(32)`, then the causal mask, then softmax, then the weighted values. Heads are concatenated and passed through an output projection.

## Causal masking

The mask is lower triangular. Future positions are filled with the dtype minimum before softmax, so a position cannot attend to a later position.

## Multiple heads

Four heads run in parallel and are concatenated back to 128 dimensions.

## Residual connections and layer normalization

Each block is pre-norm. The attention output is added to its block input. The feed-forward output is added to the attention residual stream. Layer norm uses an explicit mean and variance plus learned scale and shift.

## Feed-forward network

Each block expands 128 to 512, applies GELU, projects back to 128, and uses dropout.

## Cross entropy

The language-model head produces 115 logits per position. Training minimizes the mean token cross entropy between those logits and the shifted targets. The reported epoch losses are token-weighted means, in nats.

## Warmup and scheduler

Warmup lasted {config['warmup_steps']} optimizer steps during epochs 1 through 10 only. The first step used {float(config['learning_rate']) / int(config['warmup_steps'])}, and the rate reached {config['learning_rate']} on step {config['warmup_steps']}. Cosine decay then reached {config['minimum_learning_rate']} on step 15630. Epochs 11 through 20 did not warm up again. They held the learning rate restored from the epoch 10 optimizer, {summary['restored_learning_rate']}. The complete history is `task1_llm/drashti/outputs/epochs_1_to_20/learning_rate_history.csv`.

## Metrics

The values below are from the selected checkpoint, epoch {summary['best_epoch']}.

- Perplexity = exp(validation cross entropy) = {metrics['perplexity']}
- Bits per character = validation cross entropy / ln(2) = {metrics['bits_per_character']}
- Generalization gap = validation cross entropy minus training cross entropy = {metrics['generalization_gap']}
- Top-1 accuracy = correct argmax predictions / evaluated validation targets = {metrics['top1_accuracy']}
- Distinct-n = unique generated character n-grams / total generated character n-grams, pooled over the saved continuations.
- Repeated 4-gram rate: {summary['repeated_4gram_definition']} Measured value: {metrics['repeated_4gram_rate']}
- Gradient norm is the mean pre-clip global L2 norm over optimizer steps 1 through 20 = {metrics['gradient_norm']}
- Training tokens per second counts target tokens during optimizer steps only = {metrics['training_tokens_per_second']}
- Generation tokens per second is a separate greedy timing = {metrics['generation_tokens_per_second']}
- Parameter count = {metrics['parameter_count']}
- NaN count = {metrics['nan_count']}
- Loss spike count = {metrics['loss_spike_count']}
- Peak RSS bytes = {metrics['peak_memory_bytes']}
- Epochs 1 through 10 training seconds = {summary['original_training_seconds']}
- Epochs 11 through 20 training seconds = {summary['training_seconds']}
- Combined training seconds = {summary['combined_training_seconds']}

Distinct-1, distinct-2, and distinct-3 use {summary['generation_continuation_count']} continuations of {summary['generation_characters_each']} new characters. Half of the samples are greedy and half use temperature {config['generation_temperature']}.

## Generation

Greedy decoding takes the argmax. Temperature sampling divides logits by {config['generation_temperature']} and draws with a seeded CPU generator. Prompts are the first {config['generation_prompt_characters']} characters of validation stories. The selected checkpoint is `{summary['checkpoint']}`.

## Actual architecture and training configuration

- Member: {config['member']}
- Continuation experiment: {config['experiment_name']}
- Device: {config['device']}
- Epochs completed: {summary['epochs']}
- Global steps completed: {summary['global_step']}
- Batch size: {config['batch_size']}
- Seed: {config['seed']}
- Optimizer: {config['optimizer']} with weight decay {config['weight_decay']}
- Gradient clip: {config['gradient_clip_norm']}
- Selected epoch: {summary['best_epoch']}

## What changing a value does

Increasing the embedding dimension, head count, block count, or feed-forward dimension adds capacity and compute. Decreasing them does the opposite. A longer sequence makes attention more expensive because scores grow with the square of the sequence length. A larger learning rate or a shorter warmup moves the updates faster at the start. More dropout makes the tiny overfit harder and regularizes a longer run. These are consequences of the formulas, not a claim about which setting is best.

YOUR INPUT REQUIRED: Add any personal explanation you want to give in the viva. Do not treat this sheet as that explanation.
"""
