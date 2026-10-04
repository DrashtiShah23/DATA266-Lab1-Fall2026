"""Final Task 1 training run.

Launch this only after the readiness audit reports HARDWARE_GATE=PASS.
A failed or interrupted run appends a failure record and does not rewrite the log.
"""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from lab1.checkpoints import build_checkpoint_metadata, write_checkpoint_metadata
from lab1.experiment_log import append_log_record, create_raw_log, relative_log_path
from lab1.git_info import git_commit_hash
from lab1.manifest import build_manifest, relative_manifest_path, write_manifest
from lab1.paths import repo_root, resolve_repo_path, to_repo_relative
from lab1.seed import seed_everything

from task1_llm.drashti.src.acquire import load_tinystories
from task1_llm.drashti.src.model.gpt import GPT, parameter_count, spec_from_config
from task1_llm.drashti.src.settings import load_task1_settings
from task1_llm.drashti.src.train.corpus import iter_windows, load_corpus, prompts_from_validation, steps_per_pass
from task1_llm.drashti.src.train.device_info import memory_snapshot, resolve_device, synchronize, training_hardware
from task1_llm.drashti.src.train.documents import write_task1_documents
from task1_llm.drashti.src.train.generate import generate_characters, measure_generation_tokens_per_second
from task1_llm.drashti.src.train.metrics import (
    REPEATED_4GRAM_DEFINITION,
    bits_per_character,
    distinct_n,
    generalization_gap,
    loss_spike_count,
    mean_repeated_4gram_rate,
    perplexity,
    top1_accuracy,
)
from task1_llm.drashti.src.train.plots import write_series_svg
from task1_llm.drashti.src.train.schedule import WarmupCosineSchedule
from task1_llm.drashti.src.vocab import vocabulary_from_payload


def train(config_path: str = "configs/task1_drashti_train.json", repo: Path | None = None) -> dict[str, Any]:
    """Train for the configured epoch count and write the Task 1 evidence."""
    root = repo or repo_root()
    config = load_task1_settings(config_path, repo=root)
    if int(config["epochs"]) < 10:
        raise RuntimeError("Refusing to start a final run with fewer than 10 epochs.")
    device = resolve_device(str(config["device"]))
    corpus = load_corpus(config, root)
    if int(corpus["train_selected"]) != 100000 or int(corpus["validation_selected"]) != 10000:
        raise RuntimeError("The member split is not 100000 training and 10000 validation stories.")
    seed_everything(int(config["seed"]))
    model = GPT(spec_from_config(config, int(corpus["vocab_size"]))).to(device)
    counted = parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(float(config["adamw_beta1"]), float(config["adamw_beta2"])),
        weight_decay=float(config["weight_decay"]),
    )
    train_steps_per_epoch = steps_per_pass(int(corpus["train_windows"].shape[0]), int(config["batch_size"]))
    total_steps = train_steps_per_epoch * int(config["epochs"])
    if int(config["warmup_steps"]) >= total_steps:
        raise RuntimeError("warmup_steps must be smaller than the total number of optimizer steps.")
    schedule = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(config["warmup_steps"]),
        total_steps=total_steps,
        target_learning_rate=float(config["learning_rate"]),
        minimum_learning_rate=float(config["minimum_learning_rate"]),
    )
    hardware = training_hardware(device)
    log_path = create_raw_log(
        config["log_directory"],
        config["experiment_name"],
        config,
        git_commit_hash(root),
        hardware,
        repo=root,
    )
    append_log_record(
        log_path,
        {
            "record_type": "split_policy",
            "train_selected_stories": int(corpus["train_selected"]),
            "validation_selected_stories": int(corpus["validation_selected"]),
            "train_eligible_windows": int(corpus["train_windows"].shape[0]),
            "validation_eligible_windows": int(corpus["validation_windows"].shape[0]),
            "train_skipped_short_stories": int(corpus["train_skipped"]),
            "validation_skipped_short_stories": int(corpus["validation_skipped"]),
            "policy": corpus["policy"],
            "steps_per_epoch": train_steps_per_epoch,
            "total_steps": total_steps,
            "parameter_count": counted,
        },
    )
    epoch_rows: list[dict[str, float]] = []
    learning_rates: list[float] = []
    gradient_norms: list[float] = []
    nan_count = 0
    spike_count = 0
    token_total = 0
    step_seconds_total = 0.0
    max_step_loss = None
    min_step_loss = None
    global_step = 0
    peak_memory = memory_snapshot(device)
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    final_checkpoint = None
    try:
        for epoch in range(1, int(config["epochs"]) + 1):
            model.train()
            epoch_loss_sum = 0.0
            epoch_tokens = 0
            epoch_losses: list[float] = []
            epoch_grad_norms: list[float] = []
            for inputs_cpu, targets_cpu in iter_windows(
                corpus["train_windows"],
                int(config["batch_size"]),
                seed=int(config["seed"]),
                epoch=epoch,
                shuffle=True,
            ):
                learning_rate = schedule.apply()
                inputs = inputs_cpu.to(device)
                targets = targets_cpu.to(device)
                synchronize(device)
                step_started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                _logits, loss = model(inputs, targets)
                if not torch.isfinite(loss):
                    nan_count += 1
                    _fail(log_path, "non-finite training loss", epoch, global_step, nan_count)
                loss.backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])))
                if not math.isfinite(grad_norm):
                    nan_count += 1
                    _fail(log_path, "non-finite gradient norm", epoch, global_step, nan_count)
                optimizer.step()
                synchronize(device)
                step_seconds = time.perf_counter() - step_started
                schedule.advance()
                global_step += 1
                tokens = int(targets.numel())
                loss_value = float(loss.detach().cpu())
                epoch_loss_sum += loss_value * tokens
                epoch_tokens += tokens
                token_total += tokens
                step_seconds_total += step_seconds
                epoch_losses.append(loss_value)
                epoch_grad_norms.append(grad_norm)
                learning_rates.append(learning_rate)
                gradient_norms.append(grad_norm)
                max_step_loss = loss_value if max_step_loss is None else max(max_step_loss, loss_value)
                min_step_loss = loss_value if min_step_loss is None else min(min_step_loss, loss_value)
                append_log_record(
                    log_path,
                    {
                        "record_type": "train_step",
                        "epoch": epoch,
                        "global_step": global_step,
                        "training_cross_entropy": loss_value,
                        "learning_rate": learning_rate,
                        "gradient_norm": grad_norm,
                        "nan_events": nan_count,
                        "tokens": tokens,
                        "step_seconds": step_seconds,
                        "tokens_per_second": tokens / step_seconds,
                        "elapsed_seconds": time.perf_counter() - wall_start,
                    },
                )
            train_loss = epoch_loss_sum / epoch_tokens
            spikes = loss_spike_count(epoch_losses, float(config["loss_spike_margin_nats"]))
            spike_count += spikes
            validation = _validate(model, corpus["validation_windows"], int(config["batch_size"]), device)
            if validation["parameters_changed"]:
                _fail(log_path, "validation updated parameters", epoch, global_step, nan_count)
            checkpoint = _save_checkpoint(
                model, optimizer, schedule, config, root, epoch, global_step, log_path
            )
            final_checkpoint = checkpoint
            memory = memory_snapshot(device)
            peak_memory = _max_memory(peak_memory, memory)
            row = {
                "epoch": float(epoch),
                "global_step": float(global_step),
                "training_cross_entropy": train_loss,
                "validation_cross_entropy": validation["cross_entropy"],
                "validation_top1_accuracy": validation["top1_accuracy"],
                "mean_gradient_norm": sum(epoch_grad_norms) / len(epoch_grad_norms),
                "learning_rate_end": learning_rates[-1],
                "loss_spike_count": float(spikes),
            }
            epoch_rows.append(row)
            append_log_record(
                log_path,
                {
                    "record_type": "epoch",
                    "epoch": epoch,
                    "global_step": global_step,
                    "training_cross_entropy": train_loss,
                    "validation_cross_entropy": validation["cross_entropy"],
                    "validation_top1_accuracy": validation["top1_accuracy"],
                    "learning_rate": learning_rates[-1],
                    "mean_gradient_norm": row["mean_gradient_norm"],
                    "loss_spike_count": spikes,
                    "nan_events": nan_count,
                    "parameters_changed_during_validation": False,
                    "checkpoint": checkpoint,
                    "elapsed_seconds": time.perf_counter() - wall_start,
                    "peak_rss_bytes": memory["peak_rss_bytes"],
                },
            )
            print(
                f"EPOCH {epoch} train={train_loss:.6f} val={validation['cross_entropy']:.6f} step={global_step}",
                flush=True,
            )
        training_seconds = time.perf_counter() - wall_start
        if global_step != total_steps or len(epoch_rows) != int(config["epochs"]):
            _fail(log_path, "training ended before the configured epochs", int(config["epochs"]), global_step, nan_count)
        _reload_matches(model, final_checkpoint, corpus["validation_windows"], device, root)
        summary = _finish(
            model=model,
            config=config,
            config_path=config_path,
            repo=root,
            corpus=corpus,
            device=device,
            hardware=hardware,
            log_path=log_path,
            epoch_rows=epoch_rows,
            learning_rates=learning_rates,
            gradient_norms=gradient_norms,
            nan_count=nan_count,
            spike_count=spike_count,
            token_total=token_total,
            step_seconds_total=step_seconds_total,
            max_step_loss=float(max_step_loss),
            min_step_loss=float(min_step_loss),
            global_step=global_step,
            peak_memory=peak_memory,
            started_at=started_at,
            training_seconds=training_seconds,
            checkpoint=final_checkpoint,
            parameter_count_value=counted,
        )
        return summary
    except Exception as exc:
        if not isinstance(exc, SystemExit):
            append_log_record(
                log_path,
                {
                    "record_type": "failure",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "global_step": global_step,
                    "nan_events": nan_count,
                },
            )
        raise


def _validate(model: torch.nn.Module, windows: torch.Tensor, batch_size: int, device: torch.device) -> dict[str, float | bool]:
    model.eval()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    loss_sum = 0.0
    tokens = 0
    correct = 0
    with torch.no_grad():
        for inputs_cpu, targets_cpu in iter_windows(windows, batch_size, seed=0, epoch=0, shuffle=False):
            inputs = inputs_cpu.to(device)
            targets = targets_cpu.to(device)
            logits, loss = model(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Validation loss was not finite.")
            count = int(targets.numel())
            loss_sum += float(loss.detach().cpu()) * count
            tokens += count
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
    changed = any(not torch.equal(left, right.detach()) for left, right in zip(before, model.parameters()))
    model.train()
    return {
        "cross_entropy": loss_sum / tokens,
        "top1_accuracy": top1_accuracy(correct, tokens),
        "parameters_changed": changed,
    }


def _save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: WarmupCosineSchedule,
    config: dict[str, Any],
    repo: Path,
    epoch: int,
    global_step: int,
    log_path: Path,
) -> str:
    directory = resolve_repo_path(config["checkpoint_directory"], repo=repo)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = directory / f"{config['experiment_name']}_{stamp}_epoch{epoch:02d}_step{global_step}.pt"
    if path.exists():
        raise RuntimeError("Refusing to overwrite a checkpoint.")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "schedule_completed_steps": schedule.completed_steps,
            "epoch": epoch,
            "global_step": global_step,
            "config_file": "configs/task1_drashti_train.json",
            "seed": int(config["seed"]),
        },
        path,
    )
    relative = to_repo_relative(path, repo=repo)
    metadata = build_checkpoint_metadata(
        member=config["member"],
        task=config["task"],
        experiment_name=config["experiment_name"],
        git_commit=git_commit_hash(repo),
        config_file="configs/task1_drashti_train.json",
        seed=int(config["seed"]),
        epoch=epoch,
        checkpoint_file=relative,
    )
    metadata_path = write_checkpoint_metadata(config["checkpoint_directory"], metadata, repo=repo)
    append_log_record(
        log_path,
        {
            "record_type": "checkpoint",
            "epoch": epoch,
            "global_step": global_step,
            "checkpoint": relative,
            "metadata": to_repo_relative(metadata_path, repo=repo),
        },
    )
    return relative


def _reload_matches(model: torch.nn.Module, checkpoint: str, windows: torch.Tensor, device: torch.device, repo: Path) -> None:
    loaded = torch.load(resolve_repo_path(checkpoint, repo=repo), map_location=device, weights_only=False)
    fresh = GPT(model.spec).to(device)
    fresh.load_state_dict(loaded["model_state"])
    fresh.eval()
    model.eval()
    chunk = windows[:4].to(dtype=torch.long, device=device)
    inputs = chunk[:, :-1].contiguous()
    with torch.no_grad():
        original, _loss = model(inputs)
        restored, _loss = fresh(inputs)
    if not torch.allclose(original, restored, atol=1e-5, rtol=1e-4):
        raise RuntimeError("Reloaded checkpoint outputs did not match.")
    model.load_state_dict(loaded["model_state"])


def _finish(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    config_path: str,
    repo: Path,
    corpus: dict[str, Any],
    device: torch.device,
    hardware: dict[str, Any],
    log_path: Path,
    epoch_rows: list[dict[str, float]],
    learning_rates: list[float],
    gradient_norms: list[float],
    nan_count: int,
    spike_count: int,
    token_total: int,
    step_seconds_total: float,
    max_step_loss: float,
    min_step_loss: float,
    global_step: int,
    peak_memory: dict[str, Any],
    started_at: str,
    training_seconds: float,
    checkpoint: str,
    parameter_count_value: int,
) -> dict[str, Any]:
    output_dir = resolve_repo_path(config["output_directory"], repo=repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_history(output_dir, epoch_rows, learning_rates, gradient_norms)
    write_series_svg(
        output_dir / "training_loss.svg",
        [("training cross entropy", [row["training_cross_entropy"] for row in epoch_rows])],
        "Training cross entropy",
        "nats",
    )
    write_series_svg(
        output_dir / "validation_loss.svg",
        [("validation cross entropy", [row["validation_cross_entropy"] for row in epoch_rows])],
        "Validation cross entropy",
        "nats",
    )
    write_series_svg(
        output_dir / "combined_loss.svg",
        [
            ("training", [row["training_cross_entropy"] for row in epoch_rows]),
            ("validation", [row["validation_cross_entropy"] for row in epoch_rows]),
        ],
        "Training and validation cross entropy",
        "nats",
    )
    write_series_svg(
        output_dir / "learning_rate.svg",
        [("learning rate", learning_rates)],
        "Learning rate",
        "learning rate",
    )
    write_series_svg(
        output_dir / "gradient_norm.svg",
        [("mean gradient norm", [row["mean_gradient_norm"] for row in epoch_rows])],
        "Epoch mean gradient norm",
        "L2 norm",
    )
    samples, continuations = _generate_samples(model, config, repo, checkpoint, device)
    throughput = measure_generation_tokens_per_second(
        model,
        samples[0]["prompt_ids"],
        new_characters=int(config["generation_max_new_characters"]),
        repeats=5,
    )
    for sample in samples:
        del sample["prompt_ids"]
    samples_path = output_dir / "generated_samples.json"
    samples_path.write_text(json.dumps(samples, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    final = epoch_rows[-1]
    train_loss = float(final["training_cross_entropy"])
    val_loss = float(final["validation_cross_entropy"])
    metrics = {
        "training_cross_entropy": train_loss,
        "validation_cross_entropy": val_loss,
        "perplexity": perplexity(val_loss),
        "bits_per_character": bits_per_character(val_loss),
        "generalization_gap": generalization_gap(val_loss, train_loss),
        "top1_accuracy": float(final["validation_top1_accuracy"]),
        "distinct_1": distinct_n(continuations, 1),
        "distinct_2": distinct_n(continuations, 2),
        "distinct_3": distinct_n(continuations, 3),
        "repeated_4gram_rate": mean_repeated_4gram_rate(continuations),
        "gradient_norm": sum(gradient_norms) / len(gradient_norms),
        "loss_spike_count": spike_count,
        "nan_count": nan_count,
        "parameter_count": parameter_count_value,
        "training_tokens_per_second": token_total / step_seconds_total,
        "generation_tokens_per_second": float(throughput["tokens_per_second_median"]),
        "peak_memory_bytes": peak_memory["peak_rss_bytes"],
        "total_training_time_seconds": training_seconds,
    }
    metrics_path = _write_metrics_csv(repo, config, checkpoint, relative_log_path(log_path, repo=repo), metrics, global_step)
    ended_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "config": config,
        "config_path": config_path,
        "hardware": hardware,
        "metrics": metrics,
        "epoch_rows": epoch_rows,
        "samples": samples,
        "checkpoint": checkpoint,
        "raw_log": relative_log_path(log_path, repo=repo),
        "metrics_file": metrics_path,
        "parameter_count": parameter_count_value,
        "global_step": global_step,
        "epochs": int(config["epochs"]),
        "train_selected": int(corpus["train_selected"]),
        "validation_selected": int(corpus["validation_selected"]),
        "train_eligible": int(corpus["train_windows"].shape[0]),
        "validation_eligible": int(corpus["validation_windows"].shape[0]),
        "train_skipped": int(corpus["train_skipped"]),
        "validation_skipped": int(corpus["validation_skipped"]),
        "vocab_size": int(corpus["vocab_size"]),
        "started_at": started_at,
        "ended_at": ended_at,
        "training_seconds": training_seconds,
        "max_step_loss": max_step_loss,
        "min_step_loss": min_step_loss,
        "generation_throughput": throughput,
        "repeated_4gram_definition": REPEATED_4GRAM_DEFINITION,
        "generation_continuation_count": len(continuations),
        "generation_characters_each": int(config["generation_max_new_characters"]),
        "peak_memory": peak_memory,
        "output_directory": config["output_directory"],
    }
    manifest = build_manifest(
        member=config["member"],
        task=config["task"],
        experiment_name=config["experiment_name"],
        git_commit=git_commit_hash(repo),
        config_file=config_path,
        environment_file=config["environment_file"],
        raw_log=summary["raw_log"],
        checkpoint=checkpoint,
        metrics_file=metrics_path,
        outputs_directory=config["output_directory"],
        hardware=hardware,
        start_time=started_at,
        end_time=ended_at,
        total_training_time=training_seconds,
    )
    manifest_path = write_manifest(config["manifest_directory"], manifest, repo=repo)
    summary["manifest"] = relative_manifest_path(manifest_path, repo=repo)
    write_task1_documents(summary, repo)
    append_log_record(
        log_path,
        {
            "record_type": "completed",
            "epochs": int(config["epochs"]),
            "global_step": global_step,
            "checkpoint": checkpoint,
            "metrics_file": metrics_path,
            "manifest": summary["manifest"],
            "training_cross_entropy": train_loss,
            "validation_cross_entropy": val_loss,
            "total_training_time_seconds": training_seconds,
        },
    )
    print("TASK1_TRAINING=COMPLETE", flush=True)
    return summary


def _generate_samples(
    model: torch.nn.Module,
    config: dict[str, Any],
    repo: Path,
    checkpoint: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[str]]:
    vocabulary = vocabulary_from_payload(
        json.loads(resolve_repo_path(config["vocab_metadata"], repo=repo).read_text(encoding="utf-8"))
    )
    split = json.loads(resolve_repo_path(config["split_metadata"], repo=repo).read_text(encoding="utf-8"))
    dataset, _revision = load_tinystories(config, repo)
    prompts = prompts_from_validation(
        dataset,
        split["validation_indices"],
        config["text_field"],
        int(config["generation_prompt_characters"]),
        int(config["generation_sample_count"]),
    )
    model.eval()
    samples: list[dict[str, Any]] = []
    continuations: list[str] = []
    for index, prompt in enumerate(prompts):
        prompt_ids = vocabulary.encode(prompt)
        for method, temperature, bit in (
            ("greedy", None, 0),
            ("temperature", float(config["generation_temperature"]), 1),
        ):
            sample_seed = int(config["seed"]) * 100000 + index * 10 + bit
            generated = generate_characters(
                model,
                prompt_ids,
                max_new_characters=int(config["generation_max_new_characters"]),
                method=method,
                temperature=temperature,
                seed=sample_seed,
            )
            continuation = vocabulary.decode(generated[len(prompt_ids) :])
            samples.append(
                {
                    "prompt": prompt,
                    "prompt_ids": prompt_ids,
                    "continuation": continuation,
                    "generated_text": prompt + continuation,
                    "checkpoint": checkpoint,
                    "method": method,
                    "temperature": temperature,
                    "max_new_characters": int(config["generation_max_new_characters"]),
                    "actual_new_characters": len(continuation),
                    "sample_seed": sample_seed,
                }
            )
            continuations.append(continuation)
    _ = device
    return samples, continuations


def _write_history(
    output_dir: Path,
    epoch_rows: list[dict[str, float]],
    learning_rates: list[float],
    gradient_norms: list[float],
) -> None:
    with (output_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)
    with (output_dir / "learning_rate_history.csv").open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle)
        writer.writerow(["global_step", "learning_rate"])
        for step, learning_rate in enumerate(learning_rates, start=1):
            writer.writerow([step, f"{learning_rate:.12g}"])
    with (output_dir / "gradient_norm_history.csv").open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle)
        writer.writerow(["global_step", "gradient_norm"])
        for step, grad_norm in enumerate(gradient_norms, start=1):
            writer.writerow([step, f"{grad_norm:.12g}"])


def _write_metrics_csv(
    repo: Path,
    config: dict[str, Any],
    checkpoint: str,
    raw_log: str,
    metrics: dict[str, float | int],
    global_step: int,
) -> str:
    path = repo / "task1_llm" / "drashti" / "metrics_report.csv"
    definitions = {
        "training_cross_entropy": "Token-weighted mean training cross entropy of the final epoch, in nats.",
        "validation_cross_entropy": "Token-weighted mean validation cross entropy of the final epoch, in nats.",
        "perplexity": "exp(validation cross entropy).",
        "bits_per_character": "validation cross entropy / ln(2).",
        "generalization_gap": "final-epoch validation cross entropy minus final-epoch training cross entropy.",
        "top1_accuracy": "Correct argmax next-character predictions divided by evaluated validation targets.",
        "distinct_1": "Unique generated character 1-grams divided by total generated character 1-grams.",
        "distinct_2": "Unique generated character 2-grams divided by total generated character 2-grams.",
        "distinct_3": "Unique generated character 3-grams divided by total generated character 3-grams.",
        "repeated_4gram_rate": REPEATED_4GRAM_DEFINITION,
        "gradient_norm": "Mean pre-clip global L2 gradient norm over all training steps.",
        "loss_spike_count": "Steps whose loss is at least loss_spike_margin_nats above that epoch's median step loss.",
        "nan_count": "Non-finite training loss or gradient events. A non-zero count stops the run.",
        "parameter_count": "Trainable parameter count of the GPT.",
        "training_tokens_per_second": "Target tokens processed during optimizer steps divided by optimizer-step wall time.",
        "generation_tokens_per_second": "Median of 5 greedy generation timings, new characters divided by synchronized wall time.",
        "peak_memory_bytes": "Peak process RSS in bytes from ru_maxrss during the training process.",
        "total_training_time_seconds": "Wall time from the start of epoch 1 through the end of epoch 10 validation.",
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "metric",
                "value",
                "experiment_name",
                "member",
                "dataset_revision",
                "split_seed",
                "epoch",
                "global_step",
                "checkpoint",
                "raw_log",
                "definition",
            ]
        )
        for name, value in metrics.items():
            writer.writerow(
                [
                    name,
                    value,
                    config["experiment_name"],
                    config["member"],
                    config["dataset_revision"],
                    config["split_seed"],
                    config["epochs"],
                    global_step,
                    checkpoint,
                    raw_log,
                    definitions[name],
                ]
            )
    return to_repo_relative(path, repo=repo)


def _max_memory(current: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in new.items():
        if isinstance(value, int) and (merged.get(key) is None or value > merged[key]):
            merged[key] = value
    return merged


def _fail(log_path: Path, reason: str, epoch: int, global_step: int, nan_count: int) -> None:
    append_log_record(
        log_path,
        {
            "record_type": "failure",
            "reason": reason,
            "epoch": epoch,
            "global_step": global_step,
            "nan_events": nan_count,
        },
    )
    raise RuntimeError(reason)


def main() -> None:
    train()


if __name__ == "__main__":
    main()
