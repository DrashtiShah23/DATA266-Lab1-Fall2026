"""Continue the finished 10-epoch GPT run through epoch 20.

This process never rewrites the original raw log, manifest, checkpoints, or
generation file. It starts from the epoch 10 checkpoint and trains epochs
11 through 20.
"""

from __future__ import annotations

import csv
import json
import math
import shutil
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

from task1_llm.drashti.src.model.gpt import GPT, parameter_count, spec_from_config
from task1_llm.drashti.src.settings import load_task1_settings
from task1_llm.drashti.src.train.corpus import iter_windows, load_corpus, steps_per_pass
from task1_llm.drashti.src.train.device_info import memory_snapshot, resolve_device, synchronize, training_hardware
from task1_llm.drashti.src.train.documents import write_continuation_documents
from task1_llm.drashti.src.train.generate import measure_generation_tokens_per_second
from task1_llm.drashti.src.train.metrics import (
    REPEATED_4GRAM_DEFINITION,
    bits_per_character,
    distinct_n,
    generalization_gap,
    loss_spike_count,
    mean_repeated_4gram_rate,
    perplexity,
)
from task1_llm.drashti.src.train.plots import write_series_svg
from task1_llm.drashti.src.train.run import _fail, _generate_samples, _max_memory, _reload_matches, _validate
from task1_llm.drashti.src.train.schedule import held_continuation_learning_rate, learning_rate_for_step

RESUME_EPOCH = 10
FINAL_EPOCH = 20
ORIGINAL_CHECKPOINT = "task1_llm/drashti/checkpoints/task1_drashti_train_20260928T070850656698Z_epoch10_step15630.pt"
ORIGINAL_LOG = "reproducibility/raw_logs/task1_drashti_train_20260928T064215673720Z_5f5c9173.log"
ORIGINAL_MANIFEST = "reproducibility/manifests/drashti_task1_task1_drashti_train_20260928T070917995838Z_bb880136.manifest.json"
SCHEDULE_NOTE = (
    "The epoch 10 checkpoint stored schedule_completed_steps at 15630, which is the last step of the original "
    "10-epoch cosine. Evaluating that cosine at any later step index stays at the minimum learning rate because "
    "progress is clamped at 1. Epochs 11 through 20 therefore hold the learning rate found in the restored "
    "optimizer, 0.00003. Warmup is not repeated. A new cosine stretched over 31260 steps is not used: at step "
    "15630 that curve would still be near mid-decay and would raise the learning rate above the restored value."
)


def continue_training(
    config_path: str = "configs/task1_drashti_train_continue.json",
    repo: Path | None = None,
) -> dict[str, Any]:
    """Train epochs 11 through 20 from the epoch 10 checkpoint."""
    root = repo or repo_root()
    config = load_task1_settings(config_path, repo=root)
    if int(config["epochs"]) != FINAL_EPOCH:
        raise RuntimeError("The continuation config must request 20 total epochs.")
    if config["scheduler"] != "hold_restored_learning_rate":
        raise RuntimeError("The continuation scheduler must hold the restored learning rate.")
    if config["resume_checkpoint"] != ORIGINAL_CHECKPOINT:
        raise RuntimeError("The continuation config does not point at the epoch 10 checkpoint.")
    _archive_epoch10_evidence(root)
    checkpoint_dir = resolve_repo_path(config["checkpoint_directory"], repo=root)
    if list(checkpoint_dir.glob("*_epoch20_step31260.pt")):
        raise RuntimeError("An epoch 20 checkpoint already exists. Refusing to train again.")
    prior = _read_original_run(root)
    device = resolve_device(str(config["device"]))
    corpus = load_corpus(config, root)
    if int(corpus["train_selected"]) != 100000 or int(corpus["validation_selected"]) != 10000:
        raise RuntimeError("The member split is not 100000 training and 10000 validation stories.")
    audit = _audit_checkpoint(root, device)
    print(
        "CHECKPOINT_AUDIT",
        json.dumps({key: value for key, value in audit.items() if key != "payload"}, sort_keys=True),
        flush=True,
    )
    seed_everything(int(config["seed"]))
    model = GPT(spec_from_config(config, int(corpus["vocab_size"]))).to(device)
    counted = parameter_count(model)
    if counted != 839168:
        raise RuntimeError("The GPT parameter count changed.")
    model.load_state_dict(audit["payload"]["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(audit["restored_learning_rate"]),
        betas=(float(config["adamw_beta1"]), float(config["adamw_beta2"])),
        weight_decay=float(config["weight_decay"]),
    )
    optimizer.load_state_dict(audit["payload"]["optimizer_state"])
    learning_rate = held_continuation_learning_rate(float(optimizer.param_groups[0]["lr"]))
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
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
            "record_type": "checkpoint_audit",
            "resume_checkpoint": ORIGINAL_CHECKPOINT,
            "present": audit["present"],
            "absent": audit["absent"],
            "restored_learning_rate": learning_rate,
            "schedule_completed_steps": audit["schedule_completed_steps"],
            "warmup_rerun": False,
            "schedule_note": SCHEDULE_NOTE,
            "random_state_restored": False,
        },
    )
    train_steps_per_epoch = steps_per_pass(int(corpus["train_windows"].shape[0]), int(config["batch_size"]))
    if train_steps_per_epoch != 1563:
        raise RuntimeError("Steps per epoch changed.")
    global_step = int(audit["global_step"])
    nan_count = 0
    spike_count = 0
    token_total = 0
    step_seconds_total = 0.0
    continuation_rates: list[float] = []
    continuation_grads: list[float] = []
    new_rows: list[dict[str, float]] = []
    saved_checkpoints: dict[int, str] = {}
    max_step_loss = None
    min_step_loss = None
    peak_memory = memory_snapshot(device)
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    epoch20_checkpoint = None
    try:
        for epoch in range(RESUME_EPOCH + 1, FINAL_EPOCH + 1):
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
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
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
                global_step += 1
                tokens = int(targets.numel())
                loss_value = float(loss.detach().cpu())
                epoch_loss_sum += loss_value * tokens
                epoch_tokens += tokens
                token_total += tokens
                step_seconds_total += step_seconds
                epoch_losses.append(loss_value)
                epoch_grad_norms.append(grad_norm)
                continuation_rates.append(learning_rate)
                continuation_grads.append(grad_norm)
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
            checkpoint = _save_continuation_checkpoint(
                model, optimizer, config, root, epoch, global_step, log_path, learning_rate
            )
            saved_checkpoints[epoch] = checkpoint
            if epoch == FINAL_EPOCH:
                epoch20_checkpoint = checkpoint
            memory = memory_snapshot(device)
            peak_memory = _max_memory(peak_memory, memory)
            row = {
                "epoch": float(epoch),
                "global_step": float(global_step),
                "training_cross_entropy": train_loss,
                "validation_cross_entropy": validation["cross_entropy"],
                "validation_top1_accuracy": validation["top1_accuracy"],
                "mean_gradient_norm": sum(epoch_grad_norms) / len(epoch_grad_norms),
                "learning_rate_end": learning_rate,
                "loss_spike_count": float(spikes),
            }
            new_rows.append(row)
            append_log_record(
                log_path,
                {
                    "record_type": "epoch",
                    "epoch": epoch,
                    "global_step": global_step,
                    "training_cross_entropy": train_loss,
                    "validation_cross_entropy": validation["cross_entropy"],
                    "validation_top1_accuracy": validation["top1_accuracy"],
                    "learning_rate": learning_rate,
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
        if global_step != 31260 or len(new_rows) != 10 or epoch20_checkpoint is None:
            _fail(log_path, "continuation ended before epoch 20", FINAL_EPOCH, global_step, nan_count)
        _reload_matches(model, epoch20_checkpoint, corpus["validation_windows"], device, root)
        summary = _finish_continuation(
            model=model,
            config=config,
            config_path=config_path,
            repo=root,
            corpus=corpus,
            device=device,
            hardware=hardware,
            log_path=log_path,
            prior=prior,
            new_rows=new_rows,
            continuation_rates=continuation_rates,
            continuation_grads=continuation_grads,
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
            epoch20_checkpoint=epoch20_checkpoint,
            parameter_count_value=counted,
            restored_learning_rate=learning_rate,
            saved_checkpoints=saved_checkpoints,
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


def _archive_epoch10_evidence(repo: Path) -> None:
    """Copy the finished 10-epoch evidence. Existing archive files stay as they are."""
    relative_paths = [
        "task1_llm/drashti/results.md",
        "task1_llm/drashti/failure_analysis.md",
        "task1_llm/drashti/VIVA_PREP.md",
        "task1_llm/drashti/metrics_report.csv",
        "task1_llm/drashti/outputs/epoch_metrics.csv",
        "task1_llm/drashti/outputs/learning_rate_history.csv",
        "task1_llm/drashti/outputs/gradient_norm_history.csv",
        "task1_llm/drashti/outputs/generated_samples.json",
        "task1_llm/drashti/outputs/training_loss.svg",
        "task1_llm/drashti/outputs/validation_loss.svg",
        "task1_llm/drashti/outputs/combined_loss.svg",
        "task1_llm/drashti/outputs/learning_rate.svg",
        "task1_llm/drashti/outputs/gradient_norm.svg",
    ]
    for relative in relative_paths:
        source = repo / relative
        if not source.is_file():
            raise RuntimeError(f"Missing epoch 10 evidence: {relative}")
        destination = repo / "task1_llm" / "drashti" / "evidence_epoch10" / Path(relative).name
        if relative.startswith("task1_llm/drashti/outputs/"):
            destination = repo / "task1_llm" / "drashti" / "evidence_epoch10" / "outputs" / Path(relative).name
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _read_original_run(repo: Path) -> dict[str, Any]:
    log_path = resolve_repo_path(ORIGINAL_LOG, repo=repo)
    header_time = None
    completed = None
    failures = 0
    tokens = 0
    step_seconds = 0.0
    grad_sum = 0.0
    grad_count = 0
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            kind = record["record_type"]
            if kind == "header" and header_time is None:
                header_time = record["timestamp"]
            elif kind == "train_step":
                tokens += int(record["tokens"])
                step_seconds += float(record["step_seconds"])
                grad_sum += float(record["gradient_norm"])
                grad_count += 1
            elif kind == "completed":
                completed = record
            elif kind == "failure":
                failures += 1
    if failures or completed is None or header_time is None:
        raise RuntimeError("The original raw log is not a completed 10-epoch run.")
    if int(completed["epochs"]) != 10 or int(completed["global_step"]) != 15630:
        raise RuntimeError("The original log did not finish at epoch 10 step 15630.")
    rows = _read_epoch_rows(repo / "task1_llm" / "drashti" / "outputs" / "epoch_metrics.csv")
    if [int(row["epoch"]) for row in rows] != list(range(1, 11)):
        raise RuntimeError("The original epoch table is not epochs 1 through 10.")
    metrics = _read_metrics(repo / "task1_llm" / "drashti" / "metrics_report.csv")
    rates = _read_series(repo / "task1_llm" / "drashti" / "outputs" / "learning_rate_history.csv")
    grads = _read_series(repo / "task1_llm" / "drashti" / "outputs" / "gradient_norm_history.csv")
    if len(rates) != 15630 or len(grads) != 15630 or grad_count != 15630:
        raise RuntimeError("The original histories do not cover 15630 steps.")
    return {
        "started_at": header_time,
        "training_seconds": float(completed["total_training_time_seconds"]),
        "rows": rows,
        "metrics": metrics,
        "rates": rates,
        "grads": grads,
        "tokens": tokens,
        "step_seconds": step_seconds,
        "grad_sum": grad_sum,
        "peak_rss_bytes": int(float(metrics["peak_memory_bytes"])),
        "nan_count": int(float(metrics["nan_count"])),
        "spike_count": int(float(metrics["loss_spike_count"])),
    }


def _audit_checkpoint(repo: Path, device: torch.device) -> dict[str, Any]:
    path = resolve_repo_path(ORIGINAL_CHECKPOINT, repo=repo)
    payload = torch.load(path, map_location=device, weights_only=False)
    present = sorted(key for key in ("model_state", "optimizer_state", "schedule_completed_steps", "epoch", "global_step", "config_file", "seed") if key in payload)
    absent = [
        "python_random_state",
        "torch_random_state",
        "cuda_or_mps_random_state",
        "full_scheduler_object",
        "full_training_config",
    ]
    required = ("model_state", "optimizer_state", "schedule_completed_steps", "epoch", "global_step", "config_file", "seed")
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError("Epoch 10 checkpoint is missing required state: " + ", ".join(missing))
    if int(payload["epoch"]) != 10 or int(payload["global_step"]) != 15630:
        raise RuntimeError("Checkpoint is not the epoch 10 step 15630 state.")
    if int(payload["schedule_completed_steps"]) != 15630:
        raise RuntimeError("Saved schedule progress is not 15630 completed steps.")
    if payload["config_file"] != "configs/task1_drashti_train.json" or int(payload["seed"]) != 266:
        raise RuntimeError("Checkpoint config path or seed does not match the original run.")
    learning_rate = float(payload["optimizer_state"]["param_groups"][0]["lr"])
    if learning_rate != 0.00003:
        raise RuntimeError("Restored optimizer learning rate is not the finished cosine minimum.")
    finished = learning_rate_for_step(
        15630,
        warmup_steps=200,
        total_steps=15630,
        target_learning_rate=0.0003,
        minimum_learning_rate=0.00003,
    )
    if finished != 0.00003:
        raise RuntimeError("The original cosine no longer sits at its minimum after step 15630.")
    return {
        "payload": payload,
        "present": present,
        "absent": absent,
        "restored_learning_rate": learning_rate,
        "schedule_completed_steps": 15630,
        "global_step": 15630,
    }


def _save_continuation_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    repo: Path,
    epoch: int,
    global_step: int,
    log_path: Path,
    learning_rate: float,
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
            "schedule_completed_steps": 15630,
            "continuation_policy": "hold_restored_learning_rate",
            "continuation_learning_rate": learning_rate,
            "warmup_rerun": False,
            "epoch": epoch,
            "global_step": global_step,
            "config_file": "configs/task1_drashti_train_continue.json",
            "seed": int(config["seed"]),
            "resumed_from": ORIGINAL_CHECKPOINT,
            "random_state_restored": False,
        },
        path,
    )
    relative = to_repo_relative(path, repo=repo)
    metadata = build_checkpoint_metadata(
        member=config["member"],
        task=config["task"],
        experiment_name=config["experiment_name"],
        git_commit=git_commit_hash(repo),
        config_file="configs/task1_drashti_train_continue.json",
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


def _finish_continuation(
    *,
    model: torch.nn.Module,
    config: dict[str, Any],
    config_path: str,
    repo: Path,
    corpus: dict[str, Any],
    device: torch.device,
    hardware: dict[str, Any],
    log_path: Path,
    prior: dict[str, Any],
    new_rows: list[dict[str, float]],
    continuation_rates: list[float],
    continuation_grads: list[float],
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
    epoch20_checkpoint: str,
    parameter_count_value: int,
    restored_learning_rate: float,
    saved_checkpoints: dict[int, str],
) -> dict[str, Any]:
    rows = prior["rows"] + new_rows
    best = min(rows, key=lambda row: (float(row["validation_cross_entropy"]), float(row["epoch"])))
    best_epoch = int(best["epoch"])
    if best_epoch <= 10:
        best_checkpoint = ORIGINAL_CHECKPOINT
    else:
        best_checkpoint = saved_checkpoints[best_epoch]
    epoch20_samples, epoch20_continuations = _generate_samples(model, config, repo, epoch20_checkpoint, device)
    if best_checkpoint == epoch20_checkpoint:
        model_for_eval = model
        samples = epoch20_samples
        continuations = epoch20_continuations
    else:
        _load_model_state(model, best_checkpoint, repo, device)
        _reload_matches(model, best_checkpoint, corpus["validation_windows"], device, repo)
        samples, continuations = _generate_samples(model, config, repo, best_checkpoint, device)
        model_for_eval = model
    validation = _validate(model_for_eval, corpus["validation_windows"], int(config["batch_size"]), device)
    if abs(float(validation["cross_entropy"]) - float(best["validation_cross_entropy"])) > 1e-4:
        raise RuntimeError("Recomputed validation loss did not match the selected epoch.")
    prompt_ids = list(samples[0]["prompt_ids"])
    throughput = measure_generation_tokens_per_second(
        model_for_eval,
        prompt_ids,
        new_characters=int(config["generation_max_new_characters"]),
        repeats=5,
    )
    output_dir = repo / "task1_llm" / "drashti" / "outputs" / "epochs_1_to_20"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_combined_history(output_dir, rows, prior["rates"] + continuation_rates, prior["grads"] + continuation_grads)
    _write_marked_plots(output_dir, rows, prior["rates"] + continuation_rates)
    best_samples_path = _write_samples(repo / "task1_llm" / "drashti" / "outputs" / "generated_samples_best_checkpoint.json", samples)
    epoch20_samples_path = _write_samples(repo / "task1_llm" / "drashti" / "outputs" / "generated_samples_epoch20.json", epoch20_samples)
    train_loss = float(best["training_cross_entropy"])
    val_loss = float(validation["cross_entropy"])
    epoch20 = rows[-1]
    combined_seconds = float(prior["training_seconds"]) + training_seconds
    combined_tokens = int(prior["tokens"]) + token_total
    combined_step_seconds = float(prior["step_seconds"]) + step_seconds_total
    combined_grad = (float(prior["grad_sum"]) + sum(continuation_grads)) / (15630 + len(continuation_grads))
    peak_bytes = max(int(prior["peak_rss_bytes"]), int(peak_memory["peak_rss_bytes"] or 0))
    metrics = {
        "training_cross_entropy": train_loss,
        "validation_cross_entropy": val_loss,
        "perplexity": perplexity(val_loss),
        "bits_per_character": bits_per_character(val_loss),
        "generalization_gap": generalization_gap(val_loss, train_loss),
        "top1_accuracy": float(validation["top1_accuracy"]),
        "distinct_1": distinct_n(continuations, 1),
        "distinct_2": distinct_n(continuations, 2),
        "distinct_3": distinct_n(continuations, 3),
        "repeated_4gram_rate": mean_repeated_4gram_rate(continuations),
        "gradient_norm": combined_grad,
        "loss_spike_count": int(prior["spike_count"]) + spike_count,
        "nan_count": int(prior["nan_count"]) + nan_count,
        "parameter_count": parameter_count_value,
        "training_tokens_per_second": combined_tokens / combined_step_seconds,
        "generation_tokens_per_second": float(throughput["tokens_per_second_median"]),
        "peak_memory_bytes": peak_bytes,
        "total_training_time_seconds": combined_seconds,
        "training_time_epochs_1_to_10_seconds": float(prior["training_seconds"]),
        "training_time_epochs_11_to_20_seconds": training_seconds,
        "selected_epoch": best_epoch,
    }
    metrics_path = _write_metrics_csv(repo, config, best_checkpoint, relative_log_path(log_path, repo=repo), metrics, int(best["global_step"]))
    ended_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "config": config,
        "config_path": config_path,
        "hardware": hardware,
        "metrics": metrics,
        "epoch_rows": rows,
        "samples": [{key: value for key, value in sample.items() if key != "prompt_ids"} for sample in samples],
        "checkpoint": best_checkpoint,
        "epoch20_checkpoint": epoch20_checkpoint,
        "best_epoch": best_epoch,
        "best_validation_loss": float(best["validation_cross_entropy"]),
        "selection_reason": "Lowest validation cross entropy across epochs 1 through 20. An equal loss would keep the earlier epoch. Epoch 20 is saved even when it is not selected.",
        "raw_log": relative_log_path(log_path, repo=repo),
        "original_raw_log": ORIGINAL_LOG,
        "original_manifest": ORIGINAL_MANIFEST,
        "original_checkpoint": ORIGINAL_CHECKPOINT,
        "metrics_file": metrics_path,
        "parameter_count": parameter_count_value,
        "global_step": global_step,
        "epochs": FINAL_EPOCH,
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
        "original_training_seconds": float(prior["training_seconds"]),
        "combined_training_seconds": combined_seconds,
        "original_started_at": prior["started_at"],
        "max_step_loss": max_step_loss,
        "min_step_loss": min_step_loss,
        "generation_throughput": throughput,
        "repeated_4gram_definition": REPEATED_4GRAM_DEFINITION,
        "generation_continuation_count": len(continuations),
        "generation_characters_each": int(config["generation_max_new_characters"]),
        "peak_memory": peak_memory,
        "schedule_note": SCHEDULE_NOTE,
        "restored_learning_rate": restored_learning_rate,
        "overfitting_note": _overfitting_note(rows),
        "epoch10_metrics": prior["metrics"],
        "epoch10_train_loss": float(rows[9]["training_cross_entropy"]),
        "epoch10_val_loss": float(rows[9]["validation_cross_entropy"]),
        "epoch20_train_loss": float(epoch20["training_cross_entropy"]),
        "epoch20_val_loss": float(epoch20["validation_cross_entropy"]),
        "epoch20_perplexity": perplexity(float(epoch20["validation_cross_entropy"])),
        "epoch20_bits_per_character": bits_per_character(float(epoch20["validation_cross_entropy"])),
        "epoch20_top1": float(epoch20["validation_top1_accuracy"]),
        "epoch20_distinct_1": distinct_n(epoch20_continuations, 1),
        "epoch20_distinct_2": distinct_n(epoch20_continuations, 2),
        "epoch20_distinct_3": distinct_n(epoch20_continuations, 3),
        "epoch20_repeated_4gram_rate": mean_repeated_4gram_rate(epoch20_continuations),
        "best_samples_path": best_samples_path,
        "epoch20_samples_path": epoch20_samples_path,
        "output_directory": "task1_llm/drashti/outputs/epochs_1_to_20",
    }
    manifest = build_manifest(
        member=config["member"],
        task=config["task"],
        experiment_name=config["experiment_name"],
        git_commit=git_commit_hash(repo),
        config_file=config_path,
        environment_file=config["environment_file"],
        raw_log=summary["raw_log"],
        checkpoint=best_checkpoint,
        metrics_file=metrics_path,
        outputs_directory=summary["output_directory"],
        hardware=hardware,
        start_time=prior["started_at"],
        end_time=ended_at,
        total_training_time=combined_seconds,
    )
    manifest_path = write_manifest(config["manifest_directory"], manifest, repo=repo)
    summary["manifest"] = relative_manifest_path(manifest_path, repo=repo)
    write_continuation_documents(summary, repo)
    append_log_record(
        log_path,
        {
            "record_type": "completed",
            "epochs": FINAL_EPOCH,
            "global_step": global_step,
            "selected_epoch": best_epoch,
            "checkpoint": best_checkpoint,
            "epoch20_checkpoint": epoch20_checkpoint,
            "metrics_file": metrics_path,
            "manifest": summary["manifest"],
            "training_cross_entropy": train_loss,
            "validation_cross_entropy": val_loss,
            "continuation_training_time_seconds": training_seconds,
            "combined_training_time_seconds": combined_seconds,
        },
    )
    print("TASK1_CONTINUATION=COMPLETE", flush=True)
    return summary


def _overfitting_note(rows: list[dict[str, float]]) -> str:
    epoch10 = rows[9]
    epoch20 = rows[-1]
    best = min(rows, key=lambda row: (float(row["validation_cross_entropy"]), float(row["epoch"])))
    change = float(epoch20["validation_cross_entropy"]) - float(epoch10["validation_cross_entropy"])
    if change < 0:
        direction = "Validation loss was lower at epoch 20 than at epoch 10, so it continued improving after epoch 10."
    elif change > 0:
        direction = "Validation loss was higher at epoch 20 than at epoch 10, so it worsened after epoch 10."
    else:
        direction = "Validation loss at epoch 20 equalled validation loss at epoch 10."
    rises = []
    for previous, current in zip(rows, rows[1:]):
        if int(current["epoch"]) <= 10:
            continue
        if float(current["validation_cross_entropy"]) > float(previous["validation_cross_entropy"]):
            rises.append(str(int(current["epoch"])))
    if rises:
        rise_note = "Validation loss rose at epochs " + ", ".join(rises) + "."
    else:
        rise_note = "Validation loss did not rise between any consecutive epochs from 10 through 20."
    return (
        f"Epoch 10 validation cross entropy was {float(epoch10['validation_cross_entropy'])}. "
        f"Epoch 20 validation cross entropy was {float(epoch20['validation_cross_entropy'])}. "
        f"The change from epoch 10 to epoch 20 was {change}. {direction} {rise_note} "
        f"The lowest validation loss was epoch {int(best['epoch'])} at {float(best['validation_cross_entropy'])}. "
        "Training cross entropy stays above validation cross entropy at every epoch because training is measured with dropout on and validation is measured in eval mode."
    )


def _load_model_state(model: torch.nn.Module, checkpoint: str, repo: Path, device: torch.device) -> None:
    payload = torch.load(resolve_repo_path(checkpoint, repo=repo), map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])


def _write_samples(path: Path, samples: list[dict[str, Any]]) -> str:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path.name}.")
    cleaned = [{key: value for key, value in sample.items() if key != "prompt_ids"} for sample in samples]
    path.write_text(json.dumps(cleaned, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return to_repo_relative(path, repo=repo_root())


def _write_combined_history(
    output_dir: Path,
    rows: list[dict[str, float]],
    learning_rates: list[float],
    gradient_norms: list[float],
) -> None:
    if len(learning_rates) != 31260 or len(gradient_norms) != 31260 or len(rows) != 20:
        raise RuntimeError("Combined history does not cover 20 epochs.")
    with (output_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
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


def _write_marked_plots(output_dir: Path, rows: list[dict[str, float]], learning_rates: list[float]) -> None:
    epoch_marker = 9 / 19
    rate_marker = 15629 / 31259
    write_series_svg(
        output_dir / "training_loss.svg",
        [("training cross entropy", [float(row["training_cross_entropy"]) for row in rows])],
        "Training cross entropy, epochs 1-20",
        "nats",
        marker_fraction=epoch_marker,
        marker_label="epoch 10",
    )
    write_series_svg(
        output_dir / "validation_loss.svg",
        [("validation cross entropy", [float(row["validation_cross_entropy"]) for row in rows])],
        "Validation cross entropy, epochs 1-20",
        "nats",
        marker_fraction=epoch_marker,
        marker_label="epoch 10",
    )
    write_series_svg(
        output_dir / "combined_loss.svg",
        [
            ("training", [float(row["training_cross_entropy"]) for row in rows]),
            ("validation", [float(row["validation_cross_entropy"]) for row in rows]),
        ],
        "Training and validation cross entropy, epochs 1-20",
        "nats",
        marker_fraction=epoch_marker,
        marker_label="epoch 10",
    )
    write_series_svg(
        output_dir / "learning_rate.svg",
        [("learning rate", learning_rates)],
        "Learning rate, steps 1-31260",
        "learning rate",
        marker_fraction=rate_marker,
        marker_label="epoch 10",
    )
    write_series_svg(
        output_dir / "gradient_norm.svg",
        [("mean gradient norm", [float(row["mean_gradient_norm"]) for row in rows])],
        "Epoch mean gradient norm, epochs 1-20",
        "L2 norm",
        marker_fraction=epoch_marker,
        marker_label="epoch 10",
    )


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
        "training_cross_entropy": "Token-weighted mean training cross entropy of the selected epoch, in nats. Training mode uses dropout.",
        "validation_cross_entropy": "Token-weighted validation cross entropy recomputed from the selected checkpoint, in nats.",
        "perplexity": "exp(validation cross entropy).",
        "bits_per_character": "validation cross entropy / ln(2).",
        "generalization_gap": "selected-epoch validation cross entropy minus selected-epoch training cross entropy.",
        "top1_accuracy": "Correct argmax next-character predictions divided by evaluated validation targets.",
        "distinct_1": "Unique generated character 1-grams divided by total generated character 1-grams.",
        "distinct_2": "Unique generated character 2-grams divided by total generated character 2-grams.",
        "distinct_3": "Unique generated character 3-grams divided by total generated character 3-grams.",
        "repeated_4gram_rate": REPEATED_4GRAM_DEFINITION,
        "gradient_norm": "Mean pre-clip global L2 gradient norm over optimizer steps 1 through 31260.",
        "loss_spike_count": "Sum of per-epoch spike counts for epochs 1 through 20.",
        "nan_count": "Non-finite training loss or gradient events across both runs.",
        "parameter_count": "Trainable parameter count of the GPT.",
        "training_tokens_per_second": "Target tokens during optimizer steps 1 through 31260 divided by those steps' wall time.",
        "generation_tokens_per_second": "Median of 5 greedy generation timings on the selected checkpoint.",
        "peak_memory_bytes": "Larger of the epoch 1-10 process peak RSS and the epoch 11-20 process peak RSS.",
        "total_training_time_seconds": "Sum of the epoch 1-10 training wall time and the epoch 11-20 training wall time. The pause between processes is excluded.",
        "training_time_epochs_1_to_10_seconds": "Wall time of the original run from the start of epoch 1 through the end of epoch 10 validation.",
        "training_time_epochs_11_to_20_seconds": "Wall time of the continuation from the start of epoch 11 through the end of epoch 20 validation.",
        "selected_epoch": "Epoch whose validation cross entropy was lowest. Final generation and validation metrics use that checkpoint.",
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
                    metrics["selected_epoch"],
                    global_step,
                    checkpoint,
                    raw_log,
                    definitions[name],
                ]
            )
    archived = repo / "task1_llm" / "drashti" / "evidence_epoch10" / "metrics_report.csv"
    if not archived.is_file():
        raise RuntimeError("Refusing to replace metrics_report.csv before the epoch 10 copy exists.")
    return to_repo_relative(path, repo=repo)


def _read_epoch_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="\n") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def _read_series(path: Path) -> list[float]:
    with path.open(encoding="utf-8", newline="\n") as handle:
        reader = csv.DictReader(handle)
        return [float(row["learning_rate"] if "learning_rate" in row else row["gradient_norm"]) for row in reader]


def _read_metrics(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="\n") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def main() -> None:
    continue_training()


if __name__ == "__main__":
    main()
