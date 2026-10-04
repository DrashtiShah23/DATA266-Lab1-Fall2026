"""Train the three Task 2 sentiment models on the Phase 5 cache.

The official test split is not read during training or checkpoint selection.
It is read only by ``predict_official_test``, after a checkpoint has been chosen
from validation macro F1.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lab1.config import load_config
from lab1.experiment_log import append_log_record, create_raw_log, relative_log_path
from lab1.exclusive import write_unique_text
from lab1.git_info import git_commit_hash
from lab1.manifest import build_manifest, relative_manifest_path, write_manifest
from lab1.paths import repo_root, resolve_repo_path, to_repo_relative
from lab1.checkpoints import build_checkpoint_metadata, write_checkpoint_metadata
from lab1.seed import seed_everything
from task1_llm.drashti.src.train.device_info import (
    memory_snapshot,
    resolve_device,
    synchronize,
    training_hardware,
)
from task2_sentiment.drashti.src.checkpoints_io import (
    cpu_state_dict,
    load_model_weights,
    save_checkpoint,
)
from task2_sentiment.drashti.src.interface import example_id
from task2_sentiment.drashti.src.models import build_model, zero_padding_row
from task2_sentiment.drashti.src.populations import (
    assert_phase5_populations,
    example_ids_for,
    load_model_ready,
    reject_test_split,
    slice_name,
)
from task2_sentiment.drashti.src.predictions import write_prediction_file
from task2_sentiment.drashti.src.selection import (
    accuracy,
    binary_macro_f1,
    predict_label,
    probabilities_from_logits,
    select_checkpoint,
    should_stop,
)

CONFIGS = (
    "configs/task2_drashti_baseline.json",
    "configs/task2_drashti_cnn.json",
    "configs/task2_drashti_gru.json",
)
EXTRA_FIELDS = (
    "model_name",
    "role",
    "architecture",
    "vocab_size",
    "padding_id",
    "dataset_revision",
    "preprocessing_config",
    "vocabulary_file",
    "model_ready_cache",
    "validation_split_seed",
    "hidden_dimension",
    "num_layers",
    "kernel_sizes",
    "conv_channels",
    "pooling",
    "activation",
    "classifier_head",
    "weight_decay",
    "gradient_clip_norm",
    "min_epochs",
    "early_stopping_patience",
    "early_stopping_metric",
    "decision_threshold",
    "loss",
)
TOTAL_GATE_SECONDS = 28800


def load_model_config(config_path: str, repo: Path | None = None) -> dict[str, Any]:
    """Load one sentiment config and require the Task 2 training fields."""
    config = load_config(config_path, repo=repo)
    missing = [field for field in EXTRA_FIELDS if field not in config]
    if missing:
        raise ValueError("Model config is missing fields: " + ", ".join(missing))
    if config["early_stopping_metric"] != "validation_macro_f1":
        raise ValueError("Checkpoint selection must use validation macro F1.")
    if int(config["padding_id"]) != 0:
        raise ValueError("Padding id must be 0 to match the Phase 5 vocabulary.")
    if config["architecture"] == "gru" and int(config["num_layers"]) != 1:
        raise ValueError("The recurrent model is defined as one GRU layer.")
    return config


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def make_optimizer(model: nn.Module, config: dict[str, Any], learning_rate: float | None = None) -> torch.optim.Optimizer:
    """AdamW. Embedding weights are not decayed, and the padding row stays zero."""
    if config["optimizer"] != "AdamW":
        raise ValueError("The sentiment configs use AdamW.")
    embedding = list(model.embedding.parameters())
    embedding_ids = {id(parameter) for parameter in embedding}
    others = [parameter for parameter in model.parameters() if id(parameter) not in embedding_ids]
    return torch.optim.AdamW(
        [
            {"params": embedding, "weight_decay": 0.0},
            {"params": others, "weight_decay": float(config["weight_decay"])},
        ],
        lr=float(config["learning_rate"] if learning_rate is None else learning_rate),
    )


def take_batch(
    split_cache: dict[str, Any],
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = split_cache["input_ids"].index_select(0, indices).to(device=device, dtype=torch.long)
    labels = split_cache["labels"].index_select(0, indices).to(device=device, dtype=torch.long)
    lengths = split_cache["content_length"].index_select(0, indices).to(device=device, dtype=torch.long)
    return ids, labels, lengths


def iter_training_batches(
    split_name: str,
    split_cache: dict[str, Any],
    batch_size: int,
    order: torch.Tensor,
    device: torch.device,
):
    """Yield training batches. The official test split is rejected."""
    reject_test_split(split_name)
    rows = int(order.shape[0])
    for start in range(0, rows, batch_size):
        yield take_batch(split_cache, order[start : start + batch_size], device)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: float) -> None:
    if not math.isfinite(value):
        raise RuntimeError("Encountered a non-finite loss.")


def _run_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    ids: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
    clip_norm: float,
) -> tuple[float, float, int]:
    """One forward, and a backward when an optimizer is supplied."""
    if optimizer is None:
        logits = model(ids, lengths)
    else:
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids, lengths)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    value = float(loss.detach().cpu())
    _finite(value)
    grad_norm = 0.0
    if optimizer is not None:
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm))
        optimizer.step()
        zero_padding_row(model)
    return value, grad_norm, int(labels.shape[0])


def readiness_check(
    config: dict[str, Any],
    development: dict[str, Any],
    device: torch.device,
    repo: Path,
) -> dict[str, Any]:
    """One real development batch through forward, backward, save, and reload."""
    seed_everything(int(config["seed"]))
    model = build_model(config).to(device)
    optimizer = make_optimizer(model, config)
    batch_size = min(int(config["batch_size"]), int(development["input_ids"].shape[0]))
    indices = torch.arange(batch_size)
    ids, labels, lengths = take_batch(development, indices, device)
    model.train()
    loss_value, grad_norm, count = _run_step(
        model, optimizer, ids, labels, lengths, float(config["gradient_clip_norm"])
    )
    embedding_grad = model.embedding.weight.grad
    if embedding_grad is None or float(embedding_grad.detach()[1:].abs().sum()) == 0.0:
        raise RuntimeError("The embedding did not receive a gradient.")
    if float(model.embedding.weight.detach()[0].abs().sum()) != 0.0:
        raise RuntimeError("The padding embedding row is not zero.")
    directory = resolve_repo_path(config["checkpoint_directory"], repo=repo)
    path = save_checkpoint(
        directory,
        config["experiment_name"] + "_readiness",
        {
            "model_state": cpu_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "epoch": 0,
            "purpose": "readiness",
            "config_file": config["experiment_name"],
        },
    )
    reloaded = build_model(config).to(device)
    load_model_weights(reloaded, path)
    model.eval()
    reloaded.eval()
    with torch.no_grad():
        original = model(ids, lengths)
        copied = reloaded(ids, lengths)
    if not torch.allclose(original, copied, rtol=1e-4, atol=1e-5):
        raise RuntimeError("The reloaded model did not match the saved model.")
    probabilities = probabilities_from_logits(original)
    if not torch.allclose(probabilities.sum(dim=-1), torch.ones(batch_size, device=device), atol=1e-5):
        raise RuntimeError("Class probabilities do not sum to one.")
    prefix = development["example_id_prefix"]
    ids_attached = [example_id(prefix, int(development["source_index"][row])) for row in range(batch_size)]
    return {
        "status": "PASS",
        "batch_rows": count,
        "input_shape": list(ids.shape),
        "label_shape": list(labels.shape),
        "output_shape": list(original.shape),
        "loss": loss_value,
        "gradient_norm": grad_norm,
        "checkpoint": to_repo_relative(path, repo=repo),
        "probability_sum_ok": True,
        "example_ids_attached": ids_attached[:3],
        "example_id_count": len(ids_attached),
    }


def overfit_check(config: dict[str, Any], development: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """DEVELOPMENT OVERFIT TEST on a small real subset. Not a final score."""
    seed_everything(int(config["seed"]))
    sample_count = int(config["overfit_examples"])
    steps = int(config["overfit_steps"])
    model = build_model(config).to(device)
    optimizer = make_optimizer(model, config, learning_rate=float(config["overfit_learning_rate"]))
    model.train()
    initial = None
    final = None
    batch_size = min(32, sample_count)
    for step in range(steps):
        start = (step * batch_size) % sample_count
        indices = torch.arange(start, start + batch_size) % sample_count
        ids, labels, lengths = take_batch(development, indices, device)
        loss_value, _grad, _count = _run_step(
            model, optimizer, ids, labels, lengths, float(config["gradient_clip_norm"])
        )
        if step == 0:
            initial = loss_value
        final = loss_value
    return {
        "label": "DEVELOPMENT OVERFIT TEST",
        "sample_count": sample_count,
        "optimization_steps": steps,
        "initial_loss": initial,
        "final_loss": final,
        "loss_decreased": bool(final < initial),
    }


def benchmark_model(
    config: dict[str, Any],
    development: dict[str, Any],
    validation: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Time real optimization steps and estimate a full maximum-epoch run."""
    seed_everything(int(config["seed"]))
    model = build_model(config).to(device)
    optimizer = make_optimizer(model, config)
    batch_size = int(config["batch_size"])
    timed_steps = int(config["benchmark_steps"])
    model.train()
    for warmup in range(2):
        indices = torch.arange(warmup * batch_size, (warmup + 1) * batch_size)
        ids, labels, lengths = take_batch(development, indices, device)
        _run_step(model, optimizer, ids, labels, lengths, float(config["gradient_clip_norm"]))
    synchronize(device)
    started = time.perf_counter()
    seen = 0
    for step in range(timed_steps):
        start = ((step + 2) * batch_size) % (int(development["rows"]) - batch_size)
        indices = torch.arange(start, start + batch_size)
        ids, labels, lengths = take_batch(development, indices, device)
        _value, _grad, count = _run_step(
            model, optimizer, ids, labels, lengths, float(config["gradient_clip_norm"])
        )
        seen += count
    synchronize(device)
    train_seconds = time.perf_counter() - started
    model.eval()
    synchronize(device)
    val_started = time.perf_counter()
    val_seen = 0
    with torch.no_grad():
        for step in range(5):
            indices = torch.arange(step * batch_size, (step + 1) * batch_size)
            ids, labels, lengths = take_batch(validation, indices, device)
            _value, _grad, count = _run_step(model, None, ids, labels, lengths, float(config["gradient_clip_norm"]))
            val_seen += count
    synchronize(device)
    val_seconds = time.perf_counter() - val_started
    train_rows = int(development["rows"])
    val_rows = int(validation["rows"])
    train_steps = math.ceil(train_rows / batch_size)
    val_steps = math.ceil(val_rows / batch_size)
    epoch_seconds = (train_seconds / timed_steps) * train_steps + (val_seconds / 5) * val_steps
    epochs = int(config["epochs"])
    estimated = epoch_seconds * epochs
    return {
        "device": device.type,
        "parameter_count": parameter_count(model),
        "batch_size": batch_size,
        "examples_per_second": seen / train_seconds,
        "steps_per_epoch": train_steps,
        "estimated_epoch_seconds": epoch_seconds,
        "estimated_runtime_seconds": estimated,
        "max_epochs": epochs,
        "gate_limit_seconds": int(config["hardware_gate_seconds"]),
        "gate": "PASS" if estimated <= float(config["hardware_gate_seconds"]) else "BLOCKED",
    }


def _probe_device(config: dict[str, Any], development: dict[str, Any]) -> tuple[torch.device, str | None]:
    requested = resolve_device(str(config["device"]))
    try:
        readiness_probe = build_model(config).to(requested)
        indices = torch.arange(min(8, int(config["batch_size"])))
        ids, labels, lengths = take_batch(development, indices, requested)
        readiness_probe.train()
        _run_step(readiness_probe, make_optimizer(readiness_probe, config), ids, labels, lengths, 1.0)
        return requested, None
    except Exception as exc:
        if requested.type != "mps":
            raise
        cpu = torch.device("cpu")
        fallback = build_model(config).to(cpu)
        ids, labels, lengths = take_batch(development, torch.arange(min(8, int(config["batch_size"]))), cpu)
        fallback.train()
        _run_step(fallback, make_optimizer(fallback, config), ids, labels, lengths, 1.0)
        return cpu, f"{type(exc).__name__}: {exc}"


def preflight_model(config: dict[str, Any], cache: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Readiness, overfit, and the hardware estimate for one architecture."""
    device, device_note = _probe_device(config, cache["development"])
    ready = readiness_check(config, cache["development"], device, repo)
    overfit = overfit_check(config, cache["development"], device)
    benchmark = benchmark_model(config, cache["development"], cache["validation"], device)
    return {
        "model_name": config["model_name"],
        "experiment_name": config["experiment_name"],
        "device_requested": config["device"],
        "device_used": device.type,
        "device_note": device_note,
        "parameter_count": parameter_count(build_model(config)),
        "readiness": ready,
        "overfit": overfit,
        "benchmark": benchmark,
    }


def _evaluate(model: nn.Module, split_cache: dict[str, Any], config: dict[str, Any], device: torch.device) -> dict[str, float]:
    reject_test_split("validation")
    model.eval()
    total_loss = 0.0
    seen = 0
    labels_out: list[torch.Tensor] = []
    predictions_out: list[torch.Tensor] = []
    rows = int(split_cache["rows"])
    order = torch.arange(rows)
    with torch.no_grad():
        for ids, labels, lengths in iter_training_batches("validation", split_cache, int(config["batch_size"]), order, device):
            logits = model(ids, lengths)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            value = float(loss.detach().cpu())
            _finite(value)
            total_loss += value * int(labels.shape[0])
            seen += int(labels.shape[0])
            probabilities = probabilities_from_logits(logits)
            predictions = predict_label(probabilities[:, 1], float(config["decision_threshold"]))
            labels_out.append(labels.detach().cpu())
            predictions_out.append(predictions.detach().cpu())
    labels_all = torch.cat(labels_out)
    predictions_all = torch.cat(predictions_out)
    return {
        "validation_loss": total_loss / seen,
        "validation_accuracy": accuracy(labels_all, predictions_all),
        "validation_macro_f1": binary_macro_f1(labels_all, predictions_all),
        "rows": float(seen),
    }


def _train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    development: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]) + epoch)
    order = torch.randperm(int(development["rows"]), generator=generator)
    total_loss = 0.0
    seen = 0
    nan_count = 0
    grad_norms: list[float] = []
    steps = math.ceil(int(development["rows"]) / int(config["batch_size"]))
    synchronize(device)
    started = time.perf_counter()
    for step, (ids, labels, lengths) in enumerate(
        iter_training_batches("development", development, int(config["batch_size"]), order, device),
        start=1,
    ):
        try:
            loss_value, grad_norm, count = _run_step(
                model, optimizer, ids, labels, lengths, float(config["gradient_clip_norm"])
            )
        except RuntimeError:
            nan_count += 1
            raise
        total_loss += loss_value * count
        seen += count
        grad_norms.append(grad_norm)
        if step % 200 == 0 or step == steps:
            print(
                f"{config['model_name']} epoch {epoch} step {step}/{steps} loss {loss_value:.4f}",
                flush=True,
            )
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "training_loss": total_loss / seen,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "gradient_norm": sum(grad_norms) / len(grad_norms),
        "runtime_seconds": elapsed,
        "examples_per_second": seen / elapsed,
        "nan_count": float(nan_count),
        "examples": float(seen),
    }


def train_model(
    config: dict[str, Any],
    cache: dict[str, Any],
    device: torch.device,
    repo: Path,
    config_path: str,
) -> dict[str, Any]:
    """Train on development rows and select a checkpoint with validation macro F1."""
    seed_everything(int(config["seed"]))
    hardware = training_hardware(device)
    log_path = create_raw_log(
        config["log_directory"],
        config["experiment_name"],
        config,
        git_commit_hash(repo),
        hardware,
        repo=repo,
    )
    started_at = _now()
    started = time.perf_counter()
    try:
        model = build_model(config).to(device)
        optimizer = make_optimizer(model, config)
        count = parameter_count(model)
        epoch_rows: list[dict[str, Any]] = []
        best_f1 = None
        stall = 0
        checkpoints: dict[int, str] = {}
        for epoch in range(1, int(config["epochs"]) + 1):
            epoch_started = time.perf_counter()
            trained = _train_epoch(model, optimizer, cache["development"], config, device, epoch)
            before = cpu_state_dict(model)
            validated = _evaluate(model, cache["validation"], config, device)
            for key, value in model.state_dict().items():
                if not torch.equal(value.detach().cpu(), before[key]):
                    raise RuntimeError("Validation changed model parameters.")
            model.train()
            synchronize(device)
            runtime = time.perf_counter() - epoch_started
            record = {
                "epoch": epoch,
                "training_loss": trained["training_loss"],
                "validation_loss": validated["validation_loss"],
                "validation_accuracy": validated["validation_accuracy"],
                "validation_macro_f1": validated["validation_macro_f1"],
                "learning_rate": trained["learning_rate"],
                "gradient_norm": trained["gradient_norm"],
                "runtime_seconds": runtime,
                "train_runtime_seconds": trained["runtime_seconds"],
                "examples_per_second": trained["examples_per_second"],
                "nan_count": int(trained["nan_count"]),
            }
            directory = resolve_repo_path(config["checkpoint_directory"], repo=repo)
            checkpoint_path = save_checkpoint(
                directory,
                f"{config['experiment_name']}_epoch{epoch}",
                {
                    "model_state": cpu_state_dict(model),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "global_step": epoch * math.ceil(int(cache["development"]["rows"]) / int(config["batch_size"])),
                    "config_file": config_path,
                    "seed": int(config["seed"]),
                    "decision_threshold": float(config["decision_threshold"]),
                    "validation_macro_f1": record["validation_macro_f1"],
                    "architecture": config["architecture"],
                    "vocab_size": int(config["vocab_size"]),
                },
            )
            relative_checkpoint = to_repo_relative(checkpoint_path, repo=repo)
            checkpoints[epoch] = relative_checkpoint
            record["checkpoint"] = relative_checkpoint
            metadata = build_checkpoint_metadata(
                member=config["member"],
                task=config["task"],
                experiment_name=config["experiment_name"],
                git_commit=git_commit_hash(repo),
                config_file=config_path,
                seed=int(config["seed"]),
                epoch=epoch,
                checkpoint_file=relative_checkpoint,
            )
            write_checkpoint_metadata(config["checkpoint_directory"], metadata, repo=repo)
            epoch_rows.append(record)
            append_log_record(log_path, {"record_type": "epoch", "timestamp": _now(), **record})
            print(
                f"{config['model_name']} epoch {epoch} train {record['training_loss']:.4f} "
                f"val {record['validation_loss']:.4f} acc {record['validation_accuracy']:.4f} "
                f"macro_f1 {record['validation_macro_f1']:.4f}",
                flush=True,
            )
            score = record["validation_macro_f1"]
            if best_f1 is None or score > best_f1:
                best_f1 = score
                stall = 0
            else:
                stall += 1
            if should_stop(epoch, stall, int(config["min_epochs"]), int(config["early_stopping_patience"])):
                append_log_record(
                    log_path,
                    {
                        "record_type": "early_stop",
                        "timestamp": _now(),
                        "epoch": epoch,
                        "epochs_without_improvement": stall,
                        "metric": "validation_macro_f1",
                    },
                )
                break
        selected = select_checkpoint(epoch_rows)
        training_seconds = time.perf_counter() - started
        training_memory = memory_snapshot(device)
        prediction_path, prediction_count, first_id, last_id = predict_official_test(
            config,
            cache["test"],
            resolve_repo_path(selected["checkpoint"], repo=repo),
            device,
            repo,
        )
        metrics_text = "epoch,training_loss,validation_loss,validation_accuracy,validation_macro_f1,learning_rate,gradient_norm,runtime_seconds,examples_per_second,nan_count\n"
        for row in epoch_rows:
            metrics_text += (
                f"{row['epoch']},{row['training_loss']},{row['validation_loss']},{row['validation_accuracy']},"
                f"{row['validation_macro_f1']},{row['learning_rate']},{row['gradient_norm']},"
                f"{row['runtime_seconds']},{row['examples_per_second']},{row['nan_count']}\n"
            )
        metrics_path = write_unique_text(
            resolve_repo_path(config["output_directory"], repo=repo),
            config["experiment_name"] + "_epochs",
            ".csv",
            metrics_text,
        )
        ended_at = _now()
        manifest = build_manifest(
            member=config["member"],
            task=config["task"],
            experiment_name=config["experiment_name"],
            git_commit=git_commit_hash(repo),
            config_file=config_path,
            environment_file=config["environment_file"],
            raw_log=relative_log_path(log_path, repo=repo),
            checkpoint=selected["checkpoint"],
            metrics_file=to_repo_relative(metrics_path, repo=repo),
            outputs_directory=config["output_directory"],
            hardware=hardware,
            start_time=started_at,
            end_time=ended_at,
            total_training_time=training_seconds,
        )
        manifest["model_name"] = config["model_name"]
        manifest["dataset_revision"] = config["dataset_revision"]
        manifest["preprocessing_config"] = config["preprocessing_config"]
        manifest["vocabulary_file"] = config["vocabulary_file"]
        manifest["prediction_file"] = prediction_path
        manifest["parameter_count"] = count
        manifest_path = write_manifest(config["manifest_directory"], manifest, repo=repo)
        optimizer_examples = sum(float(row["examples_per_second"]) * float(row["train_runtime_seconds"]) for row in epoch_rows)
        optimizer_seconds = sum(float(row["train_runtime_seconds"]) for row in epoch_rows)
        summary = {
            "model_name": config["model_name"],
            "role": config["role"],
            "experiment_name": config["experiment_name"],
            "config_file": config_path,
            "architecture": config["architecture"],
            "embedding_dimension": int(config["embedding_dimension"]),
            "hidden_dimension": int(config["hidden_dimension"]),
            "num_layers": int(config["num_layers"]),
            "kernel_sizes": list(config["kernel_sizes"]),
            "pooling": config["pooling"],
            "dropout": float(config["dropout"]),
            "activation": config["activation"],
            "classifier_head": config["classifier_head"],
            "parameter_count": count,
            "optimizer": config["optimizer"],
            "learning_rate": float(config["learning_rate"]),
            "batch_size": int(config["batch_size"]),
            "max_epochs": int(config["epochs"]),
            "min_epochs": int(config["min_epochs"]),
            "early_stopping_patience": int(config["early_stopping_patience"]),
            "weight_decay": float(config["weight_decay"]),
            "gradient_clip_norm": float(config["gradient_clip_norm"]),
            "decision_threshold": float(config["decision_threshold"]),
            "threshold_tuned_on": config["threshold_tuned_on"],
            "seed": int(config["seed"]),
            "dataset_revision": config["dataset_revision"],
            "preprocessing_config": config["preprocessing_config"],
            "vocabulary_file": config["vocabulary_file"],
            "selected_epoch": int(selected["epoch"]),
            "selection_criterion": "validation_macro_f1",
            "selected_validation_macro_f1": float(selected["validation_macro_f1"]),
            "checkpoint": selected["checkpoint"],
            "epoch_checkpoints": {str(epoch): path for epoch, path in checkpoints.items()},
            "epochs": epoch_rows,
            "raw_log": relative_log_path(log_path, repo=repo),
            "metrics_file": to_repo_relative(metrics_path, repo=repo),
            "manifest": relative_manifest_path(manifest_path, repo=repo),
            "prediction_file": prediction_path,
            "prediction_rows": prediction_count,
            "first_example_id": first_id,
            "last_example_id": last_id,
            "hardware": hardware,
            "device_used": device.type,
            "start_time": started_at,
            "end_time": ended_at,
            "training_seconds": training_seconds,
            "examples_per_second": optimizer_examples / optimizer_seconds,
            "training_memory": training_memory,
            "peak_rss_bytes": training_memory.get("peak_rss_bytes"),
        }
        summary_path = write_unique_text(
            resolve_repo_path(config["output_directory"], repo=repo),
            config["experiment_name"] + "_summary",
            ".json",
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        )
        summary["summary_file"] = to_repo_relative(summary_path, repo=repo)
        append_log_record(
            log_path,
            {
                "record_type": "completed",
                "timestamp": _now(),
                "selected_epoch": summary["selected_epoch"],
                "selected_validation_macro_f1": summary["selected_validation_macro_f1"],
                "checkpoint": summary["checkpoint"],
                "prediction_file": prediction_path,
                "training_seconds": training_seconds,
                "parameter_count": count,
                "examples_per_second": summary["examples_per_second"],
                "peak_rss_bytes": summary["peak_rss_bytes"],
            },
        )
        print(f"TASK2_MODEL_SUMMARY={summary['summary_file']}", flush=True)
        return summary
    except Exception as exc:
        append_log_record(log_path, {"record_type": "failure", "timestamp": _now(), "error": str(exc)})
        raise


def predict_official_test(
    config: dict[str, Any],
    test_cache: dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    repo: Path,
) -> tuple[str, int, str, str]:
    """Score the official test cache in its stored order."""
    model = build_model(config).to(device)
    load_model_weights(model, checkpoint)
    model.eval()
    rows_out: list[dict[str, object]] = []
    relative_checkpoint = to_repo_relative(checkpoint, repo=repo)
    rows = int(test_cache["rows"])
    batch_size = int(config["batch_size"])
    with torch.no_grad():
        for start in range(0, rows, batch_size):
            indices = torch.arange(start, min(start + batch_size, rows))
            ids, labels, lengths = take_batch(test_cache, indices, device)
            logits = model(ids, lengths)
            probabilities = probabilities_from_logits(logits).detach().cpu()
            predictions = predict_label(probabilities[:, 1], float(config["decision_threshold"]))
            logits_cpu = logits.detach().cpu()
            labels_cpu = labels.detach().cpu()
            lengths_cpu = lengths.detach().cpu()
            for offset, row_index in enumerate(indices.tolist()):
                source_index = int(test_cache["source_index"][row_index])
                rows_out.append(
                    {
                        "example_id": example_id("test", source_index),
                        "source_index": source_index,
                        "true_label": int(labels_cpu[offset]),
                        "predicted_label": int(predictions[offset]),
                        "positive_probability": f"{float(probabilities[offset, 1]):.8f}",
                        "negative_probability": f"{float(probabilities[offset, 0]):.8f}",
                        "logit_negative": f"{float(logits_cpu[offset, 0]):.8f}",
                        "logit_positive": f"{float(logits_cpu[offset, 1]):.8f}",
                        "processed_token_length": int(test_cache["token_count"][row_index]),
                        "content_length": int(lengths_cpu[offset]),
                        "length_slice": slice_name(int(test_cache["slice_code"][row_index])),
                        "contains_negation": int(test_cache["negation"][row_index]),
                        "model_name": config["model_name"],
                        "checkpoint": relative_checkpoint,
                    }
                )
    output_dir = resolve_repo_path(config["output_directory"], repo=repo) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{config['experiment_name']}_test_predictions.csv"
    if path.exists():
        raise RuntimeError("The prediction file already exists and will not be overwritten.")
    write_prediction_file(path, rows_out)
    return to_repo_relative(path, repo=repo), len(rows_out), str(rows_out[0]["example_id"]), str(rows_out[-1]["example_id"])


def _write_json(directory: str, prefix: str, payload: dict[str, Any], repo: Path) -> str:
    path = write_unique_text(
        resolve_repo_path(directory, repo=repo),
        prefix,
        ".json",
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
    )
    return to_repo_relative(path, repo=repo)


def run_preflight(repo: Path) -> list[dict[str, Any]]:
    first = load_model_config(CONFIGS[0], repo=repo)
    cache = load_model_ready(first["model_ready_cache"], repo)
    counts = assert_phase5_populations(cache)
    if int(cache["maximum_length"]) != int(first["sequence_length"]):
        raise RuntimeError("Config sequence length does not match the Phase 5 cache.")
    if int(cache["vocabulary_size"]) != int(first["vocab_size"]):
        raise RuntimeError("Config vocabulary size does not match the Phase 5 cache.")
    if cache.get("pretrained_vectors_loaded") is not False:
        raise RuntimeError("The Phase 5 cache reports loaded external vectors.")
    reports = []
    for config_path in CONFIGS:
        config = load_model_config(config_path, repo=repo)
        print(f"PREFLIGHT {config['model_name']}", flush=True)
        report = preflight_model(config, cache, repo)
        report["populations"] = counts
        reports.append(report)
        print(
            f"PREFLIGHT {config['model_name']} gate {report['benchmark']['gate']} "
            f"estimate_s {report['benchmark']['estimated_runtime_seconds']:.1f} "
            f"ex_per_s {report['benchmark']['examples_per_second']:.1f}",
            flush=True,
        )
    path = _write_json(first["output_directory"], "task2_preflight", {"models": reports, "populations": counts}, repo)
    print(f"TASK2_PREFLIGHT={path}", flush=True)
    return reports


def _stream(command: list[str], repo: Path) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=environment,
    )
    captured: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        captured.append(line)
    return process.wait(), "".join(captured)


def run_all() -> int:
    """Preflight every model, then train only if the hardware gate passes."""
    repo = repo_root()
    reports = run_preflight(repo)
    estimates = [float(report["benchmark"]["estimated_runtime_seconds"]) for report in reports]
    blocked = [report["model_name"] for report in reports if report["benchmark"]["gate"] != "PASS"]
    if blocked or sum(estimates) > TOTAL_GATE_SECONDS:
        print("HARDWARE_GATE=BLOCKED", flush=True)
        print("REQUIRED_TRAINING_COMMAND=python3 -m task2_sentiment.drashti.src.train_models run", flush=True)
        return 2
    if any(not report["overfit"]["loss_decreased"] for report in reports):
        print("OVERFIT_CHECK=FAIL", flush=True)
        return 3
    for report in reports:
        if report["device_used"] != report["device_requested"]:
            print(
                f"DEVICE_FALLBACK {report['model_name']} {report['device_requested']} -> {report['device_used']}",
                flush=True,
            )
    summaries = []
    for config_path, report in zip(CONFIGS, reports):
        command = [
            sys.executable,
            "-m",
            "task2_sentiment.drashti.src.train_models",
            "train",
            config_path,
            report["device_used"],
        ]
        code, output = _stream(command, repo)
        if code != 0:
            print(f"TRAINING_FAILED {config_path} exit {code}", flush=True)
            return code
        marker = "TASK2_MODEL_SUMMARY="
        summary_path = next(line.split(marker, 1)[1].strip() for line in output.splitlines() if marker in line)
        summaries.append(json.loads(resolve_repo_path(summary_path, repo=repo).read_text(encoding="utf-8")))
    from task2_sentiment.drashti.src.report_docs import write_phase6_documents
    from task2_sentiment.drashti.src.predictions import ids_match, read_prediction_ids

    sequences = [read_prediction_ids(resolve_repo_path(item["prediction_file"], repo=repo)) for item in summaries]
    aligned = ids_match(sequences)
    write_phase6_documents(repo, reports, summaries, aligned)
    print(f"ALIGNED_PREDICTIONS={'PASS' if aligned else 'FAIL'}", flush=True)
    print("TASK2_TRAINING=COMPLETE", flush=True)
    return 0 if aligned else 4


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] == "run":
        return run_all()
    if arguments[0] == "preflight":
        run_preflight(repo_root())
        return 0
    if arguments[0] == "train":
        config_path = arguments[1]
        device_name = arguments[2] if len(arguments) > 2 else None
        repo = repo_root()
        config = load_model_config(config_path, repo=repo)
        cache = load_model_ready(config["model_ready_cache"], repo)
        assert_phase5_populations(cache)
        device = resolve_device(device_name or str(config["device"]))
        train_model(config, cache, device, repo, config_path)
        return 0
    raise SystemExit("Use run, preflight, or train.")


if __name__ == "__main__":
    raise SystemExit(main())
