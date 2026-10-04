"""Training readiness audit and a short real-data benchmark.

This is not the final 10-epoch run. Its raw log stays separate from the final log.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from lab1.experiment_log import append_log_record, create_raw_log, relative_log_path
from lab1.git_info import git_commit_hash
from lab1.paths import repo_root, resolve_repo_path, to_repo_relative
from lab1.runtime import measure_runtime
from lab1.seed import seed_everything

from task1_llm.drashti.src.io_utils import write_repo_json
from task1_llm.drashti.src.model.gpt import GPT, parameter_count, spec_from_config
from task1_llm.drashti.src.settings import load_task1_settings
from task1_llm.drashti.src.train.corpus import load_corpus, steps_per_pass
from task1_llm.drashti.src.train.device_info import memory_snapshot, resolve_device, synchronize, training_hardware
from task1_llm.drashti.src.train.metrics import finite_number, perplexity
from task1_llm.drashti.src.train.schedule import learning_rate_for_step

REPORT_RELATIVE = "task1_llm/drashti/data_processed/readiness_report.json"
BENCHMARK_STEPS = 20


def audit(config_path: str = "configs/task1_drashti_train.json", repo: Path | None = None) -> dict[str, Any]:
    """Run the pre-training checks and estimate the 10-epoch runtime."""
    root = repo or repo_root()
    config = load_task1_settings(config_path, repo=root)
    _require_final_training_config(config)
    device = resolve_device(str(config["device"]))
    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'} {name}: {detail}", flush=True)

    corpus = load_corpus(config, root)
    record(
        "real_tinystories_loaded",
        corpus["dataset_revision"] == config["dataset_revision"] and int(corpus["train_windows"].shape[0]) > 0,
        f"revision {corpus['dataset_revision']}",
    )
    record(
        "member_split_counts",
        int(corpus["train_selected"]) == 100000 and int(corpus["validation_selected"]) == 10000,
        (
            f"selected train {corpus['train_selected']}, selected validation {corpus['validation_selected']}, "
            f"eligible train windows {int(corpus['train_windows'].shape[0])}, "
            f"skipped train {corpus['train_skipped']}, skipped validation {corpus['validation_skipped']}"
        ),
    )
    record(
        "vocabulary_size",
        int(corpus["vocab_size"]) == 115,
        f"vocab_size {corpus['vocab_size']}",
    )
    record(
        "sequence_length",
        int(corpus["sequence_length"]) == int(config["sequence_length"])
        and int(corpus["train_windows"].shape[1]) == int(config["sequence_length"]) + 1,
        f"sequence_length {config['sequence_length']}",
    )
    seed_everything(int(config["seed"]))
    model = GPT(spec_from_config(config, int(corpus["vocab_size"]))).to(device)
    counted = parameter_count(model)
    record("model_loads", counted > 0 and model.spec.sequence_length == int(config["sequence_length"]), f"parameters {counted}")
    source = (root / "task1_llm/drashti/src/model/gpt.py").read_text(encoding="utf-8")
    forbidden = (
        "Multihead" + "Attention",
        "nn." + "Transformer(",
        "nn." + "TransformerEncoder",
        "nn." + "TransformerDecoder",
        "scaled_dot_product_" + "attention",
    )
    record("causal_attention_source", all(name not in source for name in forbidden), "manual attention source check")
    inputs, targets = _first_batch(corpus["train_windows"], device)
    cpu_model = GPT(model.spec)
    cpu_model.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    causal_ok = _future_token_is_hidden(cpu_model, inputs.cpu())
    record("causal_attention_behavior", causal_ok, "eval-mode future token leaves earlier logits unchanged")
    model.train()
    _logits, loss = model(inputs, targets)
    record("cross_entropy_finite", bool(torch.isfinite(loss).item()), f"loss {float(loss.detach().cpu()):.6f}")
    loss.backward()
    grad_ok = any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    record("backpropagation", grad_ok, "at least one finite gradient")
    before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        betas=(float(config["adamw_beta1"]), float(config["adamw_beta2"])),
        weight_decay=float(config["weight_decay"]),
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
    optimizer.step()
    after = next(model.parameters()).detach()
    record("optimizer_step", not torch.equal(before, after), "a parameter changed")
    warmup_first = learning_rate_for_step(
        0,
        warmup_steps=int(config["warmup_steps"]),
        total_steps=int(config["warmup_steps"]) + 10,
        target_learning_rate=float(config["learning_rate"]),
        minimum_learning_rate=float(config["minimum_learning_rate"]),
    )
    warmup_second = learning_rate_for_step(
        1,
        warmup_steps=int(config["warmup_steps"]),
        total_steps=int(config["warmup_steps"]) + 10,
        target_learning_rate=float(config["learning_rate"]),
        minimum_learning_rate=float(config["minimum_learning_rate"]),
    )
    cosine_early = learning_rate_for_step(
        int(config["warmup_steps"]),
        warmup_steps=int(config["warmup_steps"]),
        total_steps=int(config["warmup_steps"]) + 10,
        target_learning_rate=float(config["learning_rate"]),
        minimum_learning_rate=float(config["minimum_learning_rate"]),
    )
    cosine_late = learning_rate_for_step(
        int(config["warmup_steps"]) + 9,
        warmup_steps=int(config["warmup_steps"]),
        total_steps=int(config["warmup_steps"]) + 10,
        target_learning_rate=float(config["learning_rate"]),
        minimum_learning_rate=float(config["minimum_learning_rate"]),
    )
    record(
        "scheduler_step",
        warmup_second > warmup_first and cosine_late < cosine_early,
        f"warmup {warmup_first:.8f} -> {warmup_second:.8f}; cosine {cosine_early:.8f} -> {cosine_late:.8f}",
    )
    checkpoint_ok, checkpoint_detail = _checkpoint_roundtrip(cpu_model, inputs.cpu(), root)
    record("checkpoint_write_and_restore", checkpoint_ok, checkpoint_detail)
    validation_ok = _validation_does_not_update(model, corpus["validation_windows"], device)
    record("validation_loop", validation_ok, "eval mode left parameters unchanged")
    metric_ok = _metric_accumulation_matches(cpu_model, corpus["train_windows"][:8])
    record("metric_accumulation", metric_ok, "token-weighted cross entropy matches the batch mean")
    log_ok, log_detail = _raw_log_roundtrip(config, root, device)
    record("raw_logging", log_ok, log_detail)
    hardware = training_hardware(device)
    record(
        "hardware_detection",
        hardware["torch_available"] is True and hardware["selected_device"] == device.type,
        f"device {device.type}, torch {hardware['torch_version']}",
    )
    with measure_runtime() as timing:
        time.sleep(0.01)
    record("runtime_measurement", timing["elapsed_seconds"] is not None and timing["elapsed_seconds"] >= 0, "elapsed seconds recorded")
    memory = memory_snapshot(device)
    record(
        "peak_memory",
        memory["peak_rss_bytes"] is not None,
        f"peak_rss_bytes {memory['peak_rss_bytes']}",
    )
    record("nan_detection", (not finite_number(float("nan"))) and finite_number(1.0), "NaN is rejected and a finite value is accepted")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])))
    record("gradient_norm", math.isfinite(grad_norm), f"norm {grad_norm:.6f}")

    benchmark = _benchmark(config, corpus, device, root)
    train_steps_per_epoch = steps_per_pass(int(corpus["train_windows"].shape[0]), int(config["batch_size"]))
    validation_steps_per_epoch = steps_per_pass(int(corpus["validation_windows"].shape[0]), int(config["batch_size"]))
    total_steps = train_steps_per_epoch * int(config["epochs"])
    estimated_seconds = (
        total_steps * benchmark["seconds_per_train_step"]
        + validation_steps_per_epoch * int(config["epochs"]) * benchmark["seconds_per_validation_step"]
    )
    gate_limit = float(config["hardware_gate_max_seconds"])
    audit_passed = all(item["passed"] for item in checks)
    gate = "PASS" if audit_passed and estimated_seconds <= gate_limit else "BLOCKED"
    report = {
        "audit_passed": audit_passed,
        "hardware_gate": gate,
        "hardware_gate_max_seconds": gate_limit,
        "device": device.type,
        "parameter_count": counted,
        "train_selected_stories": int(corpus["train_selected"]),
        "validation_selected_stories": int(corpus["validation_selected"]),
        "train_eligible_windows": int(corpus["train_windows"].shape[0]),
        "validation_eligible_windows": int(corpus["validation_windows"].shape[0]),
        "train_skipped_short_stories": int(corpus["train_skipped"]),
        "validation_skipped_short_stories": int(corpus["validation_skipped"]),
        "benchmark_steps": BENCHMARK_STEPS,
        "benchmark_tokens_per_second": benchmark["tokens_per_second"],
        "seconds_per_train_step": benchmark["seconds_per_train_step"],
        "seconds_per_validation_step": benchmark["seconds_per_validation_step"],
        "steps_per_epoch": train_steps_per_epoch,
        "validation_steps_per_epoch": validation_steps_per_epoch,
        "total_steps": total_steps,
        "estimated_10_epoch_seconds": estimated_seconds,
        "checks": checks,
        "hardware": hardware,
        "readiness_log": benchmark["log"],
        "final_training_launched": False,
    }
    write_repo_json(REPORT_RELATIVE, report, root)
    print(f"HARDWARE_GATE={gate}", flush=True)
    print(f"ESTIMATED_10_EPOCH_SECONDS={estimated_seconds:.1f}", flush=True)
    return report


def _require_final_training_config(config: dict[str, Any]) -> None:
    if int(config["epochs"]) < 10:
        raise RuntimeError("The final training config must request at least 10 epochs.")
    if config["optimizer"] != "AdamW":
        raise RuntimeError("The training config optimizer must be AdamW.")
    if config["scheduler"] != "linear_warmup_then_cosine":
        raise RuntimeError("The training config scheduler must be linear_warmup_then_cosine.")


def _first_batch(windows: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = windows[:8].to(dtype=torch.long, device=device)
    return chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()


@torch.no_grad()
def _future_token_is_hidden(model: nn.Module, inputs: torch.Tensor) -> bool:
    model.eval()
    original, _loss = model(inputs)
    changed = inputs.clone()
    changed[:, -1] = (changed[:, -1] + 1) % model.spec.vocab_size
    updated, _loss = model(changed)
    earlier = torch.allclose(original[:, :-1], updated[:, :-1], atol=1e-5, rtol=1e-4)
    last_changed = not torch.allclose(original[:, -1], updated[:, -1], atol=1e-5, rtol=1e-4)
    model.train()
    return bool(earlier and last_changed)


def _checkpoint_roundtrip(model: nn.Module, inputs: torch.Tensor, repo: Path) -> tuple[bool, str]:
    model.eval()
    with torch.no_grad():
        expected, _loss = model(inputs)
    path = resolve_repo_path("task1_llm/drashti/data_processed/readiness_checkpoint.pt", repo=repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, path)
    fresh = GPT(model.spec).to(inputs.device)
    loaded = torch.load(path, map_location=inputs.device, weights_only=True)
    fresh.load_state_dict(loaded["model_state"])
    fresh.eval()
    with torch.no_grad():
        restored, _loss = fresh(inputs)
    matches = torch.equal(expected, restored)
    model.train()
    return matches, to_repo_relative(path, repo=repo)


@torch.no_grad()
def _validation_does_not_update(model: nn.Module, windows: torch.Tensor, device: torch.device) -> bool:
    model.eval()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    chunk = windows[:4].to(dtype=torch.long, device=device)
    inputs, targets = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
    _logits, loss = model(inputs, targets)
    unchanged = all(torch.equal(left, right) for left, right in zip(before, model.parameters()))
    model.train()
    return bool(torch.isfinite(loss).item() and unchanged)


def _metric_accumulation_matches(model: nn.Module, windows: torch.Tensor) -> bool:
    """Pooled token mean must match cross entropy on the concatenated rows."""
    model.eval()
    pieces = []
    weighted_sum = 0.0
    token_count = 0
    with torch.no_grad():
        for start in (0, 4):
            chunk = windows[start : start + 4].to(dtype=torch.long)
            inputs, targets = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
            _logits, loss = model(inputs, targets)
            pieces.append((inputs, targets))
            weighted_sum += float(loss.item()) * int(targets.numel())
            token_count += int(targets.numel())
        combined_inputs = torch.cat([item[0] for item in pieces], dim=0)
        combined_targets = torch.cat([item[1] for item in pieces], dim=0)
        _logits, combined = model(combined_inputs, combined_targets)
    pooled = weighted_sum / token_count
    model.train()
    return math.isfinite(pooled) and math.isclose(pooled, float(combined.item()), rel_tol=0, abs_tol=1e-5)


def _raw_log_roundtrip(config: dict[str, Any], repo: Path, device: torch.device) -> tuple[bool, str]:
    hardware = training_hardware(device)
    first = create_raw_log(config["log_directory"], "task1_drashti_readiness_probe", config, git_commit_hash(repo), hardware, repo=repo)
    second = create_raw_log(config["log_directory"], "task1_drashti_readiness_probe", config, git_commit_hash(repo), hardware, repo=repo)
    append_log_record(first, {"record_type": "probe", "note": "readiness logging check"})
    text = first.read_text(encoding="utf-8")
    return first != second and "probe" in text, relative_log_path(first, repo=repo)


def _benchmark(config: dict[str, Any], corpus: dict[str, Any], device: torch.device, repo: Path) -> dict[str, Any]:
    seed_everything(int(config["seed"]))
    model = GPT(spec_from_config(config, int(corpus["vocab_size"]))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]))
    hardware = training_hardware(device)
    log_path = create_raw_log(
        config["log_directory"],
        "task1_drashti_readiness_benchmark",
        config,
        git_commit_hash(repo),
        hardware,
        repo=repo,
    )
    windows = corpus["train_windows"]
    batch_size = int(config["batch_size"])
    sequence_length = int(config["sequence_length"])
    step_seconds: list[float] = []
    tokens_per_second: list[float] = []
    model.train()
    for step in range(BENCHMARK_STEPS):
        start = (step * batch_size) % max(1, int(windows.shape[0]) - batch_size)
        chunk = windows[start : start + batch_size].to(dtype=torch.long, device=device)
        inputs, targets = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
        synchronize(device)
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(inputs, targets)
        if not torch.isfinite(loss):
            append_log_record(log_path, {"record_type": "failure", "reason": "non-finite benchmark loss"})
            raise RuntimeError("Benchmark loss was not finite.")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"])))
        optimizer.step()
        synchronize(device)
        elapsed = time.perf_counter() - started
        tokens = int(targets.numel())
        step_seconds.append(elapsed)
        tokens_per_second.append(tokens / elapsed)
        append_log_record(
            log_path,
            {
                "record_type": "benchmark_step",
                "global_step": step + 1,
                "training_cross_entropy": float(loss.detach().cpu()),
                "gradient_norm": grad_norm,
                "tokens": tokens,
                "step_seconds": elapsed,
                "tokens_per_second": tokens / elapsed,
            },
        )
    validation_chunk = corpus["validation_windows"][:batch_size].to(dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        synchronize(device)
        started = time.perf_counter()
        _logits, loss = model(validation_chunk[:, :-1], validation_chunk[:, 1:])
        synchronize(device)
        validation_seconds = time.perf_counter() - started
    if not torch.isfinite(loss):
        raise RuntimeError("Benchmark validation loss was not finite.")
    _ = perplexity(float(loss.detach().cpu()))
    mean_step = sum(step_seconds) / len(step_seconds)
    mean_rate = (batch_size * sequence_length) / mean_step
    return {
        "seconds_per_train_step": mean_step,
        "seconds_per_validation_step": validation_seconds,
        "tokens_per_second": mean_rate,
        "log": relative_log_path(log_path, repo=repo),
    }


def main() -> None:
    report = audit()
    if not report["audit_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
