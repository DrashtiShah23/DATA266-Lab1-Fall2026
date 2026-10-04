"""Tiny overfit check. This is not the Phase 4 training run."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from lab1.paths import repo_root

from task1_llm.drashti.src.io_utils import write_repo_json
from task1_llm.drashti.src.model.gpt import GPT, parameter_count, spec_from_config
from task1_llm.drashti.src.model.real_batch import load_real_training_batch
from task1_llm.drashti.src.settings import load_task1_settings

OVERFIT_STEPS = 80
OVERFIT_LEARNING_RATE = 3e-3
OVERFIT_BATCH_SIZE = 4


def run_tiny_overfit(repo: Path | None = None, config_path: str = "configs/task1_drashti_data.json") -> dict[str, float | int | bool]:
    """Train on a few repeated real sequences and report the measured losses."""
    root = repo or repo_root()
    config = load_task1_settings(config_path, repo=root)
    inputs, targets, vocab_size = load_real_training_batch(config, root, batch_size=OVERFIT_BATCH_SIZE)
    torch.manual_seed(config["seed"])
    model = GPT(spec_from_config(config, vocab_size))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=OVERFIT_LEARNING_RATE)
    losses: list[float] = []
    for _step in range(OVERFIT_STEPS):
        optimizer.zero_grad(set_to_none=True)
        _logits, loss = model(inputs, targets)
        if loss is None or not torch.isfinite(loss):
            raise RuntimeError("Overfit loss was missing or non-finite.")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    start = losses[0]
    end = losses[-1]
    report = {
        "steps": OVERFIT_STEPS,
        "learning_rate": OVERFIT_LEARNING_RATE,
        "batch_size": OVERFIT_BATCH_SIZE,
        "sequence_length": int(inputs.shape[1]),
        "vocab_size": vocab_size,
        "parameter_count": parameter_count(model),
        "initial_loss": start,
        "final_loss": end,
        "loss_decreased": end < start,
    }
    write_repo_json("task1_llm/drashti/data_processed/overfit_report.json", report, root)
    return report


def main() -> int:
    report = run_tiny_overfit()
    print(json.dumps(report, sort_keys=True))
    print("TINY_OVERFIT=PASS" if report["loss_decreased"] else "TINY_OVERFIT=FAIL")
    return 0 if report["loss_decreased"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
