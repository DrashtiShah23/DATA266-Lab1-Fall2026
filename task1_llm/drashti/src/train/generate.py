"""Greedy and temperature decoding for the character GPT."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from task1_llm.drashti.src.train.device_info import synchronize


@torch.no_grad()
def generate_characters(
    model: nn.Module,
    prompt_ids: list[int],
    *,
    max_new_characters: int,
    method: str,
    temperature: float | None,
    seed: int,
) -> list[int]:
    """Append ``max_new_characters`` ids to ``prompt_ids``.

    ``greedy`` takes the argmax. ``temperature`` draws from the softmax of
    logits divided by ``temperature``. Sampling uses a CPU generator.
    """
    if method not in ("greedy", "temperature"):
        raise ValueError("method must be greedy or temperature.")
    if method == "temperature" and (temperature is None or temperature <= 0):
        raise ValueError("temperature sampling requires a positive temperature.")
    if max_new_characters < 1:
        raise ValueError("max_new_characters must be positive.")
    model.eval()
    device = next(model.parameters()).device
    sequence_length = model.spec.sequence_length
    generated = list(prompt_ids)
    for offset in range(max_new_characters):
        window = generated[-sequence_length:]
        tokens = torch.tensor([window], dtype=torch.long, device=device)
        logits, _loss = model(tokens)
        next_logits = logits[0, -1].detach().float().cpu()
        if method == "greedy":
            nxt = int(torch.argmax(next_logits).item())
        else:
            probabilities = torch.softmax(next_logits / float(temperature), dim=-1)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + offset)
            nxt = int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())
        generated.append(nxt)
    return generated


@torch.no_grad()
def measure_generation_tokens_per_second(
    model: nn.Module,
    prompt_ids: list[int],
    *,
    new_characters: int,
    repeats: int,
) -> dict[str, Any]:
    """Time greedy generation. This measurement is not the training throughput."""
    device = next(model.parameters()).device
    rates: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        generate_characters(
            model,
            prompt_ids,
            max_new_characters=new_characters,
            method="greedy",
            temperature=None,
            seed=0,
        )
        synchronize(device)
        elapsed = time.perf_counter() - started
        rates.append(new_characters / elapsed)
    ordered = sorted(rates)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 == 1 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return {
        "method": "greedy",
        "new_characters": new_characters,
        "repeats": repeats,
        "tokens_per_second_each_run": rates,
        "tokens_per_second_median": median,
    }
