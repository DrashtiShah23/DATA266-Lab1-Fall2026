"""Linear warmup followed by cosine decay.

The warmup unit is an optimizer step. Step 0, the first optimizer step, uses
``target_learning_rate / warmup_steps``. The rate then increases linearly and
equals the target on the last warmup step. Later steps follow cosine decay
from the target down to ``minimum_learning_rate`` on the final optimizer step.
"""

from __future__ import annotations

import math


def learning_rate_for_step(
    step_index: int,
    *,
    warmup_steps: int,
    total_steps: int,
    target_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    """Return the learning rate used for optimizer step ``step_index`` (0-based)."""
    if isinstance(step_index, bool) or not isinstance(step_index, int) or step_index < 0:
        raise ValueError("step_index must be a non-negative integer.")
    if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
        raise ValueError("warmup_steps must be a positive integer.")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps < warmup_steps:
        raise ValueError("total_steps must be an integer >= warmup_steps.")
    if target_learning_rate <= 0 or minimum_learning_rate < 0:
        raise ValueError("Learning rates must be positive, and the minimum must be >= 0.")
    if minimum_learning_rate > target_learning_rate:
        raise ValueError("minimum_learning_rate cannot exceed the target learning rate.")
    if step_index < warmup_steps:
        return target_learning_rate * float(step_index + 1) / float(warmup_steps)
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 1:
        return target_learning_rate
    progress = (step_index - warmup_steps) / float(decay_steps - 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_learning_rate + (target_learning_rate - minimum_learning_rate) * cosine


class WarmupCosineSchedule:
    """Applies ``learning_rate_for_step`` to one optimizer."""

    def __init__(
        self,
        optimizer: object,
        *,
        warmup_steps: int,
        total_steps: int,
        target_learning_rate: float,
        minimum_learning_rate: float,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.target_learning_rate = target_learning_rate
        self.minimum_learning_rate = minimum_learning_rate
        self.completed_steps = 0

    def current_learning_rate(self) -> float:
        return learning_rate_for_step(
            self.completed_steps,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            target_learning_rate=self.target_learning_rate,
            minimum_learning_rate=self.minimum_learning_rate,
        )

    def apply(self) -> float:
        learning_rate = self.current_learning_rate()
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate

    def advance(self) -> float:
        """Record that one optimizer step finished and return the next rate."""
        self.completed_steps += 1
        return self.current_learning_rate()


def held_continuation_learning_rate(restored_learning_rate: float) -> float:
    """Learning rate for epochs 11 through 20.

    The epoch 10 checkpoint stored ``schedule_completed_steps`` equal to the
    original 10-epoch step count, so the original cosine has already reached
    its minimum. Later indexes on that same schedule stay at the minimum
    because cosine progress is clamped at 1. This function returns that
    restored rate and does not open a new warmup.

    A cosine restretched over 20 epochs is not used. At step 15630 that
    restretched curve is still near the middle of the decay and would raise
    the learning rate above the value stored in the optimizer.
    """
    if isinstance(restored_learning_rate, bool) or not isinstance(restored_learning_rate, (int, float)):
        raise ValueError("restored_learning_rate must be a positive number.")
    if restored_learning_rate <= 0:
        raise ValueError("restored_learning_rate must be positive.")
    return float(restored_learning_rate)
