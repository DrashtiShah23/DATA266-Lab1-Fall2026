"""GPT language model built from embeddings, attention, and feed-forward blocks.

Attention is computed with matrix multiplication, scaling, a causal mask,
and softmax. This file does not call a prebuilt attention or Transformer module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from lab1.config import ConfigError


@dataclass(frozen=True)
class GPTSpec:
    """Architecture values for one GPT."""

    vocab_size: int
    sequence_length: int
    embedding_dimension: int
    num_heads: int
    num_blocks: int
    feedforward_dimension: int
    dropout: float


def spec_from_config(config: dict, vocab_size: int) -> GPTSpec:
    """Read the Task 1 architecture from the config and the measured vocabulary."""
    dimensions = config["model_dimensions"]
    if not isinstance(dimensions, dict):
        raise ConfigError("model_dimensions must be an object.")
    spec = GPTSpec(
        vocab_size=vocab_size,
        sequence_length=int(config["sequence_length"]),
        embedding_dimension=int(config["embedding_dimension"]),
        num_heads=int(dimensions["num_heads"]),
        num_blocks=int(dimensions["num_blocks"]),
        feedforward_dimension=int(dimensions["feedforward_dimension"]),
        dropout=float(config["dropout"]),
    )
    if spec.embedding_dimension % spec.num_heads != 0:
        raise ConfigError("embedding_dimension must be divisible by num_heads.")
    if min(spec.vocab_size, spec.sequence_length, spec.num_blocks, spec.feedforward_dimension) < 1:
        raise ConfigError("Architecture sizes must be positive.")
    return spec


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using explicit score computation."""

    def __init__(self, spec: GPTSpec):
        super().__init__()
        self.num_heads = spec.num_heads
        self.head_dimension = spec.embedding_dimension // spec.num_heads
        self.query = nn.Linear(spec.embedding_dimension, spec.embedding_dimension)
        self.key = nn.Linear(spec.embedding_dimension, spec.embedding_dimension)
        self.value = nn.Linear(spec.embedding_dimension, spec.embedding_dimension)
        self.output_projection = nn.Linear(spec.embedding_dimension, spec.embedding_dimension)
        self.attention_dropout = nn.Dropout(spec.dropout)
        mask = torch.tril(torch.ones(spec.sequence_length, spec.sequence_length))
        self.register_buffer("causal_mask", mask.view(1, 1, spec.sequence_length, spec.sequence_length), persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _channels = hidden.shape
        query = self._split_heads(self.query(hidden), batch_size, sequence_length)
        key = self._split_heads(self.key(hidden), batch_size, sequence_length)
        value = self._split_heads(self.value(hidden), batch_size, sequence_length)
        scores = query @ key.transpose(-2, -1)
        scores = scores / (self.head_dimension ** 0.5)
        causal = self.causal_mask[:, :, :sequence_length, :sequence_length]
        scores = scores.masked_fill(causal == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights)
        mixed = weights @ value
        combined = mixed.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return self.output_projection(combined)

    def _split_heads(self, projected: torch.Tensor, batch_size: int, sequence_length: int) -> torch.Tensor:
        return projected.view(batch_size, sequence_length, self.num_heads, self.head_dimension).transpose(1, 2)


class LayerNorm(nn.Module):
    """Layer normalization with learnable scale and shift."""

    def __init__(self, dimension: int, epsilon: float = 1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(dimension))
        self.beta = nn.Parameter(torch.zeros(dimension))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        mean = hidden.mean(dim=-1, keepdim=True)
        variance = hidden.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (hidden - mean) / torch.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta


class FeedForward(nn.Module):
    """Position-wise feed-forward network with a GELU nonlinearity."""

    def __init__(self, spec: GPTSpec):
        super().__init__()
        self.expand = nn.Linear(spec.embedding_dimension, spec.feedforward_dimension)
        self.project = nn.Linear(spec.feedforward_dimension, spec.embedding_dimension)
        self.dropout = nn.Dropout(spec.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = self.expand(hidden)
        hidden = torch.nn.functional.gelu(hidden)
        hidden = self.project(hidden)
        return self.dropout(hidden)


class TransformerBlock(nn.Module):
    """Pre-norm block with residual attention and residual feed-forward paths."""

    def __init__(self, spec: GPTSpec):
        super().__init__()
        self.attention_norm = LayerNorm(spec.embedding_dimension)
        self.attention = CausalSelfAttention(spec)
        self.feedforward_norm = LayerNorm(spec.embedding_dimension)
        self.feedforward = FeedForward(spec)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        hidden = hidden + self.feedforward(self.feedforward_norm(hidden))
        return hidden


class GPT(nn.Module):
    """Character-level GPT. Token ids in, next-character logits out."""

    def __init__(self, spec: GPTSpec):
        super().__init__()
        self.spec = spec
        self.token_embedding = nn.Embedding(spec.vocab_size, spec.embedding_dimension)
        self.position_embedding = nn.Embedding(spec.sequence_length, spec.embedding_dimension)
        self.embedding_dropout = nn.Dropout(spec.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(spec) for _ in range(spec.num_blocks))
        self.final_norm = LayerNorm(spec.embedding_dimension)
        self.language_model_head = nn.Linear(spec.embedding_dimension, spec.vocab_size, bias=False)
        self.apply(_init_module)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence).")
        _batch_size, sequence_length = token_ids.shape
        if sequence_length > self.spec.sequence_length:
            raise ValueError("Sequence is longer than the configured context length.")
        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        logits = self.language_model_head(hidden)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def parameter_count(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _init_module(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
