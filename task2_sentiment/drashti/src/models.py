"""Three from-scratch Yelp sentiment classifiers.

Each model owns a randomly initialized embedding table. Nothing in this
module loads an external embedding matrix or a pretrained language model.
"""

from __future__ import annotations

import torch
from torch import nn

from task2_sentiment.drashti.src.vocabulary import PAD_ID


def _init_embedding(embedding: nn.Embedding) -> None:
    nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
    with torch.no_grad():
        embedding.weight[PAD_ID].zero_()


def _content_mask(content_length: torch.Tensor, length: int) -> torch.Tensor:
    positions = torch.arange(length, device=content_length.device).unsqueeze(0)
    return positions < content_length.unsqueeze(1)


class MeanPoolClassifier(nn.Module):
    """Masked average of token embeddings, then a linear classifier."""

    def __init__(self, vocab_size: int, embedding_dimension: int, dropout: float):
        super().__init__()
        self.architecture = "mean_pool"
        self.embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dimension, 2)
        _init_embedding(self.embedding)

    def forward(self, input_ids: torch.Tensor, content_length: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        mask = _content_mask(content_length, input_ids.shape[1]).unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / content_length.clamp(min=1).unsqueeze(1).to(embedded.dtype)
        return self.classifier(self.dropout(pooled))


class ConvClassifier(nn.Module):
    """Parallel convolutions over token embeddings with masked max pooling."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dimension: int,
        conv_channels: int,
        kernel_sizes: list[int],
        dropout: float,
    ):
        super().__init__()
        self.architecture = "conv"
        self.kernel_sizes = list(kernel_sizes)
        self.embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx=PAD_ID)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(embedding_dimension, conv_channels, kernel_size=kernel) for kernel in self.kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(conv_channels * len(self.kernel_sizes), 2)
        _init_embedding(self.embedding)

    def forward(self, input_ids: torch.Tensor, content_length: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = [self._pool(convolution, embedded, content_length) for convolution in self.convolutions]
        return self.classifier(self.dropout(torch.cat(pooled, dim=1)))

    def _pool(self, convolution: nn.Conv1d, embedded: torch.Tensor, content_length: torch.Tensor) -> torch.Tensor:
        activated = torch.relu(convolution(embedded))
        kernel = convolution.kernel_size[0]
        valid = content_length - kernel + 1
        positions = torch.arange(activated.shape[2], device=activated.device).unsqueeze(0)
        invalid = positions >= valid.unsqueeze(1)
        activated = activated.masked_fill(invalid.unsqueeze(1), torch.finfo(activated.dtype).min)
        pooled = activated.max(dim=2).values
        pooled = torch.where(valid.unsqueeze(1) > 0, pooled, torch.zeros_like(pooled))
        return pooled


class GRUClassifier(nn.Module):
    """One-layer GRU. The classifier reads the last real hidden state."""

    def __init__(self, vocab_size: int, embedding_dimension: int, hidden_dimension: int, dropout: float):
        super().__init__()
        self.architecture = "gru"
        self.embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx=PAD_ID)
        self.gru = nn.GRU(embedding_dimension, hidden_dimension, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dimension, 2)
        _init_embedding(self.embedding)

    def forward(self, input_ids: torch.Tensor, content_length: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            content_length.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _output, hidden = self.gru(packed)
        return self.classifier(self.dropout(hidden[-1]))


def build_model(config: dict) -> nn.Module:
    """Build the architecture named by the config."""
    vocab_size = int(config["vocab_size"])
    embedding_dimension = int(config["embedding_dimension"])
    dropout = float(config["dropout"])
    architecture = config["architecture"]
    if architecture == "mean_pool":
        return MeanPoolClassifier(vocab_size, embedding_dimension, dropout)
    if architecture == "conv":
        return ConvClassifier(
            vocab_size,
            embedding_dimension,
            int(config["conv_channels"]),
            [int(size) for size in config["kernel_sizes"]],
            dropout,
        )
    if architecture == "gru":
        return GRUClassifier(vocab_size, embedding_dimension, int(config["hidden_dimension"]), dropout)
    raise ValueError(f"Unknown architecture {architecture}.")


def zero_padding_row(model: nn.Module) -> None:
    """Keep the padding embedding at zero after optimizer weight decay."""
    embedding = getattr(model, "embedding", None)
    if embedding is None:
        return
    with torch.no_grad():
        embedding.weight[PAD_ID].zero_()
