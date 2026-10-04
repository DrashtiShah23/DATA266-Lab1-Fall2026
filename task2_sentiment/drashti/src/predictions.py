"""Aligned prediction files for the three sentiment models.

The files store ids, labels, and scores. They do not store review text.
"""

from __future__ import annotations

import csv
from pathlib import Path


PREDICTION_FIELDS = (
    "example_id",
    "source_index",
    "true_label",
    "predicted_label",
    "positive_probability",
    "negative_probability",
    "logit_negative",
    "logit_positive",
    "processed_token_length",
    "content_length",
    "length_slice",
    "contains_negation",
    "model_name",
    "checkpoint",
)


def write_prediction_file(path: Path, rows: list[dict[str, object]]) -> None:
    """Write prediction rows in the given order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            missing = [field for field in PREDICTION_FIELDS if field not in row]
            if missing:
                raise ValueError("Prediction row is missing fields: " + ", ".join(missing))
            writer.writerow({field: row[field] for field in PREDICTION_FIELDS})


def read_prediction_ids(path: Path) -> list[str]:
    """Return example ids in file order."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != PREDICTION_FIELDS:
            raise ValueError("Prediction file schema does not match the required columns.")
        return [row["example_id"] for row in reader]


def ids_match(sequences: list[list[str]]) -> bool:
    """True when every sequence is the same length and the same order."""
    if len(sequences) < 2:
        return False
    first = sequences[0]
    return all(len(other) == len(first) and other == first for other in sequences[1:])
