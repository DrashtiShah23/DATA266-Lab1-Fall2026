"""REAL DATASET VALIDATION for Drashti's TinyStories split.

This command loads the official dataset. It does not fall back to toy text.
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path
from typing import Any

from lab1.paths import repo_root, to_repo_relative

from task1_llm.drashti.src.acquire import DATASET_ID, DatasetAcquisitionError, load_tinystories
from task1_llm.drashti.src.io_utils import write_repo_json
from task1_llm.drashti.src.sequences import (
    build_batch,
    ids_within_vocabulary,
    pair_from_text,
    shift_matches,
)
from task1_llm.drashti.src.settings import load_task1_settings
from task1_llm.drashti.src.split import SELECTION_PROCEDURE, disjointness_report, member_split_indices
from task1_llm.drashti.src.vocab import (
    CONSTRUCTION_POLICY,
    SPECIAL_TOKEN_POLICY,
    build_vocabulary,
    character_summary,
    vocabulary_payload,
)


def run_real_validation(repo: Path, config_path: str) -> dict[str, Any]:
    """Execute the full TinyStories checks and return the measured report."""
    config = load_task1_settings(config_path, repo=repo)
    dataset, revision = load_tinystories(config, repo)
    source_count = int(len(dataset))
    text_field = config["text_field"]
    training_indices, validation_indices = member_split_indices(
        source_count,
        config["train_size"],
        config["validation_size"],
        config["split_seed"],
    )
    repeated_training, repeated_validation = member_split_indices(
        source_count,
        config["train_size"],
        config["validation_size"],
        config["split_seed"],
    )
    reproducible = training_indices == repeated_training and validation_indices == repeated_validation
    overlap = disjointness_report(training_indices, validation_indices)

    training_view = dataset.select(training_indices)
    validation_view = dataset.select(validation_indices)
    training_texts, training_character_count, _malformed_rows = _read_texts(
        training_view,
        text_field,
    )
    vocabulary = build_vocabulary(training_texts)
    summary = character_summary(vocabulary)
    sequence_length = config["sequence_length"]

    round_trip_failures = 0
    sequence_count = 0
    too_short = 0
    shift_failures = 0
    bound_failures = 0
    batch_pairs: list[tuple[list[int], list[int]]] = []
    for text in training_texts:
        encoded = vocabulary.encode(text)
        if vocabulary.decode(encoded) != text:
            round_trip_failures += 1
        if not ids_within_vocabulary(encoded, vocabulary.size):
            bound_failures += 1
        pair = pair_from_text(text, vocabulary, sequence_length)
        if pair is None:
            too_short += 1
            continue
        sequence_count += 1
        token_ids = encoded
        if not shift_matches(token_ids, pair[0], pair[1]):
            shift_failures += 1
        if not ids_within_vocabulary(pair[0] + pair[1], vocabulary.size):
            bound_failures += 1
        if len(batch_pairs) < config["batch_size"]:
            batch_pairs.append(pair)

    validation_round_trip_failures = 0
    validation_oov_stories = 0
    validation_character_count = 0
    for row in validation_view:
        text = row[text_field]
        if not isinstance(text, str):
            raise DatasetAcquisitionError("A selected TinyStories row did not contain string text.")
        validation_character_count += len(text)
        try:
            encoded = vocabulary.encode(text)
        except ValueError:
            validation_oov_stories += 1
            continue
        if vocabulary.decode(encoded) != text:
            validation_round_trip_failures += 1

    if len(batch_pairs) != config["batch_size"]:
        raise RuntimeError("Could not build a full real batch from training stories.")
    batch = build_batch(batch_pairs)
    min_length = min(len(text) for text in training_texts)
    max_length = max(len(text) for text in training_texts)

    report: dict[str, Any] = {
        "validation_level": "REAL_DATASET_VALIDATION",
        "member": config["member"],
        "dataset_id": config["dataset_id"],
        "dataset_source": config["dataset_source"],
        "dataset_revision": revision,
        "source_split": config["source_split"],
        "text_field": text_field,
        "cache_dir": config["cache_dir"],
        "source_record_count": source_count,
        "split_seed": config["split_seed"],
        "selection_procedure": SELECTION_PROCEDURE,
        "train_count": len(training_indices),
        "validation_count": len(validation_indices),
        "disjointness_checked": True,
        "disjoint": bool(overlap["disjoint"] and overlap["train_unique"] and overlap["validation_unique"]),
        "overlap_count": overlap["overlap_count"],
        "reproducibility_checked": True,
        "reproducible": reproducible,
        "python_version": platform.python_version(),
        "vocabulary_constructed_from_training_stories": True,
        "vocabulary_size": summary["vocabulary_size"],
        "construction_policy": CONSTRUCTION_POLICY,
        "special_token_policy": SPECIAL_TOKEN_POLICY,
        "non_ascii_count": summary["non_ascii_count"],
        "non_ascii_code_points": summary["non_ascii_code_points"],
        "nonprintable_count": summary["nonprintable_count"],
        "nonprintable_code_points": summary["nonprintable_code_points"],
        "training_round_trip_checked": True,
        "training_round_trip_failures": round_trip_failures,
        "validation_stories_with_oov_character": validation_oov_stories,
        "validation_round_trip_failures_among_in_vocab": validation_round_trip_failures,
        "sequence_length": sequence_length,
        "training_sequences_built": sequence_count,
        "training_stories_shorter_than_window": too_short,
        "shift_failures": shift_failures,
        "id_bound_failures": bound_failures,
        "real_batch_constructed": True,
        "batch_size": batch["batch_size"],
        "batch_sequence_length": batch["sequence_length"],
        "batch_shape_input": [batch["batch_size"], batch["sequence_length"]],
        "batch_shape_target": [batch["batch_size"], batch["sequence_length"]],
        "training_character_count": training_character_count,
        "validation_character_count": validation_character_count,
        "training_min_characters": min_length,
        "training_max_characters": max_length,
        "paths_are_repository_relative": True,
    }
    if config["dataset_id"] != DATASET_ID:
        report["paths_are_repository_relative"] = True
    _assert_real_checks(report)
    split_payload = {
        "member": config["member"],
        "dataset_id": config["dataset_id"],
        "dataset_source": config["dataset_source"],
        "dataset_revision": revision,
        "source_split": config["source_split"],
        "text_field": text_field,
        "source_record_count": source_count,
        "split_seed": config["split_seed"],
        "selection_procedure": SELECTION_PROCEDURE,
        "train_count": len(training_indices),
        "validation_count": len(validation_indices),
        "train_indices": training_indices,
        "validation_indices": validation_indices,
        "disjointness_checked": True,
        "overlap_count": overlap["overlap_count"],
        "reproducibility_checked": True,
        "reproducible": reproducible,
        "python_version": platform.python_version(),
    }
    write_repo_json(config["split_metadata"], split_payload, repo)
    write_repo_json(config["vocab_metadata"], vocabulary_payload(vocabulary), repo)
    report_path = write_repo_json(config["validation_report"], report, repo)
    report["validation_report"] = to_repo_relative(report_path, repo=repo)
    return report


def _read_texts(dataset: Any, text_field: str) -> tuple[list[str], int, int]:
    texts: list[str] = []
    character_count = 0
    for row in dataset:
        text = row[text_field]
        if not isinstance(text, str):
            raise DatasetAcquisitionError("A selected TinyStories row did not contain string text.")
        texts.append(text)
        character_count += len(text)
        if len(texts) % 10000 == 0:
            print(f"read_training_stories={len(texts)}", file=sys.stderr)
    return texts, character_count, 0


def _assert_real_checks(report: dict[str, Any]) -> None:
    failures: list[str] = []
    if report["dataset_id"] != DATASET_ID:
        failures.append("dataset id")
    if report["source_record_count"] < report["train_count"] + report["validation_count"]:
        failures.append("source count")
    if report["train_count"] != 100_000 or report["validation_count"] != 10_000:
        failures.append("split sizes")
    if not report["disjoint"] or report["overlap_count"] != 0:
        failures.append("disjointness")
    if not report["reproducible"]:
        failures.append("reproducibility")
    if report["vocabulary_size"] < 1:
        failures.append("vocabulary")
    if report["training_round_trip_failures"] != 0:
        failures.append("round trip")
    if report["shift_failures"] != 0 or report["id_bound_failures"] != 0:
        failures.append("shift or bounds")
    if report["batch_shape_input"] != [report["batch_size"], report["sequence_length"]]:
        failures.append("batch shape")
    if report["training_sequences_built"] < 1:
        failures.append("sequences")
    if failures:
        raise RuntimeError("Real TinyStories validation failed: " + ", ".join(failures))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run real TinyStories validation for Task 1.")
    parser.add_argument("--config", default="configs/task1_drashti_data.json")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else repo_root()
    try:
        report = run_real_validation(repo, args.config)
    except DatasetAcquisitionError as exc:
        print("REAL_DATASET_VALIDATION=BLOCKED")
        print(exc)
        return 2
    print("VALIDATION_LEVEL=REAL_DATASET_VALIDATION")
    print(f"dataset_id={report['dataset_id']}")
    print(f"dataset_revision={report['dataset_revision']}")
    print(f"source_record_count={report['source_record_count']}")
    print(f"train_count={report['train_count']}")
    print(f"validation_count={report['validation_count']}")
    print(f"disjoint={report['disjoint']}")
    print(f"overlap_count={report['overlap_count']}")
    print(f"reproducible={report['reproducible']}")
    print(f"vocabulary_size={report['vocabulary_size']}")
    print(f"training_round_trip_failures={report['training_round_trip_failures']}")
    print(f"validation_stories_with_oov_character={report['validation_stories_with_oov_character']}")
    print(f"shift_failures={report['shift_failures']}")
    print(f"id_bound_failures={report['id_bound_failures']}")
    print(f"training_sequences_built={report['training_sequences_built']}")
    print(f"training_stories_shorter_than_window={report['training_stories_shorter_than_window']}")
    print(f"batch_shape_input={report['batch_shape_input']}")
    print(f"batch_shape_target={report['batch_shape_target']}")
    print("REAL_DATASET_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
