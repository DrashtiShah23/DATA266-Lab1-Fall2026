"""Build the Yelp Polarity inventory, vocabulary, and model-ready ids.

The official test split is not used to fit the vocabulary, choose the sequence
length, or choose slice cutoffs. No embedding matrix is created.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from lab1.paths import resolve_repo_path

from task2_sentiment.drashti.src.acquire import EMBEDDING_POLICY, LABEL_MEANING, LABEL_SOURCE
from task2_sentiment.drashti.src.interface import make_batch
from task2_sentiment.drashti.src.numericalize import numericalize
from task2_sentiment.drashti.src.plots import write_class_bars, write_length_histogram
from task2_sentiment.drashti.src.preprocess import NEGATION_TOKENS, STOPWORDS, contains_negation, preprocess_text
from task2_sentiment.drashti.src.quality import _plain_label, audit_dataset_split, is_malformed, record_issues
from task2_sentiment.drashti.src.slices import assign_slices
from task2_sentiment.drashti.src.split import development_validation_indices, split_is_partition
from task2_sentiment.drashti.src.stats import nearest_rank, summarize_lengths
from task2_sentiment.drashti.src.vocabulary import build_vocabulary

_SLICE_CODE = {
    "short": 1,
    "medium": 2,
    "long": 3,
}


def prepare_task2(config: dict[str, Any], dataset: Any, repo: Path) -> dict[str, Any]:
    """Run the full real-data preparation and write the Phase 5 outputs."""
    text_field = config["text_field"]
    label_field = config["label_field"]
    print("AUDIT train", flush=True)
    train_audit = audit_dataset_split(dataset["train"], text_field, label_field)
    print("AUDIT test", flush=True)
    test_audit = audit_dataset_split(dataset["test"], text_field, label_field)
    train_count = len(dataset["train"])
    test_count = len(dataset["test"])
    development, validation = development_validation_indices(
        train_count,
        float(config["validation_fraction"]),
        int(config["validation_split_seed"]),
    )
    repeated_development, repeated_validation = development_validation_indices(
        train_count,
        float(config["validation_fraction"]),
        int(config["validation_split_seed"]),
    )
    if development != repeated_development or validation != repeated_validation:
        raise RuntimeError("The training-only validation split was not reproducible.")
    if not split_is_partition(train_count, development, validation):
        raise RuntimeError("The development and validation indices are not a partition of the official train split.")
    development_set = set(development)
    benchmark = _benchmark(dataset["train"], text_field, label_field)
    print(
        f"BENCHMARK rows={benchmark['rows']} seconds={benchmark['seconds']:.3f} rows_per_second={benchmark['rows_per_second']:.1f}",
        flush=True,
    )
    print("TOKENIZE train", flush=True)
    train_token_lengths, train_negation, train_failures, vocab_counts, vocab_documents = _scan_train(
        dataset["train"],
        text_field,
        label_field,
        development_set,
    )
    if len(train_token_lengths) != train_count:
        raise RuntimeError("The training scan did not cover every official training row.")
    development_lengths = [train_token_lengths[index] for index in development if train_token_lengths[index] is not None]
    ordered_lengths = sorted(development_lengths)
    maximum_length = max(1, nearest_rank(ordered_lengths, int(config["sequence_length_percentile"])))
    low_cutoff = nearest_rank(ordered_lengths, 25)
    high_cutoff = nearest_rank(ordered_lengths, 75)
    vocabulary = build_vocabulary(
        vocab_counts,
        int(config["minimum_token_frequency"]),
        int(config["maximum_vocabulary_size"]),
        vocab_documents,
    )
    if vocabulary.encode_token("phase5leakagetokenzz") != 1:
        raise RuntimeError("A token absent from training was added to the vocabulary.")
    print("NUMERICALIZE train", flush=True)
    development_cache, validation_cache = _numericalize_train(
        dataset["train"],
        development,
        validation,
        train_token_lengths,
        train_negation,
        text_field,
        label_field,
        vocabulary,
        maximum_length,
        low_cutoff,
        high_cutoff,
    )
    print("NUMERICALIZE test", flush=True)
    test_cache, test_lengths, test_negation, test_only_types, test_token_occurrences, test_unknown_occurrences = _numericalize_test(
        dataset["test"],
        text_field,
        label_field,
        vocabulary,
        vocab_counts,
        maximum_length,
        low_cutoff,
        high_cutoff,
    )
    if len(vocabulary) != vocabulary.metadata()["size"]:
        raise RuntimeError("Vocabulary size changed after the test split was scanned.")
    ready = {
        "development": development_cache,
        "validation": validation_cache,
        "test": test_cache,
        "maximum_length": maximum_length,
        "low_cutoff": low_cutoff,
        "high_cutoff": high_cutoff,
        "vocabulary_size": len(vocabulary),
    }
    _save_cache(config, repo, vocabulary, ready)
    batch = make_batch(development_cache, int(config["batch_size"]), "train")
    report = _report(
        config=config,
        train_audit=train_audit,
        test_audit=test_audit,
        train_count=train_count,
        test_count=test_count,
        development=development,
        validation=validation,
        development_lengths=development_lengths,
        train_token_lengths=train_token_lengths,
        train_negation=train_negation,
        train_failures=train_failures,
        maximum_length=maximum_length,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        vocabulary_size=len(vocabulary),
        vocab_documents=vocab_documents,
        ready=ready,
        batch=batch,
        benchmark=benchmark,
        test_only_types=test_only_types,
        test_token_occurrences=test_token_occurrences,
        test_unknown_occurrences=test_unknown_occurrences,
        test_lengths=test_lengths,
        test_negation=test_negation,
    )
    report["columns"] = {
        "train": list(dataset["train"].column_names),
        "test": list(dataset["test"].column_names),
    }
    _write_outputs(config, repo, report, train_audit, test_audit)
    return report


def _benchmark(split: Any, text_field: str, label_field: str) -> dict[str, float]:
    started = time.perf_counter()
    rows = 0
    for batch in split.iter(batch_size=1000):
        for text, label in zip(batch[text_field], batch[label_field]):
            label = _plain_label(label)
            if isinstance(text, str) and isinstance(label, int) and not isinstance(label, bool):
                preprocess_text(text)
            rows += 1
            if rows >= 1000:
                elapsed = time.perf_counter() - started
                return {"rows": 1000, "seconds": elapsed, "rows_per_second": 1000 / elapsed if elapsed else 0.0}
    elapsed = time.perf_counter() - started
    return {"rows": rows, "seconds": elapsed, "rows_per_second": rows / elapsed if elapsed else 0.0}


def _scan_train(
    split: Any,
    text_field: str,
    label_field: str,
    development_set: set[int],
) -> tuple[list[int | None], bytearray, int, Counter[str], int]:
    lengths: list[int | None] = []
    negation = bytearray()
    failures = 0
    counts: Counter[str] = Counter()
    documents = 0
    seen = 0
    for batch in split.iter(batch_size=1000):
        for text, label in zip(batch[text_field], batch[label_field]):
            issues = record_issues(text, label)
            if is_malformed(issues):
                lengths.append(None)
                negation.append(0)
            else:
                try:
                    tokens = preprocess_text(text)
                except Exception:
                    failures += 1
                    lengths.append(None)
                    negation.append(0)
                else:
                    lengths.append(len(tokens))
                    negation.append(1 if contains_negation(tokens) else 0)
                    if seen in development_set:
                        counts.update(tokens)
                        documents += 1
            seen += 1
            if seen % 50000 == 0:
                print(f"TOKENIZE train {seen}", flush=True)
    return lengths, negation, failures, counts, documents


def _numericalize_train(
    split: Any,
    development: list[int],
    validation: list[int],
    token_lengths: list[int | None],
    negation_flags: bytearray,
    text_field: str,
    label_field: str,
    vocabulary: Any,
    maximum_length: int,
    low_cutoff: int,
    high_cutoff: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    development_ready = [index for index in development if token_lengths[index]]
    validation_ready = [index for index in validation if token_lengths[index]]
    caches = {
        "development": _empty_cache(len(development_ready), maximum_length, "train"),
        "validation": _empty_cache(len(validation_ready), maximum_length, "train"),
    }
    positions = {
        "development": {index: row for row, index in enumerate(development_ready)},
        "validation": {index: row for row, index in enumerate(validation_ready)},
    }
    seen = 0
    for batch in split.iter(batch_size=1000):
        for text, label in zip(batch[text_field], batch[label_field]):
            label = _plain_label(label)
            for name in ("development", "validation"):
                row = positions[name].get(seen)
                if row is not None:
                    _write_row(
                        caches[name],
                        row,
                        seen,
                        text,
                        label,
                        vocabulary,
                        maximum_length,
                        low_cutoff,
                        high_cutoff,
                        negation_flags,
                    )
            seen += 1
            if seen % 50000 == 0:
                print(f"NUMERICALIZE train {seen}", flush=True)
    return caches["development"], caches["validation"]


def _empty_cache(rows: int, maximum_length: int, split_name: str) -> dict[str, Any]:
    return {
        "input_ids": torch.zeros((rows, maximum_length), dtype=torch.int16),
        "labels": torch.zeros(rows, dtype=torch.int8),
        "content_length": torch.zeros(rows, dtype=torch.int16),
        "token_count": torch.zeros(rows, dtype=torch.int32),
        "source_index": torch.zeros(rows, dtype=torch.int32),
        "negation": torch.zeros(rows, dtype=torch.int8),
        "slice_code": torch.zeros(rows, dtype=torch.int8),
        "truncated_rows": 0,
        "padded_rows": 0,
        "rows": rows,
        "example_id_prefix": split_name,
    }


def _write_row(
    cache: dict[str, Any],
    row: int,
    source: int,
    text: str,
    label: int,
    vocabulary: Any,
    maximum_length: int,
    low_cutoff: int,
    high_cutoff: int,
    negation_flags: bytearray,
) -> None:
    tokens = preprocess_text(text)
    coded = numericalize(tokens, vocabulary, maximum_length)
    cache["input_ids"][row] = torch.tensor(coded["input_ids"], dtype=torch.int16)
    cache["labels"][row] = int(label)
    cache["content_length"][row] = int(coded["content_length"])
    cache["token_count"][row] = int(coded["token_count"])
    cache["source_index"][row] = source
    cache["negation"][row] = int(negation_flags[source])
    assigned = assign_slices(int(coded["token_count"]), bool(negation_flags[source]), low_cutoff, high_cutoff)
    cache["slice_code"][row] = _SLICE_CODE[assigned[0]]
    cache["truncated_rows"] += int(bool(coded["truncated"]))
    cache["padded_rows"] += int(bool(coded["padded"]))


def _numericalize_test(
    split: Any,
    text_field: str,
    label_field: str,
    vocabulary: Any,
    training_counts: Counter[str],
    maximum_length: int,
    low_cutoff: int,
    high_cutoff: int,
) -> tuple[dict[str, Any], list[int | None], bytearray, int, int, int]:
    ready_indices: list[int] = []
    lengths: list[int | None] = []
    negation = bytearray()
    test_only: set[str] = set()
    occurrences = 0
    unknown_occurrences = 0
    seen = 0
    for batch in split.iter(batch_size=1000):
        for text, label in zip(batch[text_field], batch[label_field]):
            label = _plain_label(label)
            issues = record_issues(text, label)
            if is_malformed(issues):
                lengths.append(None)
                negation.append(0)
            else:
                tokens = preprocess_text(text)
                lengths.append(len(tokens))
                negation.append(1 if contains_negation(tokens) else 0)
                if tokens:
                    ready_indices.append(seen)
                for token in tokens:
                    occurrences += 1
                    if vocabulary.encode_token(token) == 1:
                        unknown_occurrences += 1
                        if training_counts[token] == 0:
                            test_only.add(token)
            seen += 1
            if seen % 20000 == 0:
                print(f"SCAN test {seen}", flush=True)
    cache = _empty_cache(len(ready_indices), maximum_length, "test")
    position = {index: row for row, index in enumerate(ready_indices)}
    seen = 0
    for batch in split.iter(batch_size=1000):
        for text, label in zip(batch[text_field], batch[label_field]):
            label = _plain_label(label)
            row = position.get(seen)
            if row is not None:
                _write_row(cache, row, seen, text, label, vocabulary, maximum_length, low_cutoff, high_cutoff, negation)
            seen += 1
    return cache, lengths, negation, len(test_only), occurrences, unknown_occurrences


def _save_cache(config: dict[str, Any], repo: Path, vocabulary: Any, ready: dict[str, Any]) -> None:
    directory = resolve_repo_path(config["processed_cache"], repo=repo)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "development": ready["development"],
        "validation": ready["validation"],
        "test": ready["test"],
        "maximum_length": ready["maximum_length"],
        "vocabulary_size": ready["vocabulary_size"],
        "embedding_policy": EMBEDDING_POLICY,
        "pretrained_vectors_loaded": False,
    }
    torch.save(payload, directory / "model_ready.pt")
    metadata = vocabulary.metadata()
    metadata["token_to_id"] = vocabulary.token_to_id
    (directory / "vocabulary.json").write_text(json.dumps(metadata, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def _length_block(values: list[int]) -> dict[str, float | int]:
    return summarize_lengths(values)


def _without_lengths(audit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if key not in ("character_lengths", "word_lengths")}


def _coverage(cache: dict[str, Any], maximum_length: int) -> dict[str, float | int]:
    rows = int(cache["rows"])
    truncated = int(cache["truncated_rows"])
    padded = int(cache["padded_rows"])
    return {
        "model_ready_rows": rows,
        "maximum_length": maximum_length,
        "truncated_rows": truncated,
        "truncated_percentage": truncated / rows if rows else 0.0,
        "rows_fitting_without_truncation": rows - truncated,
        "percentage_fitting_without_truncation": (rows - truncated) / rows if rows else 0.0,
        "padded_rows": padded,
        "padded_percentage": padded / rows if rows else 0.0,
    }


def _slice_counts(lengths: list[int | None], negation: bytearray, indices: list[int] | None, low: int, high: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    chosen = range(len(lengths)) if indices is None else indices
    for index in chosen:
        length = lengths[index]
        if length is None:
            continue
        for name in assign_slices(length, bool(negation[index]), low, high):
            counts[name] += 1
    return dict(counts)


def _report(**kwargs: Any) -> dict[str, Any]:
    config = kwargs["config"]
    train_audit = kwargs["train_audit"]
    test_audit = kwargs["test_audit"]
    development_lengths = kwargs["development_lengths"]
    train_lengths = [value for value in kwargs["train_token_lengths"] if value is not None]
    batch = kwargs["batch"]
    ready = kwargs["ready"]
    return {
        "dataset_id": config["dataset_id"],
        "dataset_source": config["dataset_source"],
        "dataset_revision": config["dataset_revision"],
        "text_field": config["text_field"],
        "label_field": config["label_field"],
        "label_source": LABEL_SOURCE,
        "label_meaning": {str(key): value for key, value in LABEL_MEANING.items()},
        "cache_dir": config["cache_dir"],
        "embedding_policy": EMBEDDING_POLICY,
        "pretrained_vectors_loaded": False,
        "stemming_or_lemmatization": config["stemming_or_lemmatization"],
        "stopword_count": len(STOPWORDS),
        "negation_tokens_preserved": sorted(NEGATION_TOKENS),
        "train_records": kwargs["train_count"],
        "test_records": kwargs["test_count"],
        "train_quality": _without_lengths(train_audit),
        "test_quality": _without_lengths(test_audit),
        "train_character_lengths": _length_block(train_audit["character_lengths"]),
        "train_word_lengths": _length_block(train_audit["word_lengths"]),
        "test_character_lengths": _length_block(test_audit["character_lengths"]),
        "test_word_lengths": _length_block(test_audit["word_lengths"]),
        "train_processed_token_lengths": _length_block(train_lengths),
        "development_processed_token_lengths": _length_block(development_lengths),
        "validation_split": {
            "source": "official train split only",
            "seed": int(config["validation_split_seed"]),
            "fraction": float(config["validation_fraction"]),
            "procedure": "Fisher-Yates shuffle of official train indices with random.Random(seed); validation is the first floor(fraction * train_count) indices; both lists are then sorted",
            "development_count": len(kwargs["development"]),
            "validation_count": len(kwargs["validation"]),
            "disjoint": True,
            "covers_official_train": True,
            "reproducible": True,
            "test_rows_used": 0,
        },
        "vocabulary": {
            "fit_on": "development portion of the official train split",
            "training_only": True,
            "documents_fit": kwargs["vocab_documents"],
            "minimum_frequency": int(config["minimum_token_frequency"]),
            "maximum_size": int(config["maximum_vocabulary_size"]),
            "reserved_tokens": ["<pad>", "<unk>"],
            "size": kwargs["vocabulary_size"],
            "test_rows_used_for_fitting": 0,
        },
        "sequence_length": {
            "selected": kwargs["maximum_length"],
            "rule": config["sequence_length_rule"],
            "percentile": int(config["sequence_length_percentile"]),
            "basis": "development-training preprocessed token counts",
            "development": _coverage(ready["development"], kwargs["maximum_length"]),
            "validation": _coverage(ready["validation"], kwargs["maximum_length"]),
            "test": _coverage(ready["test"], kwargs["maximum_length"]),
        },
        "slices": {
            "low_cutoff": kwargs["low_cutoff"],
            "high_cutoff": kwargs["high_cutoff"],
            "rule": config["slice_policy"],
            "development": _slice_counts(kwargs["train_token_lengths"], kwargs["train_negation"], kwargs["development"], kwargs["low_cutoff"], kwargs["high_cutoff"]),
            "validation": _slice_counts(kwargs["train_token_lengths"], kwargs["train_negation"], kwargs["validation"], kwargs["low_cutoff"], kwargs["high_cutoff"]),
            "test": _slice_counts(kwargs["test_lengths"], kwargs["test_negation"], None, kwargs["low_cutoff"], kwargs["high_cutoff"]),
        },
        "preprocessing_failures": kwargs["train_failures"],
        "tokenization_failures": kwargs["train_failures"],
        "test_only_vocabulary_types": kwargs["test_only_types"],
        "test_token_occurrences": kwargs["test_token_occurrences"],
        "test_unknown_token_occurrences": kwargs["test_unknown_occurrences"],
        "absent_token_encodes_to_unknown": True,
        "batch": {
            "input_ids_shape": list(batch["input_ids"].shape),
            "labels_shape": list(batch["labels"].shape),
            "content_length_shape": list(batch["content_length"].shape),
            "example_ids": batch["example_ids"],
            "first_example_id": batch["example_ids"][0],
            "last_example_id": batch["example_ids"][-1],
        },
        "benchmark": kwargs["benchmark"],
        "rows_deleted": 0,
    }


def _write_outputs(config: dict[str, Any], repo: Path, report: dict[str, Any], train_audit: dict[str, Any], test_audit: dict[str, Any]) -> None:
    output_dir = resolve_repo_path(config["output_directory"], repo=repo)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_length_histogram(
        output_dir / "review_length_distribution.svg",
        train_audit["character_lengths"],
        "Official training review character length",
        "characters",
    )
    write_class_bars(
        output_dir / "class_distribution.svg",
        [
            ("train", [train_audit["class_counts"]["0"], train_audit["class_counts"]["1"]]),
            ("test", [test_audit["class_counts"]["0"], test_audit["class_counts"]["1"]]),
        ],
        ["negative (0)", "positive (1)"],
        "Yelp Polarity class counts",
    )
    _write_json(output_dir / "data_quality_summary.json", {"train": report["train_quality"], "test": report["test_quality"]})
    _write_json(
        output_dir / "review_length_statistics.json",
        {
            "character_length_unit": "Python len of the official review string",
            "word_length_unit": "whitespace-separated pieces of the official review string, before preprocessing",
            "processed_token_length_unit": "tokens after preprocessing, before truncation",
            "train_character_lengths": report["train_character_lengths"],
            "train_word_lengths": report["train_word_lengths"],
            "test_character_lengths": report["test_character_lengths"],
            "test_word_lengths": report["test_word_lengths"],
            "train_processed_token_lengths": report["train_processed_token_lengths"],
            "development_processed_token_lengths": report["development_processed_token_lengths"],
        },
    )
    _write_json(
        output_dir / "class_statistics.json",
        {
            "label_meaning": report["label_meaning"],
            "label_source": report["label_source"],
            "train": report["train_quality"]["class_counts"],
            "train_percentages": report["train_quality"]["class_percentages"],
            "test": report["test_quality"]["class_counts"],
            "test_percentages": report["test_quality"]["class_percentages"],
            "train_imbalance_ratio": report["train_quality"]["class_imbalance_ratio_majority_over_minority"],
            "test_imbalance_ratio": report["test_quality"]["class_imbalance_ratio_majority_over_minority"],
        },
    )
    _write_json(
        output_dir / "preprocessing_statistics.json",
        {
            "stemming_or_lemmatization": report["stemming_or_lemmatization"],
            "negation_tokens_preserved": report["negation_tokens_preserved"],
            "stopword_count": report["stopword_count"],
            "preprocessing_failures": report["preprocessing_failures"],
            "tokenization_failures": report["tokenization_failures"],
            "vocabulary": report["vocabulary"],
            "sequence_length": report["sequence_length"],
            "embedding_policy": report["embedding_policy"],
            "benchmark": report["benchmark"],
        },
    )
    _write_json(output_dir / "slice_statistics.json", report["slices"])
    _write_json(output_dir / "real_validation_report.json", report)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
    personal_prefixes = ("/" + "Users" + "/", "/" + "home" + "/")
    if any(prefix in text for prefix in personal_prefixes):
        raise RuntimeError("Refusing to write a report that contains a personal path.")
    path.write_text(text + "\n", encoding="utf-8")
