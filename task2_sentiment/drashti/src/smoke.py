"""Smoke checks on a small prefix of the real Yelp Polarity training split.

This is not the full-dataset validation. It does not fit the Phase 6 vocabulary
and it does not write the model-ready cache.
"""

from __future__ import annotations

from lab1.paths import repo_root

from task2_sentiment.drashti.src.acquire import load_yelp_polarity
from task2_sentiment.drashti.src.interface import example_id, make_batch
from task2_sentiment.drashti.src.numericalize import convert_label, numericalize
from task2_sentiment.drashti.src.preprocess import preprocess_text
from task2_sentiment.drashti.src.settings import load_task2_settings
from task2_sentiment.drashti.src.vocabulary import build_vocabulary

SMOKE_ROWS = 32


def main() -> None:
    config = load_task2_settings()
    dataset = load_yelp_polarity(config, repo_root())
    rows: list[tuple[str, int]] = []
    for batch in dataset["train"].iter(batch_size=SMOKE_ROWS):
        for text, label in zip(batch[config["text_field"]], batch[config["label_field"]]):
            if not isinstance(text, str) or isinstance(label, bool) or not isinstance(label, int):
                raise RuntimeError("The real smoke prefix contained a row the loader could not read.")
            rows.append((text, label))
            if len(rows) == SMOKE_ROWS:
                break
        break
    if len(rows) != SMOKE_ROWS:
        raise RuntimeError("The real Yelp training split did not yield 32 rows.")
    from collections import Counter

    counts: Counter[str] = Counter()
    token_rows = []
    for text, _label in rows:
        tokens = preprocess_text(text)
        token_rows.append(tokens)
        counts.update(tokens)
    vocabulary = build_vocabulary(counts, minimum_frequency=1, maximum_size=1000, documents_fit=SMOKE_ROWS)
    coded = [numericalize(tokens, vocabulary, 16) for tokens in token_rows]
    import torch

    cache = {
        "input_ids": torch.tensor([row["input_ids"] for row in coded], dtype=torch.int16),
        "labels": torch.tensor([convert_label(label) for _text, label in rows], dtype=torch.int8),
        "content_length": torch.tensor([row["content_length"] for row in coded], dtype=torch.int16),
        "source_index": torch.arange(SMOKE_ROWS, dtype=torch.int32),
    }
    batch = make_batch(cache, batch_size=4, id_prefix="train")
    if list(batch["input_ids"].shape) != [4, 16]:
        raise RuntimeError("Smoke batch input shape was not [4, 16].")
    if batch["example_ids"] != [example_id("train", index) for index in range(4)]:
        raise RuntimeError("Smoke example ids were not stable.")
    print("SMOKE_TEST_VALIDATION=PASS", flush=True)
    print(f"SMOKE_ROWS={SMOKE_ROWS}", flush=True)
    print(f"SMOKE_VOCABULARY_SIZE={len(vocabulary)}", flush=True)
    print("SMOKE_BATCH_SHAPE=4,16", flush=True)


if __name__ == "__main__":
    main()
