"""SMOKE TEST VALIDATION for the Task 1 data pipeline.

Uses a small prefix of the real TinyStories train split. This is not the
100,000/10,000 member split and does not mark Phase 2 complete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lab1.paths import repo_root

from task1_llm.drashti.src.acquire import DatasetAcquisitionError, load_tinystories
from task1_llm.drashti.src.sequences import build_batch, pair_from_text, shift_matches
from task1_llm.drashti.src.settings import load_task1_settings
from task1_llm.drashti.src.vocab import build_vocabulary

SMOKE_STORY_LIMIT = 32


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Task 1 data smoke test on a few real stories.")
    parser.add_argument("--config", default="configs/task1_drashti_data.json")
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve() if args.repo_root else repo_root()
    config = load_task1_settings(args.config, repo=repo)
    try:
        dataset, revision = load_tinystories(config, repo)
    except DatasetAcquisitionError as exc:
        print("SMOKE_TEST_VALIDATION=BLOCKED")
        print(exc)
        return 2
    text_field = config["text_field"]
    texts: list[str] = []
    scanned = 0
    for row in dataset:
        scanned += 1
        text = row[text_field]
        if isinstance(text, str) and text:
            texts.append(text)
        if len(texts) >= SMOKE_STORY_LIMIT:
            break
        if scanned >= 200 and len(texts) < 2:
            break
    if len(texts) < 2:
        print("SMOKE_TEST_VALIDATION=FAIL")
        print("Real TinyStories text was not retrieved.")
        return 1
    vocabulary = build_vocabulary(texts)
    first = texts[0]
    if vocabulary.decode(vocabulary.encode(first)) != first:
        print("SMOKE_TEST_VALIDATION=FAIL")
        print("Encode and decode did not round-trip the first real story.")
        return 1
    pairs = []
    for text in texts:
        pair = pair_from_text(text, vocabulary, config["sequence_length"])
        if pair is None:
            continue
        encoded = vocabulary.encode(text)
        if not shift_matches(encoded, pair[0], pair[1]):
            print("SMOKE_TEST_VALIDATION=FAIL")
            print("Target shift did not match the real story.")
            return 1
        pairs.append(pair)
        if len(pairs) >= min(config["batch_size"], 4):
            break
    if not pairs:
        print("SMOKE_TEST_VALIDATION=FAIL")
        print("No real story in the smoke sample was long enough for a sequence.")
        return 1
    batch = build_batch(pairs)
    print("VALIDATION_LEVEL=SMOKE_TEST_VALIDATION")
    print(f"dataset_id={config['dataset_id']}")
    print(f"dataset_revision={revision}")
    print(f"real_stories_used={len(texts)}")
    print(f"smoke_vocabulary_size={vocabulary.size}")
    print(f"batch_shape_input={[batch['batch_size'], batch['sequence_length']]}")
    print("note=Reduced real TinyStories prefix. Not the 100000/10000 member split.")
    print("SMOKE_TEST_VALIDATION=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
