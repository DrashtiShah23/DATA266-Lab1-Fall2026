"""Full Yelp Polarity validation for Phase 5.

Unit tests and the 32-row smoke command are not this check.
"""

from __future__ import annotations

from lab1.paths import repo_root

from task2_sentiment.drashti.src.acquire import load_yelp_polarity
from task2_sentiment.drashti.src.pipeline import prepare_task2
from task2_sentiment.drashti.src.settings import load_task2_settings


def main() -> None:
    config = load_task2_settings()
    dataset = load_yelp_polarity(config, repo_root())
    report = prepare_task2(config, dataset, repo_root())
    print("REAL_DATASET_VALIDATION=PASS", flush=True)
    print(f"TRAIN_RECORDS={report['train_records']}", flush=True)
    print(f"TEST_RECORDS={report['test_records']}", flush=True)
    print(f"VOCABULARY_SIZE={report['vocabulary']['size']}", flush=True)
    print(f"MAXIMUM_SEQUENCE_LENGTH={report['sequence_length']['selected']}", flush=True)
    print(f"BATCH_SHAPE={report['batch']['input_ids_shape']}", flush=True)


if __name__ == "__main__":
    main()
