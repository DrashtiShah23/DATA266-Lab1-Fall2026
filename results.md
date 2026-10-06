# Task 2 Results

Three models were trained from scratch: a mean-embedding MLP baseline, a TextCNN, and a bidirectional GRU. No pretrained embeddings or language models were used. Text was lowercased, punctuation/special characters were removed, stopwords were removed, and sequences were truncated/padded to MAXLEN=160.

The complete per-model metrics are in `metrics_report.csv` and `outputs/metrics_report.json`. Exact hardware is recorded in each model record and `hardware_disclosure.json`. Unedited epoch logs are in `raw_logs/`; checkpoints are in `checkpoints/`; and the required 20-error candidate review is `outputs/error_review_20.csv`.

The highest test macro-F1 in this run was `bigru` with `0.9494`. Explain why the models differ, discuss calibration/robustness slices, and complete the manual error-review fields before submission.
