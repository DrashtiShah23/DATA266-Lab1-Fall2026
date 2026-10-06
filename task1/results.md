# Task 1 Results

Architecture: character-level GPT from scratch with learned token and positional embeddings, manually implemented causal multi-head self-attention, layer normalization, feed-forward network, residual connections, tied language-model head, AdamW, linear warm-up, and cosine scheduling.

Configuration: {
  "block_size": 128,
  "batch_size": 64,
  "n_embd": 256,
  "n_head": 4,
  "n_layer": 4,
  "dropout": 0.1,
  "lr": 0.0003,
  "weight_decay": 0.1,
  "warmup_steps": 500,
  "epochs": 10,
  "max_train_sequences": 100000,
  "max_val_sequences": 10000,
  "eval_batches": 100,
  "sample_tokens": 300
}

The executed metrics are in `metrics_report.csv` and `outputs/metrics.json`; the unedited epoch log is `raw_logs/training_log.json`; the checkpoint is `checkpoints/tiny_gpt.pt`; and the sample/loss curve are in `outputs/`.

Before submission, explain the observed loss/generalization gap and complete `failure_analysis.md` using the three actual snippets in `outputs/failure_cases.csv`.
