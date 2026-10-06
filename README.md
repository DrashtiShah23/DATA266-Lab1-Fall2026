# DATA 266 Lab 1 — Deep Learning Experiments

This repository contains the team implementation and evaluation for DATA 266 Lab 1. The lab consists of three from-scratch deep-learning tasks: a character-level GPT-style language model, Yelp Polarity sentiment classification, and unpaired Monet/photo image translation with CycleGAN.

## Contributors

- **Sayani Brahmachari**
- **Drashti Shah**

Repository: [DATA266-Lab1-Fall2026](https://github.com/DrashtiShah23/DATA266-Lab1-Fall2026)

## Tasks

### Task 1 — Character-level GPT

Task 1 implements a GPT-style language model from scratch at the character level. The model uses learned token and positional embeddings, manually implemented causal multi-head self-attention, layer normalization, feed-forward layers, residual connections, cross-entropy loss, AdamW optimization, warm-up, and cosine learning-rate scheduling.

Recorded configuration:

- 100,000 training sequences and 10,000 validation sequences
- Block size: 128
- Embedding size: 256
- 4 attention heads and 4 layers
- 10 training epochs
- 3,236,864 trainable parameters

Recorded final metrics include validation loss **0.9381**, perplexity **2.5550**, bits per character **1.3533**, validation accuracy **0.7048**, generation throughput **489.7 tokens/s**, and peak memory **795.4 MB**.

### Task 2 — Yelp Polarity Sentiment Classification

Task 2 compares three models trained from scratch without pretrained embeddings or pretrained language models:

1. Mean-embedding MLP baseline
2. TextCNN
3. Bidirectional GRU

The run used 560,000 training examples, 38,000 test examples, a vocabulary of 214,511 tokens, and sequences padded/truncated to 160 tokens.

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.9316 | 0.9316 | 0.9808 | 0.9811 | 0.8632 |
| TextCNN | 0.9321 | 0.9320 | 0.9854 | 0.9856 | 0.8665 |
| BiGRU | **0.9494** | **0.9494** | **0.9893** | **0.9896** | **0.8989** |

The BiGRU achieved the strongest recorded test performance, with a peak memory use of 1,684.8 MB and throughput of 30,846 examples/s.

### Task 3 — Unpaired Monet/Photo Translation

Task 3 implements CycleGAN with two generators, two discriminators, adversarial least-squares loss, cycle-consistency loss, identity loss, replay buffers, checkpoint selection, and bidirectional translation.

The final executed notebook recorded:

- Python 3.11.17 and PyTorch 2.14.1+cu130
- NVIDIA GeForce RTX 5090 with CUDA
- 256×256 RGB images
- 300 Monet images and 7,038 photo images
- Batch size 1, 1,000 steps per epoch, and a 100-epoch training workflow
- 28,275,336 total parameters
- TA evaluation using 300 images in each real/generated set
- Photo → Monet FID: **95.355329**, MiFID: **0.406077**
- Monet → Photo FID: **99.562875**, MiFID: **0.416181**
- Average FID: **97.459102**, Average MiFID: **0.411129**

The notebook also creates the required `submission.csv` with ID 1 and the reported average FID/MiFID values.

## Reproducibility

The experiments were executed with fixed seed 266. The main deliverable is the executed all-parts Jupyter notebook. Supporting artifacts include model checkpoints, raw training logs, metric reports, generated images, failure analysis, and error-review files.

The recorded Task 1 and Task 2 environment used CUDA on an NVIDIA GeForce RTX 4090. The final Task 3 notebook used CUDA on an NVIDIA GeForce RTX 5090. Exact Python and PyTorch versions are recorded in the notebook and report.

## Repository contents

```text
.
├── README.md
├── DATA266_Lab1_All_Parts_Docker_executed.ipynb
├── task1_llm/
│   ├── checkpoints/
│   ├── outputs/
│   ├── raw_logs/
│   ├── metrics_report.csv
│   ├── results.md
│   └── failure_analysis.md
├── task2_sentiment/
│   ├── checkpoints/
│   ├── outputs/
│   ├── raw_logs/
│   ├── metrics_report.csv
│   ├── results.md
│   └── failure_analysis.md
└── task3_gan/
    ├── checkpoints/
    ├── outputs/
    ├── full_metrics_report.csv
    ├── submission.csv
    └── human_audit_template.csv
```

## Notes

- Raw datasets, credentials, tokens, and machine-specific absolute paths should not be committed.
- The notebook should be opened and executed in order so that setup, training, evaluation, and output-generation cells run consistently.
- The PDF/LaTeX technical report is provided as supplementary documentation for the experiment results.
