# DATA 266 Lab 1

Deep Learning Experiments

This repository contains the team implementation, training artifacts, evaluation results, and reproducibility evidence for DATA 266 Lab 1.

The lab consists of three deep learning tasks:

1. Character level language modeling with a decoder only Transformer
2. Yelp Polarity sentiment classification using multiple neural architectures
3. Unpaired Monet and photograph image translation using CycleGAN

Each team member independently implements and trains models for all three tasks. Member specific source code, checkpoints, outputs, metrics, failure analyses, and results are organized within the corresponding task directories.

## Contributors

Drashti Shah

Sayani Brahmachari

Repository: DATA266 Lab1 Fall2026

## Repository Structure

```text
DATA266_Lab1/
├── task1_llm/
│   ├── data/
│   └── drashti/
│       ├── src/
│       ├── checkpoints/
│       ├── outputs/
│       ├── metrics_report.csv
│       ├── failure_analysis.md
│       └── results.md
│
├── task2_sentiment/
│   ├── data/
│   └── drashti/
│       ├── src/
│       ├── checkpoints/
│       ├── outputs/
│       ├── metrics_report.csv
│       ├── failure_analysis.md
│       └── results.md
│
├── task3_gan/
│   ├── data/
│   │   ├── monet_jpg/
│   │   └── photo_jpg/
│   └── drashti/
│       ├── src/
│       │   └── task3_gan.ipynb
│       ├── checkpoints/
│       │   └── refine_epoch_005.pt
│       ├── outputs/
│       │   ├── pred_A2B/
│       │   ├── pred_B2A/
│       │   ├── training_history.csv
│       │   ├── generator_loss.png
│       │   ├── cycle_loss.png
│       │   ├── identity_loss.png
│       │   ├── discriminator_a_loss.png
│       │   └── discriminator_b_loss.png
│       ├── evaluate_local.py
│       ├── submission.csv
│       ├── metrics_report.csv
│       ├── full_metrics_report.csv
│       ├── failure_analysis.md
│       └── results.md
│
├── reproducibility/
│   ├── manifests/
│   └── raw_logs/
│
├── report/
├── lab1/
├── .gitignore
├── .gitattributes
└── README.md
```

## Task 1: Character Level Language Model

Task 1 implements a decoder only Transformer language model from scratch for character level next token prediction.

Drashti's implementation is located at:

`task1_llm/drashti/`

The directory contains the source implementation, trained checkpoint, generated samples, training curves, metrics, failure analysis, and results documentation.

Primary files:

`task1_llm/drashti/metrics_report.csv`

`task1_llm/drashti/failure_analysis.md`

`task1_llm/drashti/results.md`

The retained training artifacts include the completed training continuation through epoch 20.

## Task 2: Yelp Polarity Sentiment Classification

Task 2 performs binary sentiment classification on the Yelp Polarity dataset using neural models trained from scratch.

Drashti's implementation is located at:

`task2_sentiment/drashti/`

Three independently trained architectures are evaluated:

1. Baseline mean embedding classifier
2. TextCNN
3. Bidirectional GRU

Evaluation artifacts include classification metrics, confidence intervals, calibration analysis, confusion matrices, ROC and PR curves, slice metrics, statistical comparison, prediction auditing, and manual error review.

Primary files:

`task2_sentiment/drashti/metrics_report.csv`

`task2_sentiment/drashti/failure_analysis.md`

`task2_sentiment/drashti/results.md`

## Task 3: CycleGAN Image Style Transfer

Task 3 implements CycleGAN for unpaired image translation between photographs and Monet paintings.

Drashti's implementation is located at:

`task3_gan/drashti/`

The complete executed notebook is:

`task3_gan/drashti/src/task3_gan.ipynb`

### Final Configuration

| Configuration | Value |
| --- | ---: |
| Training epochs | 100 |
| Steps per epoch | 1000 |
| Image size | 256 x 256 |
| Batch size | 1 |
| Generator base channels | 64 |
| Discriminator base channels | 64 |
| Residual blocks | 9 |
| Cycle weight | 10.0 |
| Identity weight | 1.0 |
| Generator learning rate | 0.0002 |
| Discriminator learning rate | 0.0001 |
| Adam beta1 | 0.5 |
| Adam beta2 | 0.999 |
| Replay buffer capacity | 50 |
| EMA decay | 0.999 |
| Parameter count | 28,275,336 |

### Training

The final recorded run completed 100 epochs.

| Metric | Epoch 1 | Epoch 100 |
| --- | ---: | ---: |
| Generator loss | 7.1919 | 2.6298 |
| Cycle loss | 0.5876 | 0.1486 |
| Identity loss | 0.5753 | 0.2305 |
| Discriminator A loss | 0.3197 | 0.1433 |
| Discriminator B loss | 0.3248 | 0.1766 |

Recorded training time was 7866.6 seconds, approximately 2.185 hours.

The complete epoch history is stored in:

`task3_gan/drashti/outputs/training_history.csv`

### Generated Images

Photo to Monet predictions:

`task3_gan/drashti/outputs/pred_A2B/`

Retained predictions: 300

Monet to Photo predictions:

`task3_gan/drashti/outputs/pred_B2A/`

Retained predictions: 300

### Checkpoint

The retained Task 3 checkpoint is:

`task3_gan/drashti/checkpoints/refine_epoch_005.pt`

The checkpoint is managed using Git LFS.

### Final Evaluation

| Metric | Photo to Monet | Monet to Photo | Overall |
| --- | ---: | ---: | ---: |
| FID | 95.355329 | 99.562875 | 97.459102 |
| MiFID | 0.406077 | 0.416181 | 0.411129 |
| KID | | | 0.052 |
| Generative precision | | | 0.63 |
| Generative recall | | | 0.51 |
| Cycle reconstruction L1 | | | 0.1486 |
| LPIPS | | | 0.30 |
| Content preservation cosine similarity | | | 0.81 |

Additional evaluation values are stored in:

`task3_gan/drashti/full_metrics_report.csv`

The submitted FID and MiFID values are preserved in:

`task3_gan/drashti/submission.csv`

### Kaggle

Team: `Pair_Programming_Team_12`

Leaderboard rank: 23

Leaderboard score: -48.9351

### Analysis

Detailed results:

`task3_gan/drashti/results.md`

Failure and stability analysis:

`task3_gan/drashti/failure_analysis.md`

### Local Evaluation

From the repository root:

```bash
python task3_gan/drashti/evaluate_local.py
```

## Reproducibility

Experiment manifests are stored under:

`reproducibility/manifests/`

Raw training logs are stored under:

`reproducibility/raw_logs/`

Task 3 manifest:

`reproducibility/manifests/drashti_task3_manifest.json`

Task 3 training log:

`reproducibility/raw_logs/task3_drashti_training.log`

## Data

Raw datasets and machine specific dataset caches are excluded from version control where appropriate.

Task 3 expects:

```text
task3_gan/data/
├── monet_jpg/
└── photo_jpg/
```

Local Hugging Face caches and downloaded dataset caches are excluded through `.gitignore`.

## Large Files

Large Task 3 model checkpoint files are managed using Git LFS.

Before cloning the checkpoint:

```bash
git lfs install
```

After cloning:

```bash
git lfs pull
```

## Final Report

The combined team report is stored under:

`report/`

## Version Control Notes

The repository excludes machine specific and generated development artifacts including operating system metadata, Python bytecode, cache directories, local dataset caches, credentials, and secrets.

Large trained model artifacts are managed through Git LFS where required.
