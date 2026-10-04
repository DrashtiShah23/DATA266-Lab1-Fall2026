# Task 1 results

The assignment minimum was 10 epochs. The same GPT was then trained through epoch 20 by resuming the epoch 10 checkpoint. The epoch 10 evidence is preserved and is not replaced by this file.

## Architecture

- Embedding dimension: 128
- Heads: 4
- Blocks: 4
- Feed-forward dimension: 512
- Dropout: 0.1
- Sequence length: 128
- Vocabulary size: 115
- Trainable parameters: 839168
- Attention is manual scaled dot-product with a causal mask. Token and position embeddings are learned. The language-model head is untied. The architecture was not changed for the continuation.

## Hyperparameters

- Optimizer: AdamW
- Target learning rate of the original cosine: 0.0003
- Minimum learning rate: 3e-05
- Warmup steps, used only in the original run: 200
- Original scheduler: linear warmup then cosine, completed at step 15630
- Continuation scheduler: hold_restored_learning_rate
- Weight decay: 0.1
- AdamW betas: 0.9, 0.95
- Gradient clip norm: 1.0
- Batch size: 64
- Total epochs: 20
- Seed: 266
- Dropout: 0.1

## Continuation learning rate

The epoch 10 checkpoint stored schedule_completed_steps at 15630, which is the last step of the original 10-epoch cosine. Evaluating that cosine at any later step index stays at the minimum learning rate because progress is clamped at 1. Epochs 11 through 20 therefore hold the learning rate found in the restored optimizer, 0.00003. Warmup is not repeated. A new cosine stretched over 31260 steps is not used: at step 15630 that curve would still be near mid-decay and would raise the learning rate above the restored value.

- Restored optimizer learning rate: 3e-05
- Warmup was not run again.
- Python, NumPy, and PyTorch random state were not in the epoch 10 checkpoint. Epoch shuffle order still follows `random.Random(seed + epoch)`. Dropout after the resume is a new random stream.

## Dataset and split

- Dataset: roneneldan/TinyStories
- Revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
- Split seed: 266
- Selected training stories: 100000
- Selected validation stories: 10000
- Eligible training windows: 99987
- Eligible validation windows: 9999
- Training stories skipped as shorter than 129 characters: 13
- Validation stories skipped as shorter than 129 characters: 1
- Short stories remain in the 100,000 and 10,000 split. They are not padded.

## Training procedure

Epochs 1 through 10 used AdamW, linear warmup, and cosine decay. Epochs 11 through 20 resumed the epoch 10 model and optimizer and held the restored learning rate. Validation ran in eval mode at the end of every epoch and did not update parameters. Final metrics use the checkpoint with the lowest validation cross entropy, not automatically epoch 20.

- Resumed from: `task1_llm/drashti/checkpoints/task1_drashti_train_20260928T070850656698Z_epoch10_step15630.pt`
- Lowest validation epoch: 20
- Lowest validation cross entropy: 0.5075388650844569
- Selected checkpoint: `task1_llm/drashti/checkpoints/task1_drashti_train_continue_20260928T075219828443Z_epoch20_step31260.pt`
- Selection rule: Lowest validation cross entropy across epochs 1 through 20. An equal loss would keep the earlier epoch. Epoch 20 is saved even when it is not selected.
- Epoch 20 checkpoint, preserved either way: `task1_llm/drashti/checkpoints/task1_drashti_train_continue_20260928T075219828443Z_epoch20_step31260.pt`

## Hardware

- Selected device: mps
- CPU model: Apple M4
- GPU model: Apple M4
- GPU memory bytes: None
- Unified memory bytes: 17179869184
- PyTorch: 2.14.0
- Memory note: Apple Silicon uses unified memory. No separate GPU memory figure is exposed, so gpu_memory_bytes is null.

## Checkpoint, time, and evidence

- Original raw log: `reproducibility/raw_logs/task1_drashti_train_20260928T064215673720Z_5f5c9173.log`
- Original manifest: `reproducibility/manifests/drashti_task1_task1_drashti_train_20260928T070917995838Z_bb880136.manifest.json`
- Continuation raw log: `reproducibility/raw_logs/task1_drashti_train_continue_20260928T072242123664Z_ab61faf8.log`
- Continuation manifest: `reproducibility/manifests/drashti_task1_task1_drashti_train_continue_20260928T075254203860Z_aea815fb.manifest.json`
- Metrics file: `task1_llm/drashti/metrics_report.csv`
- Epoch 10 evidence copy: `task1_llm/drashti/evidence_epoch10/`
- Epochs 1 through 10 training seconds: 1594.9798009169972
- Epochs 11 through 20 training seconds: 1777.7800418330007
- Combined training seconds: 3372.759842749998
- Continuation start: 2026-09-28T07:22:42.123966+00:00
- Continuation end: 2026-09-28T07:52:54.149116+00:00
- Combined training time adds the two measured training intervals. It does not include the pause between the two processes or generation time.

## Epochs 1 through 20

| Epoch | Training cross entropy | Validation cross entropy | Generalization gap |
| --- | --- | --- | --- |
| 1 | 1.50692863695 | 0.871306754617 | -0.635621882334 |
| 2 | 0.84529794947 | 0.700056158658 | -0.145241790812 |
| 3 | 0.733153118341 | 0.634799134142 | -0.0983539841986 |
| 4 | 0.677261230223 | 0.599151564939 | -0.0781096652842 |
| 5 | 0.642337053754 | 0.571665456461 | -0.0706715972933 |
| 6 | 0.618030920304 | 0.553243366626 | -0.0647875536779 |
| 7 | 0.600063943851 | 0.541586950965 | -0.0584769928852 |
| 8 | 0.586641517478 | 0.531890681814 | -0.0547508356633 |
| 9 | 0.577550007651 | 0.525709643973 | -0.0518403636771 |
| 10 | 0.571886139489 | 0.522515400068 | -0.0493707394209 |
| 11 | 0.569076009389 | 0.521106406479 | -0.0479696029104 |
| 12 | 0.566925961504 | 0.518550539186 | -0.048375422318 |
| 13 | 0.564818205493 | 0.517022267057 | -0.0477959384363 |
| 14 | 0.562948268321 | 0.515172920998 | -0.0477753473224 |
| 15 | 0.561071337975 | 0.513674696692 | -0.047396641283 |
| 16 | 0.559455175482 | 0.511904993967 | -0.0475501815153 |
| 17 | 0.558067929024 | 0.510936805666 | -0.047131123358 |
| 18 | 0.556427602314 | 0.509364187986 | -0.0470634143286 |
| 19 | 0.55501605984 | 0.509341607941 | -0.0456744518992 |
| 20 | 0.553519908691 | 0.507538865084 | -0.0459810436065 |

Epoch 10 validation cross entropy was 0.5225154000683443. Epoch 20 validation cross entropy was 0.5075388650844569. The change from epoch 10 to epoch 20 was -0.014976534983887402. Validation loss was lower at epoch 20 than at epoch 10, so it continued improving after epoch 10. Validation loss did not rise between any consecutive epochs from 10 through 20. The lowest validation loss was epoch 20 at 0.5075388650844569. Training cross entropy stays above validation cross entropy at every epoch because training is measured with dropout on and validation is measured in eval mode.

## Metrics from the selected checkpoint

| Metric | Value |
| --- | --- |
| training_cross_entropy | 0.5535199086909827 |
| validation_cross_entropy | 0.5075388650844569 |
| perplexity | 1.6611977279949182 |
| bits_per_character | 0.7322238037157587 |
| generalization_gap | -0.04598104360652577 |
| top1_accuracy | 0.8423092309230923 |
| distinct_1 | 0.011197916666666667 |
| distinct_2 | 0.07887840670859539 |
| distinct_3 | 0.20121308016877637 |
| repeated_4gram_rate | 0.1496815286624204 |
| gradient_norm | 1.2916139577911667 |
| loss_spike_count | 89 |
| nan_count | 0 |
| parameter_count | 839168 |
| training_tokens_per_second | 78682.28343091988 |
| generation_tokens_per_second | 173.83107985502133 |
| peak_memory_bytes | 532414464 |
| total_training_time_seconds | 3372.759842749998 |
| training_time_epochs_1_to_10_seconds | 1594.9798009169972 |
| training_time_epochs_11_to_20_seconds | 1777.7800418330007 |
| selected_epoch | 20 |

Perplexity is exp(validation cross entropy). Bits per character is validation cross entropy divided by ln(2). The generalization gap is the selected epoch's validation cross entropy minus that epoch's training cross entropy.

Training cross entropy is measured in train mode, with dropout active. Validation cross entropy is measured in eval mode. The selected validation loss was recomputed from the loaded checkpoint.

For one generated continuation, a 4-gram position is repeated when the same four characters occurred at a strictly earlier position in that same continuation. The sample rate is repeated positions divided by the number of 4-gram positions. The reported rate is the unweighted mean of the sample rates. Prompts are excluded.

Distinct-n and the repeated 4-gram rate for the selected checkpoint use 24 continuations of 160 new characters each. Prompts are excluded. The protocol matches the epoch 10 generation: 12 validation prompts, greedy and temperature 0.8.

Gradient norm is the mean pre-clip global L2 over all 31260 optimizer steps. Loss spikes are the sum of the per-epoch counts from epochs 1 through 20. NaN count is the sum of both runs.

Continuation step loss minimum: 0.4411871135234833. Continuation step loss maximum: 0.6828513741493225.

## Epoch 10 compared with epoch 20

| Measurement | Epoch 10 | Epoch 20 |
| --- | --- | --- |
| Training cross entropy | 0.5718861394892722 | 0.5535199086909827 |
| Validation cross entropy | 0.5225154000683443 | 0.5075388650844569 |
| Perplexity | 1.686263947922645 | 1.6611977279949182 |
| Bits per character | 0.753830376466713 | 0.7322238037157587 |
| Top-1 accuracy | 0.8377564318931893 | 0.8423092309230923 |
| Distinct-1 | 0.011197916666666667 | 0.011197916666666667 |
| Distinct-2 | 0.08176100628930817 | 0.07887840670859539 |
| Distinct-3 | 0.20490506329113925 | 0.20121308016877637 |
| Repeated 4-gram rate | 0.17197452229299368 | 0.1496815286624204 |

Epoch 10 generation metrics are the preserved values from the original run. Epoch 20 generation metrics are a new sample set from the epoch 20 checkpoint using the same protocol.

## Plots

Updated curves for epochs 1 through 20, with a marker at the end of epoch 10:

- `task1_llm/drashti/outputs/epochs_1_to_20/training_loss.svg`
- `task1_llm/drashti/outputs/epochs_1_to_20/validation_loss.svg`
- `task1_llm/drashti/outputs/epochs_1_to_20/combined_loss.svg`
- `task1_llm/drashti/outputs/epochs_1_to_20/learning_rate.svg`
- `task1_llm/drashti/outputs/epochs_1_to_20/gradient_norm.svg`

The original 10-epoch SVG files remain in `task1_llm/drashti/outputs/` and in `task1_llm/drashti/evidence_epoch10/outputs/`.

## Generation samples

- Epoch 10 samples, unchanged: `task1_llm/drashti/outputs/generated_samples.json`
- Selected checkpoint samples: `task1_llm/drashti/outputs/generated_samples_best_checkpoint.json`
- Epoch 20 samples: `task1_llm/drashti/outputs/generated_samples_epoch20.json`

## Evidence paths

- Continuation config: `configs/task1_drashti_train_continue.json`
- Original config: `configs/task1_drashti_train.json`
- Original raw log: `reproducibility/raw_logs/task1_drashti_train_20260928T064215673720Z_5f5c9173.log`
- Continuation raw log: `reproducibility/raw_logs/task1_drashti_train_continue_20260928T072242123664Z_ab61faf8.log`
- Selected checkpoint: `task1_llm/drashti/checkpoints/task1_drashti_train_continue_20260928T075219828443Z_epoch20_step31260.pt`
- Metrics: `task1_llm/drashti/metrics_report.csv`
- Epoch table: `task1_llm/drashti/outputs/epochs_1_to_20/epoch_metrics.csv`
- Learning-rate history: `task1_llm/drashti/outputs/epochs_1_to_20/learning_rate_history.csv`
- Epoch 10 failure analysis, unchanged: `task1_llm/drashti/failure_analysis.md`
- Selected-checkpoint failure candidates: `task1_llm/drashti/failure_analysis_best_checkpoint.md`

## Personal interpretation

The loss curves in `task1_llm/drashti/outputs/epochs_1_to_20/` are the clearest story. Validation cross entropy falls from about 0.87 after epoch 1 to 0.52 at epoch 10, then keeps edging down to 0.5075 at epoch 20. Training loss stays a little above validation the whole way, so the model is not memorizing the 100k-story split; the gap is small and stable rather than blowing up. That matches the generalization gap metric of about +0.046 at the selected checkpoint.

Strengths. Next-character accuracy around 0.84 and perplexity about 1.66 mean the model usually knows the next letter in TinyStories-style prose. Bits per character under 0.74 is in line with a small character GPT on this domain. Training was stable after the first epoch: NaN count is 0, and the gradient-norm plot shows the big spikes concentrated early, then settling. Greedy samples still look like children’s stories rather than random characters.

Weaknesses. Distinct-n and the repeated 4-gram rate show that “looks like English” is not the same as “tells a clean story.” The failure-analysis candidates are the proof: greedy decoding locks onto a local phrase (“He was so excited…”, “the box and the box”) and keeps replaying it. Coherence also drifts (zoo prompt becomes cars and trucks). So the metrics that look strong are mostly local character prediction; long-range story sense is still weak.

Limitations. This is a 839k-parameter character model with context 128, trained on one member split of TinyStories. It will not transfer to adult prose, and greedy decoding is a harsh lens—temperature sampling can look less stuck, but that does not fix the underlying preference for safe, repeated n-grams. Metrics are also tied to this vocabulary and split; a different seed or tokenizer would move the numbers.

Were epochs 11–20 worth it? Yes, but modestly. Validation CE improved from 0.5225 (epoch 10) to 0.5075 (epoch 20), and epoch 20 was the best of the twenty, so selection was not just “train longer for free.” The learning-rate curve for the continuation is flat at 3e-05, so those epochs are fine-tuning at a small step size rather than a second warm start. The cost was about another 30 minutes on MPS. I would keep epoch 20 as the reported checkpoint; I would not expect another ten epochs at the same LR to buy a similar jump.
