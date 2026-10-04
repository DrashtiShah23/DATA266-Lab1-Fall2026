# Task 2 results

These sections record the training run. Strengths, weaknesses, limitations, and comparative conclusions are not written here.

## Dataset

Yelp Polarity revision `bbf1c97a1f0cf005e5aded43839fd814654a1557`.
Development source rows: 504000.
Development model-ready rows: 503972.
Validation source rows: 56000.
Validation model-ready rows: 55999.
Official test rows: 38000.
Excluded empty processed training rows: 29 (28 development, 1 validation).

## Preprocessing reference

The models use `configs/task2_drashti_preprocess.json`, `task2_sentiment/drashti/data_processed/vocabulary.json`, and `task2_sentiment/drashti/data_processed/model_ready.pt`.
Vocabulary size 30,002. Padding id 0. Maximum length 207. Embeddings are learned from scratch.

## Architectures and hyperparameters

baseline is the baseline. Architecture mean_pool. Embedding dimension 64. Hidden dimension 64. Layers 1. Kernel sizes none. Pooling masked_mean. Dropout 0.2. Activation none. Classifier linear. Parameter count 1920258. Optimizer AdamW. Learning rate 0.001. Batch size 256. Maximum epochs 4. Minimum epochs 2. Early stopping patience 2 on validation_macro_f1. Weight decay 0.01 on non-embedding parameters. Gradient clip 1.0. Decision threshold 0.5 (not_tuned). Seed 266.

experimental_1 is the experimental_model_1. Architecture conv. Embedding dimension 64. Hidden dimension 192. Layers 1. Kernel sizes 3, 4, 5. Pooling masked_max. Dropout 0.3. Activation relu. Classifier linear. Parameter count 1969858. Optimizer AdamW. Learning rate 0.001. Batch size 256. Maximum epochs 4. Minimum epochs 2. Early stopping patience 2 on validation_macro_f1. Weight decay 0.01 on non-embedding parameters. Gradient clip 1.0. Decision threshold 0.5 (not_tuned). Seed 266.

experimental_2 is the experimental_model_2. Architecture gru. Embedding dimension 64. Hidden dimension 64. Layers 1. Kernel sizes none. Pooling last_hidden. Dropout 0.2. Activation tanh. Classifier linear. Parameter count 1945218. Optimizer AdamW. Learning rate 0.001. Batch size 128. Maximum epochs 4. Minimum epochs 2. Early stopping patience 2 on validation_macro_f1. Weight decay 0.01 on non-embedding parameters. Gradient clip 1.0. Decision threshold 0.5 (not_tuned). Seed 266.

## Training procedure

Each model trains on the development model-ready rows. Validation macro F1 selects the checkpoint. The official test labels are not used for that choice, for early stopping, or for the decision threshold. The threshold is 0.5 and was not tuned.
Early stopping requires at least the configured minimum epochs and then stops after two epochs without a validation macro F1 improvement, with a maximum of four epochs.

## Selected epochs

baseline: epoch 2, validation macro F1 0.934730967265112, checkpoint `task2_sentiment/drashti/checkpoints/task2_drashti_baseline_epoch2_20260928T193624583319Z_c94637b8.pt`.
experimental_1: epoch 2, validation macro F1 0.9444762526490748, checkpoint `task2_sentiment/drashti/checkpoints/task2_drashti_cnn_epoch2_20260928T193909317882Z_7ac08cc1.pt`.
experimental_2: epoch 3, validation macro F1 0.9520347891034799, checkpoint `task2_sentiment/drashti/checkpoints/task2_drashti_gru_epoch3_20260928T201357720891Z_18f97cc3.pt`.

## Hardware

baseline: device mps, processor Apple M4, pytorch 2.14.0, unified memory bytes 17179869184.
experimental_1: device mps, processor Apple M4, pytorch 2.14.0, unified memory bytes 17179869184.
experimental_2: device mps, processor Apple M4, pytorch 2.14.0, unified memory bytes 17179869184.

## Parameter counts and training time

baseline: 1920258 parameters, training seconds 77.33121366699925, examples per second 27234.255966464625, peak RSS bytes 591642624.
experimental_1: 1969858 parameters, training seconds 251.07398004099377, examples per second 8370.095910942235, peak RSS bytes 678936576.
experimental_2: 1945218 parameters, training seconds 2587.2080364590074, examples per second 790.9226942567968, peak RSS bytes 892993536.

## Epoch records

### baseline

| Epoch | Training loss | Validation loss | Validation accuracy | Validation macro F1 | Runtime seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.24938924425711842 | 0.18325309529143505 | 0.932748799085698 | 0.9327487475946189 | 22.05777912499616 |
| 2 | 0.17266251075819686 | 0.17902397998393793 | 0.934730977338881 | 0.934730967265112 | 18.47051349999674 |
| 3 | 0.1636343842623577 | 0.1797843454402173 | 0.9345702601832175 | 0.9345702554886326 | 16.806418707987177 |
| 4 | 0.1588298936469833 | 0.18203142653545287 | 0.9342131109484098 | 0.934212619854091 | 19.015089084001374 |

### experimental_1

| Epoch | Training loss | Validation loss | Validation accuracy | Validation macro F1 | Runtime seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.20576956431070334 | 0.14778598975320678 | 0.9417489598028537 | 0.9417475138227118 | 63.69570912500785 |
| 2 | 0.13045114836991095 | 0.1409461776009798 | 0.944481151449133 | 0.9444762526490748 | 61.538796875000116 |
| 3 | 0.0980997423769566 | 0.14866841201424144 | 0.9435882783621136 | 0.9435782653775762 | 62.68208241600951 |
| 4 | 0.07128116372014238 | 0.16917939968176315 | 0.9392132002357185 | 0.9391944636861795 | 62.12840312499611 |

### experimental_2

| Epoch | Training loss | Validation loss | Validation accuracy | Validation macro F1 | Runtime seconds |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.17497400625860487 | 0.13967541132685937 | 0.9446775835282772 | 0.944675258800538 | 676.9989082079992 |
| 2 | 0.11413560046582877 | 0.12492511313260331 | 0.9517312809157307 | 0.9517282736152306 | 630.5798650840006 |
| 3 | 0.08612021370545082 | 0.12734511414408314 | 0.9520348577653173 | 0.9520347891034799 | 651.6987745840015 |
| 4 | 0.06478814726834341 | 0.14605412983754273 | 0.949266951195557 | 0.9492655336860671 | 627.0489127079927 |

## Prediction and checkpoint locations

baseline predictions: `task2_sentiment/drashti/outputs/predictions/task2_drashti_baseline_test_predictions.csv` (38000 rows).
baseline checkpoint: `task2_sentiment/drashti/checkpoints/task2_drashti_baseline_epoch2_20260928T193624583319Z_c94637b8.pt`.
baseline raw log: `reproducibility/raw_logs/task2_drashti_baseline_20260928T193543273145Z_c1ebea86.log`.
baseline manifest: `reproducibility/manifests/drashti_task2_task2_drashti_baseline_20260928T193701902818Z_8f223a83.manifest.json`.
experimental_1 predictions: `task2_sentiment/drashti/outputs/predictions/task2_drashti_cnn_test_predictions.csv` (38000 rows).
experimental_1 checkpoint: `task2_sentiment/drashti/checkpoints/task2_drashti_cnn_epoch2_20260928T193909317882Z_7ac08cc1.pt`.
experimental_1 raw log: `reproducibility/raw_logs/task2_drashti_cnn_20260928T193703312855Z_9f57791e.log`.
experimental_1 manifest: `reproducibility/manifests/drashti_task2_task2_drashti_cnn_20260928T194116460199Z_c9730844.manifest.json`.
experimental_2 predictions: `task2_sentiment/drashti/outputs/predictions/task2_drashti_gru_test_predictions.csv` (38000 rows).
experimental_2 checkpoint: `task2_sentiment/drashti/checkpoints/task2_drashti_gru_epoch3_20260928T201357720891Z_18f97cc3.pt`.
experimental_2 raw log: `reproducibility/raw_logs/task2_drashti_gru_20260928T194117690709Z_888f1f25.log`.
experimental_2 manifest: `reproducibility/manifests/drashti_task2_task2_drashti_gru_20260928T202431697416Z_14aa0074.manifest.json`.

Aligned example-id order: yes.
First example id: test-0. Last example id: test-37999.

## External resource audit

pretrained embedding files: PASS
GloVe: PASS
word2vec: PASS
FastText pretrained vectors: PASS
BERT: PASS
RoBERTa: PASS
DistilBERT: PASS
GPT based text model: PASS
sentence transformer: PASS
Hugging Face AutoModel: PASS

## Development overfit tests

These figures are labeled DEVELOPMENT OVERFIT TEST. They are not final model performance.

baseline: samples 64, steps 40, initial loss 0.698765754699707, final loss 0.005877777934074402, decreased True.
experimental_1: samples 64, steps 40, initial loss 0.6976368427276611, final loss 0.0, decreased True.
experimental_2: samples 64, steps 40, initial loss 0.6909219026565552, final loss 3.688598008011468e-05, decreased True.

## Official test evaluation

Population: 38,000 official test rows. Positive class is integer 1. Negative class is integer 0. Hard labels use positive probability >= 0.5. These figures did not change the selected checkpoints.

PR area is average precision, the step-function area under the precision-recall curve. It is not trapezoidal integration. ECE uses 15 equal-width bins on the probability of the predicted class. Bootstrap intervals are percentile intervals from 1,000 resamples with numpy PCG64 seed 266.

| Metric | baseline | experimental_1 | experimental_2 |
| --- | --- | --- | --- |
| Accuracy | 0.937105 | 0.946447 | 0.953947 |
| Precision macro | 0.937126 | 0.946545 | 0.953971 |
| Precision micro | 0.937105 | 0.946447 | 0.953947 |
| Precision weighted | 0.937126 | 0.946545 | 0.953971 |
| Recall macro | 0.937105 | 0.946447 | 0.953947 |
| Recall micro | 0.937105 | 0.946447 | 0.953947 |
| Recall weighted | 0.937105 | 0.946447 | 0.953947 |
| F1 macro | 0.937105 | 0.946444 | 0.953947 |
| F1 micro | 0.937105 | 0.946447 | 0.953947 |
| F1 weighted | 0.937105 | 0.946444 | 0.953947 |
| ROC AUC | 0.981706 | 0.987713 | 0.990843 |
| PR average precision | 0.981503 | 0.988050 | 0.990918 |
| MCC | 0.874232 | 0.892992 | 0.907918 |
| Brier | 0.047841 | 0.040567 | 0.035093 |
| ECE | 0.007711 | 0.004536 | 0.012525 |
| Parameters | 1920258 | 1969858 | 1945218 |
| Training seconds | 77.331 | 251.074 | 2587.208 |
| Examples per second | 27234.256 | 8370.096 | 790.923 |
| Peak RSS bytes | 591642624 | 678936576 | 892993536 |

### Confusion counts

baseline: TN 17871, FP 1129, FN 1261, TP 17739.
experimental_1: TN 17842, FP 1158, FN 877, TP 18123.
experimental_2: TN 18193, FP 807, FN 943, TP 18057.

### Bootstrap 95 percent intervals

baseline: accuracy [0.934789, 0.939290], macro F1 [0.934785, 0.939290], MCC [0.869574, 0.878615].
experimental_1: accuracy [0.944210, 0.948605], macro F1 [0.944206, 0.948603], MCC [0.888485, 0.897278].
experimental_2: accuracy [0.951868, 0.956054], macro F1 [0.951868, 0.956047], MCC [0.903751, 0.912110].

### McNemar

baseline versus experimental_1: both correct 34769, baseline only 841, experimental only 1196, both wrong 1194, statistic 61.519882, p-value 4.38295e-15.

YOUR INPUT REQUIRED: Interpret this McNemar result in the context of the model comparison.

baseline versus experimental_2: both correct 35006, baseline only 604, experimental only 1244, both wrong 1146, statistic 220.952922, p-value 5.6044e-50.

YOUR INPUT REQUIRED: Interpret this McNemar result in the context of the model comparison.

### Robustness slices

| Model | Slice | Count | Macro F1 | Error rate |
| --- | --- | --- | --- | --- |
| baseline | short | 9911 | 0.933297 | 0.064272 |
| baseline | medium | 18751 | 0.939359 | 0.060637 |
| baseline | long | 9338 | 0.931289 | 0.065967 |
| baseline | contains_negation | 30968 | 0.932303 | 0.066488 |
| baseline | empty_after_preprocessing | 0 |  |  |
| experimental_1 | short | 9911 | 0.940515 | 0.057108 |
| experimental_1 | medium | 18751 | 0.951416 | 0.048584 |
| experimental_1 | long | 9338 | 0.938085 | 0.059756 |
| experimental_1 | contains_negation | 30968 | 0.944667 | 0.054508 |
| experimental_1 | empty_after_preprocessing | 0 |  |  |
| experimental_2 | short | 9911 | 0.952286 | 0.046009 |
| experimental_2 | medium | 18751 | 0.956211 | 0.043784 |
| experimental_2 | long | 9338 | 0.947335 | 0.050653 |
| experimental_2 | contains_negation | 30968 | 0.952798 | 0.046370 |
| experimental_2 | empty_after_preprocessing | 0 |  |  |

contains_negation overlaps the length slices. short, medium, and long partition the 38,000 test rows. empty_after_preprocessing has 0 official test rows, so it has no F1 or error rate.

Error-review model: `experimental_2`. The 20 cases are in `task2_sentiment/drashti/failure_analysis.md` and `task2_sentiment/drashti/outputs/error_review_20.csv`.

Evidence directory: `task2_sentiment/drashti/outputs/evaluation/`. The metric table is `task2_sentiment/drashti/metrics_report.csv`.

## Strengths

YOUR INPUT REQUIRED

## Weaknesses

YOUR INPUT REQUIRED

## Limitations

YOUR INPUT REQUIRED

## Comparative conclusions

YOUR INPUT REQUIRED
