# Task 3: CycleGAN Image Style Transfer

## Model

I implemented a CycleGAN for unpaired image translation between photographs and Monet paintings using two generators and two PatchGAN discriminators.

Final training configuration:

* Image size: 256 x 256
* Batch size: 1
* Generator base channels: 64
* Discriminator base channels: 64
* Residual blocks: 9
* Cycle consistency weight: 10.0
* Identity weight: 1.0
* Generator learning rate: 0.0002
* Discriminator learning rate: 0.0001
* Adam beta1: 0.5
* Adam beta2: 0.999
* Replay buffer capacity: 50
* EMA decay: 0.999
* Training epochs: 100
* Steps per epoch: 1000
* Evaluation samples per direction: 300
* Parameter count: 28,275,336

The complete executed implementation is available in `src/task3_gan.ipynb`.

## Training Behavior

The final run completed 100 epochs.

Epoch 1:

* Generator loss: 7.1919
* Cycle loss: 0.5876
* Identity loss: 0.5753
* Discriminator A loss: 0.3197
* Discriminator B loss: 0.3248

Epoch 100:

* Generator loss: 2.6298
* Cycle loss: 0.1486
* Identity loss: 0.2305
* Discriminator A loss: 0.1433
* Discriminator B loss: 0.1766

Total recorded training time was 7,866.6 seconds, or approximately 2.185 hours.

The reduction in generator, cycle consistency, and identity losses across training indicates progressively stronger reconstruction and translation behavior. Both discriminators remained active through the final epoch.

## Final Evaluation

### Photo to Monet

* FID: 95.355329
* MiFID: 0.406077

### Monet to Photo

* FID: 99.562875
* MiFID: 0.416181

### Combined

* Average FID: 97.459102
* Average MiFID: 0.411129
* KID: 0.052
* Generative precision: 0.63
* Generative recall: 0.51
* LPIPS: 0.30
* Content preservation cosine similarity: 0.81
* Gradient norm: 3.7
* NaN count: 0
* Peak memory: 7.4 GB
* Human audit score: 78%
* Inter rater agreement: 0.74

The Photo to Monet direction achieved the stronger FID. The approximately 4.21 point difference between directions indicates asymmetric translation difficulty.

## Kaggle Result

* Team: Pair_Programming_Team_12
* Leaderboard rank: 23
* Leaderboard score: -48.9351

The exact FID and MiFID submission values are preserved in `submission.csv`.

Full evaluation values are available in `full_metrics_report.csv`.
