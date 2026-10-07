# Task 3: Failure and Stability Analysis

## Training Stability

Training remained stable across the recorded 100 epochs.

Generator loss decreased from 7.1919 at epoch 1 to 2.6298 at epoch 100.

Cycle consistency loss decreased from 0.5876 to 0.1486, while identity loss decreased from 0.5753 to 0.2305.

The final discriminator losses were 0.1433 for discriminator A and 0.1766 for discriminator B.

The recorded NaN count was 0 and the gradient norm was 3.7.

## Direction Asymmetry

Photo to Monet achieved an FID of 95.355329, while Monet to Photo achieved 99.562875.

The Monet to Photo direction was therefore the weaker mapping by approximately 4.21 FID points. This indicates that the two translation directions did not have identical difficulty.

## Perceptual and Distributional Quality

The final KID was 0.052. Generative precision was 0.63 and generative recall was 0.51.

The difference between precision and recall suggests that generated samples achieved stronger fidelity than coverage of the complete target distribution.

LPIPS was 0.30 and content preservation cosine similarity was 0.81.

The content similarity result indicates substantial preservation of input structure, while the LPIPS result reflects perceptual changes introduced by the style translation.

## Human Evaluation

The blinded human audit score was 78%, with inter rater agreement of 0.74.

This indicates generally positive visual assessment with meaningful agreement between raters, while also leaving observable cases where style quality, content preservation, or artifacts could be improved.

## Limitations

The remaining gap between the two translation directions suggests that future work should investigate direction specific optimization.

Further improvements could also explore stronger content preservation constraints, additional regularization, and systematic tuning of the adversarial versus reconstruction loss balance.
