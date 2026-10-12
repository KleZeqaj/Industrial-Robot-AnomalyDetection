# Industrial Robot Anomaly Detection

**Adversarial Autoencoders with score fusion for anomaly detection on KUKA industrial robot sensor data.**

Project for the *Machine Learning in Applications* course (AM01), Politecnico di Torino.

**Authors:** Elziario Paduano, Marco Pavanati, Klejsi Zeqaj

---

## Overview

Industrial robots continuously produce multivariate sensor data (joint positions, velocities, motor currents, power-related signals). Subtle changes in these signals, such as slower movements or abnormal energy consumption, can be early signs of degradation. Detecting them is hard because anomalies are rare, only weakly separated from normal behavior, and may involve just a few sensor channels.

This project studies **reconstruction-based anomaly detection** on KUKA robot data. The main question: *does adding an adversarial component to a standard Autoencoder improve detection performance?*

We compare:

- **AE baseline**: a fully connected Autoencoder scored by top-k reconstruction error.
- **AAE + Fusion** (proposed): an Adversarial Autoencoder whose three anomaly scores (reconstruction error, latent-space outlier density, prior deviation) are combined by a logistic-regression fusion model.

The AAE + Fusion model reaches **F1 = 0.943** and **PR AUC = 0.976**, against F1 = 0.737 and PR AUC = 0.850 for the AE baseline.

## Problem formulation

- Point-wise binary anomaly detection on feature vectors `x ∈ R^d`.
- Labels: `y = 0` normal behavior, `y = 1` slow/anomalous behavior.
- **Semi-supervised setting**: training uses *only normal* samples; validation and test sets contain both normal and anomalous samples.
- Detection is score-based: `ŷ(x) = 1[s(x) > τ]`, where the models are not used as direct binary classifiers.

## Method

### Dataset split

| Model | Train | Validation | Test |
|---|---|---|---|
| AE baseline | normal only | normal + anomalous (threshold calibration) | normal + anomalous |
| AAE + Fusion | normal only | split into two equal halves: (1) train the logistic-regression fusion model, (2) hyperparameter tuning and model selection | normal + anomalous |

### Preprocessing

Fitted **only on the training data**, then applied to validation and test data:

- Feature-wise min-max normalization (very small ranges replaced by a constant).
- Optional feature filtering: correlation pruning (absolute Pearson correlation above a threshold), near-constant feature removal, and percentile-based clipping.

### Autoencoder (baseline)

- Encoder: `d → 72 → 64 → 56 → 48 → 32 → 24 → 16 → ℓ`, with the decoder mirroring it.
- ReLU on hidden layers, linear latent and output layers.
- Loss: mean squared reconstruction error.
- **Score:** mean of the top-k largest feature-wise squared errors, which emphasizes localized failures on a subset of channels.
- **Threshold:** calibrated on validation data, choosing the (recall, precision) point closest to the ideal top-right corner of the PR curve.
- The baseline uses the **top 40 features** ranked by the reconstruction-separation statistic.

### Adversarial Autoencoder

Same encoder/decoder as the AE, plus a **latent-space discriminator** that separates samples of the prior `N(0, I)` from encoder outputs. Training alternates between:

1. Updating encoder and decoder with the reconstruction loss.
2. Updating the discriminator to tell prior samples from encoded samples.
3. Updating the encoder to fool the discriminator.

The adversarial objective is hinge-style, with an **R1 gradient penalty** on the discriminator.

### Score fusion

After training, the AAE produces three anomaly scores:

| Score | Definition | Meaning |
|---|---|---|
| `s_rec` | top-k mean squared reconstruction error | how poorly the sample is reconstructed |
| `s_out` | `-log p̂(z)` (density estimated on normal training latents, Gaussian kernel) | how unlikely the latent code is among normal samples |
| `s_lat` | `‖z‖²` | deviation from the Gaussian prior |

A **logistic-regression** model (class-weighted, 4 parameters, no hyperparameters to tune) fuses them into a single signed score `s_AAE(x) = wᵀu(x) + b`, with decision boundary `ŷ = 1[s_AAE > 0]`.

### Evaluation

Because of the strong class imbalance, accuracy alone is not used. Metrics: precision, recall, F1, Fβ, balanced accuracy, PR AUC and average precision.

## Results

### AE baseline vs. AAE + Fusion

| Model | Precision | Recall | F1-Score | PR AUC |
|---|---|---|---|---|
| AE Baseline | 0.688 | 0.794 | 0.737 | 0.850 |
| **AAE + Fusion** | **0.932** | **0.956** | **0.943** | **0.976** |

### Ablation study

| Model | Precision | Recall | F1-Score | PR AUC |
|---|---|---|---|---|
| AAE (Standalone) | 0.873 | 0.863 | 0.868 | 0.932 |
| AE + Fusion | 0.883 | 0.867 | 0.875 | 0.948 |
| **AAE + Fusion** | **0.932** | **0.956** | **0.943** | **0.976** |

Both the adversarial regularization and the score fusion contribute. Fusion greatly increases recall, and the fused AAE beats the fused AE on all metrics. The AAE latent space is also more compact and isotropic than the AE one, consistent with prior matching toward a standard Gaussian.

### Sensitivity to top-k

| Top-k | F1-Score |
|---|---|
| k = 1 | 0.944 |
| k = 2 | 0.944 |
| k = 3 | 0.942 |
| k = 5 | 0.940 |
| k = 9 | 0.937 |

Here `k = 9` is the total number of features retained after near-constant feature removal, i.e. the mean error over all features. F1 stays stable across `k`, so the model needs little fine-tuning of this parameter.

## Getting started

> Adjust the commands below to match the actual scripts/notebooks in this repository.

```bash
# Clone the repository
git clone https://github.com/KleZeqaj/Industrial-Robot-AnomalyDetection.git
cd Industrial-Robot-AnomalyDetection

# Install dependencies
pip install -r requirements.txt
```

Then run the notebooks or scripts for, in order: preprocessing and feature selection, AE baseline training and evaluation, AAE training, fusion calibration and evaluation.

## Future work

- Prior distributions other than the standard Gaussian, to better match the physical properties of the robot.
- More advanced non-linear models for score fusion.
- Testing the AAE + Fusion architecture in a real-time, edge-computing industrial setting for predictive maintenance.

## References

1. V. Chandola, A. Banerjee, V. Kumar, "Anomaly detection: A survey," *ACM Computing Surveys*, 2009.
2. M. Sakurada, T. Yairi, "Anomaly detection using autoencoders with nonlinear dimensionality reduction," *MLSDA 2014*.
3. A. Makhzani et al., "Adversarial autoencoders," arXiv:1511.05644, 2015.
4. M. M. Breunig et al., "LOF: identifying density-based local outliers," *SIGMOD Rec.*, 29(2), 2000.
5. S. Weglarczyk, "Kernel density estimation and its application," *ITM Web of Conferences*, 23, 2018.
6. T. Saito, M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, 2015.
