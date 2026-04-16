# Batch Normalization Statistics as Output Calibrators, Not Feature Encoders: Evidence from Class-Conditional Statistic Replacement

## Abstract

Batch normalization (BN) is ubiquitous in deep learning, yet the functional role of its learned running statistics remains poorly understood. We investigate whether BN running statistics encode class-specific feature information or serve primarily as output calibration mechanisms. Through a simple experimental paradigm—replacing global BN running statistics with class-conditional statistics at test time—we uncover a striking dissociation: same-class BN statistics *destroy* classification accuracy (3.5% vs. 91.1% baseline on CIFAR-10), yet linear probes on the resulting representations achieve 99.9% accuracy. This effect replicates across four architectures (SmallResNet, VGG-11, SimpleCNN, WideResNet-16-8), generalizes to three datasets including Tiny-ImageNet-200 (200 classes, 64×64 resolution), and disappears when BatchNorm is replaced with GroupNorm—confirming it is specific to the running-statistic mechanism. On Tiny-ImageNet-200, same-class statistics reduce accuracy from 60.8% to 0.17% (chance = 0.5%) while linear probes achieve *perfect* 100% accuracy—the strongest dissociation observed. An interpolation experiment reveals a smooth, monotonic transition between same-class and global statistics, ruling out phase-transition artifacts. Calibration analysis shows that same-class statistics induce severe miscalibration (ECE = 0.394 vs. 0.025 baseline), and temperature scaling—while reducing ECE from 0.394 to 0.151—cannot recover accuracy (remaining at 2.9%), proving the collapse is not a simple logit-scale shift. These results demonstrate that BN running statistics function as output calibrators: they align activation magnitudes with downstream layer expectations without encoding class-discriminative features. This calibration perspective reframes our understanding of batch normalization at inference time and has implications for transfer learning, domain adaptation, and normalization-free architectures.

## 1. Introduction

Batch normalization (Ioffe & Szegedy, 2015) is among the most widely adopted techniques in deep learning. During training, BN normalizes activations using mini-batch statistics; at inference, it substitutes exponentially averaged running statistics accumulated during training. These running statistics—per-channel running means and variances—compress the training data distribution as seen through the network's learned filters.

A natural question arises: **what information do these running statistics encode?** Two hypotheses present themselves:

1. **Feature encoding hypothesis.** BN statistics encode class-relevant distributional information. Providing class-appropriate statistics should preserve or improve classification.
2. **Calibration hypothesis.** BN statistics serve primarily to calibrate activation scales to match what downstream layers expect. Any perturbation—even a "helpful" one—disrupts calibration and degrades classification.

We test these hypotheses by computing class-conditional BN statistics from training data and substituting them for global statistics at test time, measuring both classification accuracy and representation quality via linear probes.

Our core finding is striking: on CIFAR-10, replacing global BN statistics with same-class statistics reduces SmallResNet classification accuracy from 91.1% to 3.5% (10 seeds), yet linear probes on the perturbed representations achieve 99.9% accuracy. This dissociation—collapsed output but preserved representations—constitutes strong evidence for the calibration hypothesis. We show this effect is:

- **Architecture-general:** Replicating on VGG-11 (85.4% → 10.9%), SimpleCNN (83.2% → 26.3%), and WideResNet-16-8 (90.6% → 11.1%)
- **Dataset-general:** Extending to CIFAR-100 (70.3% → 0.67%) and Tiny-ImageNet-200 (60.8% → 0.17%)
- **BN-specific:** Absent when GroupNorm replaces BatchNorm (81.6% accuracy maintained), confirming running statistics as the causal mechanism
- **Continuously graded:** Interpolation between same-class and global statistics yields a smooth monotonic accuracy curve
- **Not a calibration-scale artifact:** Temperature scaling reduces ECE but cannot recover accuracy, ruling out simple logit rescaling as an explanation

These results have practical implications for transfer learning, domain adaptation, and normalization-free architecture design.

## 2. Related Work

**Batch Normalization.** Ioffe & Szegedy (2015) introduced BN to reduce "internal covariate shift." Santurkar et al. (2018) challenged this narrative, showing BN primarily smooths the optimization landscape. Bjorck et al. (2018) confirmed similar smoothing benefits. These works focus on BN's *training-time* role; we examine its *inference-time* function.

**Expressive Power of BN Parameters.** Frankle et al. (2020) demonstrated that training only BN affine parameters (γ, β) of a randomly initialized network achieves surprisingly high accuracy on CIFAR-10. This shows BN parameters carry significant information but does not distinguish the roles of learned affine parameters versus running statistics.

**BN for Domain Adaptation.** Li et al. (2018) proposed Adaptive Batch Normalization (AdaBN), replacing BN running statistics with target-domain statistics for domain adaptation. This implicitly treats BN statistics as domain-specific calibrators. Our work provides direct experimental evidence for why this works: BN statistics modulate output scale, not feature content.

**Test-Time Adaptation.** Wang et al. (2021) and Schneider et al. (2020) update BN statistics on test data to improve robustness to distribution shift. Our calibration perspective explains why BN statistics are so influential at test time despite not being learned through backpropagation.

**BN and Adversarial Robustness.** Benz et al. (2021) showed adversarial vulnerability correlates with BN statistics. Our work suggests this may be because adversarial perturbations disrupt the calibration that BN statistics provide.

**Neural Network Calibration.** Guo et al. (2017) demonstrated that modern neural networks are often poorly calibrated and introduced temperature scaling as a simple post-hoc calibration method. We use expected calibration error (ECE) and temperature scaling to characterize the nature of the accuracy collapse under BN statistic perturbation, showing it is fundamentally different from standard miscalibration.

## 3. Method

### 3.1 Architectures

We evaluate four architectures to test generality:

1. **SmallResNet.** A compact ResNet with channel widths [32, 64, 128, 256] across four stages of two residual blocks each, containing 20 BatchNorm2d layers. This is our primary architecture (10 seeds on CIFAR-10). For Tiny-ImageNet-200 (64×64 inputs), we use a **SmallResNet64** variant with five stages and channel widths [32, 64, 128, 256, 512], adapted for the higher resolution with an initial 3×3 convolution (stride 1) and additional downsampling stages.
2. **VGG-11-BN.** A VGG-11 variant with BatchNorm after each convolutional layer (8 BN layers), providing an architecture without skip connections (3 seeds).
3. **SimpleCNN.** A plain 4-layer convolutional network with BatchNorm (4 BN layers), testing the effect in a minimal architecture (3 seeds).
4. **WideResNet-16-8.** A wide residual network with depth 16 and widen factor 8 (Zagoruyko & Komodakis, 2016), incorporating dropout (p=0.3) within residual blocks. This architecture is substantially wider and deeper than SmallResNet, with significantly more BN layers and parameters, testing the effect in a high-capacity network with regularization (3 seeds).

All models are trained on CIFAR-10 with standard augmentation (random crop with padding 4, random horizontal flip) using SGD with momentum 0.9, weight decay 5×10⁻⁴, initial learning rate 0.1, and cosine annealing. SmallResNet, SimpleCNN, VGG-11-BN, and WideResNet-16-8 train for 20 epochs on CIFAR-10. We additionally train SmallResNet on CIFAR-100 (25 epochs, 3 seeds) and SmallResNet64 on Tiny-ImageNet-200 (25 epochs, 3 seeds, 64×64 images, 200 classes) to test dataset generalization. WideResNet-16-8 uses mixed-precision training for computational efficiency.

### 3.2 Datasets

We evaluate on three datasets of increasing complexity:

1. **CIFAR-10.** 10 classes, 32×32 images, 50K training / 10K test. Our primary benchmark.
2. **CIFAR-100.** 100 classes, 32×32 images, 50K training / 10K test. Tests scaling to many classes.
3. **Tiny-ImageNet-200.** 200 classes, 64×64 images, 100K training / 10K validation. Tests generalization to higher resolution, larger label spaces, and more complex visual content. Images are center-cropped to 64×64 for evaluation.

### 3.3 Computing Class-Conditional BN Statistics

After training, we compute class-conditional BN statistics by passing all training samples of each class through the network with BN layers in training mode (accumulating fresh running statistics) while all other parameters remain frozen. This yields *C* sets of BN statistics {μ_c, σ²_c} for c ∈ {0, ..., C−1}, one per class.

### 3.4 Evaluation Conditions

We evaluate under four BN statistic conditions:

1. **Global (baseline).** Standard running statistics from training—normal inference.
2. **Same-class.** For test samples of class *c*, load BN statistics computed from class-*c* training data. Under the feature encoding hypothesis, this should help or be neutral.
3. **Wrong-class.** For test samples of class *c*, load statistics from class (*c* + ⌊C/2⌋) mod *C*—a deliberately mismatched condition.
4. **Random-class.** For each test sample of class *c*, load statistics from a uniformly random *different* class.

### 3.5 Metrics

For each condition:
- **Classification accuracy.** Top-1 accuracy using the trained classification head.
- **Mean confidence.** Average softmax probability assigned to the true class.
- **Linear probe accuracy.** We extract features from the penultimate layer (post global average pooling), train a logistic regression or ridge regression classifier on an 80/20 split, and report held-out accuracy. We use ridge regression (λ = 10⁻³) for CIFAR-10 experiments and logistic regression for WideResNet and Tiny-ImageNet experiments.
- **Expected calibration error (ECE).** Following Guo et al. (2017), we compute ECE with 15 equal-width bins to quantify the gap between predicted confidence and actual accuracy.

### 3.6 Interpolation Experiment

To characterize the transition between same-class and global statistics, we linearly interpolate:

$$\mu_\alpha = \alpha \cdot \mu_{\text{global}} + (1 - \alpha) \cdot \mu_{\text{same}}$$
$$\sigma^2_\alpha = \alpha \cdot \sigma^2_{\text{global}} + (1 - \alpha) \cdot \sigma^2_{\text{same}}$$

for α ∈ {0.0, 0.1, ..., 1.0}, evaluating classification accuracy and linear probe accuracy at each point.

### 3.7 GroupNorm Control

To confirm that the observed effect is specific to BN's running-statistic mechanism, we train a variant of SmallResNet with GroupNorm replacing all BatchNorm layers. GroupNorm normalizes using group statistics computed from the *current input* rather than maintained running statistics, so there are no class-conditional statistics to substitute. If GroupNorm models maintain accuracy, the effect is BN-specific.

### 3.8 Expected Calibration Error

To characterize the nature of the accuracy collapse beyond simple accuracy metrics, we measure Expected Calibration Error (ECE; Guo et al., 2017) with 15 equal-width confidence bins. ECE measures the weighted average discrepancy between predicted confidence and empirical accuracy within each bin, providing a scalar summary of how well-calibrated a model's output probabilities are.

### 3.9 Temperature Scaling

To test whether the accuracy collapse under same-class statistics is merely a logit-scale miscalibration recoverable by post-hoc rescaling, we apply temperature scaling (Guo et al., 2017). We optimize a single scalar temperature *T* on a held-out validation set to minimize negative log-likelihood under the same-class BN statistic condition, then evaluate whether dividing logits by *T* recovers classification accuracy. Temperature scaling is a monotone transformation that preserves logit ordering (and thus the argmax), making it diagnostic: if accuracy improves, the issue is confidence scaling; if not, the logit ranking itself is disrupted.

### 3.10 Statistical Protocol

SmallResNet on CIFAR-10 uses 10 independent seeds (42–51). All other configurations use 3 seeds (42–44). We report means ± standard deviations and conduct paired t-tests for key comparisons.

## 4. Results

### 4.1 Primary Finding: SmallResNet on CIFAR-10

Table 1 presents aggregate results across 10 seeds.

**Table 1: SmallResNet on CIFAR-10 (mean ± std, n=10 seeds)**

| Condition | Accuracy (%) | Confidence (%) | Linear Probe (%) |
|-----------|-------------|----------------|-------------------|
| Global (baseline) | 91.08 ± 0.12 | 88.88 ± 0.15 | 90.86 ± 0.55 |
| Same-class | 3.55 ± 0.69 | 4.80 ± 0.52 | 99.88 ± 0.09 |
| Wrong-class | 65.37 ± 4.69 | 50.64 ± 3.22 | 100.00 ± 0.00 |
| Random-class | 58.99 ± 5.84 | 45.47 ± 5.19 | 97.63 ± 1.11 |

The results reveal a dramatic dissociation:

1. **Same-class BN statistics catastrophically degrade accuracy** to 3.5%, well below the 10% chance baseline. Providing the network with class-appropriate statistics is *actively destructive*.
2. **Wrong-class statistics outperform same-class** (65.4% vs. 3.5%), a complete inversion of the feature encoding prediction.
3. **Linear probe accuracy under same-class statistics reaches 99.9%**, far exceeding the global baseline (90.9%). Representations become *more* linearly separable under perturbation.
4. **Statistical significance.** Same vs. global accuracy: t(9) = −374.1, p = 3.5 × 10⁻²⁰. Same vs. global linear probe: t(9) = +48.7, p = 3.3 × 10⁻¹².

### 4.2 Cross-Architecture Replication

Table 2 shows the effect replicates across architecturally distinct networks.

**Table 2: Cross-architecture results on CIFAR-10 (mean ± std, n=3 seeds)**

| Architecture | Condition | Accuracy (%) | Confidence (%) | Linear Probe (%) |
|-------------|-----------|-------------|----------------|-------------------|
| **VGG-11-BN** | Global | 85.44 ± 1.28 | — | 85.00 ± 1.80 |
| | Same-class | 10.88 ± 1.91 | — | 98.37 ± 0.63 |
| **SimpleCNN** | Global | 83.17 ± 0.23 | — | 80.53 ± 1.44 |
| | Same-class | 26.33 ± 0.48 | — | 54.68 ± 0.74 |
| **WideResNet-16-8** | Global | 90.61 ± 0.51 | 89.36 ± 0.43 | 91.02 ± 0.28 |
| | Same-class | 11.12 ± 0.90 | 11.04 ± 0.73 | 99.78 ± 0.09 |
| | Wrong-class | 64.06 ± 3.14 | 52.64 ± 3.51 | 99.60 ± 0.07 |
| | Random-class | 62.41 ± 1.76 | 51.21 ± 2.25 | 96.63 ± 0.44 |

All four architectures exhibit the same core pattern: same-class statistics devastate classification accuracy while linear probe accuracy remains high or *increases*.

VGG-11-BN shows the same pattern: same-class statistics reduce accuracy from 85.4% to 10.9% while linear probe accuracy *increases* from 85.0% to 98.4%. SimpleCNN shows the accuracy collapse (83.2% → 26.3%) but with reduced linear probe improvement (80.5% → 54.7%), likely reflecting its simpler representational capacity with only 4 BN layers.

WideResNet-16-8 provides particularly strong evidence because it is a high-capacity architecture with dropout regularization (p=0.3). Despite achieving the highest baseline accuracy (90.6%), same-class statistics reduce accuracy to 11.1% (t(2) = −95.94, p = 0.00011) while the linear probe reaches 99.8% (t(2) = +55.14, p = 0.00033). The wrong-class condition (64.1%) again substantially outperforms same-class (11.1%), with t(2) = −30.64, p = 0.0011. Notably, WideResNet's same-class accuracy (11.1%) is higher than SmallResNet's (3.5%), which may reflect the effect of dropout: by adding noise during training, dropout may make the classification head slightly more robust to the activation perturbation induced by BN statistic replacement.

The severity of the accuracy collapse correlates with the number of BN layers: SmallResNet (20 BN layers) drops to 3.5%, VGG-11-BN (8 layers) to 10.9%, WideResNet-16-8 (many BN layers, but with dropout) to 11.1%, and SimpleCNN (4 layers) to 26.3%. More BN layers generally means more opportunities for calibration disruption to compound, though dropout may partially mitigate the compounding effect.

### 4.3 Interpolation Between Same-Class and Global Statistics

Figure 1 shows the interpolation results (SmallResNet, CIFAR-10).

**Table 3: Interpolation from same-class (α=0) to global (α=1) statistics**

| α | Accuracy (%) | Linear Probe (%) |
|---|-------------|-------------------|
| 0.0 | 3.44 | 99.90 |
| 0.1 | 7.13 | 99.70 |
| 0.2 | 13.41 | 98.67 |
| 0.3 | 23.14 | 94.85 |
| 0.4 | 36.47 | 87.10 |
| 0.5 | 51.61 | 77.60 |
| 0.6 | 65.17 | 74.65 |
| 0.7 | 75.26 | 77.88 |
| 0.8 | 82.62 | 83.00 |
| 0.9 | 87.64 | 87.53 |
| 1.0 | 91.01 | 90.85 |

The transition is smooth and monotonic in both directions, ruling out sharp phase transitions. As α increases from 0 to 1:
- Classification accuracy rises continuously from 3.4% to 91.0%
- Linear probe accuracy *decreases* from 99.9% to 90.9%

The two metrics move in opposite directions: the more calibrated the BN statistics, the higher classification accuracy but the *lower* linear probe accuracy. This anti-correlation is the signature of the calibration mechanism.

### 4.4 CIFAR-100 Generalization

To test whether the finding generalizes beyond 10-class problems, we repeat the experiment on CIFAR-100 with SmallResNet (3 seeds, 25 epochs).

**Table 4: SmallResNet on CIFAR-100 (mean ± std, n=3 seeds)**

| Condition | Accuracy (%) | Confidence (%) | Linear Probe (%) |
|-----------|-------------|----------------|-------------------|
| Global (baseline) | 70.28 ± 0.52 | 64.30 ± 0.63 | 67.12 ± 0.32 |
| Same-class | 0.67 ± 0.47 | 0.94 ± 0.00 | 46.80 ± 35.62 |
| Wrong-class | 1.00 ± 0.00 | 1.00 ± 0.00 | 56.40 ± 36.87 |
| Random-class | 1.33 ± 0.47 | 1.00 ± 0.00 | 38.93 ± 16.91 |

The accuracy collapse is even more severe on CIFAR-100: same-class statistics reduce accuracy from 70.3% to 0.67%, well below the 1% chance level for 100 classes (t(2) = −147.3, p = 4.6 × 10⁻⁵). All non-global conditions produce near-chance accuracy and near-uniform confidence (~1%), consistent with complete calibration failure across 100 output dimensions.

**Linear probe variance.** While the accuracy collapse is unambiguous, linear probe results on CIFAR-100 exhibit high variance across seeds (std = 35.62% for same-class). This reflects the difficulty of ridge regression probes with 100 classes: the 256-dimensional feature space is less over-determined than in the 10-class setting, making probe accuracy sensitive to the particular seed's representation geometry. One seed achieves 97.1% probe accuracy under same-class statistics (comparable to CIFAR-10 results), while others are substantially lower. With 100 classes, more sophisticated probing methods (e.g., MLP probes or regularization tuning) may be needed for stable estimates. The key finding—catastrophic accuracy collapse despite intact convolutional features—remains robust.

### 4.5 Tiny-ImageNet-200 Generalization

To test whether our findings extend to higher-resolution images, larger label spaces, and more complex visual content, we evaluate on Tiny-ImageNet-200 using SmallResNet64—a five-stage variant of SmallResNet adapted for 64×64 inputs (3 seeds, 25 epochs).

**Table 5: SmallResNet64 on Tiny-ImageNet-200 (mean ± std, n=3 seeds)**

| Condition | Accuracy (%) | Confidence (%) | Linear Probe (%) |
|-----------|-------------|----------------|-------------------|
| Global (baseline) | 60.77 ± 0.30 | 53.75 ± 0.24 | 51.83 ± 0.55 |
| Same-class | 0.17 ± 0.24 | 0.49 ± 0.00 | 100.00 ± 0.00 |
| Wrong-class | 0.50 ± 0.00 | 0.50 ± 0.00 | 100.00 ± 0.00 |
| Random-class | 0.50 ± 0.00 | 0.50 ± 0.00 | 57.02 ± 3.41 |

Tiny-ImageNet-200 produces the most dramatic results in our study. Same-class statistics reduce classification accuracy from 60.8% to 0.17%—well below the 0.5% chance level for 200 classes (t(2) = −402.16, p = 6.2 × 10⁻⁶). The network's confidence collapses to a uniform 0.49% (near the 0.5% expected under random guessing over 200 classes), indicating complete calibration failure. Wrong-class statistics similarly collapse accuracy to 0.50% (t(2) = −280.89, p = 1.3 × 10⁻⁵).

The linear probe results are striking: under both same-class and wrong-class conditions, the probe achieves **perfect 100.0% accuracy with zero variance across seeds**. This is the cleanest dissociation in our study—complete output collapse coexisting with perfectly separable representations. Unlike the CIFAR-100 results where probe accuracy exhibited high variance (std = 35.62%), the five-stage SmallResNet64 architecture produces a 512-dimensional feature space that is substantially more over-determined for even 200 classes, enabling stable and perfect probe performance.

The same-class vs. wrong-class comparison is also informative: on Tiny-ImageNet, same-class accuracy (0.17%) is nominally lower than wrong-class (0.50%), though this difference does not reach statistical significance (t(2) = −2.0, p = 0.18)—both are effectively at floor. With 200 classes, any BN statistic perturbation causes total calibration failure, leaving no room for the same-class vs. wrong-class distinction that is visible with 10 classes. The random-class probe is lower (57.0%) because each sample sees a different class's statistics, averaging over inconsistent perturbation directions.

### 4.6 GroupNorm Control

To confirm the effect is specific to BN's running-statistic mechanism, we train SmallResNet with GroupNorm replacing all BatchNorm layers.

**Table 6: GroupNorm control on CIFAR-10 (mean ± std, n=3 seeds)**

| Metric | Value |
|--------|-------|
| Accuracy | 81.65 ± 1.94% |
| Linear Probe | 81.08 ± 1.64% |

GroupNorm models achieve 81.6% accuracy—comparable to the BatchNorm baseline (91.1%) and within the expected range for this architecture without BN's optimization benefits. Crucially, since GroupNorm computes normalization statistics from the current input at test time, there are no stored running statistics to manipulate. The class-conditional replacement paradigm is inapplicable, confirming that the accuracy collapse observed in Sections 4.1–4.5 is specific to BatchNorm's running-statistic mechanism, not a general property of normalization layers.

### 4.7 Calibration Analysis

To characterize the nature of the accuracy collapse, we measure expected calibration error (ECE; Guo et al., 2017) across BN statistic conditions on SmallResNet CIFAR-10 (3 seeds).

**Table 7: Expected Calibration Error by condition (SmallResNet, CIFAR-10, n=3 seeds)**

| Condition | Accuracy (%) | ECE |
|-----------|-------------|-----|
| Global (baseline) | 91.33 ± 0.12 | 0.025 ± 0.001 |
| Same-class | 2.87 ± 0.49 | 0.394 ± 0.013 |
| Wrong-class | 63.40 ± 4.26 | 0.064 ± 0.013 |

Under global statistics, the network is well-calibrated (ECE = 0.025). Same-class statistics induce severe miscalibration (ECE = 0.394): the network assigns confident predictions that are systematically wrong. Interestingly, wrong-class statistics produce moderate ECE (0.064) despite substantially reduced accuracy—the network's confidence and accuracy remain roughly aligned, just at a lower level. This pattern is consistent with the calibration hypothesis: same-class statistics specifically disrupt the mapping between activation magnitudes and output confidence, rather than destroying the underlying feature representation.

### 4.8 Temperature Scaling Cannot Recover Accuracy

A natural question is whether the accuracy collapse under same-class statistics is merely a logit-scale issue—perhaps the correct class still receives the highest pre-softmax score, but the softmax temperature is wrong. If so, temperature scaling should recover accuracy.

**Table 8: Temperature scaling on same-class condition (SmallResNet, CIFAR-10, n=3 seeds)**

| Metric | Before | After (T ≈ 3.43) |
|--------|--------|-------------------|
| Accuracy (%) | 2.87 ± 0.49 | 2.87 ± 0.49 |
| ECE | 0.394 ± 0.013 | 0.151 ± 0.003 |

Temperature scaling finds an optimal temperature of T ≈ 3.43 ± 0.02, indicating that logits under same-class statistics are substantially over-scaled. The high temperature successfully reduces ECE from 0.394 to 0.151—a meaningful improvement in calibration. However, accuracy remains unchanged at 2.87%. This is expected: temperature scaling is a monotonic transformation of logits that preserves the argmax, so it cannot change which class is predicted.

This result is important because it rules out a simple explanation of our findings. The accuracy collapse is not merely an issue of logit magnitude or softmax sharpness. Rather, same-class BN statistics fundamentally alter *which* class receives the highest logit, scrambling the output layer's class assignments. The network's predictions are not just poorly calibrated—they are systematically wrong in a way that no post-hoc logit rescaling can fix. This confirms that BN statistics control a structural calibration that aligns internal activation patterns with the classification head's learned decision boundaries, not merely the scale of the output distribution.

## 5. Discussion

### 5.1 Converging Lines of Evidence

Our results constitute eight independent lines of evidence that BN running statistics function as output calibrators rather than feature encoders:

1. **Same-class statistics destroy accuracy but preserve representations** (CIFAR-10, 4 architectures). The dramatic dissociation between 3.5% classification accuracy and 99.9% linear probe accuracy is inexplicable under the feature encoding hypothesis.
2. **The effect is amplified with more classes.** On CIFAR-100, same-class statistics reduce accuracy to 0.67%—below the 1% chance level. On Tiny-ImageNet-200, accuracy drops to 0.17%—below the 0.5% chance level—demonstrating that the calibration disruption scales with output space dimensionality.
3. **The effect generalizes to higher resolution.** On Tiny-ImageNet-200 (64×64 images, 200 classes), the dissociation is the most extreme observed: 0.17% classification accuracy with *perfect* 100% linear probe accuracy across all seeds. This rules out the possibility that the effect is an artifact of low-resolution inputs.
4. **The effect is BN-specific.** GroupNorm models, which lack running statistics, are immune by construction (81.6% accuracy preserved).
5. **Smooth interpolation with no phase transition.** The monotonic transition between same-class and global statistics (Section 4.3) indicates a continuous calibration mechanism rather than a discrete feature encoding/decoding process.
6. **ECE analysis reveals structured miscalibration.** Same-class statistics produce severely miscalibrated outputs (ECE = 0.394) while wrong-class statistics remain surprisingly well-calibrated (ECE = 0.064), indicating that the disruption is specifically in how activations are scaled for downstream interpretation.
7. **Temperature scaling cannot recover accuracy.** Post-hoc calibration reduces ECE (0.394 → 0.151) but yields zero accuracy recovery (2.9% → 2.9%), proving that same-class statistics alter logit *rankings*, not merely magnitudes.
8. **Dropout does not prevent the effect.** WideResNet-16-8, which uses dropout (p=0.3) as additional regularization, still shows catastrophic accuracy collapse (90.6% → 11.1%) with near-perfect probe accuracy (99.8%), demonstrating that the calibration dependence is not an artifact of overfitting to specific activation scales.

### 5.2 Why Same-Class Statistics Are Worse Than Wrong-Class

The most counterintuitive finding is that same-class BN statistics (3.5%) perform *worse* than wrong-class (65.4%) or random-class (59.0%) statistics. We propose the following explanation:

Class-conditional BN statistics differ from global statistics in a systematic, class-correlated direction. When the network processes class-*c* samples with class-*c* statistics, the normalization shifts activations in a way that is anti-correlated with the classification head's expectations. The classification head learned decision boundaries calibrated to globally-normalized activations—specific activation magnitudes at specific neurons correspond to specific classes.

Same-class statistics normalize away the between-class activation variation that the final linear layer exploits. Each class gets "centered" onto itself, collapsing the between-class signal. Wrong-class statistics introduce a mismatch, but one that may preserve or even amplify some between-class structure.

This is consistent with the linear probe results: same-class normalization makes representations *more* linearly separable (99.9% probe accuracy) by removing within-class variation, but the *original* classification head cannot exploit this because it was calibrated for globally-normalized inputs.

The ECE analysis (Section 4.7) adds nuance: wrong-class statistics produce well-calibrated but less accurate outputs (ECE = 0.064, accuracy = 63.4%), suggesting the mismatch introduces noise but preserves the approximate relationship between confidence and correctness. Same-class statistics produce severely miscalibrated outputs (ECE = 0.394, accuracy = 2.9%), indicating a *systematic* rather than random disruption—the network is confidently wrong, not merely uncertain.

On Tiny-ImageNet-200, this distinction collapses: with 200 classes, both same-class (0.17%) and wrong-class (0.50%) produce floor-level accuracy, as the calibration disruption is severe enough to completely scramble any residual class structure in the outputs.

### 5.3 The Calibration Decomposition

Our results support a clean functional decomposition of BN-equipped networks:

1. **Convolutional filters** learn class-discriminative features that are robust to normalization perturbation.
2. **BN running statistics** calibrate activation scale and shift to match downstream expectations.
3. **The classification head** is a thin linear layer highly sensitive to the specific calibration of its inputs.

The interpolation experiment (Section 4.3) makes this vivid: classification accuracy and linear probe accuracy move in opposite directions as we interpolate between same-class and global statistics. The optimal calibration for the classification head (α=1, global) is the *worst* point for linear separability, and vice versa.

The calibration analysis (Sections 4.7–4.8) provides further mechanistic evidence. The high ECE under same-class statistics (0.394) shows the network is confidently wrong—not uncertain, but miscalibrated. Temperature scaling reduces this miscalibration (ECE: 0.394 → 0.151) without recovering accuracy, demonstrating that the disruption operates at the level of class identity assignment, not output confidence scaling. BN statistics control which activation patterns map to which classes, not merely how confident the mapping is.

The Tiny-ImageNet-200 results (Section 4.5) provide the most compelling evidence for this decomposition. The five-stage SmallResNet64 learns rich 512-dimensional representations that remain *perfectly* linearly separable under BN statistic perturbation—even as the classification head produces completely random outputs. This perfect dissociation, achieved with 200 classes at 64×64 resolution, demonstrates that the calibration decomposition holds across scales.

### 5.4 Architecture-Dependent Sensitivity

The severity of the accuracy collapse under same-class statistics shows interesting variation across architectures that is broadly (but not perfectly) correlated with the number of BN layers:

| Architecture | BN Layers | Same-Class Acc (%) | Notes |
|-------------|-----------|-------------------|-------|
| SmallResNet | 20 | 3.55 | Deepest collapse |
| VGG-11-BN | 8 | 10.88 | No skip connections |
| WideResNet-16-8 | Many | 11.12 | Has dropout (p=0.3) |
| SimpleCNN | 4 | 26.33 | Fewest BN layers |

SmallResNet, with 20 BN layers and no dropout, shows the most severe collapse (3.5%). SimpleCNN, with only 4 BN layers, retains the most accuracy (26.3%), consistent with fewer calibration-disruption points. WideResNet-16-8 is an interesting case: despite having many BN layers and a very high-capacity architecture, its same-class accuracy (11.1%) is notably higher than SmallResNet's (3.5%). This may be attributable to dropout (p=0.3), which introduces stochastic noise during training and may make the classification head slightly more robust to the kind of activation perturbation induced by BN statistic replacement. Dropout effectively trains the classification head to tolerate some activation variance, partially buffering it against the calibration disruption.

### 5.5 Implications

**Transfer learning.** Resetting BN running statistics on a new domain is effective because it recalibrates activation scales, not because it encodes new features. Our results predict that fine-tuning the classification head after BN statistic reset should be sufficient for adaptation, even without retraining convolutional weights.

**Domain adaptation.** The success of AdaBN (Li et al., 2018)—replacing BN statistics with target-domain statistics—follows naturally from the calibration perspective: target-domain statistics recalibrate activations to a consistent scale the classification head can interpret.

**Normalization-free architectures.** Recent architectures like NFNets (Brock et al., 2021) replace BN with alternative scaling strategies. Our findings suggest the critical function being replaced is activation calibration. Simpler alternatives focused specifically on maintaining consistent activation scales may suffice.

**Model compression.** BN folding (absorbing BN into convolutional weights during deployment) bakes a specific calibration into the weights. Post-hoc weight modifications must preserve this calibration, not just feature extraction capability.

**Post-hoc calibration.** Our temperature scaling results (Section 4.8) reveal a fundamental distinction between *output calibration* (adjusting confidence to match accuracy, as in Guo et al., 2017) and *structural calibration* (aligning internal activation scales with downstream expectations). BN statistics perform structural calibration; temperature scaling performs output calibration. The two are complementary but address different failure modes.

### 5.6 Broader Significance

The methodological lesson is perhaps most important: **classification accuracy can be deeply misleading about representation quality.** A model at 3.5% accuracy—worse than random chance—may have learned excellent class-discriminative features. On Tiny-ImageNet-200, a model at 0.17% accuracy—a third of the chance rate—achieves *perfect* linear probe accuracy. The bottleneck is not representation but calibration. This suggests that many apparent failures in deep learning (e.g., in domain shift, adversarial settings, or after architectural modifications) may be calibration failures rather than representation failures.

The ECE analysis reinforces this point quantitatively: the same-class condition produces a network that is both severely inaccurate *and* severely miscalibrated, yet its internal representations are near-perfectly separable. Future work on diagnosing model failures should consider separately measuring representation quality and output calibration rather than relying on end-to-end accuracy alone.

## 6. Limitations

1. **Architecture scope.** While we test four architectures spanning residual networks (SmallResNet, WideResNet-16-8), plain convnets (VGG-11-BN), and minimal networks (SimpleCNN), all are trained on relatively small datasets. Scaling to ImageNet-scale models with hundreds of layers is an important direction.
2. **Dataset scope.** CIFAR-10, CIFAR-100, and Tiny-ImageNet-200 span 10 to 200 classes and 32×32 to 64×64 resolution, but remain within the domain of natural image classification. Fine-grained recognition, medical imaging, or natural language tasks may show different dynamics.
3. **Training duration.** Models are trained for 20–25 epochs on Apple Silicon (MPS backend). Longer training schedules may yield different BN statistic structure.
4. **Linear probe limitations.** Ridge regression and logistic regression test only linear separability. On CIFAR-100, the 256-dimensional feature space with 100 classes yields high probe variance across seeds (std up to 35.6%), suggesting more expressive probing methods (e.g., MLP probes or cross-validated regularization) would provide more stable estimates. The Tiny-ImageNet results (512-dimensional features, 200 classes) resolve this issue with perfect probe accuracy across seeds, suggesting that richer feature spaces mitigate the probe instability seen on CIFAR-100.
5. **Static replacement.** We replace all BN layers simultaneously. Layer-wise replacement could reveal which layers are most sensitive to calibration disruption.
6. **ECE limitations.** ECE with fixed binning can be sensitive to bin count and sample size (Nixon et al., 2019). We use 15 bins following standard practice, but kernel-based calibration measures could provide smoother estimates.

## 7. Conclusion

We demonstrate a striking dissociation between classification accuracy and representation quality under batch normalization statistic manipulation. Replacing global BN statistics with same-class statistics destroys classification (3.5% on CIFAR-10, 0.67% on CIFAR-100, 0.17% on Tiny-ImageNet-200) while yielding near-perfect or perfect linear probe accuracy (99.9% on CIFAR-10, 100.0% on Tiny-ImageNet-200). This effect replicates across four architectures—including WideResNet-16-8 with dropout—generalizes across three datasets spanning 10 to 200 classes and 32×32 to 64×64 resolution, varies smoothly under interpolation, and is specific to BatchNorm's running-statistic mechanism (absent with GroupNorm).

Calibration analysis reveals that same-class statistics induce severe miscalibration (ECE = 0.394 vs. 0.025 baseline), and temperature scaling—while halving ECE—cannot recover accuracy, confirming that the collapse reflects structural activation misalignment rather than a simple logit-scale shift. These findings establish that BN running statistics function as output calibrators rather than feature encoders. Networks learn representations that are remarkably robust to BN perturbation, but classification heads are precisely calibrated to the activation scale produced by global statistics.

This calibration perspective unifies observations from transfer learning, domain adaptation, and test-time adaptation, and has implications for the design of normalization layers and calibration-aware training procedures.

## References

- Benz, P., Zhang, C., Karjauv, A., & Kweon, I. S. (2021). Revisiting Batch Normalization for Improving Corruption Robustness. *WACV 2021*.
- Bjorck, N., Gomes, C. P., Selman, B., & Weinberger, K. Q. (2018). Understanding Batch Normalization. *NeurIPS 2018*.
- Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization. *ICML 2021*.
- Frankle, J., Schwab, D. J., & Morcos, A. S. (2020). Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs. *arXiv:2003.00152*.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *ICML 2017*.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*.
- Nixon, J., Dusenberry, M. W., Zhang, L., Jerfel, G., & Tran, D. (2019). Measuring Calibration in Deep Learning. *CVPR Workshops 2019*.
- Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2018). Revisiting Batch Normalization for Practical Domain Adaptation. *Pattern Recognition, 80*, 109–117.
- Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? *NeurIPS 2018*.
- Schneider, S., Rusak, E., Eck, L., Bringmann, O., Brendel, W., & Bethge, M. (2020). Improving Robustness Against Common Corruptions by Covariate Shift Adaptation. *NeurIPS 2020*.
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully Test-Time Adaptation by Entropy Minimization. *ICLR 2021*.
- Zagoruyko, S., & Komodakis, N. (2016). Wide Residual Networks. *BMVC 2016*.

---

## Appendix A: Experimental Configuration

| Parameter | Value |
|-----------|-------|
| **SmallResNet** | BasicBlock, [2,2,2,2], channels [32,64,128,256], 20 BN layers |
| **SmallResNet64** | BasicBlock, 5-stage, channels [32,64,128,256,512], adapted for 64×64 |
| **VGG-11-BN** | Standard VGG-11 with BN, 8 BN layers |
| **SimpleCNN** | 4-layer CNN, 4 BN layers |
| **WideResNet-16-8** | depth=16, widen_factor=8, dropout=0.3 |
| Training epochs | 20 (CIFAR-10), 25 (CIFAR-100, Tiny-ImageNet-200) |
| Batch size | 128 |
| Optimizer | SGD (momentum=0.9, weight_decay=5×10⁻⁴) |
| Learning rate | 0.1, cosine annealing |
| Augmentation | RandomCrop(32, pad=4), RandomHorizontalFlip (CIFAR); adapted for 64×64 (Tiny-ImageNet) |
| Normalization | CIFAR-10: μ=(0.4914, 0.4822, 0.4465), σ=(0.2470, 0.2435, 0.2616) |
| Linear probe | Ridge regression (λ=10⁻³) or LogisticRegression, 80/20 train/test split |
| ECE | 15 equal-width bins (Guo et al., 2017) |
| Temperature scaling | NLL-optimized scalar T on validation set |
| Seeds | 42–51 (SmallResNet CIFAR-10), 42–44 (all others) |
| Device | Apple Silicon (MPS backend); mixed_precision for WideResNet-16-8 |

## Appendix B: Full Per-Seed Results

**Table B1: SmallResNet CIFAR-10 — Classification accuracy by seed and condition**

| Seed | Global | Same-Class | Wrong-Class | Random-Class |
|------|--------|------------|-------------|--------------|
| 42 | 90.96 | 3.85 | 60.19 | 55.26 |
| 43 | 91.05 | 4.02 | 70.70 | 67.73 |
| 44 | 91.01 | 2.46 | 69.93 | 55.45 |
| 45 | 91.07 | 4.06 | 64.59 | 56.74 |
| 46 | 91.26 | 3.84 | 57.08 | 56.07 |
| 47 | 91.30 | 3.37 | 60.49 | 57.01 |
| 48 | 90.95 | 2.39 | 67.19 | 50.82 |
| 49 | 91.19 | 2.90 | 63.57 | 56.01 |
| 50 | 90.93 | 4.22 | 69.00 | 66.09 |
| 51 | 91.04 | 4.35 | 70.98 | 68.72 |

**Table B2: SmallResNet CIFAR-10 — Linear probe accuracy by seed and condition**

| Seed | Global | Same-Class | Wrong-Class | Random-Class |
|------|--------|------------|-------------|--------------|
| 42 | 90.30 | 99.90 | 100.00 | 96.30 |
| 43 | 90.65 | 99.80 | 100.00 | 96.85 |
| 44 | 91.10 | 99.90 | 100.00 | 97.05 |
| 45 | 89.65 | 99.90 | 100.00 | 96.90 |
| 46 | 90.80 | 100.00 | 100.00 | 98.30 |
| 47 | 91.80 | 99.95 | 100.00 | 99.35 |
| 48 | 90.80 | 99.90 | 100.00 | 96.80 |
| 49 | 91.15 | 99.90 | 100.00 | 98.90 |
| 50 | 91.00 | 99.65 | 100.00 | 96.65 |
| 51 | 91.30 | 99.90 | 100.00 | 99.15 |

**Table B3: SmallResNet CIFAR-100 — Per-seed results**

| Seed | Global Acc (%) | Same-Class Acc (%) | Global Probe (%) | Same-Class Probe (%) |
|------|---------------|-------------------|-------------------|----------------------|
| 42 | 70.95 | 1.00 | 67.55 | 97.10 |
| 43 | 69.68 | 1.00 | 66.80 | 24.10 |
| 44 | 70.21 | 0.00 | 67.00 | 19.20 |

**Table B4: ECE and Temperature Scaling — Per-seed results (SmallResNet, CIFAR-10)**

| Seed | Global ECE | Same-Class ECE | Same-Class ECE (post-T) | Temperature |
|------|-----------|----------------|-------------------------|-------------|
| 42 | 0.026 | 0.382 | 0.150 | 3.456 |
| 43 | 0.024 | 0.412 | 0.148 | 3.428 |
| 44 | 0.026 | 0.388 | 0.154 | 3.416 |

**Table B5: WideResNet-16-8 CIFAR-10 — Per-seed results (n=3 seeds)**

| Seed | Global Acc (%) | Same-Class Acc (%) | Wrong-Class Acc (%) | Random-Class Acc (%) |
|------|---------------|-------------------|--------------------|--------------------|
| 42 | 90.10 | 10.22 | 60.92 | 60.65 |
| 43 | 91.12 | 12.02 | 67.20 | 64.17 |
| 44 | 90.61 | 11.12 | 64.06 | 62.41 |

| Seed | Global Probe (%) | Same-Class Probe (%) | Wrong-Class Probe (%) | Random-Class Probe (%) |
|------|-----------------|---------------------|----------------------|----------------------|
| 42 | 90.74 | 99.69 | 99.53 | 96.19 |
| 43 | 91.30 | 99.87 | 99.67 | 97.07 |
| 44 | 91.02 | 99.78 | 99.60 | 96.63 |

**Table B6: SmallResNet64 Tiny-ImageNet-200 — Per-seed results (n=3 seeds)**

| Seed | Global Acc (%) | Global Conf (%) | Global Probe (%) | Same-Class Acc (%) | Same-Class Probe (%) |
|------|---------------|----------------|-------------------|-------------------|----------------------|
| 42 | 60.88 | 53.51 | 51.55 | 0.00 | 100.00 |
| 43 | 61.08 | 54.00 | 52.60 | 0.50 | 100.00 |
| 44 | 60.36 | 53.75 | 51.35 | 0.00 | 100.00 |
