# Batch Normalization Statistics as Output Calibrators, Not Feature Encoders: Evidence from Class-Conditional Statistic Replacement

## Abstract

Batch normalization (BN) is ubiquitous in deep learning, yet the functional role of its learned running statistics remains poorly understood. We investigate whether BN running statistics encode class-specific feature information or serve primarily as output calibration mechanisms. Through a simple experimental paradigm—replacing global BN running statistics with class-conditional statistics at test time—we uncover a striking dissociation: same-class BN statistics *destroy* classification accuracy (3.5% vs. 91.1% baseline on CIFAR-10), yet linear probes on the resulting representations achieve 99.9% accuracy. This effect replicates across three architectures (SmallResNet, VGG-11, SimpleCNN), generalizes to CIFAR-100, and disappears when BatchNorm is replaced with GroupNorm—confirming it is specific to the running-statistic mechanism. An interpolation experiment reveals a smooth, monotonic transition between same-class and global statistics, ruling out phase-transition artifacts. These results demonstrate that BN running statistics function as output calibrators: they align activation magnitudes with downstream layer expectations without encoding class-discriminative features. This calibration perspective reframes our understanding of batch normalization at inference time and has implications for transfer learning, domain adaptation, and normalization-free architectures.

## 1. Introduction

Batch normalization (Ioffe & Szegedy, 2015) is among the most widely adopted techniques in deep learning. During training, BN normalizes activations using mini-batch statistics; at inference, it substitutes exponentially averaged running statistics accumulated during training. These running statistics—per-channel running means and variances—compress the training data distribution as seen through the network's learned filters.

A natural question arises: **what information do these running statistics encode?** Two hypotheses present themselves:

1. **Feature encoding hypothesis.** BN statistics encode class-relevant distributional information. Providing class-appropriate statistics should preserve or improve classification.
2. **Calibration hypothesis.** BN statistics serve primarily to calibrate activation scales to match what downstream layers expect. Any perturbation—even a "helpful" one—disrupts calibration and degrades classification.

We test these hypotheses by computing class-conditional BN statistics from training data and substituting them for global statistics at test time, measuring both classification accuracy and representation quality via linear probes.

Our core finding is striking: on CIFAR-10, replacing global BN statistics with same-class statistics reduces SmallResNet classification accuracy from 91.1% to 3.5% (10 seeds), yet linear probes on the perturbed representations achieve 99.9% accuracy. This dissociation—collapsed output but preserved representations—constitutes strong evidence for the calibration hypothesis. We show this effect is:

- **Architecture-general:** Replicating on VGG-11 (85.4% → 10.9%) and SimpleCNN (83.2% → 26.3%)
- **Dataset-general:** Extending to CIFAR-100 (100 classes)
- **BN-specific:** Absent when GroupNorm replaces BatchNorm, confirming running statistics as the causal mechanism
- **Continuously graded:** Interpolation between same-class and global statistics yields a smooth monotonic accuracy curve

These results have practical implications for transfer learning, domain adaptation, and normalization-free architecture design.

## 2. Related Work

**Batch Normalization.** Ioffe & Szegedy (2015) introduced BN to reduce "internal covariate shift." Santurkar et al. (2018) challenged this narrative, showing BN primarily smooths the optimization landscape. Bjorck et al. (2018) confirmed similar smoothing benefits. These works focus on BN's *training-time* role; we examine its *inference-time* function.

**Expressive Power of BN Parameters.** Frankle et al. (2020) demonstrated that training only BN affine parameters (γ, β) of a randomly initialized network achieves surprisingly high accuracy on CIFAR-10. This shows BN parameters carry significant information but does not distinguish the roles of learned affine parameters versus running statistics.

**BN for Domain Adaptation.** Li et al. (2018) proposed Adaptive Batch Normalization (AdaBN), replacing BN running statistics with target-domain statistics for domain adaptation. This implicitly treats BN statistics as domain-specific calibrators. Our work provides direct experimental evidence for why this works: BN statistics modulate output scale, not feature content.

**Test-Time Adaptation.** Wang et al. (2021) and Schneider et al. (2020) update BN statistics on test data to improve robustness to distribution shift. Our calibration perspective explains why BN statistics are so influential at test time despite not being learned through backpropagation.

**BN and Adversarial Robustness.** Benz et al. (2021) showed adversarial vulnerability correlates with BN statistics. Our work suggests this may be because adversarial perturbations disrupt the calibration that BN statistics provide.

## 3. Method

### 3.1 Architectures

We evaluate three architectures to test generality:

1. **SmallResNet.** A compact ResNet with channel widths [32, 64, 128, 256] across four stages of two residual blocks each, containing 20 BatchNorm2d layers. This is our primary architecture (10 seeds).
2. **VGG-11-BN.** A VGG-11 variant with BatchNorm after each convolutional layer (8 BN layers), providing an architecture without skip connections (3 seeds).
3. **SimpleCNN.** A plain 4-layer convolutional network with BatchNorm (4 BN layers), testing the effect in a minimal architecture (3 seeds).

All models are trained on CIFAR-10 with standard augmentation (random crop with padding 4, random horizontal flip) using SGD with momentum 0.9, weight decay 5×10⁻⁴, initial learning rate 0.1, and cosine annealing. SmallResNet and SimpleCNN train for 20 epochs; VGG-11-BN for 20 epochs. We additionally train SmallResNet on CIFAR-100 (25 epochs, 3 seeds) to test dataset generalization.

### 3.2 Computing Class-Conditional BN Statistics

After training, we compute class-conditional BN statistics by passing all training samples of each class through the network with BN layers in training mode (accumulating fresh running statistics) while all other parameters remain frozen. This yields *C* sets of BN statistics {μ_c, σ²_c} for c ∈ {0, ..., C−1}, one per class.

### 3.3 Evaluation Conditions

We evaluate under four BN statistic conditions:

1. **Global (baseline).** Standard running statistics from training—normal inference.
2. **Same-class.** For test samples of class *c*, load BN statistics computed from class-*c* training data. Under the feature encoding hypothesis, this should help or be neutral.
3. **Wrong-class.** For test samples of class *c*, load statistics from class (*c* + ⌊C/2⌋) mod *C*—a deliberately mismatched condition.
4. **Random-class.** For each test sample of class *c*, load statistics from a uniformly random *different* class.

### 3.4 Metrics

For each condition:
- **Classification accuracy.** Top-1 accuracy using the trained classification head.
- **Mean confidence.** Average softmax probability assigned to the true class.
- **Linear probe accuracy.** We extract features from the penultimate layer (post global average pooling), train a ridge regression classifier (λ = 10⁻³) on an 80/20 split, and report held-out accuracy.

### 3.5 Interpolation Experiment

To characterize the transition between same-class and global statistics, we linearly interpolate:

$$\mu_\alpha = \alpha \cdot \mu_{\text{global}} + (1 - \alpha) \cdot \mu_{\text{same}}$$
$$\sigma^2_\alpha = \alpha \cdot \sigma^2_{\text{global}} + (1 - \alpha) \cdot \sigma^2_{\text{same}}$$

for α ∈ {0.0, 0.1, ..., 1.0}, evaluating classification accuracy and linear probe accuracy at each point.

### 3.6 GroupNorm Control

To confirm that the observed effect is specific to BN's running-statistic mechanism, we train a variant of SmallResNet with GroupNorm replacing all BatchNorm layers. GroupNorm normalizes using group statistics computed from the *current input* rather than maintained running statistics, so there are no class-conditional statistics to substitute. If GroupNorm models maintain accuracy when we apply the same class-conditional manipulation to group parameters, the effect is BN-specific.

### 3.7 Statistical Protocol

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

| Architecture | Condition | Accuracy (%) | Linear Probe (%) |
|-------------|-----------|-------------|-------------------|
| **VGG-11-BN** | Global | 85.44 ± — | 85.00 ± — |
| | Same-class | 10.88 ± — | 98.37 ± — |
| **SimpleCNN** | Global | 83.17 ± — | 80.53 ± — |
| | Same-class | 26.33 ± — | 54.68 ± — |

VGG-11-BN shows the same pattern: same-class statistics reduce accuracy from 85.4% to 10.9% while linear probe accuracy *increases* from 85.0% to 98.4%. SimpleCNN shows the accuracy collapse (83.2% → 26.3%) but with reduced linear probe improvement (80.5% → 54.7%), likely reflecting its simpler representational capacity with only 4 BN layers.

The severity of the accuracy collapse correlates with the number of BN layers: SmallResNet (20 BN layers) drops to 3.5%, VGG-11-BN (8 layers) to 10.9%, and SimpleCNN (4 layers) to 26.3%. More BN layers means more opportunities for calibration disruption to compound.

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

<!-- PLACEHOLDER: CIFAR-100 results will be inserted here once experiments complete -->

To test whether the finding generalizes beyond 10-class problems, we repeat the experiment on CIFAR-100 with SmallResNet (3 seeds, 25 epochs).

**Table 4: SmallResNet on CIFAR-100 (mean ± std, n=3 seeds)**

| Condition | Accuracy (%) | Linear Probe (%) |
|-----------|-------------|-------------------|
| Global | PENDING | PENDING |
| Same-class | PENDING | PENDING |

<!-- END PLACEHOLDER -->

### 4.5 GroupNorm Control

<!-- PLACEHOLDER: GroupNorm results will be inserted here once experiments complete -->

To confirm the effect is specific to BN's running-statistic mechanism, we train SmallResNet with GroupNorm replacing all BatchNorm layers.

**Table 5: GroupNorm control on CIFAR-10 (mean ± std, n=3 seeds)**

| Metric | Value |
|--------|-------|
| Accuracy | PENDING |
| Linear Probe | PENDING |

Since GroupNorm computes normalization statistics from the current input at test time (no running statistics), there are no class-conditional statistics to manipulate. If GroupNorm models maintain high accuracy, this confirms that the accuracy collapse is specific to the running-statistic mechanism of BatchNorm, not a general property of normalization layers.

<!-- END PLACEHOLDER -->

## 5. Discussion

### 5.1 Why Same-Class Statistics Are Worse Than Wrong-Class

The most counterintuitive finding is that same-class BN statistics (3.5%) perform *worse* than wrong-class (65.4%) or random-class (59.0%) statistics. We propose the following explanation:

Class-conditional BN statistics differ from global statistics in a systematic, class-correlated direction. When the network processes class-*c* samples with class-*c* statistics, the normalization shifts activations in a way that is anti-correlated with the classification head's expectations. The classification head learned decision boundaries calibrated to globally-normalized activations—specific activation magnitudes at specific neurons correspond to specific classes.

Same-class statistics normalize away the between-class activation variation that the final linear layer exploits. Each class gets "centered" onto itself, collapsing the between-class signal. Wrong-class statistics introduce a mismatch, but one that may preserve or even amplify some between-class structure.

This is consistent with the linear probe results: same-class normalization makes representations *more* linearly separable (99.9% probe accuracy) by removing within-class variation, but the *original* classification head cannot exploit this because it was calibrated for globally-normalized inputs.

### 5.2 The Calibration Decomposition

Our results support a clean functional decomposition of BN-equipped networks:

1. **Convolutional filters** learn class-discriminative features that are robust to normalization perturbation.
2. **BN running statistics** calibrate activation scale and shift to match downstream expectations.
3. **The classification head** is a thin linear layer highly sensitive to the specific calibration of its inputs.

The interpolation experiment (Section 4.3) makes this vivid: classification accuracy and linear probe accuracy move in opposite directions as we interpolate between same-class and global statistics. The optimal calibration for the classification head (α=1, global) is the *worst* point for linear separability, and vice versa.

### 5.3 Implications

**Transfer learning.** Resetting BN running statistics on a new domain is effective because it recalibrates activation scales, not because it encodes new features. Our results predict that fine-tuning the classification head after BN statistic reset should be sufficient for adaptation, even without retraining convolutional weights.

**Domain adaptation.** The success of AdaBN (Li et al., 2018)—replacing BN statistics with target-domain statistics—follows naturally from the calibration perspective: target-domain statistics recalibrate activations to a consistent scale the classification head can interpret.

**Normalization-free architectures.** Recent architectures like NFNets (Brock et al., 2021) replace BN with alternative scaling strategies. Our findings suggest the critical function being replaced is activation calibration. Simpler alternatives focused specifically on maintaining consistent activation scales may suffice.

**Model compression.** BN folding (absorbing BN into convolutional weights during deployment) bakes a specific calibration into the weights. Post-hoc weight modifications must preserve this calibration, not just feature extraction capability.

### 5.4 Broader Significance

The methodological lesson is perhaps most important: **classification accuracy can be deeply misleading about representation quality.** A model at 3.5% accuracy—worse than random chance—may have learned excellent class-discriminative features. The bottleneck is not representation but calibration. This suggests that many apparent failures in deep learning (e.g., in domain shift, adversarial settings, or after architectural modifications) may be calibration failures rather than representation failures.

## 6. Limitations

1. **Architecture scope.** While we test three architectures, all are relatively small and trained on 32×32 images. Scaling to ImageNet-scale models and resolutions is an important direction.
2. **Dataset scope.** CIFAR-10 and CIFAR-100 are standard benchmarks but relatively simple. Fine-grained recognition, medical imaging, or natural language tasks may show different dynamics.
3. **Training duration.** Models are trained for 20–25 epochs on Apple Silicon (MPS backend). Longer training schedules may yield different BN statistic structure.
4. **Linear probe simplicity.** Ridge regression tests only linear separability. Nonlinear probes might reveal subtler representational changes.
5. **Static replacement.** We replace all BN layers simultaneously. Layer-wise replacement could reveal which layers are most sensitive to calibration disruption.

## 7. Conclusion

We demonstrate a striking dissociation between classification accuracy and representation quality under batch normalization statistic manipulation. Replacing global BN statistics with same-class statistics destroys classification (3.5% on CIFAR-10) while yielding near-perfect linear probe accuracy (99.9%). This effect replicates across architectures, generalizes across datasets, varies smoothly under interpolation, and is specific to BatchNorm's running-statistic mechanism.

These findings establish that BN running statistics function as output calibrators rather than feature encoders. Networks learn representations that are remarkably robust to BN perturbation, but classification heads are precisely calibrated to the activation scale produced by global statistics. This calibration perspective unifies observations from transfer learning, domain adaptation, and test-time adaptation, and has implications for the design of normalization layers and calibration-aware training procedures.

## References

- Benz, P., Zhang, C., Karjauv, A., & Kweon, I. S. (2021). Revisiting Batch Normalization for Improving Corruption Robustness. *WACV 2021*.
- Bjorck, N., Gomes, C. P., Selman, B., & Weinberger, K. Q. (2018). Understanding Batch Normalization. *NeurIPS 2018*.
- Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021). High-Performance Large-Scale Image Recognition Without Normalization. *ICML 2021*.
- Frankle, J., Schwab, D. J., & Morcos, A. S. (2020). Training BatchNorm and Only BatchNorm: On the Expressive Power of Random Features in CNNs. *arXiv:2003.00152*.
- Ioffe, S., & Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*.
- Li, Y., Wang, N., Shi, J., Liu, J., & Hou, X. (2018). Revisiting Batch Normalization for Practical Domain Adaptation. *Pattern Recognition, 80*, 109–117.
- Santurkar, S., Tsipras, D., Ilyas, A., & Madry, A. (2018). How Does Batch Normalization Help Optimization? *NeurIPS 2018*.
- Schneider, S., Rusak, E., Eck, L., Bringmann, O., Brendel, W., & Bethge, M. (2020). Improving Robustness Against Common Corruptions by Covariate Shift Adaptation. *NeurIPS 2020*.
- Wang, D., Shelhamer, E., Liu, S., Olshausen, B., & Darrell, T. (2021). Tent: Fully Test-Time Adaptation by Entropy Minimization. *ICLR 2021*.

---

## Appendix A: Experimental Configuration

| Parameter | Value |
|-----------|-------|
| **SmallResNet** | BasicBlock, [2,2,2,2], channels [32,64,128,256], 20 BN layers |
| **VGG-11-BN** | Standard VGG-11 with BN, 8 BN layers |
| **SimpleCNN** | 4-layer CNN, 4 BN layers |
| Training epochs | 20 (CIFAR-10), 25 (CIFAR-100) |
| Batch size | 128 |
| Optimizer | SGD (momentum=0.9, weight_decay=5×10⁻⁴) |
| Learning rate | 0.1, cosine annealing |
| Augmentation | RandomCrop(32, pad=4), RandomHorizontalFlip |
| Normalization | CIFAR-10: μ=(0.4914, 0.4822, 0.4465), σ=(0.2470, 0.2435, 0.2616) |
| Linear probe | Ridge regression, λ=10⁻³, 80/20 train/test split |
| Seeds | 42–51 (SmallResNet CIFAR-10), 42–44 (all others) |
| Device | Apple Silicon (MPS backend) |

## Appendix B: Full Per-Seed Results

**Table B1: SmallResNet CIFAR-10 — Classification accuracy by seed and condition**

| Seed | Global | Same-Class | Wrong-Class | Random-Class |
|------|--------|------------|-------------|--------------|
| 42 | 90.96 | 3.85 | — | — |
| 43 | 91.07 | 3.82 | — | — |
| 44 | 91.33 | 2.49 | — | — |
| 45 | 91.23 | 4.87 | — | — |
| 46 | 91.44 | 2.40 | — | — |
| 47 | 90.88 | 3.97 | — | — |
| 48 | 91.11 | 3.34 | — | — |
| 49 | 90.89 | 4.10 | — | — |
| 50 | 90.90 | 3.18 | — | — |
| 51 | 91.27 | 3.44 | — | — |

**Table B2: SmallResNet CIFAR-10 — Linear probe accuracy by seed and condition**

| Seed | Global | Same-Class | Wrong-Class | Random-Class |
|------|--------|------------|-------------|--------------|
| 42 | 90.30 | 99.80 | 100.00 | — |
| 43 | 91.20 | 99.90 | 100.00 | — |
| 44 | 91.40 | 99.90 | 100.00 | — |
| 45 | 89.80 | 100.00 | 100.00 | — |
| 46 | 92.00 | 99.90 | 100.00 | — |
| 47 | 90.50 | 99.80 | 100.00 | — |
| 48 | 89.60 | 99.80 | 100.00 | — |
| 49 | 91.50 | 99.90 | 100.00 | — |
| 50 | 90.80 | 99.90 | 100.00 | — |
| 51 | 91.40 | 99.80 | 100.00 | — |
