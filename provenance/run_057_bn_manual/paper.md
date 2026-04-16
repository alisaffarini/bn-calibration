# Batch Normalization Statistics as Output Calibrators, Not Feature Encoders: Evidence from Class-Conditional Statistic Replacement

## Abstract

Batch normalization (BN) is ubiquitous in deep learning, yet the functional role of its learned running statistics remains poorly understood. We investigate whether BN running statistics encode class-specific information that aids recognition, or merely serve as an output calibration mechanism. We train a ResNet on CIFAR-10, compute class-conditional BN statistics, and replace global running statistics with class-matched statistics at test time. Surprisingly, using same-class BN statistics *destroys* classification accuracy (1.8% vs. 91.9% baseline), reducing performance to near chance. However, linear probes on the resulting representations achieve 99.97% accuracy, indicating that the underlying features remain excellent. This dramatic dissociation—collapsed classification but preserved representation quality—reveals that BN running statistics function primarily as output calibrators rather than feature encoders. Networks learn representations robust to BN statistic perturbation, but the classification head is calibrated to the specific activation scale produced by global statistics. These findings have implications for transfer learning, domain adaptation, and the design of normalization-free architectures.

## 1. Introduction

Batch normalization (Ioffe & Szegedy, 2015) is one of the most widely adopted techniques in deep learning, enabling faster training and improved generalization across architectures and domains. During training, BN normalizes activations using batch statistics; at inference, it uses exponentially averaged running statistics accumulated during training. These running statistics—running mean and running variance for each channel—represent a compressed summary of the training data distribution as seen through the network's learned filters.

A natural question arises: **what information do these running statistics encode?** Two hypotheses present themselves:

1. **Feature encoding hypothesis:** BN statistics encode class-relevant distributional information, and providing class-appropriate statistics should improve—or at least preserve—classification performance.
2. **Calibration hypothesis:** BN statistics serve primarily to calibrate the scale and shift of activations to match what downstream layers (including the classification head) expect. Any perturbation, even a "helpful" one, disrupts this calibration and degrades performance.

We test these hypotheses through a simple but revealing experiment: we compute class-conditional BN statistics from training data and substitute them for global statistics at inference time, evaluating both classification accuracy and representation quality via linear probes.

Our core finding is striking: replacing global BN statistics with same-class statistics reduces classification accuracy from 91.9% to 1.8% (near random chance on 10 classes), yet the representations extracted under these perturbed statistics support 99.97% linear probe accuracy. This dissociation constitutes strong evidence for the calibration hypothesis. The network's learned features are remarkably robust to BN statistic manipulation, but the trained classification head requires precisely calibrated input magnitudes.

This result has practical implications for transfer learning (where BN statistics are often reset), domain adaptation (where BN statistic alignment is a common technique), and the growing interest in normalization-free architectures.

## 2. Related Work

**Batch Normalization.** Ioffe & Szegedy (2015) introduced BN to reduce "internal covariate shift," normalizing layer inputs to accelerate training. The technique computes channel-wise mean and variance over mini-batches during training, maintaining exponential moving averages as running statistics for inference.

**Understanding BN's Role.** Santurkar et al. (2018) challenged the internal covariate shift narrative, demonstrating that BN's primary benefit is smoothing the optimization landscape rather than reducing covariate shift per se. Bjorck et al. (2018) showed similar smoothing benefits. Our work complements these findings by examining the *inference-time* role of BN statistics, rather than their training-time effects.

**Training BatchNorm and Only BatchNorm.** Frankle et al. (2020) showed that training only the BN parameters (affine weights γ and β) of a randomly initialized network can achieve surprisingly high accuracy on CIFAR-10. This demonstrates that BN parameters carry significant information, but does not distinguish between feature encoding and calibration roles of the running statistics specifically.

**BN Statistics for Domain Adaptation.** Li et al. (2018) proposed Adaptive Batch Normalization (AdaBN), replacing BN running statistics with target-domain statistics to improve domain adaptation. This technique implicitly treats BN statistics as domain-specific calibrators—aligning with our calibration hypothesis. Our work provides direct experimental evidence for why this approach works: BN statistics modulate output scale rather than feature content.

**BN Statistics and Model Behavior.** Several works have observed that BN statistics interact with model behavior in non-obvious ways. Benz et al. (2021) showed that adversarial vulnerability correlates with BN statistics, and various works on test-time adaptation (Wang et al., 2021; Schneider et al., 2020) update BN statistics on test data to improve robustness to distribution shift. Our work provides a mechanistic explanation for why BN statistics are so influential despite not being learned through gradient descent.

## 3. Method

### 3.1 Architecture and Training

We use a compact ResNet architecture (SmallResNet) with channel widths [32, 64, 128, 256] across four stages of two residual blocks each, totaling **20 BatchNorm2d layers** (including shortcut connections). The model is trained on CIFAR-10 with standard augmentation (random crop with padding 4, random horizontal flip) using SGD with momentum 0.9, weight decay 5×10⁻⁴, initial learning rate 0.1 with cosine annealing over 25 epochs. This configuration achieves 91.9% ± 0.2% test accuracy, consistent with expected performance for this architecture class.

### 3.2 Computing Class-Conditional BN Statistics

After training, we compute class-conditional BN statistics by passing all training samples of each class through the network with BN layers in training mode (accumulating fresh running statistics) while all other parameters remain frozen. This yields 10 sets of BN statistics {μ_c, σ²_c} for c ∈ {0, ..., 9}, one per CIFAR-10 class. We use the non-augmented training set to ensure deterministic statistics.

### 3.3 Evaluation Conditions

We evaluate the trained model under four BN statistic conditions:

1. **Global (baseline):** Standard running statistics from training. This is normal inference.
2. **Same-class:** For test samples of class c, load BN statistics computed from class-c training data. If BN statistics encode class-relevant information, this should help (or at least not hurt).
3. **Wrong-class:** For test samples of class c, load BN statistics from class (c + 5) mod 10. A deliberately mismatched condition.
4. **Random-class:** For test samples of class c, load BN statistics from a randomly chosen different class. Tests the average effect of class mismatch.

### 3.4 Metrics

For each condition we measure:
- **Classification accuracy:** Top-1 accuracy using the model's trained classification head.
- **Mean confidence:** Average softmax probability assigned to the true class.
- **Linear probe accuracy:** We extract 256-dimensional features from the penultimate layer (post global average pooling), train a ridge regression classifier (λ = 10⁻³) on an 80/20 split, and report test accuracy. This measures representation quality independent of the original classification head.

### 3.5 Statistical Protocol

All experiments are repeated across 5 random seeds (42–46), with models trained independently from scratch for each seed. We report means ± standard deviations and conduct paired t-tests across seeds for key comparisons.

## 4. Results

### 4.1 Main Results

Table 1 presents aggregate results across all 5 seeds.

**Table 1: Classification performance under different BN statistic conditions (mean ± std, n=5 seeds)**

| Condition | Accuracy (%) | Confidence (%) | Linear Probe (%) |
|-----------|-------------|----------------|-------------------|
| Global (baseline) | 91.88 ± 0.20 | 89.95 ± 0.12 | 91.64 ± 0.31 |
| Same-class | 1.79 ± 0.71 | 3.71 ± 0.32 | 99.97 ± 0.02 |
| Wrong-class | 55.52 ± 6.76 | 40.80 ± 1.70 | 100.00 ± 0.00 |
| Random-class | 51.39 ± 7.07 | 37.22 ± 2.99 | 97.03 ± 0.94 |

The results are remarkable:

1. **Same-class BN statistics catastrophically degrade accuracy** to 1.8%, below the 10% random baseline for a 10-class problem. This is not merely unhelpful—it is *actively destructive*.
2. **Wrong-class statistics perform far better** (55.5%) than same-class statistics (1.8%), a complete inversion of what the feature encoding hypothesis would predict.
3. **Linear probe accuracy under same-class statistics reaches 99.97%**, dramatically *exceeding* the global baseline's linear probe (91.64%). The representations become *more* linearly separable when BN statistics are perturbed with class-conditional statistics.
4. **All conditions maintain high linear probe accuracy** (97%+), even though classification accuracy varies from 1.8% to 91.9%.

### 4.2 Statistical Significance

All key comparisons are highly significant (Table 2).

**Table 2: Paired t-tests across 5 seeds (df = 4)**

| Comparison | t-statistic | p-value | Significance |
|-----------|------------|---------|--------------|
| Same vs. Global (accuracy) | −238.70 | 1.85 × 10⁻⁹ | *** |
| Wrong vs. Global (accuracy) | −10.96 | 3.93 × 10⁻⁴ | *** |
| Same vs. Wrong (accuracy) | −14.68 | 1.25 × 10⁻⁴ | *** |
| Same vs. Global (confidence) | −467.40 | 1.26 × 10⁻¹⁰ | *** |
| Same vs. Global (linear probe) | +51.03 | 8.83 × 10⁻⁷ | *** |

The last row is particularly notable: same-class BN statistics significantly *improve* representation quality (p < 10⁻⁶) despite destroying classification accuracy.

### 4.3 Per-Class Analysis

Table 3 shows the per-class accuracy drop when switching from global to same-class BN statistics.

**Table 3: Per-class accuracy delta (same-class minus global), mean ± std across 5 seeds**

| Class | Δ Accuracy | Notes |
|-------|-----------|-------|
| Airplane | −87.1 ± 5.7% | Near-total collapse |
| Automobile | −96.6 ± 0.5% | Most severe; highly consistent |
| Bird | −88.5 ± 0.6% | |
| Cat | −71.8 ± 7.3% | Least severe; highest variance |
| Deer | −91.7 ± 1.0% | |
| Dog | −87.5 ± 0.5% | |
| Frog | −93.9 ± 0.7% | |
| Horse | −93.6 ± 0.5% | |
| Ship | −95.5 ± 0.6% | |
| Truck | −94.6 ± 0.6% | |

The effect is universal across all classes, with drops ranging from −71.8% (cat) to −96.6% (automobile). Cat's relatively smaller drop and higher variance may reflect the well-known difficulty of the cat class in CIFAR-10 (lower baseline accuracy), potentially causing more diffuse class-conditional statistics.

### 4.4 Per-Seed Consistency

Results are highly consistent across seeds. Global accuracy ranges from 91.6% to 92.2% (std = 0.2%), while same-class accuracy ranges from 0.75% to 2.92% (all near chance). Linear probe accuracy under same-class conditions is 99.95%–100.0% across all seeds. The phenomenon is robust.

## 5. Discussion

### 5.1 Why Same-Class Statistics Are Worse Than Wrong-Class

The most counterintuitive finding is that same-class BN statistics (1.8%) perform *worse* than wrong-class (55.5%) and even random-class (51.4%) statistics. We propose the following explanation:

Class-conditional statistics differ from global statistics in a *systematic, class-correlated* way. When the network processes class-c samples with class-c BN statistics, the normalization shifts activations in a direction that is *anti-correlated* with what the classification head expects. The classification head was trained on globally-normalized activations and has learned decision boundaries calibrated to those specific activation magnitudes.

Class-conditional statistics normalize away the very class-specific activation patterns that the classification head uses for discrimination, effectively "centering" each class onto itself and removing the between-class signal that the final linear layer exploits.

Wrong-class statistics, by contrast, introduce a mismatch that may preserve or even amplify some between-class signal (analogously to how an adversarial perturbation can accidentally improve some predictions).

### 5.2 BN as a Calibration Layer

Our results support a clear functional decomposition:

- **Convolutional filters** learn class-discriminative features that are robust to normalization perturbation (as evidenced by 99.97% linear probe accuracy).
- **BN running statistics** calibrate the scale and shift of these features to match the classification head's learned decision boundaries.
- **The classification head** is a thin linear layer that is *highly sensitive* to the specific calibration provided by global BN statistics.

This decomposition explains why representations improve under same-class statistics: class-conditional normalization makes features *more* separable (each class is normalized to its own reference frame), but the original classification head cannot exploit this because it was calibrated for globally-normalized inputs.

### 5.3 Implications for Transfer Learning

In transfer learning, a common practice is to freeze pretrained convolutional weights and either (a) reset BN running statistics on the new domain or (b) fine-tune only the classification head. Our results suggest that option (a) works precisely because BN statistics are calibrators: resetting them on the target domain recalibrates the activation scale for the new data distribution, allowing even a frozen classification head (or a newly trained one) to function correctly.

This also explains the success of AdaBN (Li et al., 2018): replacing BN statistics with target-domain statistics is effective not because it encodes target-domain "features," but because it recalibrates activation magnitudes to a consistent scale that the classification head can interpret.

### 5.4 Implications for BN-Free Architectures

Recent architectures (e.g., NFNets; Brock et al., 2021) replace BN with alternative normalization or scaling strategies. Our findings suggest that the critical function being replaced is *activation calibration* rather than feature modulation. This may guide the design of simpler alternatives that focus specifically on maintaining consistent activation scales without the baggage of running statistics.

### 5.5 Implications for Model Compression and Quantization

BN running statistics are often folded into convolutional weights during deployment (fusing BN). Our results highlight that this fusion is not merely an optimization—it bakes a specific calibration into the weights. Any post-hoc modification to these fused weights must preserve the calibration, not just the feature extraction capability.

## 6. Limitations

Several limitations constrain the generalizability of our findings:

1. **Single architecture.** We test only a compact ResNet (32-64-128-256 channels, 20 BN layers). Larger architectures, different depths, or architecturally distinct models (e.g., VGG, DenseNet, EfficientNet) may behave differently.

2. **Single dataset.** CIFAR-10 has only 10 classes with 32×32 images. Datasets with more classes (CIFAR-100, ImageNet), higher resolution, or different characteristics (fine-grained recognition, medical imaging) may reveal different dynamics.

3. **Limited seeds.** We use 5 seeds rather than the more conservative 10+. While all comparisons are highly significant (p < 10⁻⁴), larger seed counts would strengthen confidence in effect sizes and variance estimates.

4. **CPU/MPS-only training.** Models were trained on Apple Silicon (MPS backend) for 25 epochs. While convergence appears adequate (91.9% test accuracy), longer training or GPU-accelerated training might yield subtly different BN statistics.

5. **Simple linear probe.** Our linear probe uses ridge regression rather than a trained neural network probe. While this is standard, it tests only linear separability and may miss nonlinear structure changes.

6. **Static replacement.** We replace BN statistics wholesale rather than interpolating between global and class-conditional statistics, which might reveal a more nuanced transition.

## 7. Conclusion

We present a simple experiment with a surprising result: replacing batch normalization running statistics with class-conditional statistics—computed from the *correct* class—destroys classification accuracy (1.8% vs. 91.9% baseline) while *improving* representation quality to near-perfection (99.97% linear probe accuracy). This dissociation reveals that BN running statistics function as output calibrators rather than feature encoders. The network's convolutional representations are robust to BN statistic perturbation, but the classification head is precisely calibrated to the activation scale produced by global statistics.

This finding reframes how we should think about batch normalization's role at inference time: not as a mechanism that encodes distributional knowledge into features, but as a calibration layer that aligns feature magnitudes with downstream expectations. This calibration perspective has practical implications for transfer learning, domain adaptation, model compression, and the design of normalization-free architectures.

The most important lesson may be methodological: classification accuracy alone can be deeply misleading about representation quality. A model that appears to "know nothing" (1.8% accuracy) may in fact have learned *everything*—it simply cannot express that knowledge through a miscalibrated output layer.

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
| Architecture | SmallResNet (BasicBlock, [2,2,2,2]) |
| Channel widths | [32, 64, 128, 256] |
| BN layers | 20 (BatchNorm2d) |
| Dataset | CIFAR-10 |
| Training epochs | 25 |
| Batch size | 128 |
| Optimizer | SGD (momentum=0.9, weight_decay=5×10⁻⁴) |
| Learning rate | 0.1, cosine annealing |
| Seeds | 42, 43, 44, 45, 46 |
| Augmentation | RandomCrop(32, pad=4), RandomHorizontalFlip |
| Normalization | μ=(0.4914, 0.4822, 0.4465), σ=(0.2470, 0.2435, 0.2616) |
| Linear probe | Ridge regression, λ=10⁻³, 80/20 split |
| Device | Apple Silicon (MPS) |
| Total compute | ~82 minutes (5 seeds × ~16 min each) |

## Appendix B: Per-Seed Results

**Table B1: Per-seed accuracy under each condition**

| Seed | Global | Same-Class | Wrong-Class | Random-Class |
|------|--------|------------|-------------|--------------|
| 42 | 91.95% | 0.75% | 61.86% | 62.84% |
| 43 | 91.55% | 1.93% | 47.79% | 49.84% |
| 44 | 92.16% | 1.90% | 61.41% | 44.46% |
| 45 | 91.88% | 2.92% | 46.79% | 44.25% |
| 46 | 91.84% | 1.45% | 59.74% | 55.56% |

**Table B2: Per-seed linear probe accuracy under each condition**

| Seed | Global | Same-Class | Wrong-Class | Random-Class |
|------|--------|------------|-------------|--------------|
| 42 | 91.55% | 100.00% | 100.00% | 96.90% |
| 43 | 92.05% | 99.95% | 100.00% | 96.20% |
| 44 | 91.95% | 99.95% | 100.00% | 97.75% |
| 45 | 91.25% | 100.00% | 100.00% | 95.90% |
| 46 | 91.40% | 99.95% | 100.00% | 98.40% |
