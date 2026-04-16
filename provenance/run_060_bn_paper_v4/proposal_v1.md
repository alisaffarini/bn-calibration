## Research Proposal: "Batch Normalization Statistics as Implicit Calibrators: Dissecting the Feature-Confidence Coupling in Deep Networks"

### (1) EXACT NOVELTY CLAIM
We demonstrate for the first time that batch normalization statistics serve as implicit calibration parameters that are **orthogonal to feature discriminability**: corrupting BN statistics with same-class replacements catastrophically degrades model accuracy (1.8% on CIFAR-10) while preserving near-perfect linear separability (99.97%) of pre-BN features. This reveals that BN layers create a previously unknown feature-confidence decoupling that fundamentally challenges our understanding of how deep networks make predictions.

### (2) CLOSEST PRIOR WORK
- **Mehrbod et al. (2025)** "Test-Time Adaptation via Adaptive Quantile-based Feature Recalibration" - Uses BN statistics updates for test-time adaptation but doesn't investigate BN's role in calibration or the accuracy-probe disconnect
- **Golechha & Dao (2024)** "Challenges in Interpretability of Neural Representations for Images" - Discusses representation interpretability but doesn't examine BN layers as interpretability bottlenecks
- **Ioffe & Szegedy (2015)** "Batch Normalization: Accelerating Deep Network Training" - Original BN paper focuses on training dynamics, not calibration or feature-confidence decoupling

### (3) EXPECTED CONTRIBUTION
This work would contribute:
1. **New Scientific Understanding**: First evidence that BN statistics control confidence calibration independently of feature quality
2. **New Diagnostic Tool**: BN replacement as a method to diagnose feature collapse vs. calibration issues
3. **New Calibration Method**: Post-hoc BN statistics adjustment for temperature-free calibration
4. **Theoretical Insight**: Explains why BN helps generalization beyond its training stabilization benefits

### (4) HYPOTHESIS
**Primary Hypothesis**: Batch normalization statistics (running mean/variance) encode class-agnostic calibration information that modulates network confidence independently of feature discriminability, creating a two-stage prediction process: (1) discriminative feature extraction, (2) BN-mediated confidence calibration.

**Testable Predictions**:
- H1: Same-class BN replacement will degrade accuracy to <5% while maintaining >95% linear probe accuracy across all architectures and datasets
- H2: Cross-class BN replacement will show intermediate effects based on class similarity
- H3: Temperature scaling applied post-BN will recover significantly less accuracy than pre-BN temperature scaling
- H4: ECE will increase by >0.3 after BN corruption despite preserved feature quality

### (5) EXPERIMENTAL PLAN

**Datasets**: CIFAR-10, CIFAR-100, Tiny-ImageNet-200

**Models**: ResNet-20, VGG-16-BN, MobileNetV2, DenseNet-40

**Core Experiments** (10 seeds each):

1. **Baseline Characterization**:
   - Train all models normally, save BN statistics per batch
   - Measure: accuracy, ECE, per-class ECE, feature norms pre/post-BN

2. **BN Replacement Study**:
   - Same-class replacement (random batch → random batch, same class)
   - Cross-class replacement (systematic: similar classes → similar, dissimilar → dissimilar based on confusion matrix)
   - Measure: accuracy, linear probe accuracy (frozen pre-BN features), ECE

3. **Calibration Recovery**:
   - Temperature scaling on logits (baseline)
   - Temperature scaling on pre-BN features
   - BN statistics interpolation: α*original + (1-α)*corrupted
   - Measure: How much accuracy/ECE recovers with each method

4. **Statistical Analysis**:
   - Paired t-tests between conditions
   - Bootstrap confidence intervals (1000 samples)
   - Effect sizes (Cohen's d)

5. **Feature Analysis**:
   - t-SNE of features pre/post-BN
   - Centered Kernel Alignment between layer representations
   - Class separation metrics (Fisher discriminant ratio)

**Compute Estimate**: ~3.5 hours on single GPU
- Training: 4 models × 3 datasets × 10 seeds × 2 min = 240 min
- BN replacement experiments: 20 min
- Calibration experiments: 20 min
- Analysis: 10 min

**Success Metrics**:
- Reproduce >95% accuracy drop with >95% probe retention across all settings
- Show ECE increase >0.3 with statistical significance (p<0.001)
- Demonstrate BN interpolation recovers >50% of accuracy loss
- Temperature scaling on logits recovers <20% of accuracy loss

This investigation would fundamentally change how we understand BN's role in deep learning - not just as a training stabilizer, but as a critical calibration mechanism that decouples feature extraction from confidence estimation.