## Literature Summary

Based on the search results, the interpretability landscape shows several key trends:

**Calibration Research:**
- Paper 13 ("Beyond the Final Layer", 2025) examines multilingual calibration in LLMs, finding that non-English languages suffer from systematically worse calibration. They analyze intermediate representations but focus on language models, not vision models or normalization layers.
- Paper 14 ("Calibration Across Layers", 2024) investigates how calibration evolves throughout network depth, linking it to specific components like entropy neurons and unembedding matrices, but doesn't examine normalization layers specifically.

**Normalization Understanding:**
- Paper 21 ("Towards Understanding Dual BN", 2024) studies Batch Normalization in adversarial training, using separate BN statistics for adversarial vs clean samples. However, they don't investigate calibration or interpolation between statistics.

**General Interpretability Methods:**
- Papers 1-12 heavily focus on post-hoc interpretability methods (LIME, SHAP, Grad-CAM) for explaining predictions, but none examine the interpretability of normalization layers themselves or their calibration properties.

**Key Finding:** No papers explore the calibration properties of normalization layers or the trade-off between global and class-conditional statistics.

## Identified Gaps

1. **Normalization-Calibration Connection**: While Papers 13-14 study calibration across layers, they don't investigate how normalization methods (BN, LayerNorm, GroupNorm) specifically affect calibration. The role of running statistics in calibration is unexplored.

2. **Statistics Interpolation**: Paper 21 uses dual BN for different domains but doesn't explore continuous interpolation between statistics or its effect on calibration. The non-monotonic behavior you discovered in run_059 appears completely novel.

3. **Layer-Specific Normalization Behavior**: While Paper 14 shows calibration varies by layer depth, no work examines whether optimal normalization statistics should vary by layer position.

4. **Comparative Normalization Analysis**: No systematic comparison exists between BN, LayerNorm, and GroupNorm from a calibration perspective, especially regarding their statistical properties.

## Recommended Research Directions

### 1. **Layer-Adaptive Normalization for Optimal Calibration**
**Gap**: Paper 14 shows calibration varies across layers but doesn't connect this to normalization choices. Paper 21 uses fixed dual BN without exploring intermediate statistics.

**Novel Contribution**: Develop a layer-adaptive normalization that learns optimal alpha values per layer during training. Test hypothesis that early layers prefer global statistics (alpha→0) while later layers benefit from class-conditional (alpha→1).

**4-Hour Experiment**: 
- Modify ResNet-20 to have learnable alpha parameters per BN layer
- Train with joint loss: CE + λ*ECE
- Compare against fixed alpha=0.5 and your sweep results
- Measure if learned alphas match your empirical optimal values

### 2. **Normalization Statistics as Uncertainty Indicators**
**Gap**: Papers 1-12 use complex post-hoc methods for interpretability, but none examine BN statistics themselves as interpretable features. The variance between global and class-conditional stats could indicate model uncertainty.

**Novel Contribution**: Show that the divergence between global and class-conditional BN statistics correlates with prediction uncertainty and miscalibration.

**4-Hour Experiment**:
- For each test sample, compute KL divergence between stats at different alpha values
- Correlate this "normalization uncertainty" with: (a) prediction entropy, (b) correctness, (c) ECE contribution
- Test if high divergence predicts miscalibrated predictions
- Compare against standard uncertainty metrics

### 3. **Cross-Dataset Calibration Transfer via Normalization**
**Gap**: Paper 13 studies calibration across languages but not across visual domains. No work examines whether normalization statistics can improve calibration transfer.

**Novel Contribution**: Demonstrate that interpolating BN statistics improves calibration when transferring between datasets (CIFAR-10→CIFAR-100, or to SVHN).

**4-Hour Experiment**:
- Train on CIFAR-10, evaluate on CIFAR-100
- Test calibration with: (a) source stats, (b) target stats, (c) interpolated stats
- Find optimal interpolation weights for transfer
- Compare against temperature scaling and other calibration methods

### 4. **Theoretical Analysis of BN Calibration Trade-off**
**Gap**: While your empirical U-curve is novel, no theoretical framework explains why intermediate statistics improve calibration.

**Novel Contribution**: Provide theoretical analysis showing BN interpolation balances bias (from wrong statistics) and variance (from limited samples), creating optimal calibration at intermediate points.

**4-Hour Experiment**:
- Empirically verify bias-variance decomposition of calibration error
- Show global stats → high bias, class-conditional → high variance
- Measure these components across different dataset sizes
- Validate theoretical predictions against your U-curve results

### 5. **Dynamic Normalization for Test-Time Calibration**
**Gap**: Paper 15 proposes input-adaptive computation but doesn't consider normalization. No work explores test-time adjustment of normalization for calibration.

**Novel Contribution**: Develop test-time optimization of alpha based on prediction confidence, allowing dynamic calibration adjustment without retraining.

**4-Hour Experiment**:
- For each test batch, optimize alpha to minimize ECE on held-out calibration set
- Test on distribution-shifted data (corrupted CIFAR)
- Compare against test-time normalization adaptation methods
- Show this improves robustness while maintaining accuracy

Each direction builds directly on your U-curve discovery while addressing clear gaps in the literature. The experiments are designed to fit within computational constraints while producing publishable insights.