## Research Proposal: Predicting Critical BatchNorm Layers via Statistical Divergence Analysis

### (1) EXACT NOVELTY CLAIM
No prior work has demonstrated that the criticality of BatchNorm layers for model performance can be predicted by the KL divergence between their global and class-conditional statistics, revealing that BN learns a layer-specific calibration strategy where statistical divergence indicates calibration importance. Unlike existing interpretability work that treats all normalization layers equally, this shows BN develops a hierarchical calibration structure measurable through statistical analysis.

### (2) CLOSEST PRIOR WORK
- **Erdogan & Lucic 2025** ("Group Equivariant Sparse Autoencoders"): Explores how symmetries improve feature learning but focuses on representation quality, not normalization layer importance
- **Joseph et al. 2025** ("Prisma: A Comprehensive Framework for Vision Model Interpretability"): Provides tools for vision model analysis but examines attention/features, not normalization statistics as calibration mechanisms  
- **Golechha & Dao 2024** (Paper 28): Highlights challenges in interpreting hidden representations but doesn't analyze normalization layers as distinct interpretability units

**Key difference**: These papers treat normalization as implementation detail. My proposal shows BN statistics divergence predicts layer importance, revealing an interpretable calibration hierarchy.

### (3) EXPECTED CONTRIBUTION
1. **New interpretability metric**: KL divergence between global/class-conditional BN stats as layer importance predictor
2. **Novel finding**: BN learns non-uniform calibration distribution - high-divergence layers are critical checkpoints
3. **Practical tool**: Identify which BN layers to preserve during model compression/modification without exhaustive search
4. **Theoretical insight**: First evidence that normalization layers self-organize into calibration hierarchy

### (4) HYPOTHESIS
**Primary**: The performance drop from replacing a BN layer with global statistics correlates positively (Pearson r > 0.7) with the KL divergence between that layer's global and class-conditional statistics.

**Secondary**: This correlation is stronger for BN than GroupNorm/LayerNorm, indicating batch statistics uniquely capture calibration information.

### (5) EXPERIMENTAL PLAN

**Setup (1 hour)**:
- Load pretrained ResNet-50 on ImageNet validation set (5000 images, 50 classes subset)
- Implement class-conditional BN statistics collection
- Implement KL divergence calculation between multivariate Gaussians

**Experiment 1: Divergence Calculation (1 hour)**:
- For each BN layer, compute:
  - Global statistics (mean, var) across all samples
  - Per-class statistics for each of 50 classes
  - Average KL divergence: mean over classes of KL(class_stats || global_stats)
- Output: Divergence score for each of ~50 BN layers

**Experiment 2: Layer Criticality Measurement (1.5 hours)**:
- For each BN layer independently:
  - Replace with fixed global statistics
  - Measure accuracy drop on validation set
  - Restore original layer
- Correlate accuracy drops with divergence scores
- Statistical test: Pearson correlation with p-value

**Experiment 3: Cross-Normalization Testing (0.5 hours)**:
- Repeat divergence-criticality analysis for:
  - GroupNorm model (if available pretrained)
  - Or convert 5 BN layers to GN, test correlation
- Compare correlation strength across normalization types

**Analysis**:
- Plot: X-axis = KL divergence, Y-axis = accuracy drop
- Identify outlier layers (high divergence but low criticality or vice versa)
- Visualize layer depth vs divergence to find patterns

**Key metrics**: Pearson correlation coefficient, p-value, R², layer-wise accuracy drops, divergence distributions

This directly extends the "BN as calibrator" finding by showing not just that BN calibrates, but that it learns a predictable, hierarchical calibration structure measurable through statistical divergence.