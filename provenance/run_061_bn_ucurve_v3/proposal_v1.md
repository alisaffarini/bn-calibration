## Research Proposal: Layer-Specific Bias-Variance Trade-offs in Batch Normalization Reveal Non-Monotonic Calibration Optima

### (1) EXACT NOVELTY CLAIM
**First demonstration that interpolating between global and class-conditional batch normalization statistics produces a non-monotonic (U-shaped) calibration curve, with optimal calibration at intermediate alpha values (α ≈ 0.3-0.5) rather than at either extreme.** No prior work has shown that partial class-conditioning in normalization layers can improve calibration over both fully global (α=0) and fully class-conditional (α=1) statistics.

### (2) CLOSEST PRIOR WORK
- **"Towards Understanding Dual BN in Adversarial Training" (2024, Paper 21)**: Uses separate BN statistics for adversarial vs clean samples but only considers binary switching, not continuous interpolation. They don't measure calibration effects.
- **"Looking Beyond the Final Layer: Investigating Multilingual Calibration" (2025, Paper 13)**: Studies calibration across languages in LLMs but doesn't examine normalization layers or vision models.
- **"Does Calibration Emerge During Training? Insights from Layer-wise Analysis" (2024, Paper 14)**: Analyzes calibration evolution across layers but doesn't investigate normalization statistics or their interpolation.

**Key difference**: We show that BN statistics interpolation creates a previously unknown bias-variance trade-off measurable through calibration, with layer-specific optimal interpolation points.

### (3) EXPECTED CONTRIBUTION
This work would contribute **a new understanding of normalization layers as calibration mechanisms**, showing that:
1. BN layers implicitly learn a bias-variance trade-off between overfitting to class statistics (high variance) and underfitting with global statistics (high bias)
2. Optimal calibration requires layer-specific interpolation coefficients, not uniform treatment
3. This provides a new lens for understanding why BN improves generalization beyond its training stability benefits

### (4) HYPOTHESIS
**Primary hypothesis**: Interpolating BN statistics with α ∈ [0,1] produces a U-shaped ECE curve where ECE(α=0.3-0.5) < min(ECE(α=0), ECE(α=1)), and this optimal α varies monotonically with layer depth (earlier layers prefer smaller α, later layers larger α).

**Secondary hypothesis**: GroupNorm and LayerNorm, lacking running statistics, cannot exhibit this non-monotonic calibration behavior, serving as negative controls.

### (5) EXPERIMENTAL PLAN

**Models & Datasets**:
- ResNet-20 on CIFAR-10 and CIFAR-100
- 10 random seeds per configuration
- Temperature scaling applied post-hoc for fair comparison

**Core Experiment**:
1. Train standard ResNet-20 models with BN, GroupNorm, and LayerNorm
2. For BN models at test time:
   - For each BN layer independently: sweep α ∈ {0, 0.1, ..., 1.0}
   - Compute interpolated stats: μ_interp = (1-α)×μ_global + α×μ_class
   - Measure accuracy and ECE at each α configuration
3. Identify layer-specific optimal α values that minimize ECE
4. Plot ECE vs α curves for each layer, test U-shape significance

**Key Measurements**:
- ECE and accuracy at each α for each layer
- Optimal α* per layer (minimizing ECE)
- Statistical test for non-monotonicity: ECE(α*) < ECE(0) AND ECE(α*) < ECE(1) with p<0.05
- Correlation between layer depth and optimal α

**Ablations**:
- Effect of number of calibration samples (1k, 5k, 10k)
- Comparison with post-hoc temperature scaling
- Analysis of per-class calibration differences

**Compute Requirements**: ~3.5 hours
- Training: 3 models × 2 datasets × 10 seeds × 3 min = 3 hours  
- Evaluation: 60 models × 11 α values × 20 layers × 0.5s = 0.5 hours

This experiment would establish normalization layers as active calibration mechanisms and reveal a fundamental bias-variance trade-off in their operation, opening new research directions in both interpretability and architectural design.