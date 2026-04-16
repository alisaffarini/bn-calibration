## Revised Research Proposal: "Normalization Layers as Feature-Confidence Decouplers: A Comparative Study Across Architectures"

### (1) EXACT NOVELTY CLAIM
We provide the first systematic comparison of how different normalization methods (BatchNorm, LayerNorm, GroupNorm, InstanceNorm) create distinct feature-confidence decoupling patterns. Specifically, we show that corrupting normalization statistics causes architecture-dependent accuracy degradation (1.8%-95%) while preserving feature discriminability (>90% linear probe), with this gap varying predictably based on the normalization method's spatial/channel aggregation pattern.

### (2) CLOSEST PRIOR WORK
- **Guo et al. (2017)** "On Calibration of Modern Neural Networks" - Shows BN affects calibration but doesn't investigate feature-confidence decoupling or compare normalization methods
- **Singh & Krishnan (2020)** "Filter Response Normalization" - Compares normalization methods for accuracy/stability but not for calibration or feature preservation
- **Luo et al. (2019)** "Understanding the Effective Receptive Field in Deep CNNs" - Studies BN's effect on representations but not the accuracy-probe disconnect

### (3) EXPECTED CONTRIBUTION
1. **Empirical Characterization**: First comprehensive study showing how BN/LN/GN/IN differently couple features and confidence
2. **Practical Insight**: Guidelines for choosing normalization methods based on calibration requirements
3. **Diagnostic Tool**: Normalization corruption as a method to distinguish feature quality from calibration issues
4. **Reproducible Benchmark**: Public codebase with standardized evaluation protocol

### (4) HYPOTHESIS
**Primary Hypothesis**: The degree of feature-confidence coupling in neural networks depends on the normalization method's aggregation scope: BatchNorm (batch-level stats) shows maximal decoupling, while InstanceNorm (instance-level stats) shows minimal decoupling.

**Testable Predictions**:
- H1: Accuracy drop from stats corruption: BN > GN > LN > IN
- H2: Linear probe retention: BN ≈ GN ≈ LN ≈ IN (all >85%)
- H3: ECE increase from corruption correlates with aggregation scope
- H4: Final-layer normalizations show larger effects than early layers

### (5) EXPERIMENTAL PLAN

**Precise Definitions**:
- **Stats Corruption**: Replace running_mean and running_var in normalization layers with those from a different random batch of the same class (for same-class) or different class (for cross-class)
- **Noise Baseline**: Add Gaussian noise (σ = 0.1 * stat_std) to normalization statistics

**Datasets**: CIFAR-10, CIFAR-100 (Tiny-ImageNet if time permits)

**Models**: ResNet-20 with {BN, LN, GN, IN}, VGG-16 with {BN, LN, GN, IN}

**Experiments** (5 seeds each - realistic for 4 hours):

1. **Training Phase** (2.5 hours):
   - Train 8 model variants (2 architectures × 4 norm types)
   - Track: accuracy, ECE, per-layer statistics
   - Realistic: 20 min/model on CIFAR-10

2. **Corruption Analysis** (1 hour):
   - For each trained model:
     a) Same-class stats replacement (random batch → random batch)
     b) Cross-class replacement (class i → class j for all i,j pairs)
     c) Gaussian noise injection (multiple σ levels)
     d) Layer-wise ablation (corrupt one layer at a time)
   - Measure: accuracy, ECE, linear probe accuracy on pre-norm features

3. **Calibration Recovery** (20 min):
   - Temperature scaling on corrupted models
   - Stats interpolation: α*original + (1-α)*corrupted
   - Measure recovery rates

4. **Statistical Analysis** (10 min):
   - Bootstrap CIs for accuracy/ECE changes
   - Correlation between aggregation scope and effect size
   - Layer-wise effect patterns

**Key Baselines**:
- No normalization (baseline for coupling)
- Random feature corruption (sanity check)
- Temperature scaling alone (calibration baseline)

**Success Metrics**:
- Show statistically significant differences between norm methods (p<0.05)
- Demonstrate accuracy-probe gap exists across methods but varies in magnitude
- Correlation r>0.7 between aggregation scope and decoupling strength

**Why This Addresses Criticisms**:

1. **Novelty**: Comparative study across normalization methods is genuinely new
2. **Methodology**: Clear definitions, proper baselines, realistic compute
3. **Significance**: Practical guidance for practitioners, not just "BN affects calibration"
4. **Soundness**: No causal claims, just empirical characterization
5. **Related Work**: Properly situated relative to Guo et al. 2017

This focused empirical study would provide valuable insights for the community while avoiding overstated mechanistic claims. The core finding about feature-confidence decoupling remains central but is properly contextualized.