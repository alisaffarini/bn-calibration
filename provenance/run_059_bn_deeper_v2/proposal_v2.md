## Revised Research Proposal: BatchNorm Statistics Interpolation: Revealing Implicit Class Boundaries and Calibration Limits

### (1) EXACT NOVELTY CLAIM
**This is the first work to systematically study the failure modes of interpolating between global and class-conditional BatchNorm statistics, revealing that BN layers encode implicit decision boundaries that become incompatible under linear mixing.** Unlike MixStyle (Zhou et al., ICLR 2021) which interpolates instance stats for augmentation, and AdaBN (Li et al., IJCAI 2018) which switches between source/target stats, we show that **continuous interpolation between semantic groupings (class-conditional vs global) exposes fundamental incompatibilities in learned calibration spaces**.

### (2) CLOSEST PRIOR WORK
- **MixStyle (Zhou et al., ICLR 2021)**: Interpolates instance normalization statistics for domain generalization. **Key difference**: They mix random instance pairs for augmentation; we systematically interpolate between semantic groupings to understand calibration limits.
- **AdaBN (Li et al., IJCAI 2018)**: Adapts BN statistics for domain adaptation by replacing source with target stats. **Key difference**: Binary replacement vs our continuous interpolation revealing intermediate failure modes.
- **Test-Time Normalization (Schneider et al., NeurIPS 2020)**: Updates BN stats at test time. **Key difference**: They adapt to distribution shift; we analyze what happens between known distributions.

### (3) EXPECTED CONTRIBUTION
1. **Mechanistic insight**: Evidence that BN statistics encode implicit class boundaries that are **geometrically incompatible** - interpolating between them creates invalid calibration regions
2. **Practical finding**: Identification of "safe" vs "unsafe" adaptation strategies for test-time adaptation methods
3. **New benchmark**: Systematic evaluation protocol for normalization robustness
4. **Actionable outcome**: Guidelines for when BN adaptation methods (AdaBN, Tent, etc.) will likely fail

### (4) HYPOTHESIS & MECHANISM

**Proposed Mechanism**: Class-conditional BN statistics create distinct "calibration manifolds" in activation space. When networks learn with global stats, they learn one manifold; with class stats, they learn multiple disjoint manifolds. Linear interpolation creates points that lie **outside all learned manifolds**, causing miscalibration.

**H1 (Primary)**: Interpolating between global and class-conditional BN statistics (`α×class_stats + (1-α)×global_stats`) will show **non-monotonic** accuracy degradation, with worst performance at intermediate α values (0.3-0.7), not at the extremes.

**H2 (Mechanistic)**: The accuracy degradation correlates with the **Wasserstein distance** between class-conditional statistic distributions - classes with more separated statistics will show sharper degradation.

**H3 (Control)**: Interpolating between global stats and **random** statistics will show monotonic degradation, confirming that the non-monotonic behavior is specific to class structure.

### (5) EXPERIMENTAL PLAN

**Setup**:
- Pretrained ResNet-50 (ImageNet)
- Validation set: 50k images
- Compute: <4 hours on single GPU

**Experiment 1: Core Interpolation Study** (1.5 hours)
```python
# For each BN layer:
for alpha in np.linspace(0, 1, 50):
    # Three conditions:
    stats_class = alpha * class_conditional + (1-alpha) * global_stats
    stats_random = alpha * random_stats + (1-alpha) * global_stats  # CONTROL
    stats_shuffled = alpha * shuffled_class_stats + (1-alpha) * global_stats
```
- Measure: Top-1 accuracy, per-class accuracy variance, calibration error (ECE)
- Key comparison: Non-monotonic (class) vs monotonic (random) degradation

**Experiment 2: Mechanistic Analysis** (1 hour)
1. Compute pairwise Wasserstein distance between class BN distributions
2. Group classes by distance: similar vs dissimilar statistics
3. Test interpolation within similar vs dissimilar groups
4. Correlate distance with interpolation failure severity

**Experiment 3: Layer-wise Analysis** (1 hour)
- Repeat interpolation per layer independently
- Identify which layers show strongest non-monotonic behavior
- Correlate with layer depth and feature abstraction level

**Experiment 4: Practical Impact** (30 mins)
- Simulate test-time adaptation scenario:
  - Start with global stats model
  - Partially adapt toward target distribution stats
  - Show when adaptation helps vs hurts

**Key Metrics**:
- Accuracy curves (expecting ∩-shape for class condition)
- Wasserstein distance vs degradation correlation
- Layer-wise sensitivity ranking
- Calibration error (ECE) throughout interpolation

**Statistical Validation**:
- Bootstrap CIs over 5 seeds
- Permutation test for non-monotonicity
- Correlation significance tests

### ADDRESSING REVIEWER CONCERNS:

1. **"Phase transition" removed** - Now framed as non-monotonic degradation with mechanistic explanation
2. **Included all missing work** - MixStyle, AdaBN, test-time normalization
3. **Added mechanism** - Calibration manifold incompatibility explains the failure
4. **Random baseline included** - Tests specificity of the effect
5. **Practical connection** - Direct implications for when BN adaptation methods fail
6. **Grounded hypotheses** - Based on geometric view of BN statistics as defining calibration spaces

This isn't "let's try something and see" - it's testing a specific mechanistic hypothesis about how BN statistics define incompatible calibration spaces, with direct implications for domain adaptation methods.