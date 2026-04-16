## Research Proposal: Discovery and Characterization of Critical Phase Transitions in BatchNorm Statistics Interpolation

### (1) EXACT NOVELTY CLAIM
**This is the first work to systematically interpolate between global and class-conditional BatchNorm statistics via continuous parameter α ∈ [0,1] and discover sharp phase transitions in model performance.** Specifically, we introduce the interpolation scheme: `running_stats = α × class_conditional_stats + (1-α) × global_stats` and show that there exist critical values α_c where accuracy catastrophically drops, revealing fundamental limits of normalization-based calibration.

### (2) CLOSEST PRIOR WORK
Based on the scout's analysis, the most related works are:
- **Erdogan & Lucic (2025)** "Incorporating Group Symmetries in Sparse Autoencoders" - Studies disentangled representations but focuses on feature learning, not normalization calibration
- **Prisma (Joseph et al. 2025)** - Provides vision model interpretability tools but analyzes features/activations, not normalization statistics
- **Mechanistic Neural Networks (Pervez et al. 2024)** - Examines learned representations but doesn't study normalization layers

**Key difference**: All prior interpretability work treats normalization as a fixed preprocessing step. We're the first to treat BN statistics as a continuous calibration mechanism and study the interpolation space between different statistical regimes.

### (3) EXPECTED CONTRIBUTION
1. **Novel empirical finding**: Discovery of sharp phase transitions at specific α values where model accuracy drops >50% within Δα=0.05
2. **New interpretability lens**: Framework for understanding normalization layers as continuous calibrators rather than binary on/off components
3. **Practical insight**: Identification of "safe" interpolation ranges for domain adaptation and continual learning applications
4. **Theoretical foundation**: Evidence that BN statistics encode implicit decision boundaries that can be continuously deformed until critical collapse

### (4) HYPOTHESIS
**H1 (Primary)**: There exists at least one critical value α_c ∈ (0,1) where interpolating between global and class-conditional BN statistics causes a sharp phase transition (>30% accuracy drop within Δα=0.1).

**H2 (Secondary)**: The phase transition occurs earlier (lower α_c) in deeper layers compared to shallow layers, indicating that deeper BN statistics are more sensitive calibrators.

**H3 (Tertiary)**: The sharpness of the phase transition correlates with the inter-class variance of the original class-conditional statistics.

### (5) EXPERIMENTAL PLAN

**Models & Data**:
- Pretrained ResNet-50 on ImageNet (available via torchvision)
- Test on ImageNet validation set (50k images)
- Optional: CIFAR-100 for faster iteration

**Core Experiment** (2 hours):
1. Extract and store class-conditional BN stats for all 1000 classes
2. Implement α-interpolation: For each BN layer, replace stats with interpolated version
3. Sweep α from 0→1 in steps of 0.02 (50 evaluations)
4. Plot accuracy vs α, identify phase transitions via gradient: |dAcc/dα| > threshold

**Layer-wise Analysis** (1 hour):
1. Repeat interpolation for each BN layer independently
2. Create heatmap: layers × α values → accuracy
3. Test H2 by comparing α_c across layers

**Statistical Validation** (30 min):
1. Bootstrap confidence intervals over 5 random seeds
2. Compute phase transition sharpness: max|dAcc/dα|
3. Correlate with inter-class statistics variance

**Ablations** (30 min):
1. Test with partial interpolation (only mean or only variance)
2. Compare random class assignments vs true labels
3. Test on out-of-distribution data (ImageNet-C)

**Key Metrics**:
- Top-1 accuracy vs α curves
- Phase transition location α_c and sharpness
- Layer-wise sensitivity rankings
- Statistical significance via bootstrap

**Computational Requirements**:
- 1 GPU (RTX 3090 or similar)
- ~50 forward passes through ResNet-50
- No training required (inference only)
- Total time: <4 hours

This proposal directly addresses the gap identified by the scout - no existing work on BN statistics interpolation - while building on run_057's finding that BN acts as a calibrator. The phase transition framing connects to established ML concepts while being entirely novel for normalization layers.