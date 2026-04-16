## Research Proposal: "BatchNorm Statistics Encode Hierarchical Semantic Structure: A Gradient-Based Discovery Method"

### (1) EXACT NOVELTY CLAIM
We demonstrate for the first time that BatchNorm statistics encode a **hierarchical semantic structure** across network depth, where early layers require global statistics for low-level feature detection while later layers can operate with class-conditional statistics. We introduce a novel **gradient-based importance score** that identifies critical BN layers by measuring ∂Accuracy/∂α where α interpolates between global and class-conditional statistics. No prior work has: (a) shown that BN statistics themselves encode semantic information hierarchically, or (b) used gradient-based interpolation analysis to discover which normalization layers are critical for maintaining representations.

### (2) CLOSEST PRIOR WORK
1. **"Towards Automated Circuit Discovery for Mechanistic Interpretability" (Conmy et al., 2023)**: Uses activation patching and edge ablation to find important components in transformers. **Key difference**: They trace forward passes through attention/MLP blocks; we exploit the unique mathematical structure of BN statistics interpolation to identify critical layers via gradients, revealing a previously unknown hierarchical encoding in normalization layers.

2. **"Attribution-based Parameter Decomposition" (Braun et al., 2025)**: Decomposes parameters to minimize description length of mechanisms. **Key difference**: They focus on parameter decomposition for general interpretability; we specifically show that BN *statistics* (not parameters) encode semantic structure and can be interpolated to reveal layer importance.

3. **"Reclaiming Residual Knowledge: A Novel Interpretability-Driven Transfer Learning Approach" (Song et al., 2025)**: Uses SAEs for interpretable features in transfer learning. **Key difference**: They extract features via auxiliary models; we show that existing BN statistics *already* encode interpretable semantic structure without additional components.

### (3) EXPECTED CONTRIBUTION
This work would contribute:
- **New Finding**: First demonstration that normalization statistics encode hierarchical semantic information (early layers: texture/edges need global stats; late layers: objects/semantics can use class-conditional)
- **New Method**: Gradient-based importance scoring via statistics interpolation - a novel tool for discovering critical normalization layers
- **New Understanding**: Explains *why* BatchNorm helps CNNs - not just statistical benefits but semantic calibration at appropriate abstraction levels
- **Practical Impact**: Suggests new architectures could leverage this hierarchy (e.g., global BN early, class-conditional late)

### (4) HYPOTHESIS
**Primary Hypothesis**: BatchNorm layers in CNNs exhibit a monotonic importance gradient from early to late layers when measured by ∂Accuracy/∂α (sensitivity to replacing global with class-conditional statistics), with early layers showing |∂Accuracy/∂α| > 0.5 and late layers showing |∂Accuracy/∂α| < 0.1, reflecting a transition from low-level feature detection to high-level semantic processing.

**Secondary Hypothesis**: This hierarchical structure is architecture-invariant and will hold across ResNet, VGG, DenseNet, and EfficientNet architectures.

### (5) EXPERIMENTAL PLAN

**Setup**:
- Models: ResNet-18, VGG-16, DenseNet-121, EfficientNet-B0 (all pretrained on ImageNet, fine-tuned on CIFAR-10/100)
- Datasets: CIFAR-10, CIFAR-100 
- Seeds: 10 random seeds per experiment

**Main Experiments**:

1. **Layer-wise Importance Discovery** (2 hours):
   - For each BN layer i, compute importance score: I_i = |∂Accuracy/∂α_i| where α_i ∈ [0,1] interpolates between global (α=0) and class-conditional (α=1) stats
   - Use finite differences: I_i ≈ |Acc(α=0.1) - Acc(α=0.0)| / 0.1
   - Plot importance vs layer depth, test monotonicity with Spearman correlation

2. **Semantic Hierarchy Validation** (1 hour):
   - Create "hybrid" networks: use α=0 (global) for first k layers, α=1 (class-conditional) for remaining
   - Sweep k from 0 to total_layers, measure accuracy
   - Hypothesis: accuracy should increase monotonically with k, plateauing when semantic layers are reached

3. **Cross-Architecture Universality** (1 hour):
   - Repeat experiments 1-2 across all architectures
   - Test if importance curves have similar shape via Pearson correlation of importance vectors
   - Quantify: does the "transition depth" (where importance < 0.1) occur at similar relative positions?

**Ablations**:
- Compare with random statistics (baseline) vs class-conditional
- Test with shuffled class assignments to verify semantic nature
- Measure correlation between importance scores and established layer-wise metrics (e.g., CKA similarity to input)

**Metrics**:
- Primary: Layer importance scores I_i, Spearman ρ for monotonicity
- Secondary: Accuracy curves for hybrid networks, cross-architecture correlation
- Statistical: Mean ± std over 10 seeds, paired t-tests for significance

**Expected Results**:
- Early layers (conv1, conv2) show importance > 0.4
- Late layers (final blocks) show importance < 0.1  
- Transition occurs around 60-70% network depth
- Pattern holds with ρ > 0.8 correlation across architectures

This discovers a fundamental principle of how CNNs organize features hierarchically through their normalization statistics, providing new insights into why BatchNorm is so effective and suggesting new architectural designs.