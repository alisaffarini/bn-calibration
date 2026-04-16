## Literature Summary

Based on the search results, the interpretability landscape shows several key trends:

1. **Calibration Studies in Deep Networks**: Recent work has explored calibration properties, particularly in LLMs. Joshi et al. (2025) studied how calibration evolves across transformer layers, finding that calibration properties emerge gradually through network depth. However, this work focused on transformers, not CNNs with batch normalization.

2. **Batch Normalization Research**: Zhang et al. (2024) investigated Dual BN in adversarial training, where separate BN statistics are maintained for clean and adversarial samples. They assume these form two distinct domains but don't explore intermediate mixtures of statistics.

3. **Mechanistic Interpretability Tools**: Multiple frameworks have emerged (nnterp by Dumas 2025, Prisma by Joseph et al. 2025) for analyzing neural network internals, but these focus primarily on transformers and vision models, not normalization layers specifically.

4. **Trade-off Analysis**: Several papers examine various trade-offs (Samad et al. 2025 on fairness-accuracy, Yu et al. 2024 on accuracy-communication in federated learning) but none investigate calibration trade-offs in normalization layers.

## Identified Gaps

1. **No exploration of intermediate BN statistics**: While Zhang et al. (2024) use separate statistics for different domains, nobody has studied what happens with weighted combinations of global and class-conditional statistics.

2. **Missing layer-wise calibration analysis for CNNs**: Joshi et al. (2025) analyzed calibration across transformer layers, but similar analysis for CNN architectures with BN layers remains unexplored.

3. **Lack of normalization comparison for calibration**: No systematic comparison exists between BN, GroupNorm, and LayerNorm regarding their calibration properties.

4. **No connection to bias-variance trade-offs**: The interpretability literature hasn't connected normalization statistics to measurable bias-variance trade-offs.

## Recommended Research Directions

### 1. **Discover Layer-Specific Calibration Signatures in BN Statistics**
**Gap**: While Joshi et al. (2025) showed calibration evolves across transformer layers, no work examines whether different CNN layers learn different optimal mixtures of global vs class-conditional statistics for calibration.

**Concrete Experiment**: For each BN layer in ResNet-20, sweep α∈[0,1] and measure both accuracy and ECE. Plot calibration curves per layer depth. Test hypothesis: early layers prefer global statistics (α→0) for feature extraction, while later layers prefer class-conditional (α→1) for discrimination.

**Why Novel**: Extends single-domain BN analysis (Zhang et al. 2024) to continuous interpolation, revealing previously hidden non-monotonic behaviors.

### 2. **Establish GroupNorm as Calibration-Preserving Alternative to BN**
**Gap**: Despite mechanistic interpretability frameworks studying various architectures (Dumas 2025, Joseph et al. 2025), no work compares normalization methods' impact on calibration.

**Concrete Experiment**: Train identical ResNet-20 architectures with BN, GroupNorm, and LayerNorm. For BN variant, implement α-interpolation. Measure if GroupNorm naturally achieves calibration similar to optimal-α BN without requiring statistics tuning.

**Why Novel**: Could reveal that BN's calibration benefits come with complexity costs that simpler normalizations avoid.

### 3. **Develop Interpretable Calibration Probes for Normalization Layers**
**Gap**: Current interpretability tools (nnterp, Prisma) focus on attention/features but lack specialized probes for understanding normalization's role in calibration.

**Concrete Experiment**: Design probing classifiers that predict ECE from BN statistics (running mean/var) at different α values. Use SHAP values to interpret which statistics drive calibration. Validate on CIFAR-10/100.

**Why Novel**: Bridges mechanistic interpretability with calibration analysis, providing actionable insights about which BN statistics matter for uncertainty estimation.

### 4. **Demonstrate Task-Complexity Dependence of Optimal α**
**Gap**: Existing calibration work doesn't consider how task complexity affects optimal normalization strategies.

**Concrete Experiment**: Compare optimal α values between CIFAR-10 (10 classes) and CIFAR-100 (100 classes) for each layer. Test hypothesis: higher task complexity requires more class-conditional statistics (higher optimal α) in deeper layers.

**Why Novel**: Unlike fixed normalization schemes in current practice, suggests adaptive normalization based on task properties.

### 5. **Create Minimal Synthetic Tasks Exhibiting U-Curve Calibration**
**Gap**: No controlled studies isolate the conditions causing non-monotonic calibration in normalization.

**Concrete Experiment**: Design synthetic 2D classification tasks with controlled class overlap. Implement single-layer networks with BN. Systematically vary data properties (class separation, noise) to identify when U-curves emerge vs monotonic calibration.

**Why Novel**: Provides theoretical grounding for empirical U-curve findings, enabling principled normalization design.