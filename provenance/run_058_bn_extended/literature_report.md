## Literature Summary

The search results reveal that current interpretability research is heavily focused on:

1. **Mechanistic interpretability of transformers** - Papers like "Towards Automated Circuit Discovery" (Conmy et al., 2023) and "A Practical Review of Mechanistic Interpretability" (Rai et al., 2024) focus on reverse-engineering transformer internals, with most work targeting language models.

2. **Sparse autoencoders (SAEs)** for feature extraction - Multiple papers (Song et al. 2025, Erdogan & Lucic 2025) use SAEs to decompose neural activations into interpretable features, but primarily for transformers.

3. **Parameter decomposition methods** - Braun et al. (2025) introduces Attribution-based Parameter Decomposition, but focuses on minimizing mechanistic description length rather than understanding normalization layers.

4. **Domain-specific interpretability** - Papers explore interpretability in specific domains like graphs (El et al., 2025), TSP solvers (Narad et al., 2025), and biological systems (Tazza et al., 2025).

Notably, **no papers investigate BatchNorm statistics as semantic feature extractors or calibrators**, and the only BN-related paper (Xu et al., 2025) focuses on spiking neural networks in RL contexts.

## Identified Gaps

1. **No exploration of normalization layers as interpretable components** - While extensive work exists on attention mechanisms and SAEs, normalization layers (BatchNorm, LayerNorm, GroupNorm) remain unexplored as sources of interpretable features.

2. **Missing connection between statistical moments and semantic information** - No papers investigate how running statistics in BatchNorm encode class-level or semantic information.

3. **Lack of cross-architecture interpretability studies** - Most mechanistic interpretability focuses on transformers; systematic studies across CNN architectures (VGG, ResNet, DenseNet) are missing.

4. **No work on statistics interpolation for interpretability** - The idea of interpolating between global and class-conditional statistics to understand feature representations is unexplored.

## Recommended Research Directions

### 1. **BatchNorm Statistics as Universal Feature Calibrators Across Normalization Methods**
**Gap**: While your initial finding shows BN statistics can act as calibrators, no work compares this phenomenon across different normalization techniques.
**Novel Direction**: Test whether Layer Normalization, Group Normalization, and Instance Normalization exhibit similar calibration properties when their statistics are made class-conditional. This addresses what Erdogan & Lucic (2025) missed by focusing only on SAEs without considering how normalization itself encodes features.
**Why Novel**: No existing paper examines normalization statistics as a general principle for feature extraction across different normalization schemes.

### 2. **Gradient-Based Discovery of Critical BatchNorm Layers via Statistics Perturbation**
**Gap**: Current layer-wise analysis methods (e.g., activation patching in Conmy et al., 2023) don't leverage the unique properties of BN statistics.
**Novel Direction**: Develop a gradient-based importance score that identifies which BN layers are most critical by measuring the gradient of accuracy w.r.t. interpolation parameter α between global and class-conditional stats. This could reveal a hierarchy of semantic encoding across network depth.
**Why Novel**: Unlike existing circuit discovery methods that trace forward passes, this leverages the unique mathematical structure of BN to identify critical layers through statistics perturbation.

### 3. **BatchNorm Statistics Entropy as a Measure of Feature Specialization**
**Gap**: While Song et al. (2025) discuss feature consistency in SAEs, no work quantifies how specialized vs. distributed features are in normalization layers.
**Novel Direction**: Compute the entropy of class-conditional BN statistics across layers and correlate with: (a) layer depth, (b) architecture type, (c) dataset complexity (CIFAR-10 vs CIFAR-100). Low entropy indicates specialized features, high entropy indicates distributed representations.
**Why Novel**: Provides a quantitative framework for understanding feature specialization without requiring expensive SAE training, filling the gap left by current interpretability methods.

### 4. **Zero-Shot Domain Adaptation via BatchNorm Statistics Transfer**
**Gap**: Current interpretability work doesn't explore practical applications of understanding BN statistics.
**Novel Direction**: If BN stats encode semantic calibration, test whether transferring class-conditional statistics from a source domain (e.g., CIFAR-10) to a target domain (e.g., SVHN) improves zero-shot classification. Compare against standard BN adaptation methods.
**Why Novel**: Transforms your interpretability finding into a practical method, demonstrating that understanding BN statistics has direct applications beyond mechanistic insights.

### 5. **Temporal Dynamics of BatchNorm Statistics During Training**
**Gap**: All current interpretability work analyzes fixed, pretrained models. No work studies how interpretable features emerge during training.
**Novel Direction**: Track the evolution of class-conditional vs. global statistics similarity throughout training. Identify phase transitions where BN statistics become semantic calibrators. Correlate with standard training phenomena (e.g., critical periods, grokking).
**Why Novel**: Bridges the gap between Staats et al. (2024)'s analysis of singular values in pretrained models and the dynamic emergence of interpretable features during training.

Each direction can be validated with <4 hours of compute by leveraging pretrained models and focusing on specific hypotheses rather than full training runs.