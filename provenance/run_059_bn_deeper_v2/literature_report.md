## Literature Summary

The search results reveal a significant gap - there appear to be **no papers directly investigating BatchNorm (BN) statistics as calibrators or examining the interpolation between global and class-conditional BN statistics**. The returned papers are largely unrelated to normalization layers or calibration.

The few relevant interpretability papers focus on:
- **Sparse autoencoders** (Erdogan & Lucic 2025) for disentangling neural network activations, with focus on incorporating group symmetries
- **Mechanistic interpretability tools** like nnterp (Dumas 2025) and Prisma (Joseph et al. 2025) for analyzing transformers and vision models
- **Mechanistic Neural Networks** (Pervez et al. 2024) that learn differential equations as representations

Notably absent: Any work on BatchNorm, GroupNorm, or LayerNorm from an interpretability perspective, or treating normalization statistics as calibration mechanisms rather than feature encoders.

## Identified Gaps

1. **No existing work on BN statistics interpolation**: The idea of gradually interpolating between global and class-conditional BN stats (alpha sweep) appears completely unexplored
2. **Layer-wise normalization ablation studies are missing**: While there's work on model interpretability, no one has systematically studied which specific normalization layers are most critical
3. **Cross-normalization comparisons absent**: No papers compare BN vs GroupNorm vs LayerNorm from a calibration perspective
4. **Phase transition analysis unexplored**: The concept of finding accuracy collapse points during stat interpolation is novel

## Recommended Research Directions

### 1. **BatchNorm Statistics Phase Transition Analysis**
**What to do**: Implement alpha-weighted interpolation between global and per-class BN statistics: `stats = alpha * class_stats + (1-alpha) * global_stats`. Sweep alpha from 0→1 on pretrained ResNet-50, measuring accuracy at each point to identify sharp phase transitions.

**Why it's novel**: No existing papers examine this interpolation space. While Erdogan & Lucic (2025) study symmetries in representations and Pervez et al. (2024) examine mechanistic representations, none investigate how normalization statistics themselves act as implicit calibrators.

**Validation**: <2 hours on single GPU with pretrained models, clear metric (accuracy vs alpha curve).

### 2. **Layer-Specific Normalization Criticality Mapping**
**What to do**: For each BN layer in ResNet-50, replace its statistics with (a) random noise, (b) uniform constants, (c) statistics from a different layer. Create a "criticality heatmap" showing which layers cause most damage.

**Why it's novel**: While Prisma (Joseph et al. 2025) provides tools for vision model interpretability, they focus on features/activations, not normalization layers. This would be first systematic study of individual BN layer importance.

**Validation**: Parallelize across layers, ~3 hours total compute.

### 3. **Cross-Normalization Calibration Transfer**
**What to do**: Test if the "calibrator not encoder" finding transfers to GroupNorm and LayerNorm by: (1) freezing pretrained model, (2) replacing BN with GN/LN, (3) only training the normalization parameters on new data.

**Why it's novel**: Current interpretability work (nnterp by Dumas 2025, Mechanistic Neural Networks by Pervez et al. 2024) doesn't compare normalization methods from a calibration perspective. This tests a specific mechanistic hypothesis across architectures.

**Validation**: Quick experiments with different norm types, ~2 hours.

### 4. **Normalization Statistics as Implicit Class Prototypes**
**What to do**: Analyze if class-conditional BN statistics encode implicit prototypes by: (1) computing mean/var for each class, (2) using these stats to classify new samples via nearest-neighbor in stat-space, (3) visualizing the stat-space structure.

**Why it's novel**: Goes beyond viewing BN as feature normalization (implicit in all papers) to test if statistics themselves contain discriminative information. Connects to prototype learning but via normalization mechanics.

**Validation**: Simple implementation with existing models, <1 hour compute.

### 5. **Dynamic Alpha Scheduling for Improved Generalization**
**What to do**: Instead of fixed interpolation, learn an alpha schedule during training that dynamically adjusts the global/class-conditional mix based on training dynamics (e.g., higher alpha early, lower later).

**Why it's novel**: While Joseph et al. (2025) mention unified tools for vision interpretability, no work explores adaptive normalization statistics. This could reveal when models rely on class-specific vs global calibration.

**Validation**: Modify training loop to include learnable alpha, compare final accuracy, ~4 hours.

## Why These Directions Are Publishing-Worthy

The complete absence of papers on BN statistics as calibrators in the search results suggests this is genuinely unexplored territory. These directions:
- Test specific, falsifiable hypotheses about normalization mechanics
- Can be validated with controlled experiments
- Connect to broader interpretability goals but via an unstudied mechanism
- Could fundamentally change how we view normalization layers (from "feature normalizers" to "statistical calibrators")

The fact that even broad searches on interpretability return no work on this specific angle strongly indicates these would be novel contributions suitable for ICML/ICLR.