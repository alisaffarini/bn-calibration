## Literature Summary

The search results reveal several key trends in interpretability research:

1. **Applied XAI Dominance**: Most papers (Papers 1-12) focus on applying existing interpretability techniques (LIME, SHAP, Grad-CAM) to specific domains like medical imaging, malware detection, and deepfakes. These papers typically achieve 90%+ accuracy but don't advance interpretability methods themselves.

2. **Mechanistic Interpretability Tools**: Papers 20-22 (Erdogan & Lucic 2025, Dumas 2025, Joseph et al. 2025) are building infrastructure for mechanistic interpretability, including sparse autoencoders and standardized interfaces for transformer analysis.

3. **Calibration and Adaptation**: Papers 15-16 (Tareen et al. 2025, Mehrbod et al. 2025) touch on model calibration, with Paper 16 specifically mentioning batch normalization statistics updates for test-time adaptation, but neither explores BN's role in interpretability or calibration systematically.

4. **Scaling Laws**: Papers 13-14 (Mastromichalakis 2026, Alnemari et al. 2026) explore scaling behaviors but focus on optimization and small models rather than interpretability.

## Identified Gaps

Your BN-as-calibrator finding addresses a **completely unexplored area**. No paper investigates:

1. How batch normalization statistics encode class-specific information
2. The disconnect between BN replacement effects on accuracy vs. linear probe performance  
3. BN layers as interpretability bottlenecks
4. The calibration implications of BN statistics manipulation
5. Cross-class BN statistics replacement as a diagnostic tool

## Recommended Research Directions

### 1. **BN Statistics as Class Prototypes: A New Lens for Feature Collapse**
**Why Novel**: While Golechha & Dao (2024) discuss challenges in interpreting representations and Joseph et al. (2025) build vision interpretability tools, nobody has investigated whether BN statistics effectively store class prototypes. Your finding that same-class BN replacement preserves 99.97% linear probe accuracy suggests BN moments might encode minimal class information while features before BN are highly discriminative.

**Experiment**: Beyond your planned experiments, add:
- Compute cosine similarity between BN statistics (mean/var) across classes
- Visualize BN statistics in 2D using t-SNE/PCA - do they cluster by class?
- Test if BN statistics can be used as a nearest-neighbor classifier
- Compare feature discriminability before/after BN using centered kernel alignment

### 2. **Temperature-Free Calibration via BN Surgery**
**Why Novel**: Mehrbod et al. (2025) propose adaptive quantile recalibration for test-time adaptation using BN updates, but they don't explore using BN replacement for calibration. Your finding suggests a new calibration method: instead of temperature scaling, selectively replace BN statistics from well-calibrated classes to poorly-calibrated ones.

**Experiment**:
- Identify over-confident vs under-confident classes using ECE per class
- Replace BN stats from well-calibrated to poorly-calibrated classes
- Compare to temperature scaling and other calibration baselines
- Test on naturally miscalibrated scenarios (class imbalance, domain shift)

### 3. **BN-Guided Network Pruning: Exploiting the Accuracy-Probe Gap**
**Why Novel**: While Ohib et al. (2024) explore sparse federated learning and Qiu et al. (2024) replace dense layers with structured matrices, nobody has used the accuracy-probe gap as a pruning signal. If linear probes maintain 99.97% accuracy after BN corruption, the network after BN might be heavily redundant.

**Experiment**:
- Progressively prune channels/filters after BN based on probe-accuracy retention
- Compare to magnitude-based and gradient-based pruning
- Test if BN-corrupted networks are more amenable to extreme pruning
- Measure FLOPs vs accuracy trade-off compared to standard pruning

### 4. **Cross-Architecture BN Transfer as Architecture Understanding**
**Why Novel**: Papers 17 (Gambella et al. 2025) and 25 (Qiu et al. 2024) study architecture properties, but don't consider BN statistics as architecture signatures. Your multi-architecture plan (ResNet, VGG, MobileNet, DenseNet) could reveal whether BN statistics are architecture-specific or universal.

**Experiment**:
- Train all architectures, then cross-replace BN stats between architectures
- Measure accuracy drop when using ResNet BN stats in VGG vs same-architecture different-class
- Test if certain architecture pairs are more "BN-compatible"
- Investigate if BN statistics similarity predicts architecture similarity

### 5. **BN Replacement as a Debugging Tool for Dataset Bias**
**Why Novel**: While many papers use XAI for model debugging, none propose using BN replacement patterns to detect dataset biases. If certain class pairs show minimal accuracy drop under BN replacement, it might indicate the model relies on spurious features shared between those classes.

**Experiment**:
- Create accuracy drop heatmap for all class-pair BN replacements
- Identify class pairs with suspiciously low accuracy drops
- Manually inspect these pairs for visual/semantic similarities
- Validate by training on debiased datasets and comparing BN replacement patterns

Each direction builds on your core finding while exploring genuinely new territory not covered in existing literature. The 4-hour compute constraint is met since these mostly involve inference-time manipulations of pre-trained models.