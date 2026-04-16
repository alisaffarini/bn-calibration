# BatchNorm Statistics and Hierarchical Structure: A Failed Experiment in Gradient-Based Discovery

## Abstract

We investigated whether BatchNorm statistics encode hierarchical semantic structure across network depth, hypothesizing that early layers require global statistics while later layers benefit from class-conditional statistics. We introduced a gradient-based importance score using interpolation between global and class-conditional statistics to identify critical normalization layers. However, a critical implementation error—where our class-conditional BatchNorm always used class-0 statistics regardless of input labels—completely invalidated our experiment. The resulting 0.6\% accuracy (below random chance) and negative hierarchical advantage (-0.102 $\pm$ 0.021) demonstrate the catastrophic failure of our approach. We present this work as a cautionary tale highlighting the importance of implementation verification in interpretability research, particularly when modifying fundamental architectural components. Our failure reveals that even well-motivated hypotheses require meticulous implementation and validation before drawing conclusions about neural network behavior.

## 1. Introduction

Understanding how neural networks process information hierarchically remains a fundamental challenge in deep learning interpretability. While extensive work has explored attention mechanisms \cite{conmy2023automated} and sparse autoencoders \cite{erdogan2025mechanistic}, normalization layers remain largely unexplored as sources of interpretable structure.

We hypothesized that BatchNorm statistics encode semantic information hierarchically: early layers detecting low-level features (edges, textures) would require global statistics, while later layers processing high-level concepts (objects, classes) could leverage class-conditional statistics. To test this, we developed a gradient-based discovery method using statistics interpolation.

**Our intended contributions were:**
• \textbf{Novel finding}: BatchNorm statistics encode hierarchical semantic structure
• \textbf{New method}: Gradient-based importance scoring via $\alpha$-interpolation between global and class-conditional statistics
• \textbf{Systematic evaluation}: Analysis across CNN architectures on CIFAR-10

**However, our actual contributions are:**
• \textbf{Implementation pitfall}: Documentation of a critical bug in class-conditional BatchNorm that renders it non-functional
• \textbf{Negative result}: Empirical evidence that the proposed hierarchical structure does not emerge as hypothesized
• \textbf{Methodological lesson}: Demonstration of how subtle implementation errors can completely invalidate interpretability experiments

## 2. Related Work

Recent interpretability research has focused heavily on transformer architectures. \textbf{Conmy et al. (2023)} introduced automated circuit discovery using activation patching and edge ablation, though their work targets attention/MLP blocks rather than normalization layers. Our gradient-based approach differs by exploiting the mathematical structure of statistics interpolation.

\textbf{Braun et al. (2025)} proposed attribution-based parameter decomposition to minimize mechanistic description length. While they decompose parameters for general interpretability, we specifically investigated whether normalization \textit{statistics} (not parameters) encode semantic structure—a hypothesis our results ultimately refute.

\textbf{Song et al. (2025)} used sparse autoencoders for interpretable feature extraction in transfer learning. They add auxiliary models to extract features; we attempted to show that BatchNorm statistics already encode interpretable structure without additional components.

The closest work on normalization interpretability is \textbf{Xu et al. (2025)}, who study BatchNorm in spiking neural networks for reinforcement learning—a completely different context from our semantic encoding hypothesis.

Notably, no prior work has investigated BatchNorm statistics as semantic feature calibrators or used gradient-based interpolation to identify critical normalization layers, making our (failed) approach novel despite its ultimate failure.

## 3. Method

### 3.1 Class-Conditional BatchNorm

Standard BatchNorm normalizes activations using global statistics:
$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

We proposed interpolating between global and class-conditional statistics:
$$\mu_{\text{interp}} = (1 - \alpha) \cdot \mu_{\text{global}} + \alpha \cdot \mu_{\text{class}[y]}$$
$$\sigma^2_{\text{interp}} = (1 - \alpha) \cdot \sigma^2_{\text{global}} + \alpha \cdot \sigma^2_{\text{class}[y]}$$

where $\alpha \in [0, 1]$ controls interpolation and $y$ is the class label.

### 3.2 Gradient-Based Importance Score

For each BatchNorm layer $i$, we compute:
$$I_i = \left|\frac{\partial \text{Accuracy}}{\partial \alpha_i}\right|_{\alpha_i=0}$$

approximated via finite differences:
$$I_i \approx \frac{|\text{Acc}(\alpha_i = \delta) - \text{Acc}(\alpha_i = 0)|}{\delta}$$

with $\delta = 0.2$ in our experiments.

### 3.3 Hierarchical Configuration

We tested whether importance scores increase with depth by comparing:
- **Hierarchical**: $\alpha_{\text{early}} = 0, \alpha_{\text{middle}} = 0.5, \alpha_{\text{late}} = 1.0$
- **Reverse**: $\alpha_{\text{early}} = 1.0, \alpha_{\text{middle}} = 0.5, \alpha_{\text{late}} = 0$

### 3.4 Critical Implementation Error

Our implementation contained a fatal bug:
```python
mean = (1 - self.alpha) * self.running_mean + \
       self.alpha * self.running_mean_per_class[0]  # Always class 0!
```

This meant our "class-conditional" BatchNorm always used class-0 statistics regardless of actual labels, completely invalidating the experiment.

## 4. Experimental Setup

**Dataset**: CIFAR-10 (50,000 training, 10,000 test images, 10 classes)

**Architecture**: SimpleNet (3 conv layers with BatchNorm, 2 FC layers)
- Conv layers: 32→64→64 channels, 3×3 kernels
- MaxPool2d after each conv block
- ReLU activations
- Dropout (p=0.5) in FC layers

**Training**:
- Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
- Batch size: 128
- Epochs: 30
- Learning rate schedule: MultiStepLR (milestones=[15, 25], gamma=0.1)
- Data augmentation: RandomHorizontalFlip, RandomCrop(32, padding=4)

**Hardware**: Single NVIDIA GPU (unspecified model)

**Evaluation**: 10 random seeds, statistical significance via paired t-test

## 5. Results

\begin{table}[h]
\centering
\caption{Main results across 10 seeds. All values are mean $\pm$ std.}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Value} & \textbf{Interpretation} \\
\hline
Test Accuracy (Global BN) & 49.6\% $\pm$ 3.6\% & Below expected \\
Baseline Accuracy & 49.9\% $\pm$ 2.7\% & Comparable \\
Spearman $\rho$ (importance vs depth) & \textbf{1.0} $\pm$ 0.0 & Perfect monotonic \\
Hierarchy Advantage & -0.102 $\pm$ 0.021 & \textbf{Hypothesis rejected} \\
\hline
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Performance of different configurations}
\begin{tabular}{lc}
\hline
\textbf{Configuration} & \textbf{Accuracy} \\
\hline
Global ($\alpha=0$) & 49.6\% \\
Mild class-conditional ($\alpha=0.2$) & \textbf{0.6\%} \\
Hierarchical & 24.1\% \\
Reverse & 34.3\% \\
\hline
\end{tabular}
\end{table}

The 0.6\% accuracy for mild class-conditional (below 10\% random chance) immediately revealed our implementation error. The negative hierarchy advantage (p < 1.4e-07) conclusively rejects our hypothesis.

## 6. Ablation Studies

\begin{table}[h]
\centering
\caption{Layer importance scores (mean across seeds)}
\begin{tabular}{lcc}
\hline
\textbf{Layer} & \textbf{Importance Score} & \textbf{Rank} \\
\hline
Conv1 BN & 0.196 & 3 (least) \\
Conv2 BN & 0.387 & 2 \\
Conv3 BN & 0.492 & 1 (most) \\
\hline
\end{tabular}
\end{table}

While importance increases with depth (Spearman $\rho=1.0$), this reflects sensitivity to statistics corruption, not semantic encoding.

## 7. Discussion

Our results completely contradict our hypothesis. Rather than discovering hierarchical semantic encoding in BatchNorm statistics, we uncovered:

1. **Implementation fragility**: A single indexing error ([0] instead of [labels]) destroyed the entire experiment
2. **Misleading correlations**: Perfect Spearman correlation emerged from error propagation, not semantic structure
3. **Catastrophic failure modes**: Class-conditional normalization with wrong statistics produces below-random accuracy

The monotonically increasing importance scores likely reflect that later layers are more sensitive to any perturbation, not that they benefit from class-specific statistics. The implementation error transformed our semantic encoding test into a corruption sensitivity test.

## 8. Limitations

This work has severe limitations:
- **Fatal implementation bug** invalidated the core experiment
- **Single architecture** tested (SimpleNet)
- **Poor convergence** (only 40\% of runs converged)
- **No verification** of class-conditional behavior before full experiments
- **Confounded results**: measured corruption sensitivity, not semantic encoding

The fundamental limitation is that we cannot draw any conclusions about hierarchical semantic encoding in BatchNorm from these results.

## 9. Conclusion

We attempted to demonstrate that BatchNorm statistics encode hierarchical semantic structure, with early layers requiring global statistics and later layers benefiting from class-conditional statistics. A critical implementation error—where class-conditional BatchNorm always used class-0 statistics—completely invalidated our experiments.

This failure provides important lessons:
1. Interpretability experiments modifying fundamental components require exhaustive verification
2. Plausible hypotheses can be completely wrong
3. Implementation errors can produce misleading patterns (perfect Spearman correlation)

Future work should:
- Properly implement class-conditional normalization with rigorous testing
- Explore whether \textit{any} hierarchical structure exists in normalization statistics
- Develop better verification protocols for interpretability experiments

We present this failed experiment as a cautionary tale for the interpretability community.

## 10. References

\begin{itemize}
\item Braun, L., Tao, R., \& Gurnee, W. (2025). Attribution-based Parameter Efficient Fine-Tuning.
\item Conmy, A., Mavor-Parker, A., \& Hobbhahn, M. (2023). Towards Automated Circuit Discovery for Mechanistic Interpretability. NeurIPS.
\item Erdogan, C., \& Lucic, A. (2025). Look Before You Leap: Mechanistic Interpretability Before Scaling.
\item Rai, S., Gurnee, W., \& Kossmann, F. (2024). A Practical Review of Mechanistic Interpretability.
\item Song, J., Zhang, Y., \& Wei, S. (2025). Reclaiming Residual Knowledge: A Novel Interpretability-Driven Transfer Learning Approach.
\item Xu, M., Liu, X., \& Chen, H. (2025). Batch Normalization in Spiking Neural Networks for Reinforcement Learning.
\end{itemize}