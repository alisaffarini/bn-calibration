Looking at this paper against the search results and experimental outcomes:

## NOVELTY CHECK ✓
The search results confirm this is novel - no prior work explores BatchNorm statistics encoding hierarchical semantic structure or uses gradient-based interpolation to identify critical BN layers. The closest papers like "Transformer^-1" (2025) focus on adaptive computation in different contexts, and "Linear Mode Connectivity" (2019) examines optimization/pruning rather than BN statistics semantics.

## METHODOLOGY ASSESSMENT ❌❌❌

**FATAL IMPLEMENTATION BUG**: The ClassConditionalBatchNorm2d always uses class 0 statistics regardless of actual labels:
```python
mean = (1 - self.alpha) * self.running_mean + self.alpha * self.running_mean_per_class[0]
```
This completely invalidates the core experiment - you're not testing class-conditional BatchNorm at all!

**CATASTROPHIC EXPERIMENTAL FAILURE**:
- Only 5/10 seeds converged
- Class-conditional BN: 0.6% accuracy (below 10% random chance!)
- Hierarchical configuration: 0.000 accuracy
- Experiment terminated with "NO_SIGNAL: Failed criteria" and "SANITY_ABORT"
- Even baseline ResNet-18 gets only 86.6% on CIFAR-10 (should be >90%)

## REVIEWER FLAGS 🚨

1. **The experiment disproves your hypothesis**: Your results show hierarchical BN gets 0% accuracy. You cannot publish "we hypothesized X would work, but it completely fails."

2. **Implementation doesn't match claims**: Your "class-conditional" BN isn't actually class-conditional due to the bug.

3. **Training instability**: 50% convergence failure rate indicates fundamental issues.

4. **Below-baseline performance**: Standard ResNet-18 should achieve >90% on CIFAR-10, not 86.6%.

## VERDICT: NO-GO

This paper is not publishable in its current form. The fatal implementation bug means you haven't actually tested your hypothesis, and even the buggy results show complete failure (0% accuracy). You're essentially submitting a paper saying "we tried this idea and it doesn't work at all."

To salvage this:
1. Fix the class-conditional BN implementation bug
2. Debug why baseline performance is so poor
3. If the hypothesis still fails after fixes, pivot to understanding WHY it fails (that could be interesting)
4. Or abandon this direction if properly-implemented class-conditional BN still gets near-zero accuracy

No reputable venue would accept a paper with implementation bugs that invalidate the core experiment and results showing complete method failure.