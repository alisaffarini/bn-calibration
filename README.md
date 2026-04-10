# Batch Normalization Statistics as Output Calibrators, Not Feature Encoders

**Author:** Ali Saffarini, Harvard University

## Key Results

- Same-class BN statistics **destroy accuracy** (91% → 3.5%) while linear probes reach **99.9%**
- Effect replicates across 4 architectures (SmallResNet, VGG-11, SimpleCNN, WideResNet-16-8)
- Generalizes to 3 datasets: CIFAR-10, CIFAR-100 (0.67%), Tiny-ImageNet-200 (0.17% with 100% probe)
- GroupNorm control confirms effect is BN-specific
- Temperature scaling reduces ECE but cannot recover accuracy — proves structural miscalibration
- 10 seeds on primary experiment, 3 on all others

## Structure

```
paper/       LaTeX source (ICML 2025 format) + refs.bib + style files
results/     JSON results for all experiments
experiment/  Python experiment scripts
```

## Citation

Target venue: ICLR 2027
