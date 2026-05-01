# Results

Canonical experimental data cited by the paper.

## Files

- **results_v2.json** — Primary experiment: SmallResNet on CIFAR-10. 10 seeds (42-51), 20 epochs. Source for Tables 1 and 3 (interpolation). Also includes 3-seed VGG-11-BN and SimpleCNN cross-architecture replications (Table 2).
- **results_cifar100_fixed.json** — CIFAR-100 experiment: 5 seeds, logistic-regression linear probe. Source for Table 4.
- **results_tinyimagenet.json** — Tiny-ImageNet-200 experiment: 3 seeds, SmallResNet64. Source for Table 5.
- **results_wideresnet.json** — WideResNet-28-10 cross-architecture replication: 3 seeds. Source for the WRN block of Table 2.
- **results_ece_tempscaling.json** — ECE + temperature-scaling experiment: 3 seeds. Source for Tables 6 and 7.
- **results_groupnorm.json** — GroupNorm control experiment: 3 seeds. Confirms BN-specificity of the same-class effect (paper cites 81.65 ± 1.94% accuracy).

## Archive

See `archive/` for pilot runs and superseded versions.
