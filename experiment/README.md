# Experiment Code

Canonical experiment scripts. Each script is paired with one or more JSONs in `../results/`.

## Scripts

- **experiment_v2.py** — SmallResNet/CIFAR-10 (10 seeds), VGG-11-BN/SimpleCNN cross-architecture (3 seeds), interpolation. Writes `../results/results_v2.json`.
- **exp1_cifar100_fixed.py** — CIFAR-100 (5 seeds, logistic probe). Writes `../results/results_cifar100_fixed.json`.
- **exp2_wideresnet.py** — WideResNet-28-10 (3 seeds). Writes `../results/results_wideresnet.json`.
- **exp3_tinyimagenet.py** — Tiny-ImageNet-200 (3 seeds). Writes `../results/results_tinyimagenet.json`.
- **exp_ece_tempscale.py** — ECE + temperature scaling (3 seeds). Writes `../results/results_ece_tempscaling.json`.
- **exp_groupnorm.py** — GroupNorm control. Writes `../results/results_groupnorm.json`.

## Archive

See `archive/` for pilot/superseded scripts.
