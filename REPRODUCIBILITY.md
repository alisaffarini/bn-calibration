# Reproducibility — paper claim → code → result file

Every numerical claim in the paper traces to a JSON output file in
`results/`. Below maps each table, figure, and inline number to (a) the
script that produced it and (b) the JSON file containing the raw
numbers.

## Headline result (Section 4.1, Table 1: SmallResNet/CIFAR-10)

| Claim | Script | Output file | Path within JSON |
|---|---|---|---|
| Global $91.08 \pm 0.12\%$ | `experiment/experiment_v2.py` | `results/results_v2.json` | `experiment1_smallresnet_cifar10.aggregate.global.acc_mean` |
| Same-class $3.55 \pm 0.69\%$ | same | same | `experiment1_smallresnet_cifar10.aggregate.same_class.acc_mean` |
| Wrong-class $65.37 \pm 4.69\%$ | same | same | `experiment1_smallresnet_cifar10.aggregate.wrong_class.acc_mean` |
| Random-class $58.99 \pm 5.84\%$ | same | same | `experiment1_smallresnet_cifar10.aggregate.random_class.acc_mean` |
| Linear probe (post-substitution) $99.88\%$ | same | same | `experiment1_smallresnet_cifar10.aggregate.same_class.probe_acc_mean` |
| Confidence values | same | same | `*.aggregate.*.conf_mean` |
| 10 seeds (42–51) | same | same | per-seed entries under `experiment1_smallresnet_cifar10.seeds` |

## Cross-architecture (Section 4.1, Table 2)

| Architecture | Script | Output file |
|---|---|---|
| VGG-11-BN | `experiment/experiment_v2.py` | `results/results_v2.json` (`experiment2a_vgg11bn_cifar10`) |
| SimpleCNN | same | (`experiment2b_simplecnn_cifar10`) |
| WideResNet-28-10 | `experiment/exp2_wideresnet.py` | `results/results_wideresnet.json` |

Log-ratio numbers ($1.15 / 2.06 / 3.25$ for 4/8/20 BN layers) computed
in `experiment/reanalysis.py`.

## Cross-dataset (Section 4.3)

| Dataset | Script | Output file |
|---|---|---|
| CIFAR-100 | `experiment/exp1_cifar100_fixed.py` | `results/results_cifar100_fixed.json` |
| Tiny-ImageNet-200 | `experiment/exp3_tinyimagenet.py` | `results/results_tinyimagenet.json` |

## Layer-wise ablation (Section 4.4, Table 3)

| Claim | Script | Output file |
|---|---|---|
| Single-layer drops, cumulative shallow→deep, cumulative deep→shallow | `experiment/exp_layerwise.py` (3 seeds) | `results/results_layerwise_seed{42,43,44}.json` |
| $2.95\times$ super-additive factor | aggregated by `experiment/layerwise_summarize.py` | from the 3 seed files |

Run all three seeds, then `python experiment/layerwise_summarize.py` prints
the LaTeX-ready summary numbers.

## μ vs σ² disentangling (Section 4.5, Table 4)

| Claim | Script | Output file |
|---|---|---|
| Global $91.25\%$ | `experiment/exp_mu_sigma_split.py` | `results/results_mu_sigma_split_seed42.json` (`global_acc`) |
| μ-only $76.66\%$ | same | (`mu_only_acc`) |
| σ²-only $27.18\%$ | same | (`sigma_only_acc`) |
| Joint same-class $2.80\%$ | same | (`both_acc`) |

## Controls (Section 4.2)

| Control | Script | Output file |
|---|---|---|
| GroupNorm baseline ($81.65\%$ acc) | `experiment/exp_groupnorm.py` | `results/results_groupnorm.json` |
| ECE before/after temperature scaling ($0.394 \to 0.151$) | `experiment/exp_ece_tempscale.py` | `results/results_ece_tempscaling.json` |

## Interpolation sweep (Section 4.6)

| Claim | Script | Output file |
|---|---|---|
| 11-point $\alpha$ sweep, classifier acc + probe acc | `experiment/experiment_v2.py` | `results/results_v2.json` (`experiment3_interpolation`) |

## Statistical tests

Paired $t$-tests (e.g. $t(9) = -39.1$ for same-vs-wrong) and Fisher's
exact tests are computed inline in `experiment/reanalysis.py` from the
per-seed entries in the JSON files above.

## How to regenerate every paper number from scratch

```bash
# (1) train + evaluate
python experiment/experiment_v2.py 42       # SmallResNet/VGG/SimpleCNN, CIFAR-10
python experiment/exp2_wideresnet.py        # WideResNet-28-10, CIFAR-10
python experiment/exp1_cifar100_fixed.py    # SmallResNet, CIFAR-100
python experiment/exp3_tinyimagenet.py      # SmallResNet64, Tiny-ImageNet-200
python experiment/exp_groupnorm.py          # GroupNorm control
python experiment/exp_ece_tempscale.py      # temperature-scaling control
python experiment/exp_layerwise.py 42
python experiment/exp_layerwise.py 43
python experiment/exp_layerwise.py 44
python experiment/exp_mu_sigma_split.py 42

# (2) aggregate + summary tables
python experiment/reanalysis.py
python experiment/layerwise_summarize.py
```
