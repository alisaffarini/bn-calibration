# Class-Conditional Batch-Normalization Substitution

Code, results, and reproducibility artefacts for the paper *Same-Class
Batch Normalization Statistics Are Worse Than Wrong-Class: A Forensic
Study of an Asymmetric Substitution Effect.*

## Key results

- Same-class BN statistics drop accuracy from $91.1\%$ to $3.5\%$ on
  CIFAR-10/SmallResNet, while wrong-class statistics leave $65.4\%$:
  same-class is roughly $18\times$ more damaging than wrong-class.
- The post-substitution linear probe reaches $\geq 99.7\%$ on CIFAR-10
  ($100\%$ on CIFAR-100, Tiny-ImageNet-200) — the encoder is intact;
  the head is misaligned.
- Within-architecture layer-wise ablation: single-layer drops sum to
  $29.4\%$ but the all-layer drop is $86.8\%$ — a $2.95\times$
  super-additive compounding factor that replicates seed-to-seed.
- $\mu$ vs $\sigma^2$ disentangling: $\mu$-only $76.7\%$,
  $\sigma^2$-only $27.2\%$, joint same-class $2.8\%$ — neither
  statistic alone reproduces the catastrophe; the two interact
  super-additively.
- Three controls rule out distribution-shift, generic-perturbation,
  and logit-scale explanations.
- Replicates across four architectures (SmallResNet, VGG-11-BN,
  SimpleCNN, WideResNet-28-10) and three datasets (CIFAR-10,
  CIFAR-100, Tiny-ImageNet-200).

## Repository layout

```
.
├── README.md             — this file
├── REPRODUCIBILITY.md    — paper-claim → script → result-file mapping
├── requirements.txt      — Python dependencies
├── experiment/           — experiment scripts (one per experiment in the paper)
├── results/              — JSON output of every script (pre-computed)
└── paper/                — LaTeX source + bibliography
```

## Quickstart

```bash
pip install -r requirements.txt

# Headline experiment: same-/wrong-/random-class on 4 architectures
python experiment/experiment_v2.py 42

# Layer-wise ablation (3 seeds → super-additive 2.95×)
python experiment/exp_layerwise.py 42
python experiment/exp_layerwise.py 43
python experiment/exp_layerwise.py 44
python experiment/layerwise_summarize.py

# μ vs σ² disentangling
python experiment/exp_mu_sigma_split.py 42

# Controls
python experiment/exp_groupnorm.py
python experiment/exp_ece_tempscale.py

# Cross-dataset
python experiment/exp1_cifar100_fixed.py
python experiment/exp2_wideresnet.py
python experiment/exp3_tinyimagenet.py

# Cross-architecture aggregation (uses pre-computed JSON)
python experiment/reanalysis.py
```

Every script writes its JSON to `results/`. Pre-computed copies of
those files are bundled in `results/` so the analysis numbers can be
reproduced without re-running training. See `REPRODUCIBILITY.md` for a
claim-by-claim mapping from the paper to the relevant script and
output file.

## Hardware

All experiments ran on a single consumer-grade GPU (8 GB VRAM) and
complete in minutes-to-hours each. Datasets (CIFAR-10/100,
Tiny-ImageNet-200) are downloaded by the scripts on first run via
torchvision.
