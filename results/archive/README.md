# Archive

Files in this directory are valid experimental data that has been superseded by canonical versions in the parent `results/` directory.

## Files

- **results.json** — 5-seed SmallResNet/CIFAR-10 pilot (25 epochs, ridge probe). Output of `../experiment/archive/experiment.py`. Replicates canonical findings.
- **results_cifar100.json** — 3-seed CIFAR-100 pilot (ridge probe ill-posed at 100 classes). Output of `../experiment/archive/exp_cifar100.py`. Accuracy/confidence numbers valid; probe column uninformative.
- **results_v3_partial.json** — Originally `results_v3_missing.json`. Two-block file: the groupnorm block was promoted to `../results_groupnorm.json` (canonical, paper-cited). The cifar100 block is a stale partial duplicate of `results_cifar100.json`. Preserved unchanged.
