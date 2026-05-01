# Archive

Files in this directory are valid scientific work that has been superseded by canonical versions in the parent `experiment/` directory. Preserved for reproducibility and provenance.

## Files

- **experiment.py** — Original 5-seed CIFAR-10 pilot, 25 epochs, ridge-regression linear probe. Superseded by `../experiment_v2.py` (10 seeds, 20 epochs, all 4 BN conditions across architectures). Findings replicate.
- **exp_cifar100.py** — 3-seed CIFAR-100 pilot using ridge-regression linear probe. Ridge is ill-posed in 100-class regimes (256-d feature space, ~80 samples/class) so probe column is uninformative. Superseded by `../exp1_cifar100_fixed.py` which uses logistic regression for 5 seeds. Accuracy/confidence numbers from this run are still valid.
- **exp_ece.py** — Earlier ECE-only driver. Superseded by `../exp_ece_tempscale.py` which combines ECE + temperature scaling.
- **exp_tempscaling.py** — Earlier temperature-scaling driver. Superseded by `../exp_ece_tempscale.py`.
- **exp2_wideresnet_small_batch.py** — WideResNet-28-10 with `BATCH_SIZE=32` (vs canonical 128). Methodologically distinct ablation; not executed in the current paper run.
