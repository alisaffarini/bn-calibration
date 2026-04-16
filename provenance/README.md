# Provenance — BN Calibration

Raw run data from the experiment pipeline that generated this paper's results.

## Run History

| Run | Description | Key Output |
|-----|-------------|------------|
| `run_054_batchnorm` | Early automated exploration of BN statistics | Initial experiments, iteration logs |
| `run_057_bn_manual` | **Main manual run** — all final experiments | All result JSONs, training logs, WideResNet/TinyImageNet/ECE experiments |
| `run_058_bn_extended` | Extended BN experiments (automated) | Conversation log, iteration outputs |
| `run_059_bn_deeper` | Deeper BN exploration (v1 + v2) | Multiple iterations of experiment code |
| `run_060_bn_paper_v4` | Paper-writing run with literature search | Proposals, lit review, training checkpoints |
| `run_061_bn_ucurve` | U-curve BN experiment (v1 + v3) | Training outputs, conversation logs |

## Key Files

- `run_057_bn_manual/results*.json` — All experiment results (CIFAR-10, CIFAR-100, TinyImageNet, WideResNet, ECE)
- `run_057_bn_manual/*_output.log` — Training stdout for WideResNet, TinyImageNet, ECE
- `run_060_bn_paper_v4/conversation_log.md` — AI agent conversation during paper development
- `run_060_bn_paper_v4/literature_report.md` — Literature review that informed the paper

## Notes

- All experiments ran on Apple Silicon (MPS backend)
- Run 057 is the source of truth for reported results
- Earlier runs (054, 058, 059) show the research progression
