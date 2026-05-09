"""Re-analysis from existing JSON results: cross-arch asymmetry magnitude.

Pulls cross-architecture numbers from results JSONs and computes:
- BN-layer count vs same-class accuracy correlation
- same/wrong ratio across architectures
- diagnostic for theory-vs-empirics section
"""
import json
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "results"

v2 = json.load(open(BASE / "results_v2.json"))
wrn = json.load(open(BASE / "results_wideresnet.json"))
c100 = json.load(open(BASE / "results_cifar100_fixed.json"))
tin = json.load(open(BASE / "results_tinyimagenet.json"))
gn = json.load(open(BASE / "results_groupnorm.json"))
ece = json.load(open(BASE / "results_ece_tempscaling.json"))

archs = [
    ("SmallResNet", 20, v2["experiment1_smallresnet_cifar10"]["aggregate"]),
    ("VGG-11-BN",   8,  v2["experiment2a_vgg11bn_cifar10"]["aggregate"]),
    ("SimpleCNN",   4,  v2["experiment2b_simplecnn_cifar10"]["aggregate"]),
    ("WRN-28-10",   None, wrn["aggregate"]),
]

print(f"{'arch':<14} {'BN':>4} {'Glob':>7} {'Same':>7} {'Wrong':>7} {'S/W':>7} {'logG/S':>8} {'Probe(S)':>9}")
import math
for name, bn, agg in archs:
    g = agg["global"]["accuracy_mean"] * 100
    s = agg["same_class"]["accuracy_mean"] * 100
    w = agg["wrong_class"]["accuracy_mean"] * 100
    ps = agg["same_class"]["linear_probe_mean"] * 100
    log_drop = math.log(g / s)
    print(f"{name:<14} {bn!s:>4} {g:>7.2f} {s:>7.2f} {w:>7.2f} {s/w:>7.3f} {log_drop:>8.2f} {ps:>9.2f}")

# Pearson correlation: BN-layer count vs same-class accuracy (using only the 3 with known BN counts)
known = [(bn, archs[i][2]["same_class"]["accuracy_mean"]*100) for i,(_,bn,_) in enumerate(archs) if bn is not None]
n = len(known)
mx = sum(x for x,_ in known)/n
my = sum(y for _,y in known)/n
cov = sum((x-mx)*(y-my) for x,y in known)/n
sx = (sum((x-mx)**2 for x,_ in known)/n)**0.5
sy = (sum((y-my)**2 for _,y in known)/n)**0.5
print(f"\nPearson r(BN_count, same_acc) over 3 BN-counted archs: {cov/(sx*sy):.3f}")

# Class-mean separability prediction: under the linearized theory, the asymmetry
# magnitude should scale with how much "between-class" signal is in the activations.
# Probe accuracy under same-class > global is consistent with within-class variation
# being the dominant contaminating factor.
print("\n=== Key theory-relevant quantities ===")
for name, bn, agg in archs:
    g = agg["global"]["accuracy_mean"]
    s = agg["same_class"]["accuracy_mean"]
    pg = agg["global"]["linear_probe_mean"]
    ps = agg["same_class"]["linear_probe_mean"]
    print(f"{name}: probe(same)/probe(global) = {ps/pg:.3f}; classifier(same)/classifier(global) = {s/g:.3f}")
