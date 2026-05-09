"""Summarize layer-wise results across multiple seeds (mean +/- std).

Reads results_layerwise_seed{42,43,44}.json and produces:
- Per-layer mean +/- std for single-layer drops
- Cumulative-shallow and cumulative-deep mean +/- std at key k values
- Sum-of-single vs all-layer multiplicative factor across seeds

Output: prints LaTeX-ready summary numbers.
"""
import json
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent / "results"
SEEDS = [42, 43, 44]

results = {}
for s in SEEDS:
    path = BASE / f"results_layerwise_seed{s}.json"
    if not path.exists():
        print(f"WARNING: missing seed {s} at {path}")
        continue
    with open(path) as f:
        results[s] = json.load(f)

print(f"Loaded {len(results)} seeds: {sorted(results.keys())}\n")

# Aggregate global + all-layer
glob = [results[s]["global_acc"] for s in results]
allayer = [results[s]["all_layer_same_class_acc"] for s in results]
print(f"Global:    {statistics.mean(glob)*100:.2f} +/- {statistics.stdev(glob)*100 if len(glob)>1 else 0:.2f}")
print(f"All-layer: {statistics.mean(allayer)*100:.2f} +/- {statistics.stdev(allayer)*100 if len(allayer)>1 else 0:.2f}")

# Single-layer drops by index
print("\n=== Single-layer drops (seeds: {}) ===".format(sorted(results.keys())))
print(f"{'idx':>4} {'name':<22} {'mean':>7} {'std':>6}")
n_layers = len(results[SEEDS[0]]["single_layer"])
single_layer_means = []
for i in range(n_layers):
    drops = [results[s]["single_layer"][i]["drop"] for s in results]
    name = results[SEEDS[0]]["single_layer"][i]["name"]
    m = statistics.mean(drops)
    sd = statistics.stdev(drops) if len(drops) > 1 else 0.0
    print(f"{i:>4} {name:<22} {m*100:>6.2f}%  {sd*100:>5.2f}%")
    single_layer_means.append(m)

sum_means = sum(single_layer_means)
allayer_drop = statistics.mean(glob) - statistics.mean(allayer)
factor = allayer_drop / sum_means if sum_means > 0 else float('nan')
print(f"\nSum of mean single-layer drops: {sum_means*100:.2f}%")
print(f"Mean all-layer drop: {allayer_drop*100:.2f}%")
print(f"Super-additive factor: {factor:.2f}x")

# Cumulative shallow-to-deep at key k
print("\n=== Cumulative shallow-to-deep ===")
print(f"{'k':>3} {'mean':>7} {'std':>6}")
for k in [1, 5, 10, 15, 20]:
    accs = [results[s]["cum_shallow_to_deep"][k-1]["acc"] for s in results]
    m = statistics.mean(accs)
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f"{k:>3}  {m*100:>6.2f}  {sd*100:>5.2f}")

# Cumulative deep-to-shallow at key k
print("\n=== Cumulative deep-to-shallow ===")
print(f"{'k':>3} {'mean':>7} {'std':>6}")
for k in [1, 5, 10, 15, 20]:
    accs = [results[s]["cum_deep_to_shallow"][k-1]["acc"] for s in results]
    m = statistics.mean(accs)
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f"{k:>3}  {m*100:>6.2f}  {sd*100:>5.2f}")
