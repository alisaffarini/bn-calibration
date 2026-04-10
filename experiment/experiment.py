#!/usr/bin/env python3
"""
Experiment: Class-Specific Information in Batch Normalization Running Statistics
================================================================================
This experiment investigates whether BN running statistics (running_mean, running_var)
encode class-specific information in CNNs trained on CIFAR-10.

Protocol:
  1. Train a CNN with BN on CIFAR-10 to convergence
  2. Compute class-conditional BN statistics (per-class running mean/var)
  3. Evaluate with three BN stat replacement strategies:
     a. Global (baseline): use standard running stats
     b. Same-class: use class-c stats when evaluating class-c samples
     c. Wrong-class: use class-j stats (j != c) when evaluating class-c samples
     d. Random-class: use randomly chosen class stats
  4. Measure per-class accuracy, confidence, and representation quality (linear probe)
  5. Repeat across multiple seeds for statistical rigor

Author: Automated research pipeline
"""

import json
import copy
import time
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from scipy import stats

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
NUM_SEEDS = 5          # 5 seeds for statistical rigor
NUM_CLASSES = 10
BATCH_SIZE = 128
TRAIN_EPOCHS = 25      # Good convergence on CIFAR-10 with cosine schedule
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']


# ─────────────────────────────────────────────
# Model: Compact ResNet for CIFAR-10
# ─────────────────────────────────────────────
class BasicBlock(nn.Module):
    """Standard ResNet basic block with BN."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SmallResNet(nn.Module):
    """
    A compact ResNet for CIFAR-10: [2,2,2,2] blocks with channels [32,64,128,256].
    Much faster than standard ResNet-18 (channels [64,128,256,512]) while still
    having multiple BN layers to analyze.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


# ─────────────────────────────────────────────
# BN Statistics Utilities
# ─────────────────────────────────────────────
def get_bn_layers(model):
    """Return list of (name, module) for all BatchNorm2d layers."""
    return [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, nn.BatchNorm2d)]


def save_bn_stats(model):
    """Snapshot current running_mean and running_var from all BN layers."""
    stats = {}
    for name, mod in get_bn_layers(model):
        stats[name] = {
            'running_mean': mod.running_mean.clone(),
            'running_var': mod.running_var.clone(),
        }
    return stats


def load_bn_stats(model, stats):
    """Restore BN running stats from a snapshot."""
    for name, mod in get_bn_layers(model):
        mod.running_mean.copy_(stats[name]['running_mean'])
        mod.running_var.copy_(stats[name]['running_var'])


def compute_class_conditional_bn_stats(model, dataloader_by_class):
    """
    For each class c, run all class-c samples through the model in eval mode
    but with BN in train mode (to accumulate fresh statistics), then save.

    Returns: dict[class_idx] -> bn_stats_snapshot
    """
    class_stats = {}
    original_stats = save_bn_stats(model)

    for c in range(NUM_CLASSES):
        # Reset BN running stats to zero
        for name, mod in get_bn_layers(model):
            mod.running_mean.zero_()
            mod.running_var.fill_(1.0)
            mod.num_batches_tracked.zero_()

        # Set model to train mode (so BN updates running stats)
        # but we don't want dropout etc — this model has no dropout
        model.train()

        with torch.no_grad():
            for images, _ in dataloader_by_class[c]:
                images = images.to(DEVICE)
                _ = model(images)

        class_stats[c] = save_bn_stats(model)

    # Restore original stats
    load_bn_stats(model, original_stats)
    model.eval()
    return class_stats


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def get_data():
    """Load CIFAR-10 with standard augmentation for training."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # Also create a non-augmented training set for computing BN stats
    trainset_clean = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Per-class loaders from training set (clean transforms) for computing BN stats
    targets = np.array(trainset_clean.targets)
    class_loaders_train = {}
    for c in range(NUM_CLASSES):
        indices = np.where(targets == c)[0].tolist()
        subset = Subset(trainset_clean, indices)
        class_loaders_train[c] = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Per-class loaders from test set for evaluation
    test_targets = np.array(testset.targets)
    class_loaders_test = {}
    for c in range(NUM_CLASSES):
        indices = np.where(test_targets == c)[0].tolist()
        subset = Subset(testset, indices)
        class_loaders_test[c] = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader, class_loaders_train, class_loaders_test


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_model(seed):
    """Train SmallResNet on CIFAR-10 with given seed. Returns trained model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SmallResNet(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    train_loader, test_loader, _, _ = get_data()

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            # Quick test eval
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    test_correct += (outputs.argmax(1) == labels).sum().item()
                    test_total += labels.size(0)
            print(f"  Epoch {epoch+1:3d} | Train Acc: {correct/total:.4f} | "
                  f"Test Acc: {test_correct/test_total:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

    model.eval()
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate_with_bn_stats(model, class_loaders_test, bn_stats_to_use):
    """
    Evaluate model on each class using specified BN stats.

    Args:
        bn_stats_to_use: dict[class_idx] -> bn_stats (which stats to load for evaluating class c)

    Returns: per_class_accuracy, per_class_confidence, all_features, all_labels
    """
    original_stats = save_bn_stats(model)
    model.eval()

    per_class_acc = {}
    per_class_conf = {}
    all_features = []
    all_labels = []

    for c in range(NUM_CLASSES):
        # Load the specified BN stats for this class
        load_bn_stats(model, bn_stats_to_use[c])

        correct = 0
        total = 0
        confidences = []
        class_features = []

        with torch.no_grad():
            for images, labels in class_loaders_test[c]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits, feats = model(images, return_features=True)
                probs = F.softmax(logits, dim=1)

                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
                # Confidence = probability assigned to the true class
                confidences.extend(probs[range(len(labels)), labels].cpu().numpy().tolist())
                class_features.append(feats.cpu())
                all_labels.extend(labels.cpu().numpy().tolist())

        per_class_acc[c] = correct / total if total > 0 else 0.0
        per_class_conf[c] = float(np.mean(confidences))
        all_features.append(torch.cat(class_features, dim=0))

    # Restore original
    load_bn_stats(model, original_stats)
    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)

    return per_class_acc, per_class_conf, all_features, all_labels


def linear_probe(features, labels, num_classes=10):
    """
    Train a linear classifier on features to measure representation quality.
    Uses 80/20 split. Returns test accuracy.
    """
    n = len(labels)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train = features[train_idx].numpy()
    y_train = labels[train_idx]
    X_test = features[test_idx].numpy()
    y_test = labels[test_idx]

    # Simple linear classifier via closed-form (ridge regression with one-hot targets)
    lam = 1e-3
    Y_train = np.eye(num_classes)[y_train]  # one-hot
    # W = (X^T X + lambda I)^{-1} X^T Y
    XtX = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
    XtY = X_train.T @ Y_train
    W = np.linalg.solve(XtX, XtY)

    preds = X_test @ W
    acc = np.mean(preds.argmax(axis=1) == y_test)
    return float(acc)


# ─────────────────────────────────────────────
# Main Experiment
# ─────────────────────────────────────────────
def run_single_seed(seed):
    """Run the full experiment for one seed. Returns metrics dict."""
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")

    t0 = time.time()

    # 1. Train model
    print(f"\n[1/4] Training model (seed={seed})...")
    model = train_model(seed)

    # 2. Get data loaders
    _, test_loader, class_loaders_train, class_loaders_test = get_data()

    # 3. Save global BN stats (the standard running stats from training)
    print("[2/4] Saving global BN stats and computing class-conditional stats...")
    global_stats = save_bn_stats(model)

    # 4. Compute class-conditional BN stats
    class_cond_stats = compute_class_conditional_bn_stats(model, class_loaders_train)

    # 5. Evaluate under different conditions
    print("[3/4] Evaluating under different BN stat conditions...")

    # (a) Global baseline: use global stats for all classes
    global_bn_for_all = {c: global_stats for c in range(NUM_CLASSES)}
    global_acc, global_conf, global_feats, global_labels = \
        evaluate_with_bn_stats(model, class_loaders_test, global_bn_for_all)

    # (b) Same-class: use class-c stats for class-c samples
    same_class_bn = {c: class_cond_stats[c] for c in range(NUM_CLASSES)}
    same_acc, same_conf, same_feats, same_labels = \
        evaluate_with_bn_stats(model, class_loaders_test, same_class_bn)

    # (c) Wrong-class: for class c, use stats from class (c+5) % 10 (maximally different)
    wrong_class_bn = {c: class_cond_stats[(c + 5) % NUM_CLASSES] for c in range(NUM_CLASSES)}
    wrong_acc, wrong_conf, wrong_feats, wrong_labels = \
        evaluate_with_bn_stats(model, class_loaders_test, wrong_class_bn)

    # (d) Random-class: for each class c, pick a random other class's stats
    np.random.seed(seed + 1000)
    random_class_bn = {}
    for c in range(NUM_CLASSES):
        r = np.random.choice([j for j in range(NUM_CLASSES) if j != c])
        random_class_bn[c] = class_cond_stats[r]
    rand_acc, rand_conf, rand_feats, rand_labels = \
        evaluate_with_bn_stats(model, class_loaders_test, random_class_bn)

    # 6. Linear probe on representations
    print("[4/4] Running linear probes...")
    np.random.seed(seed + 2000)
    global_probe = linear_probe(global_feats, global_labels)
    same_probe = linear_probe(same_feats, same_labels)
    wrong_probe = linear_probe(wrong_feats, wrong_labels)
    rand_probe = linear_probe(rand_feats, rand_labels)

    elapsed = time.time() - t0
    print(f"\nSeed {seed} completed in {elapsed:.1f}s")

    # Aggregate per-class metrics into means
    results = {
        'seed': seed,
        'elapsed_seconds': elapsed,
        'global': {
            'per_class_acc': global_acc,
            'mean_acc': float(np.mean(list(global_acc.values()))),
            'per_class_conf': global_conf,
            'mean_conf': float(np.mean(list(global_conf.values()))),
            'linear_probe_acc': global_probe,
        },
        'same_class': {
            'per_class_acc': same_acc,
            'mean_acc': float(np.mean(list(same_acc.values()))),
            'per_class_conf': same_conf,
            'mean_conf': float(np.mean(list(same_conf.values()))),
            'linear_probe_acc': same_probe,
        },
        'wrong_class': {
            'per_class_acc': wrong_acc,
            'mean_acc': float(np.mean(list(wrong_acc.values()))),
            'per_class_conf': wrong_conf,
            'mean_conf': float(np.mean(list(wrong_conf.values()))),
            'linear_probe_acc': wrong_probe,
        },
        'random_class': {
            'per_class_acc': rand_acc,
            'mean_acc': float(np.mean(list(rand_acc.values()))),
            'per_class_conf': rand_conf,
            'mean_conf': float(np.mean(list(rand_conf.values()))),
            'linear_probe_acc': rand_probe,
        },
    }

    # Print summary for this seed
    print(f"\n  Condition     | Accuracy | Confidence | Linear Probe")
    print(f"  --------------|----------|------------|-------------")
    for cond in ['global', 'same_class', 'wrong_class', 'random_class']:
        r = results[cond]
        print(f"  {cond:14s} | {r['mean_acc']:.4f}   | {r['mean_conf']:.4f}     | {r['linear_probe_acc']:.4f}")

    return results


def compute_per_class_delta(all_results):
    """
    Compute per-class accuracy delta: same_class - global for each class.
    This shows whether using class-matched BN stats helps specifically for that class.
    """
    deltas = defaultdict(list)
    for res in all_results:
        for c in range(NUM_CLASSES):
            delta = res['same_class']['per_class_acc'][c] - res['global']['per_class_acc'][c]
            deltas[c].append(delta)
    return {c: (float(np.mean(v)), float(np.std(v))) for c, v in deltas.items()}


def main():
    print(f"Device: {DEVICE}")
    print(f"Seeds: {NUM_SEEDS}, Epochs: {TRAIN_EPOCHS}")
    print(f"Model: SmallResNet (channels 32-64-128-256)")
    print(f"BN layers: will be counted after model init")

    # Count BN layers
    tmp = SmallResNet()
    bn_count = len(get_bn_layers(tmp))
    print(f"Number of BN layers: {bn_count}")
    del tmp

    all_results = []
    seeds = list(range(42, 42 + NUM_SEEDS))

    total_start = time.time()
    for seed in seeds:
        result = run_single_seed(seed)
        all_results.append(result)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"ALL SEEDS COMPLETE ({total_elapsed:.0f}s total)")
    print(f"{'='*60}")

    # ─── Aggregate statistics across seeds ───
    conditions = ['global', 'same_class', 'wrong_class', 'random_class']
    agg = {}
    for cond in conditions:
        accs = [r[cond]['mean_acc'] for r in all_results]
        confs = [r[cond]['mean_conf'] for r in all_results]
        probes = [r[cond]['linear_probe_acc'] for r in all_results]
        agg[cond] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'confidence_mean': float(np.mean(confs)),
            'confidence_std': float(np.std(confs)),
            'linear_probe_mean': float(np.mean(probes)),
            'linear_probe_std': float(np.std(probes)),
            'raw_accs': accs,
            'raw_confs': confs,
            'raw_probes': probes,
        }

    # ─── Statistical tests (paired t-tests) ───
    stat_tests = {}

    # Same-class vs Global (accuracy)
    t_sg, p_sg = stats.ttest_rel(agg['same_class']['raw_accs'], agg['global']['raw_accs'])
    stat_tests['same_vs_global_acc'] = {'t': float(t_sg), 'p': float(p_sg)}

    # Wrong-class vs Global (accuracy)
    t_wg, p_wg = stats.ttest_rel(agg['wrong_class']['raw_accs'], agg['global']['raw_accs'])
    stat_tests['wrong_vs_global_acc'] = {'t': float(t_wg), 'p': float(p_wg)}

    # Same-class vs Wrong-class (accuracy)
    t_sw, p_sw = stats.ttest_rel(agg['same_class']['raw_accs'], agg['wrong_class']['raw_accs'])
    stat_tests['same_vs_wrong_acc'] = {'t': float(t_sw), 'p': float(p_sw)}

    # Same-class vs Global (confidence)
    t_sg_c, p_sg_c = stats.ttest_rel(agg['same_class']['raw_confs'], agg['global']['raw_confs'])
    stat_tests['same_vs_global_conf'] = {'t': float(t_sg_c), 'p': float(p_sg_c)}

    # Same-class vs Global (linear probe)
    t_sg_p, p_sg_p = stats.ttest_rel(agg['same_class']['raw_probes'], agg['global']['raw_probes'])
    stat_tests['same_vs_global_probe'] = {'t': float(t_sg_p), 'p': float(p_sg_p)}

    # Per-class deltas
    per_class_deltas = compute_per_class_delta(all_results)

    # ─── Print final summary ───
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean ± std across {0} seeds)".format(NUM_SEEDS))
    print(f"{'='*60}")
    print(f"\n{'Condition':<16} | {'Accuracy':>15} | {'Confidence':>15} | {'Linear Probe':>15}")
    print(f"{'-'*16}-+-{'-'*15}-+-{'-'*15}-+-{'-'*15}")
    for cond in conditions:
        a = agg[cond]
        print(f"{cond:<16} | {a['accuracy_mean']:.4f} ± {a['accuracy_std']:.4f} | "
              f"{a['confidence_mean']:.4f} ± {a['confidence_std']:.4f} | "
              f"{a['linear_probe_mean']:.4f} ± {a['linear_probe_std']:.4f}")

    print(f"\nStatistical Tests (paired t-tests, df={NUM_SEEDS-1}):")
    for test_name, vals in stat_tests.items():
        sig = "***" if vals['p'] < 0.001 else "**" if vals['p'] < 0.01 else "*" if vals['p'] < 0.05 else "ns"
        print(f"  {test_name:<30s}: t={vals['t']:+.3f}, p={vals['p']:.4f} {sig}")

    print(f"\nPer-class accuracy delta (same_class - global):")
    for c in range(NUM_CLASSES):
        mean_d, std_d = per_class_deltas[c]
        print(f"  {CIFAR_CLASSES[c]:>12s} (class {c}): {mean_d:+.4f} ± {std_d:.4f}")

    # ─── Save results ───
    # Clean up raw arrays for JSON serialization (convert int keys to str)
    for res in all_results:
        for cond in conditions:
            res[cond]['per_class_acc'] = {str(k): v for k, v in res[cond]['per_class_acc'].items()}
            res[cond]['per_class_conf'] = {str(k): v for k, v in res[cond]['per_class_conf'].items()}

    # Remove raw arrays from agg for clean JSON
    agg_clean = {}
    for cond in conditions:
        agg_clean[cond] = {k: v for k, v in agg[cond].items()
                           if not k.startswith('raw_')}

    final_results = {
        'config': {
            'num_seeds': NUM_SEEDS,
            'num_classes': NUM_CLASSES,
            'train_epochs': TRAIN_EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'device': DEVICE,
            'model': 'SmallResNet (32-64-128-256)',
            'bn_layers': bn_count,
        },
        'aggregate': agg_clean,
        'statistical_tests': stat_tests,
        'per_class_deltas': {CIFAR_CLASSES[c]: {'mean': d[0], 'std': d[1]}
                             for c, d in per_class_deltas.items()},
        'per_seed_results': all_results,
        'total_time_seconds': total_elapsed,
    }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print machine-readable results line
    summary = {
        'signal': True,  # Will be validated below
        'global_acc': agg_clean['global']['accuracy_mean'],
        'same_class_acc': agg_clean['same_class']['accuracy_mean'],
        'wrong_class_acc': agg_clean['wrong_class']['accuracy_mean'],
        'random_class_acc': agg_clean['random_class']['accuracy_mean'],
        'same_vs_global_p': stat_tests['same_vs_global_acc']['p'],
        'wrong_vs_global_p': stat_tests['wrong_vs_global_acc']['p'],
        'same_vs_wrong_p': stat_tests['same_vs_wrong_acc']['p'],
    }

    # Validate signal: same_class should be >= global, wrong_class should be <= global
    # (or at least the direction should be consistent)
    if (summary['same_class_acc'] >= summary['wrong_class_acc'] and
            summary['same_vs_wrong_p'] < 0.10):
        summary['signal'] = True
    else:
        summary['signal'] = False

    print(f"\nRESULTS: {json.dumps(summary)}")
    return final_results


if __name__ == '__main__':
    main()
