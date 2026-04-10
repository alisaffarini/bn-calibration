#!/usr/bin/env python3
"""
Experiment 1: CIFAR-100 with fixed linear probe (LogisticRegression instead of ridge).
SmallResNet, 5 seeds (42-46), 25 epochs, 4 BN conditions.
"""

import json
import copy
import time
import os
import sys
import gc
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 25
SEEDS = list(range(42, 47))  # 5 seeds

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class BasicBlock(nn.Module):
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
    return [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, nn.BatchNorm2d)]


def save_bn_stats(model):
    stats = {}
    for name, mod in get_bn_layers(model):
        stats[name] = {
            'running_mean': mod.running_mean.clone(),
            'running_var': mod.running_var.clone(),
        }
    return stats


def load_bn_stats(model, stats):
    for name, mod in get_bn_layers(model):
        mod.running_mean.copy_(stats[name]['running_mean'])
        mod.running_var.copy_(stats[name]['running_var'])


def compute_class_conditional_bn_stats(model, dataloader_by_class, num_classes):
    class_stats = {}
    original_stats = save_bn_stats(model)

    for c in range(num_classes):
        for name, mod in get_bn_layers(model):
            mod.running_mean.zero_()
            mod.running_var.fill_(1.0)
            mod.num_batches_tracked.zero_()
        model.train()
        with torch.no_grad():
            for images, _ in dataloader_by_class[c]:
                images = images.to(DEVICE)
                _ = model(images)
        class_stats[c] = save_bn_stats(model)

    load_bn_stats(model, original_stats)
    model.eval()
    return class_stats


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────
def get_cifar100_data():
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    num_classes = 100

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=transform_test)
    trainset_clean = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=True, download=False, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    targets = np.array(trainset_clean.targets)
    class_loaders_train = {}
    for c in range(num_classes):
        indices = np.where(targets == c)[0].tolist()
        subset = Subset(trainset_clean, indices)
        class_loaders_train[c] = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_targets = np.array(testset.targets)
    class_loaders_test = {}
    for c in range(num_classes):
        indices = np.where(test_targets == c)[0].tolist()
        subset = Subset(testset, indices)
        class_loaders_test[c] = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, test_loader, class_loaders_train, class_loaders_test, num_classes


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train_model(model, train_loader, test_loader, epochs, seed, label=""):
    torch.manual_seed(seed)
    np.random.seed(seed)

    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    test_correct += (outputs.argmax(1) == labels).sum().item()
                    test_total += labels.size(0)
            print(f"  [{label}] Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {correct/total:.4f} | Test: {test_correct/test_total:.4f}")

    model.eval()
    return model


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate_with_bn_stats(model, class_loaders_test, bn_stats_to_use, num_classes):
    original_stats = save_bn_stats(model)
    model.eval()
    per_class_acc = {}
    per_class_conf = {}
    all_features = []
    all_labels = []

    for c in range(num_classes):
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
                confidences.extend(probs[range(len(labels)), labels].cpu().numpy().tolist())
                class_features.append(feats.cpu())
                all_labels.extend(labels.cpu().numpy().tolist())

        per_class_acc[c] = correct / total if total > 0 else 0.0
        per_class_conf[c] = float(np.mean(confidences))
        all_features.append(torch.cat(class_features, dim=0))

    load_bn_stats(model, original_stats)
    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)
    return per_class_acc, per_class_conf, all_features, all_labels


def linear_probe(features, labels, num_classes=100):
    """Fixed linear probe using LogisticRegression instead of ridge."""
    n = len(labels)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]
    X_train = features[train_idx].numpy()
    y_train = labels[train_idx]
    X_test = features[test_idx].numpy()
    y_test = labels[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return float(acc)


def run_standard_eval(model, class_loaders_train, class_loaders_test, num_classes, seed):
    """Run the 4-condition evaluation (global, same, wrong, random). Returns dict."""
    global_stats = save_bn_stats(model)
    class_cond_stats = compute_class_conditional_bn_stats(model, class_loaders_train, num_classes)

    conditions = {}

    # Global
    print(f"    Evaluating global...")
    global_bn = {c: global_stats for c in range(num_classes)}
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, global_bn, num_classes)
    np.random.seed(seed + 2000)
    probe = linear_probe(feats, labs, num_classes)
    conditions['global'] = {
        'mean_acc': float(np.mean(list(acc.values()))),
        'mean_conf': float(np.mean(list(conf.values()))),
        'linear_probe_acc': probe,
    }
    print(f"      global: acc={conditions['global']['mean_acc']:.4f}, conf={conditions['global']['mean_conf']:.4f}, probe={probe:.4f}")

    # Same-class
    print(f"    Evaluating same_class...")
    same_bn = {c: class_cond_stats[c] for c in range(num_classes)}
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, same_bn, num_classes)
    np.random.seed(seed + 2001)
    probe = linear_probe(feats, labs, num_classes)
    conditions['same_class'] = {
        'mean_acc': float(np.mean(list(acc.values()))),
        'mean_conf': float(np.mean(list(conf.values()))),
        'linear_probe_acc': probe,
    }
    print(f"      same_class: acc={conditions['same_class']['mean_acc']:.4f}, conf={conditions['same_class']['mean_conf']:.4f}, probe={probe:.4f}")

    # Wrong-class
    print(f"    Evaluating wrong_class...")
    wrong_bn = {c: class_cond_stats[(c + num_classes // 2) % num_classes] for c in range(num_classes)}
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, wrong_bn, num_classes)
    np.random.seed(seed + 2002)
    probe = linear_probe(feats, labs, num_classes)
    conditions['wrong_class'] = {
        'mean_acc': float(np.mean(list(acc.values()))),
        'mean_conf': float(np.mean(list(conf.values()))),
        'linear_probe_acc': probe,
    }
    print(f"      wrong_class: acc={conditions['wrong_class']['mean_acc']:.4f}, conf={conditions['wrong_class']['mean_conf']:.4f}, probe={probe:.4f}")

    # Random-class
    print(f"    Evaluating random_class...")
    np.random.seed(seed + 1000)
    random_bn = {}
    for c in range(num_classes):
        r = np.random.choice([j for j in range(num_classes) if j != c])
        random_bn[c] = class_cond_stats[r]
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, random_bn, num_classes)
    np.random.seed(seed + 2003)
    probe = linear_probe(feats, labs, num_classes)
    conditions['random_class'] = {
        'mean_acc': float(np.mean(list(acc.values()))),
        'mean_conf': float(np.mean(list(conf.values()))),
        'linear_probe_acc': probe,
    }
    print(f"      random_class: acc={conditions['random_class']['mean_acc']:.4f}, conf={conditions['random_class']['mean_conf']:.4f}, probe={probe:.4f}")

    return conditions


# ─────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────
def aggregate_conditions(all_seed_results):
    conditions = ['global', 'same_class', 'wrong_class', 'random_class']
    agg = {}
    for cond in conditions:
        accs = [r[cond]['mean_acc'] for r in all_seed_results]
        confs = [r[cond]['mean_conf'] for r in all_seed_results]
        probes = [r[cond]['linear_probe_acc'] for r in all_seed_results]
        agg[cond] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'confidence_mean': float(np.mean(confs)),
            'confidence_std': float(np.std(confs)),
            'linear_probe_mean': float(np.mean(probes)),
            'linear_probe_std': float(np.std(probes)),
        }
    # Statistical tests
    stat_tests = {}
    accs_s = [r['same_class']['mean_acc'] for r in all_seed_results]
    accs_g = [r['global']['mean_acc'] for r in all_seed_results]
    accs_w = [r['wrong_class']['mean_acc'] for r in all_seed_results]
    probes_s = [r['same_class']['linear_probe_acc'] for r in all_seed_results]
    probes_g = [r['global']['linear_probe_acc'] for r in all_seed_results]

    if len(all_seed_results) >= 3:
        t, p = stats.ttest_rel(accs_s, accs_g)
        stat_tests['same_vs_global_acc'] = {'t': float(t), 'p': float(p)}
        t, p = stats.ttest_rel(accs_w, accs_g)
        stat_tests['wrong_vs_global_acc'] = {'t': float(t), 'p': float(p)}
        t, p = stats.ttest_rel(accs_s, accs_w)
        stat_tests['same_vs_wrong_acc'] = {'t': float(t), 'p': float(p)}
        t, p = stats.ttest_rel(probes_s, probes_g)
        stat_tests['same_vs_global_probe'] = {'t': float(t), 'p': float(p)}

    return agg, stat_tests


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    total_start = time.time()
    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")
    print()

    print("=" * 70)
    print("EXPERIMENT 1: SmallResNet on CIFAR-100 (fixed linear probe, 5 seeds)")
    print("=" * 70)

    train_loader, test_loader, cl_train, cl_test, nc = get_cifar100_data()
    all_results = []

    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        print(f"\n--- Seed {seed} ({i+1}/{len(SEEDS)}) ---")

        torch.manual_seed(seed)
        model = SmallResNet(num_classes=nc).to(DEVICE)
        model = train_model(model, train_loader, test_loader, EPOCHS, seed, f"C100 s{seed}")

        conds = run_standard_eval(model, cl_train, cl_test, nc, seed)
        conds['seed'] = seed
        all_results.append(conds)

        elapsed = time.time() - t0
        print(f"  Seed {seed} done ({elapsed:.0f}s)")

        del model
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    agg, stat_tests = aggregate_conditions(all_results)

    results = {
        'config': {
            'seeds': SEEDS,
            'epochs': EPOCHS,
            'model': 'SmallResNet',
            'dataset': 'CIFAR-100',
            'linear_probe': 'LogisticRegression(max_iter=1000, C=1.0, solver=lbfgs, multi_class=multinomial)',
            'note': 'Fixed linear probe — replaces broken ridge regression for 100 classes',
        },
        'aggregate': agg,
        'statistical_tests': stat_tests,
        'per_seed': all_results,
        'total_time_seconds': time.time() - total_start,
    }

    out_path = OUTPUT_DIR / 'results_cifar100_fixed.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"CIFAR-100 RESULTS (fixed linear probe)")
    print("=" * 70)
    for cond in ['global', 'same_class', 'wrong_class', 'random_class']:
        a = agg[cond]
        print(f"  {cond:14s} | acc={a['accuracy_mean']:.4f}±{a['accuracy_std']:.4f} | "
              f"conf={a['confidence_mean']:.4f}±{a['confidence_std']:.4f} | "
              f"probe={a['linear_probe_mean']:.4f}±{a['linear_probe_std']:.4f}")

    print(f"\nStatistical tests:")
    for key, val in stat_tests.items():
        print(f"  {key}: t={val['t']:.4f}, p={val['p']:.6f}")

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    main()
