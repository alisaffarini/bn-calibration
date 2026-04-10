#!/usr/bin/env python3
"""
Experiment V2: Expanded analysis of BN statistics as output calibrators.
Adds: multiple architectures, interpolation, GroupNorm control, CIFAR-100, 10 seeds.
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

# Epoch counts (reduced from 25 to keep under 3h total)
EPOCHS_MAIN = 20        # SmallResNet CIFAR-10, 10 seeds
EPOCHS_ARCH = 15        # VGG/SimpleCNN (3 seeds each)
EPOCHS_INTERP = 20      # Interpolation (3 seeds, reuses main training)
EPOCHS_GN = 20          # GroupNorm variant (3 seeds)
EPOCHS_C100 = 25        # CIFAR-100 (harder, needs more epochs)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

ALPHA_VALUES = [round(a * 0.1, 1) for a in range(11)]  # 0.0, 0.1, ..., 1.0

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


class VGG11BN(nn.Module):
    """VGG-11 with BatchNorm for CIFAR (smaller FC layers)."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 5
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x, return_features=False):
        out = self.features(x)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


class SimpleCNN(nn.Module):
    """3-layer CNN with BatchNorm — simple baseline."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x, return_features=False):
        out = self.pool(F.relu(self.bn1(self.conv1(x))))   # 32->16
        out = self.pool(F.relu(self.bn2(self.conv2(out))))  # 16->8
        out = self.pool(F.relu(self.bn3(self.conv3(out))))  # 8->4
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        logits = self.fc(features)
        if return_features:
            return logits, features
        return logits


class BasicBlockGN(nn.Module):
    """ResNet basic block with GroupNorm instead of BatchNorm."""
    def __init__(self, in_planes, planes, stride=1, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, planes), planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(num_groups, planes), planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.GroupNorm(min(num_groups, planes), planes),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class SmallResNetGN(nn.Module):
    """SmallResNet with GroupNorm — no running stats to replace."""
    def __init__(self, num_classes=10, num_groups=8):
        super().__init__()
        self.in_planes = 32
        self.num_groups = num_groups
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, 32), 32)
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
            layers.append(BasicBlockGN(self.in_planes, planes, s, self.num_groups))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = F.relu(self.gn1(self.conv1(x)))
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


def interpolate_bn_stats(stats_a, stats_b, alpha):
    """Return interpolated stats: alpha * stats_a + (1-alpha) * stats_b."""
    result = {}
    for name in stats_a:
        result[name] = {
            'running_mean': alpha * stats_a[name]['running_mean'] + (1 - alpha) * stats_b[name]['running_mean'],
            'running_var': alpha * stats_a[name]['running_var'] + (1 - alpha) * stats_b[name]['running_var'],
        }
    return result


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
def get_cifar_data(dataset='cifar10'):
    """Load CIFAR-10 or CIFAR-100."""
    if dataset == 'cifar10':
        DataClass = torchvision.datasets.CIFAR10
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        num_classes = 10
    else:
        DataClass = torchvision.datasets.CIFAR100
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

    trainset = DataClass(root=str(DATA_DIR), train=True, download=True, transform=transform_train)
    testset = DataClass(root=str(DATA_DIR), train=False, download=True, transform=transform_test)
    trainset_clean = DataClass(root=str(DATA_DIR), train=True, download=False, transform=transform_test)

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


def evaluate_no_bn_replacement(model, class_loaders_test, num_classes):
    """Evaluate without BN stat replacement (for GroupNorm models)."""
    model.eval()
    per_class_acc = {}
    per_class_conf = {}
    all_features = []
    all_labels = []

    for c in range(num_classes):
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

    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)
    return per_class_acc, per_class_conf, all_features, all_labels


def linear_probe(features, labels, num_classes=10):
    n = len(labels)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]
    X_train = features[train_idx].numpy()
    y_train = labels[train_idx]
    X_test = features[test_idx].numpy()
    y_test = labels[test_idx]
    lam = 1e-3
    Y_train = np.eye(num_classes)[y_train]
    XtX = X_train.T @ X_train + lam * np.eye(X_train.shape[1])
    XtY = X_train.T @ Y_train
    W = np.linalg.solve(XtX, XtY)
    preds = X_test @ W
    acc = np.mean(preds.argmax(axis=1) == y_test)
    return float(acc)


def run_standard_eval(model, class_loaders_train, class_loaders_test, num_classes, seed):
    """Run the 4-condition evaluation (global, same, wrong, random). Returns dict."""
    global_stats = save_bn_stats(model)
    class_cond_stats = compute_class_conditional_bn_stats(model, class_loaders_train, num_classes)

    conditions = {}

    # Global
    global_bn = {c: global_stats for c in range(num_classes)}
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, global_bn, num_classes)
    np.random.seed(seed + 2000)
    probe = linear_probe(feats, labs, num_classes)
    conditions['global'] = {
        'mean_acc': float(np.mean(list(acc.values()))),
        'mean_conf': float(np.mean(list(conf.values()))),
        'linear_probe_acc': probe,
    }

    # Same-class
    same_bn = {c: class_cond_stats[c] for c in range(num_classes)}
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, same_bn, num_classes)
    np.random.seed(seed + 2001)
    probe = linear_probe(feats, labs, num_classes)
    conditions['same_class'] = {
        'mean_acc': float(np.mean(list(acc.values()))),
        'mean_conf': float(np.mean(list(conf.values()))),
        'linear_probe_acc': probe,
    }

    # Wrong-class
    wrong_bn = {c: class_cond_stats[(c + num_classes // 2) % num_classes] for c in range(num_classes)}
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, wrong_bn, num_classes)
    np.random.seed(seed + 2002)
    probe = linear_probe(feats, labs, num_classes)
    conditions['wrong_class'] = {
        'mean_acc': float(np.mean(list(acc.values()))),
        'mean_conf': float(np.mean(list(conf.values()))),
        'linear_probe_acc': probe,
    }

    # Random-class
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

    return conditions, global_stats, class_cond_stats


def run_interpolation_eval(model, class_loaders_test, global_stats, class_cond_stats, num_classes, seed):
    """Evaluate accuracy and linear probe for alpha in [0, 0.1, ..., 1.0].
    alpha=1.0 means pure global stats, alpha=0.0 means pure class-conditional stats."""
    results = {}
    for alpha in ALPHA_VALUES:
        interp_bn = {}
        for c in range(num_classes):
            interp_bn[c] = interpolate_bn_stats(global_stats, class_cond_stats[c], alpha)
        acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, interp_bn, num_classes)
        np.random.seed(seed + 3000 + int(alpha * 10))
        probe = linear_probe(feats, labs, num_classes)
        results[str(alpha)] = {
            'mean_acc': float(np.mean(list(acc.values()))),
            'mean_conf': float(np.mean(list(conf.values()))),
            'linear_probe_acc': probe,
        }
        print(f"    alpha={alpha:.1f} | acc={results[str(alpha)]['mean_acc']:.4f} | probe={results[str(alpha)]['linear_probe_acc']:.4f}")
    return results


# ─────────────────────────────────────────────
# Aggregation utilities
# ─────────────────────────────────────────────
def aggregate_conditions(all_seed_results):
    """Aggregate across seeds for each condition."""
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


def aggregate_interpolation(all_interp_results):
    """Aggregate interpolation results across seeds."""
    agg = {}
    for alpha_str in [str(a) for a in ALPHA_VALUES]:
        accs = [r[alpha_str]['mean_acc'] for r in all_interp_results]
        probes = [r[alpha_str]['linear_probe_acc'] for r in all_interp_results]
        agg[alpha_str] = {
            'accuracy_mean': float(np.mean(accs)),
            'accuracy_std': float(np.std(accs)),
            'linear_probe_mean': float(np.mean(probes)),
            'linear_probe_std': float(np.std(probes)),
        }
    return agg


# ─────────────────────────────────────────────
# Main Experiments
# ─────────────────────────────────────────────
def main():
    total_start = time.time()
    results = {}

    print(f"Device: {DEVICE}")
    print(f"Data dir: {DATA_DIR}")
    print()

    # ═══════════════════════════════════════════
    # EXPERIMENT 1: SmallResNet on CIFAR-10 (10 seeds)
    # ═══════════════════════════════════════════
    print("=" * 70)
    print("EXPERIMENT 1: SmallResNet on CIFAR-10 (10 seeds)")
    print("=" * 70)

    train_loader, test_loader, cl_train, cl_test, nc = get_cifar_data('cifar10')
    seeds_main = list(range(42, 52))
    resnet_results = []
    interp_results = []

    for i, seed in enumerate(seeds_main):
        t0 = time.time()
        print(f"\n--- Seed {seed} ({i+1}/10) ---")

        torch.manual_seed(seed)
        model = SmallResNet(num_classes=nc).to(DEVICE)
        model = train_model(model, train_loader, test_loader, EPOCHS_MAIN, seed, f"ResNet s{seed}")

        conds, global_stats, class_cond_stats = run_standard_eval(model, cl_train, cl_test, nc, seed)
        conds['seed'] = seed
        resnet_results.append(conds)

        # Interpolation for first 3 seeds
        if i < 3:
            print(f"  Running interpolation experiment...")
            interp = run_interpolation_eval(model, cl_test, global_stats, class_cond_stats, nc, seed)
            interp_results.append({'seed': seed, 'alphas': interp})

        elapsed = time.time() - t0
        print(f"  Seed {seed}: acc_global={conds['global']['mean_acc']:.4f}, "
              f"acc_same={conds['same_class']['mean_acc']:.4f}, "
              f"probe_same={conds['same_class']['linear_probe_acc']:.4f} "
              f"({elapsed:.0f}s)")

        del model, global_stats, class_cond_stats
        gc.collect()
        if DEVICE == "mps":
            torch.mps.empty_cache()

    resnet_agg, resnet_stats = aggregate_conditions(resnet_results)
    interp_agg = aggregate_interpolation([r['alphas'] for r in interp_results])

    results['experiment1_smallresnet_cifar10'] = {
        'config': {'seeds': seeds_main, 'epochs': EPOCHS_MAIN, 'model': 'SmallResNet', 'dataset': 'CIFAR-10'},
        'aggregate': resnet_agg,
        'statistical_tests': resnet_stats,
        'per_seed': resnet_results,
    }
    results['experiment3_interpolation'] = {
        'config': {'seeds': [42, 43, 44], 'model': 'SmallResNet', 'dataset': 'CIFAR-10'},
        'aggregate': interp_agg,
        'per_seed': interp_results,
    }

    print(f"\n  SmallResNet CIFAR-10 aggregate:")
    for cond in ['global', 'same_class', 'wrong_class', 'random_class']:
        a = resnet_agg[cond]
        print(f"    {cond:14s} | acc={a['accuracy_mean']:.4f}±{a['accuracy_std']:.4f} | "
              f"probe={a['linear_probe_mean']:.4f}±{a['linear_probe_std']:.4f}")

    # Save incremental results
    _save_results(results)

    # ═══════════════════════════════════════════
    # EXPERIMENT 2a: VGG-11 with BN on CIFAR-10 (3 seeds)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 2a: VGG-11 with BN on CIFAR-10 (3 seeds)")
    print("=" * 70)

    vgg_results = []
    for i, seed in enumerate([42, 43, 44]):
        t0 = time.time()
        print(f"\n--- VGG-11 Seed {seed} ({i+1}/3) ---")
        torch.manual_seed(seed)
        model = VGG11BN(num_classes=nc).to(DEVICE)
        model = train_model(model, train_loader, test_loader, EPOCHS_ARCH, seed, f"VGG s{seed}")
        conds, _, _ = run_standard_eval(model, cl_train, cl_test, nc, seed)
        conds['seed'] = seed
        vgg_results.append(conds)
        elapsed = time.time() - t0
        print(f"  VGG s{seed}: acc_global={conds['global']['mean_acc']:.4f}, "
              f"acc_same={conds['same_class']['mean_acc']:.4f}, "
              f"probe_same={conds['same_class']['linear_probe_acc']:.4f} ({elapsed:.0f}s)")
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()

    vgg_agg, vgg_stats = aggregate_conditions(vgg_results)
    results['experiment2a_vgg11bn_cifar10'] = {
        'config': {'seeds': [42,43,44], 'epochs': EPOCHS_ARCH, 'model': 'VGG-11-BN', 'dataset': 'CIFAR-10'},
        'aggregate': vgg_agg,
        'statistical_tests': vgg_stats,
        'per_seed': vgg_results,
    }
    _save_results(results)

    # ═══════════════════════════════════════════
    # EXPERIMENT 2b: Simple CNN on CIFAR-10 (3 seeds)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 2b: Simple CNN on CIFAR-10 (3 seeds)")
    print("=" * 70)

    cnn_results = []
    for i, seed in enumerate([42, 43, 44]):
        t0 = time.time()
        print(f"\n--- SimpleCNN Seed {seed} ({i+1}/3) ---")
        torch.manual_seed(seed)
        model = SimpleCNN(num_classes=nc).to(DEVICE)
        model = train_model(model, train_loader, test_loader, EPOCHS_ARCH, seed, f"CNN s{seed}")
        conds, _, _ = run_standard_eval(model, cl_train, cl_test, nc, seed)
        conds['seed'] = seed
        cnn_results.append(conds)
        elapsed = time.time() - t0
        print(f"  CNN s{seed}: acc_global={conds['global']['mean_acc']:.4f}, "
              f"acc_same={conds['same_class']['mean_acc']:.4f}, "
              f"probe_same={conds['same_class']['linear_probe_acc']:.4f} ({elapsed:.0f}s)")
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()

    cnn_agg, cnn_stats = aggregate_conditions(cnn_results)
    results['experiment2b_simplecnn_cifar10'] = {
        'config': {'seeds': [42,43,44], 'epochs': EPOCHS_ARCH, 'model': 'SimpleCNN', 'dataset': 'CIFAR-10'},
        'aggregate': cnn_agg,
        'statistical_tests': cnn_stats,
        'per_seed': cnn_results,
    }
    _save_results(results)

    # ═══════════════════════════════════════════
    # EXPERIMENT 4: GroupNorm control (3 seeds)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: SmallResNet with GroupNorm (control, 3 seeds)")
    print("=" * 70)

    gn_results = []
    for i, seed in enumerate([42, 43, 44]):
        t0 = time.time()
        print(f"\n--- GroupNorm Seed {seed} ({i+1}/3) ---")
        torch.manual_seed(seed)
        model = SmallResNetGN(num_classes=nc).to(DEVICE)
        model = train_model(model, train_loader, test_loader, EPOCHS_GN, seed, f"GN s{seed}")

        # For GN, we evaluate normally (no BN stats to replace)
        acc, conf, feats, labs = evaluate_no_bn_replacement(model, cl_test, nc)
        np.random.seed(seed + 2000)
        probe = linear_probe(feats, labs, nc)
        gn_conds = {
            'seed': seed,
            'global': {
                'mean_acc': float(np.mean(list(acc.values()))),
                'mean_conf': float(np.mean(list(conf.values()))),
                'linear_probe_acc': probe,
            },
            'note': 'GroupNorm has no running stats — cannot perform BN stat replacement'
        }
        gn_results.append(gn_conds)
        elapsed = time.time() - t0
        print(f"  GN s{seed}: acc={gn_conds['global']['mean_acc']:.4f}, "
              f"probe={gn_conds['global']['linear_probe_acc']:.4f} ({elapsed:.0f}s)")
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()

    gn_accs = [r['global']['mean_acc'] for r in gn_results]
    gn_probes = [r['global']['linear_probe_acc'] for r in gn_results]
    results['experiment4_groupnorm_control'] = {
        'config': {'seeds': [42,43,44], 'epochs': EPOCHS_GN, 'model': 'SmallResNet-GN', 'dataset': 'CIFAR-10'},
        'aggregate': {
            'accuracy_mean': float(np.mean(gn_accs)),
            'accuracy_std': float(np.std(gn_accs)),
            'linear_probe_mean': float(np.mean(gn_probes)),
            'linear_probe_std': float(np.std(gn_probes)),
        },
        'per_seed': gn_results,
        'note': 'GroupNorm has no running stats to replace. This is a control showing the phenomenon is BN-specific.'
    }
    _save_results(results)

    # ═══════════════════════════════════════════
    # EXPERIMENT 5: SmallResNet on CIFAR-100 (3 seeds)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: SmallResNet on CIFAR-100 (3 seeds)")
    print("=" * 70)

    train_loader_100, test_loader_100, cl_train_100, cl_test_100, nc_100 = get_cifar_data('cifar100')
    c100_results = []
    for i, seed in enumerate([42, 43, 44]):
        t0 = time.time()
        print(f"\n--- CIFAR-100 Seed {seed} ({i+1}/3) ---")
        torch.manual_seed(seed)
        model = SmallResNet(num_classes=nc_100).to(DEVICE)
        model = train_model(model, train_loader_100, test_loader_100, EPOCHS_C100, seed, f"C100 s{seed}")

        conds, _, _ = run_standard_eval(model, cl_train_100, cl_test_100, nc_100, seed)
        conds['seed'] = seed
        c100_results.append(conds)
        elapsed = time.time() - t0
        print(f"  C100 s{seed}: acc_global={conds['global']['mean_acc']:.4f}, "
              f"acc_same={conds['same_class']['mean_acc']:.4f}, "
              f"probe_same={conds['same_class']['linear_probe_acc']:.4f} ({elapsed:.0f}s)")
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()

    c100_agg, c100_stats = aggregate_conditions(c100_results)
    results['experiment5_smallresnet_cifar100'] = {
        'config': {'seeds': [42,43,44], 'epochs': EPOCHS_C100, 'model': 'SmallResNet', 'dataset': 'CIFAR-100'},
        'aggregate': c100_agg,
        'statistical_tests': c100_stats,
        'per_seed': c100_results,
    }
    _save_results(results)

    # ═══════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════
    total_elapsed = time.time() - total_start
    results['total_time_seconds'] = total_elapsed

    _save_results(results)

    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETE ({total_elapsed/60:.1f} minutes)")
    print("=" * 70)

    # Print summary table
    print("\n=== SUMMARY: Classification Accuracy / Linear Probe ===\n")
    experiments = [
        ('SmallResNet CIFAR-10 (10 seeds)', 'experiment1_smallresnet_cifar10'),
        ('VGG-11-BN CIFAR-10 (3 seeds)', 'experiment2a_vgg11bn_cifar10'),
        ('SimpleCNN CIFAR-10 (3 seeds)', 'experiment2b_simplecnn_cifar10'),
        ('SmallResNet CIFAR-100 (3 seeds)', 'experiment5_smallresnet_cifar100'),
    ]
    for label, key in experiments:
        a = results[key]['aggregate']
        print(f"\n{label}:")
        for cond in ['global', 'same_class', 'wrong_class', 'random_class']:
            d = a[cond]
            print(f"  {cond:14s} | acc={d['accuracy_mean']:.4f}±{d['accuracy_std']:.4f} | "
                  f"probe={d['linear_probe_mean']:.4f}±{d['linear_probe_std']:.4f}")

    print(f"\nGroupNorm control:")
    ga = results['experiment4_groupnorm_control']['aggregate']
    print(f"  accuracy={ga['accuracy_mean']:.4f}±{ga['accuracy_std']:.4f} | "
          f"probe={ga['linear_probe_mean']:.4f}±{ga['linear_probe_std']:.4f}")
    print(f"  (No BN stats to replace — phenomenon is BN-specific)")

    print(f"\nInterpolation (alpha=1.0 is global, alpha=0.0 is class-conditional):")
    for alpha_str in [str(a) for a in ALPHA_VALUES]:
        d = results['experiment3_interpolation']['aggregate'][alpha_str]
        print(f"  alpha={alpha_str:>3s} | acc={d['accuracy_mean']:.4f}±{d['accuracy_std']:.4f} | "
              f"probe={d['linear_probe_mean']:.4f}±{d['linear_probe_std']:.4f}")

    print(f"\nResults saved to {OUTPUT_DIR / 'results_v2.json'}")


def _save_results(results):
    """Save results incrementally."""
    path = OUTPUT_DIR / 'results_v2.json'
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    main()
