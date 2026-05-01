#!/usr/bin/env python3
"""Experiment 1: CIFAR-100 Replication - SmallResNet, 25 epochs, 3 seeds, 4 BN conditions."""

import json
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from scipy import stats
from pathlib import Path

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
EPOCHS = 25
SEEDS = [42, 43, 44]
NUM_CLASSES = 100

# ── Model ──
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
    def __init__(self, num_classes=100):
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

# ── BN utilities ──
def get_bn_layers(model):
    return [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.BatchNorm2d)]

def save_bn_stats(model):
    stats = {}
    for name, mod in get_bn_layers(model):
        stats[name] = {'running_mean': mod.running_mean.clone(), 'running_var': mod.running_var.clone()}
    return stats

def load_bn_stats(model, stats):
    for name, mod in get_bn_layers(model):
        mod.running_mean.copy_(stats[name]['running_mean'])
        mod.running_var.copy_(stats[name]['running_var'])

def compute_class_conditional_bn_stats(model, dataloader_by_class):
    class_stats = {}
    original_stats = save_bn_stats(model)
    for c in range(NUM_CLASSES):
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

# ── Data ──
def get_data():
    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=False, download=True, transform=transform_test)
    trainset_clean = torchvision.datasets.CIFAR100(root=str(DATA_DIR), train=True, download=False, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    targets = np.array(trainset_clean.targets)
    class_loaders_train = {}
    for c in range(NUM_CLASSES):
        indices = np.where(targets == c)[0].tolist()
        class_loaders_train[c] = DataLoader(Subset(trainset_clean, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_targets = np.array(testset.targets)
    class_loaders_test = {}
    for c in range(NUM_CLASSES):
        indices = np.where(test_targets == c)[0].tolist()
        class_loaders_test[c] = DataLoader(Subset(testset, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader, class_loaders_train, class_loaders_test

# ── Training ──
def train_model(model, train_loader, test_loader, seed, label=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        correct = total = 0
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
            tc = tt = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    tc += (outputs.argmax(1) == labels).sum().item()
                    tt += labels.size(0)
            print(f"  [{label}] Epoch {epoch+1:3d}/{EPOCHS} | Train: {correct/total:.4f} | Test: {tc/tt:.4f}")
    model.eval()
    return model

# ── Evaluation ──
def evaluate_with_bn_stats(model, class_loaders_test, bn_stats_to_use):
    original_stats = save_bn_stats(model)
    model.eval()
    per_class_acc = {}
    per_class_conf = {}
    all_features = []
    all_labels = []
    for c in range(NUM_CLASSES):
        load_bn_stats(model, bn_stats_to_use[c])
        correct = total = 0
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
        per_class_conf[c] = float(np.mean(confidences)) if confidences else 0.0
        all_features.append(torch.cat(class_features, dim=0))
    load_bn_stats(model, original_stats)
    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)
    return per_class_acc, per_class_conf, all_features, all_labels

def linear_probe(features, labels, num_classes=100):
    n = len(labels)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = perm[:split], perm[split:]
    X_train, y_train = features[train_idx].numpy(), labels[train_idx]
    X_test, y_test = features[test_idx].numpy(), labels[test_idx]
    Y_train = np.eye(num_classes)[y_train]
    XtX = X_train.T @ X_train + 1e-3 * np.eye(X_train.shape[1])
    XtY = X_train.T @ Y_train
    W = np.linalg.solve(XtX, XtY)
    preds = X_test @ W
    return float(np.mean(preds.argmax(axis=1) == y_test))

def run_standard_eval(model, class_loaders_train, class_loaders_test, seed):
    global_stats = save_bn_stats(model)
    print(f"  Computing class-conditional BN stats for {NUM_CLASSES} classes...")
    class_cond_stats = compute_class_conditional_bn_stats(model, class_loaders_train)
    conditions = {}
    # Global
    global_bn = {c: global_stats for c in range(NUM_CLASSES)}
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, global_bn)
    np.random.seed(seed + 2000)
    probe = linear_probe(feats, labs)
    conditions['global'] = {'mean_acc': float(np.mean(list(acc.values()))), 'mean_conf': float(np.mean(list(conf.values()))), 'linear_probe_acc': probe}
    print(f"    global: acc={conditions['global']['mean_acc']:.4f}")
    # Same-class
    same_bn = {c: class_cond_stats[c] for c in range(NUM_CLASSES)}
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, same_bn)
    np.random.seed(seed + 2001)
    probe = linear_probe(feats, labs)
    conditions['same_class'] = {'mean_acc': float(np.mean(list(acc.values()))), 'mean_conf': float(np.mean(list(conf.values()))), 'linear_probe_acc': probe}
    print(f"    same_class: acc={conditions['same_class']['mean_acc']:.4f}")
    # Wrong-class
    wrong_bn = {c: class_cond_stats[(c + NUM_CLASSES // 2) % NUM_CLASSES] for c in range(NUM_CLASSES)}
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, wrong_bn)
    np.random.seed(seed + 2002)
    probe = linear_probe(feats, labs)
    conditions['wrong_class'] = {'mean_acc': float(np.mean(list(acc.values()))), 'mean_conf': float(np.mean(list(conf.values()))), 'linear_probe_acc': probe}
    print(f"    wrong_class: acc={conditions['wrong_class']['mean_acc']:.4f}")
    # Random-class
    np.random.seed(seed + 1000)
    random_bn = {}
    for c in range(NUM_CLASSES):
        r = np.random.choice([j for j in range(NUM_CLASSES) if j != c])
        random_bn[c] = class_cond_stats[r]
    acc, conf, feats, labs = evaluate_with_bn_stats(model, class_loaders_test, random_bn)
    np.random.seed(seed + 2003)
    probe = linear_probe(feats, labs)
    conditions['random_class'] = {'mean_acc': float(np.mean(list(acc.values()))), 'mean_conf': float(np.mean(list(conf.values()))), 'linear_probe_acc': probe}
    print(f"    random_class: acc={conditions['random_class']['mean_acc']:.4f}")
    return conditions

def aggregate_conditions(all_seed_results):
    conds_list = ['global', 'same_class', 'wrong_class', 'random_class']
    agg = {}
    for cond in conds_list:
        accs = [r[cond]['mean_acc'] for r in all_seed_results]
        confs = [r[cond]['mean_conf'] for r in all_seed_results]
        probes = [r[cond]['linear_probe_acc'] for r in all_seed_results]
        agg[cond] = {
            'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'confidence_mean': float(np.mean(confs)), 'confidence_std': float(np.std(confs)),
            'linear_probe_mean': float(np.mean(probes)), 'linear_probe_std': float(np.std(probes)),
        }
    stat_tests = {}
    if len(all_seed_results) >= 3:
        accs_s = [r['same_class']['mean_acc'] for r in all_seed_results]
        accs_g = [r['global']['mean_acc'] for r in all_seed_results]
        accs_w = [r['wrong_class']['mean_acc'] for r in all_seed_results]
        probes_s = [r['same_class']['linear_probe_acc'] for r in all_seed_results]
        probes_g = [r['global']['linear_probe_acc'] for r in all_seed_results]
        t, p = stats.ttest_rel(accs_s, accs_g); stat_tests['same_vs_global_acc'] = {'t': float(t), 'p': float(p)}
        t, p = stats.ttest_rel(accs_w, accs_g); stat_tests['wrong_vs_global_acc'] = {'t': float(t), 'p': float(p)}
        t, p = stats.ttest_rel(accs_s, accs_w); stat_tests['same_vs_wrong_acc'] = {'t': float(t), 'p': float(p)}
        t, p = stats.ttest_rel(probes_s, probes_g); stat_tests['same_vs_global_probe'] = {'t': float(t), 'p': float(p)}
    return agg, stat_tests

def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print(f"CIFAR-100 Replication: SmallResNet, {EPOCHS} epochs, seeds {SEEDS}")
    print()

    train_loader, test_loader, cl_train, cl_test = get_data()
    all_results = []

    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'='*60}")
        torch.manual_seed(seed)
        model = SmallResNet(num_classes=NUM_CLASSES).to(DEVICE)
        model = train_model(model, train_loader, test_loader, seed, f"C100 s{seed}")
        conds = run_standard_eval(model, cl_train, cl_test, seed)
        conds['seed'] = seed
        all_results.append(conds)
        elapsed = time.time() - t0
        print(f"\n  Seed {seed} done in {elapsed:.0f}s")
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()

    agg, stat_tests = aggregate_conditions(all_results)
    total_time = time.time() - t_start

    output = {
        'experiment': 'CIFAR-100 Replication',
        'config': {'model': 'SmallResNet', 'dataset': 'CIFAR-100', 'num_classes': 100, 'epochs': EPOCHS, 'seeds': SEEDS,
                   'lr': LR, 'batch_size': BATCH_SIZE, 'weight_decay': WEIGHT_DECAY},
        'aggregate': agg,
        'statistical_tests': stat_tests,
        'per_seed': all_results,
        'total_time_seconds': total_time,
    }

    out_path = OUTPUT_DIR / 'results_cifar100.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    for cond in ['global', 'same_class', 'wrong_class', 'random_class']:
        a = agg[cond]
        print(f"  {cond:14s} | acc={a['accuracy_mean']:.4f}+-{a['accuracy_std']:.4f} | probe={a['linear_probe_mean']:.4f}+-{a['linear_probe_std']:.4f}")

if __name__ == '__main__':
    main()
