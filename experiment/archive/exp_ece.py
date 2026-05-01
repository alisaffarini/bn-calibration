#!/usr/bin/env python3
"""Experiment 3: ECE (Expected Calibration Error) for all experiments.
Computes ECE (15 bins) for all conditions across SmallResNet, VGG, SimpleCNN on CIFAR-10,
plus CIFAR-100 and GroupNorm results. Retrains models and collects logits to compute ECE."""

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
from pathlib import Path

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_BINS = 15

# ── Models (same as experiment_v2.py) ──
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
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = self.avgpool(out)
        features = out.view(out.size(0), -1)
        logits = self.fc(features)
        return (logits, features) if return_features else logits

class VGG11BN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def forward(self, x, return_features=False):
        out = self.features(x); out = self.avgpool(out)
        features = out.view(out.size(0), -1); logits = self.fc(features)
        return (logits, features) if return_features else logits

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1); self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1); self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1); self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2); self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x, return_features=False):
        out = self.pool(F.relu(self.bn1(self.conv1(x))))
        out = self.pool(F.relu(self.bn2(self.conv2(out))))
        out = self.pool(F.relu(self.bn3(self.conv3(out))))
        out = self.avgpool(out); features = out.view(out.size(0), -1); logits = self.fc(features)
        return (logits, features) if return_features else logits

class BasicBlockGN(nn.Module):
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
        out = F.relu(self.gn1(self.conv1(x))); out = self.gn2(self.conv2(out)); out += self.shortcut(x)
        return F.relu(out)

class SmallResNetGN(nn.Module):
    def __init__(self, num_classes=10, num_groups=8):
        super().__init__()
        self.in_planes = 32; self.num_groups = num_groups
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(num_groups, 32), 32)
        self.layer1 = self._make_layer(32, 2, stride=1); self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2); self.layer4 = self._make_layer(256, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)); self.fc = nn.Linear(256, num_classes)
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1); layers = []
        for s in strides:
            layers.append(BasicBlockGN(self.in_planes, planes, s, self.num_groups)); self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x, return_features=False):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = self.avgpool(out); features = out.view(out.size(0), -1); logits = self.fc(features)
        return (logits, features) if return_features else logits

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
        mod.running_mean.copy_(stats[name]['running_mean']); mod.running_var.copy_(stats[name]['running_var'])

def compute_class_conditional_bn_stats(model, dataloader_by_class, num_classes):
    class_stats = {}; original_stats = save_bn_stats(model)
    for c in range(num_classes):
        for name, mod in get_bn_layers(model):
            mod.running_mean.zero_(); mod.running_var.fill_(1.0); mod.num_batches_tracked.zero_()
        model.train()
        with torch.no_grad():
            for images, _ in dataloader_by_class[c]:
                _ = model(images.to(DEVICE))
        class_stats[c] = save_bn_stats(model)
    load_bn_stats(model, original_stats); model.eval()
    return class_stats

# ── Data ──
def get_cifar_data(dataset='cifar10'):
    if dataset == 'cifar10':
        DC = torchvision.datasets.CIFAR10; mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616); nc = 10
    else:
        DC = torchvision.datasets.CIFAR100; mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761); nc = 100
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    trainset = DC(root=str(DATA_DIR), train=True, download=True, transform=transform_train)
    testset = DC(root=str(DATA_DIR), train=False, download=True, transform=transform_test)
    trainset_clean = DC(root=str(DATA_DIR), train=True, download=False, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    targets = np.array(trainset_clean.targets)
    cl_train = {c: DataLoader(Subset(trainset_clean, np.where(targets==c)[0].tolist()), batch_size=BATCH_SIZE, shuffle=False, num_workers=0) for c in range(nc)}
    test_targets = np.array(testset.targets)
    cl_test = {c: DataLoader(Subset(testset, np.where(test_targets==c)[0].tolist()), batch_size=BATCH_SIZE, shuffle=False, num_workers=0) for c in range(nc)}
    return train_loader, test_loader, cl_train, cl_test, nc

# ── Training ──
def train_model(model, train_loader, test_loader, epochs, seed, label=""):
    torch.manual_seed(seed); np.random.seed(seed)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train(); correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(); outputs = model(images); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step()
            correct += (outputs.argmax(1) == labels).sum().item(); total += labels.size(0)
        scheduler.step()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval(); tc = tt = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    tc += (model(images).argmax(1) == labels).sum().item(); tt += labels.size(0)
            print(f"  [{label}] Epoch {epoch+1:3d}/{epochs} | Train: {correct/total:.4f} | Test: {tc/tt:.4f}")
    model.eval(); return model

# ── ECE computation ──
def compute_ece(all_probs, all_labels, num_bins=15):
    """Compute Expected Calibration Error.
    all_probs: np array of shape (N, C) - predicted probabilities
    all_labels: np array of shape (N,) - true labels
    """
    confidences = np.max(all_probs, axis=1)
    predictions = np.argmax(all_probs, axis=1)
    accuracies = (predictions == all_labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    bin_details = []
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > lo) & (confidences <= hi)
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        n_in_bin = mask.sum()
        if n_in_bin > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = accuracies[mask].mean()
            ece += (n_in_bin / len(all_labels)) * abs(avg_acc - avg_conf)
            bin_details.append({'bin': i, 'lo': float(lo), 'hi': float(hi), 'n': int(n_in_bin),
                                'avg_confidence': float(avg_conf), 'avg_accuracy': float(avg_acc),
                                'gap': float(abs(avg_acc - avg_conf))})
    return float(ece), bin_details

def collect_logits_with_bn(model, class_loaders_test, bn_stats_to_use, num_classes):
    """Collect all logits and labels using specified BN stats per class."""
    original_stats = save_bn_stats(model); model.eval()
    all_logits = []; all_labels = []
    for c in range(num_classes):
        load_bn_stats(model, bn_stats_to_use[c])
        with torch.no_grad():
            for images, labels in class_loaders_test[c]:
                images = images.to(DEVICE)
                logits = model(images)
                all_logits.append(logits.cpu()); all_labels.extend(labels.numpy().tolist())
    load_bn_stats(model, original_stats)
    all_logits = torch.cat(all_logits, dim=0)
    all_probs = F.softmax(all_logits, dim=1).numpy()
    return all_probs, np.array(all_labels)

def collect_logits_no_bn(model, class_loaders_test, num_classes):
    """Collect logits without BN replacement (for GroupNorm)."""
    model.eval(); all_logits = []; all_labels = []
    for c in range(num_classes):
        with torch.no_grad():
            for images, labels in class_loaders_test[c]:
                all_logits.append(model(images.to(DEVICE)).cpu()); all_labels.extend(labels.numpy().tolist())
    all_logits = torch.cat(all_logits, dim=0)
    return F.softmax(all_logits, dim=1).numpy(), np.array(all_labels)

# ── Main ──
def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print(f"ECE Experiment: Computing ECE (15 bins) for all models/conditions\n")
    
    results = {}
    
    # ── CIFAR-10 models ──
    train_loader, test_loader, cl_train, cl_test, nc = get_cifar_data('cifar10')
    
    configs = [
        ('SmallResNet', SmallResNet, 20, [42, 43, 44]),
        ('VGG11BN', VGG11BN, 15, [42, 43, 44]),
        ('SimpleCNN', SimpleCNN, 15, [42, 43, 44]),
    ]
    
    for model_name, ModelClass, epochs, seeds in configs:
        print(f"\n{'='*60}")
        print(f"ECE for {model_name} on CIFAR-10 ({len(seeds)} seeds)")
        print(f"{'='*60}")
        
        model_results = {'per_seed': [], 'conditions': ['global', 'same_class', 'wrong_class', 'random_class']}
        
        for seed in seeds:
            print(f"\n  Seed {seed}...")
            torch.manual_seed(seed)
            model = ModelClass(num_classes=nc).to(DEVICE)
            model = train_model(model, train_loader, test_loader, epochs, seed, f"{model_name} s{seed}")
            
            global_stats = save_bn_stats(model)
            print(f"  Computing class-conditional BN stats...")
            class_cond_stats = compute_class_conditional_bn_stats(model, cl_train, nc)
            
            seed_ece = {'seed': seed}
            
            # Global
            global_bn = {c: global_stats for c in range(nc)}
            probs, labs = collect_logits_with_bn(model, cl_test, global_bn, nc)
            ece, bins = compute_ece(probs, labs, NUM_BINS)
            seed_ece['global'] = {'ece': ece, 'bins': bins}
            print(f"    global ECE: {ece:.4f}")
            
            # Same-class
            same_bn = {c: class_cond_stats[c] for c in range(nc)}
            probs, labs = collect_logits_with_bn(model, cl_test, same_bn, nc)
            ece, bins = compute_ece(probs, labs, NUM_BINS)
            seed_ece['same_class'] = {'ece': ece, 'bins': bins}
            print(f"    same_class ECE: {ece:.4f}")
            
            # Wrong-class
            wrong_bn = {c: class_cond_stats[(c + nc // 2) % nc] for c in range(nc)}
            probs, labs = collect_logits_with_bn(model, cl_test, wrong_bn, nc)
            ece, bins = compute_ece(probs, labs, NUM_BINS)
            seed_ece['wrong_class'] = {'ece': ece, 'bins': bins}
            print(f"    wrong_class ECE: {ece:.4f}")
            
            # Random-class
            np.random.seed(seed + 1000)
            random_bn = {c: class_cond_stats[np.random.choice([j for j in range(nc) if j != c])] for c in range(nc)}
            probs, labs = collect_logits_with_bn(model, cl_test, random_bn, nc)
            ece, bins = compute_ece(probs, labs, NUM_BINS)
            seed_ece['random_class'] = {'ece': ece, 'bins': bins}
            print(f"    random_class ECE: {ece:.4f}")
            
            model_results['per_seed'].append(seed_ece)
            del model, global_stats, class_cond_stats; gc.collect()
            if DEVICE == "mps": torch.mps.empty_cache()
        
        # Aggregate
        agg = {}
        for cond in ['global', 'same_class', 'wrong_class', 'random_class']:
            eces = [s[cond]['ece'] for s in model_results['per_seed']]
            agg[cond] = {'ece_mean': float(np.mean(eces)), 'ece_std': float(np.std(eces))}
        model_results['aggregate'] = agg
        results[f'{model_name}_cifar10'] = model_results
    
    # ── CIFAR-100 ──
    print(f"\n{'='*60}")
    print(f"ECE for SmallResNet on CIFAR-100 (3 seeds)")
    print(f"{'='*60}")
    
    train_loader_100, test_loader_100, cl_train_100, cl_test_100, nc_100 = get_cifar_data('cifar100')
    c100_results = {'per_seed': [], 'conditions': ['global', 'same_class', 'wrong_class', 'random_class']}
    
    for seed in [42, 43, 44]:
        print(f"\n  Seed {seed}...")
        torch.manual_seed(seed)
        model = SmallResNet(num_classes=nc_100).to(DEVICE)
        model = train_model(model, train_loader_100, test_loader_100, 25, seed, f"C100 s{seed}")
        
        global_stats = save_bn_stats(model)
        print(f"  Computing class-conditional BN stats for 100 classes...")
        class_cond_stats = compute_class_conditional_bn_stats(model, cl_train_100, nc_100)
        
        seed_ece = {'seed': seed}
        
        global_bn = {c: global_stats for c in range(nc_100)}
        probs, labs = collect_logits_with_bn(model, cl_test_100, global_bn, nc_100)
        ece, bins = compute_ece(probs, labs, NUM_BINS)
        seed_ece['global'] = {'ece': ece, 'bins': bins}
        print(f"    global ECE: {ece:.4f}")
        
        same_bn = {c: class_cond_stats[c] for c in range(nc_100)}
        probs, labs = collect_logits_with_bn(model, cl_test_100, same_bn, nc_100)
        ece, bins = compute_ece(probs, labs, NUM_BINS)
        seed_ece['same_class'] = {'ece': ece, 'bins': bins}
        print(f"    same_class ECE: {ece:.4f}")
        
        wrong_bn = {c: class_cond_stats[(c + nc_100 // 2) % nc_100] for c in range(nc_100)}
        probs, labs = collect_logits_with_bn(model, cl_test_100, wrong_bn, nc_100)
        ece, bins = compute_ece(probs, labs, NUM_BINS)
        seed_ece['wrong_class'] = {'ece': ece, 'bins': bins}
        print(f"    wrong_class ECE: {ece:.4f}")
        
        np.random.seed(seed + 1000)
        random_bn = {c: class_cond_stats[np.random.choice([j for j in range(nc_100) if j != c])] for c in range(nc_100)}
        probs, labs = collect_logits_with_bn(model, cl_test_100, random_bn, nc_100)
        ece, bins = compute_ece(probs, labs, NUM_BINS)
        seed_ece['random_class'] = {'ece': ece, 'bins': bins}
        print(f"    random_class ECE: {ece:.4f}")
        
        c100_results['per_seed'].append(seed_ece)
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()
    
    agg = {}
    for cond in ['global', 'same_class', 'wrong_class', 'random_class']:
        eces = [s[cond]['ece'] for s in c100_results['per_seed']]
        agg[cond] = {'ece_mean': float(np.mean(eces)), 'ece_std': float(np.std(eces))}
    c100_results['aggregate'] = agg
    results['SmallResNet_cifar100'] = c100_results
    
    # ── GroupNorm (baseline only — no BN replacement) ──
    print(f"\n{'='*60}")
    print(f"ECE for SmallResNet-GN on CIFAR-10 (3 seeds)")
    print(f"{'='*60}")
    
    train_loader, test_loader, cl_train, cl_test, nc = get_cifar_data('cifar10')
    gn_results = {'per_seed': []}
    
    for seed in [42, 43, 44]:
        print(f"\n  Seed {seed}...")
        torch.manual_seed(seed)
        model = SmallResNetGN(num_classes=nc).to(DEVICE)
        model = train_model(model, train_loader, test_loader, 20, seed, f"GN s{seed}")
        
        probs, labs = collect_logits_no_bn(model, cl_test, nc)
        ece, bins = compute_ece(probs, labs, NUM_BINS)
        gn_results['per_seed'].append({'seed': seed, 'baseline': {'ece': ece, 'bins': bins}})
        print(f"    baseline ECE: {ece:.4f}")
        
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()
    
    eces = [s['baseline']['ece'] for s in gn_results['per_seed']]
    gn_results['aggregate'] = {'baseline': {'ece_mean': float(np.mean(eces)), 'ece_std': float(np.std(eces))}}
    gn_results['note'] = 'GroupNorm has no running stats — only baseline ECE reported'
    results['SmallResNetGN_cifar10'] = gn_results
    
    total_time = time.time() - t_start
    results['config'] = {'num_bins': NUM_BINS, 'total_time_seconds': total_time}
    
    out_path = OUTPUT_DIR / 'results_ece.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"ECE Results saved to {out_path}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    # Summary
    print("\n=== ECE Summary ===")
    for key in results:
        if key == 'config': continue
        print(f"\n{key}:")
        if 'aggregate' in results[key]:
            for cond, vals in results[key]['aggregate'].items():
                if isinstance(vals, dict) and 'ece_mean' in vals:
                    print(f"  {cond:14s} ECE: {vals['ece_mean']:.4f} +- {vals['ece_std']:.4f}")

if __name__ == '__main__':
    main()
