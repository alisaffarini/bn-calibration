#!/usr/bin/env python3
"""Experiment 4: Temperature Scaling after BN replacement.
Train SmallResNet on CIFAR-10, replace BN with same-class stats, 
fit temperature on validation set, report recovered accuracy."""

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
EPOCHS = 20
SEEDS = [42, 43, 44]
NUM_CLASSES = 10

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
        strides = [stride] + [1] * (num_blocks - 1); layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s)); self.in_planes = planes
        return nn.Sequential(*layers)
    def forward(self, x, return_features=False):
        out = F.relu(self.bn1(self.conv1(x)))
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

def compute_class_conditional_bn_stats(model, dataloader_by_class):
    class_stats = {}; original_stats = save_bn_stats(model)
    for c in range(NUM_CLASSES):
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
def get_data():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform_test)
    trainset_clean = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform_test)
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Class-conditional loaders from training set (for BN stat computation)
    targets = np.array(trainset_clean.targets)
    cl_train = {c: DataLoader(Subset(trainset_clean, np.where(targets==c)[0].tolist()), batch_size=BATCH_SIZE, shuffle=False, num_workers=0) for c in range(NUM_CLASSES)}
    
    # Split test set into val (first half) and test (second half) for temperature scaling
    test_targets = np.array(testset.targets)
    n_test = len(testset)
    np.random.seed(999)
    perm = np.random.permutation(n_test)
    val_indices = perm[:n_test//2].tolist()
    test_indices = perm[n_test//2:].tolist()
    
    val_loader = DataLoader(Subset(testset, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    final_test_loader = DataLoader(Subset(testset, test_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Class-conditional test loaders (for BN replacement eval)
    cl_test = {}
    for c in range(NUM_CLASSES):
        indices = np.where(test_targets == c)[0].tolist()
        cl_test[c] = DataLoader(Subset(testset, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Class-conditional val/test loaders
    val_targets = test_targets[val_indices]
    test_sub_targets = test_targets[test_indices]
    
    cl_val = {}
    for c in range(NUM_CLASSES):
        c_indices = [val_indices[i] for i in range(len(val_indices)) if val_targets[i] == c]
        cl_val[c] = DataLoader(Subset(testset, c_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    cl_final_test = {}
    for c in range(NUM_CLASSES):
        c_indices = [test_indices[i] for i in range(len(test_indices)) if test_sub_targets[i] == c]
        cl_final_test[c] = DataLoader(Subset(testset, c_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return train_loader, test_loader, cl_train, cl_test, cl_val, cl_final_test, val_loader, final_test_loader

# ── Training ──
def train_model(model, train_loader, test_loader, seed, label=""):
    torch.manual_seed(seed); np.random.seed(seed)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
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
            print(f"  [{label}] Epoch {epoch+1:3d}/{EPOCHS} | Train: {correct/total:.4f} | Test: {tc/tt:.4f}")
    model.eval(); return model

# ── Temperature Scaling ──
def collect_logits_with_bn_replacement(model, class_loaders, bn_stats_per_class):
    """Collect logits using class-conditional BN stats."""
    original_stats = save_bn_stats(model); model.eval()
    all_logits = []; all_labels = []
    for c in range(NUM_CLASSES):
        load_bn_stats(model, bn_stats_per_class[c])
        with torch.no_grad():
            for images, labels in class_loaders[c]:
                logits = model(images.to(DEVICE))
                all_logits.append(logits.cpu()); all_labels.extend(labels.numpy().tolist())
    load_bn_stats(model, original_stats)
    return torch.cat(all_logits, dim=0), torch.tensor(all_labels, dtype=torch.long)

def collect_logits_global(model, loader):
    """Collect logits with global (original) BN stats."""
    model.eval(); all_logits = []; all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            all_logits.append(logits.cpu()); all_labels.extend(labels.numpy().tolist())
    return torch.cat(all_logits, dim=0), torch.tensor(all_labels, dtype=torch.long)

def fit_temperature(logits, labels, lr=0.01, max_iter=500):
    """Fit temperature parameter via NLL minimization on validation set."""
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()
    
    def eval_fn():
        optimizer.zero_grad()
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_fn)
    return temperature.item()

def compute_accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

def compute_nll(logits, labels):
    return F.cross_entropy(logits, labels).item()

def compute_ece(logits, labels, num_bins=15):
    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(1)
    accuracies = (predictions == labels).float()
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i+1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        n_in_bin = mask.sum().item()
        if n_in_bin > 0:
            avg_conf = confidences[mask].mean().item()
            avg_acc = accuracies[mask].mean().item()
            ece += (n_in_bin / len(labels)) * abs(avg_acc - avg_conf)
    return ece

def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print(f"Temperature Scaling Experiment: SmallResNet on CIFAR-10, {EPOCHS} epochs, seeds {SEEDS}\n")
    
    train_loader, test_loader, cl_train, cl_test, cl_val, cl_final_test, val_loader, final_test_loader = get_data()
    
    all_results = []
    
    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({i+1}/{len(SEEDS)})")
        print(f"{'='*60}")
        
        torch.manual_seed(seed)
        model = SmallResNet(num_classes=NUM_CLASSES).to(DEVICE)
        model = train_model(model, train_loader, test_loader, seed, f"ResNet s{seed}")
        
        global_stats = save_bn_stats(model)
        print(f"  Computing class-conditional BN stats...")
        class_cond_stats = compute_class_conditional_bn_stats(model, cl_train)
        
        result = {'seed': seed}
        
        # ── Baseline (global BN stats) ──
        print(f"\n  Evaluating baseline (global stats)...")
        logits_global_test, labels_global_test = collect_logits_global(model, final_test_loader)
        result['global_baseline'] = {
            'accuracy': compute_accuracy(logits_global_test, labels_global_test),
            'nll': compute_nll(logits_global_test, labels_global_test),
            'ece': compute_ece(logits_global_test, labels_global_test),
        }
        print(f"    Global baseline: acc={result['global_baseline']['accuracy']:.4f}, "
              f"NLL={result['global_baseline']['nll']:.4f}, ECE={result['global_baseline']['ece']:.4f}")
        
        # ── Same-class BN replacement (no temperature scaling) ──
        print(f"\n  Evaluating same-class BN replacement...")
        same_bn = {c: class_cond_stats[c] for c in range(NUM_CLASSES)}
        
        # Collect val logits for temperature fitting
        logits_same_val, labels_same_val = collect_logits_with_bn_replacement(model, cl_val, same_bn)
        # Collect test logits for evaluation
        logits_same_test, labels_same_test = collect_logits_with_bn_replacement(model, cl_final_test, same_bn)
        
        result['same_class_no_tempscale'] = {
            'accuracy': compute_accuracy(logits_same_test, labels_same_test),
            'nll': compute_nll(logits_same_test, labels_same_test),
            'ece': compute_ece(logits_same_test, labels_same_test),
        }
        print(f"    Same-class (no temp): acc={result['same_class_no_tempscale']['accuracy']:.4f}, "
              f"NLL={result['same_class_no_tempscale']['nll']:.4f}, ECE={result['same_class_no_tempscale']['ece']:.4f}")
        
        # ── Fit temperature on validation set ──
        print(f"  Fitting temperature on validation set...")
        temperature = fit_temperature(logits_same_val, labels_same_val)
        print(f"    Optimal temperature: {temperature:.4f}")
        
        # ── Apply temperature scaling to test set ──
        scaled_logits = logits_same_test / temperature
        result['same_class_with_tempscale'] = {
            'accuracy': compute_accuracy(scaled_logits, labels_same_test),
            'nll': compute_nll(scaled_logits, labels_same_test),
            'ece': compute_ece(scaled_logits, labels_same_test),
            'temperature': temperature,
        }
        print(f"    Same-class (with temp={temperature:.4f}): acc={result['same_class_with_tempscale']['accuracy']:.4f}, "
              f"NLL={result['same_class_with_tempscale']['nll']:.4f}, ECE={result['same_class_with_tempscale']['ece']:.4f}")
        
        # ── Wrong-class BN replacement + temperature ──
        print(f"\n  Evaluating wrong-class BN replacement...")
        wrong_bn = {c: class_cond_stats[(c + NUM_CLASSES // 2) % NUM_CLASSES] for c in range(NUM_CLASSES)}
        logits_wrong_val, labels_wrong_val = collect_logits_with_bn_replacement(model, cl_val, wrong_bn)
        logits_wrong_test, labels_wrong_test = collect_logits_with_bn_replacement(model, cl_final_test, wrong_bn)
        
        result['wrong_class_no_tempscale'] = {
            'accuracy': compute_accuracy(logits_wrong_test, labels_wrong_test),
            'nll': compute_nll(logits_wrong_test, labels_wrong_test),
            'ece': compute_ece(logits_wrong_test, labels_wrong_test),
        }
        
        temp_wrong = fit_temperature(logits_wrong_val, labels_wrong_val)
        scaled_wrong = logits_wrong_test / temp_wrong
        result['wrong_class_with_tempscale'] = {
            'accuracy': compute_accuracy(scaled_wrong, labels_wrong_test),
            'nll': compute_nll(scaled_wrong, labels_wrong_test),
            'ece': compute_ece(scaled_wrong, labels_wrong_test),
            'temperature': temp_wrong,
        }
        print(f"    Wrong-class (no temp): acc={result['wrong_class_no_tempscale']['accuracy']:.4f}")
        print(f"    Wrong-class (with temp={temp_wrong:.4f}): acc={result['wrong_class_with_tempscale']['accuracy']:.4f}")
        
        # ── Also temp scale the global baseline for comparison ──
        logits_global_val, labels_global_val = collect_logits_global(model, val_loader)
        temp_global = fit_temperature(logits_global_val, labels_global_val)
        scaled_global = logits_global_test / temp_global
        result['global_with_tempscale'] = {
            'accuracy': compute_accuracy(scaled_global, labels_global_test),
            'nll': compute_nll(scaled_global, labels_global_test),
            'ece': compute_ece(scaled_global, labels_global_test),
            'temperature': temp_global,
        }
        print(f"    Global (with temp={temp_global:.4f}): acc={result['global_with_tempscale']['accuracy']:.4f}, "
              f"ECE={result['global_with_tempscale']['ece']:.4f}")
        
        all_results.append(result)
        elapsed = time.time() - t0
        print(f"\n  Seed {seed} done in {elapsed:.0f}s")
        
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()
    
    # ── Aggregate ──
    total_time = time.time() - t_start
    agg = {}
    for condition in ['global_baseline', 'same_class_no_tempscale', 'same_class_with_tempscale',
                      'wrong_class_no_tempscale', 'wrong_class_with_tempscale', 'global_with_tempscale']:
        accs = [r[condition]['accuracy'] for r in all_results]
        nlls = [r[condition]['nll'] for r in all_results]
        eces = [r[condition]['ece'] for r in all_results]
        agg[condition] = {
            'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'nll_mean': float(np.mean(nlls)), 'nll_std': float(np.std(nlls)),
            'ece_mean': float(np.mean(eces)), 'ece_std': float(np.std(eces)),
        }
        if 'temperature' in all_results[0][condition]:
            temps = [r[condition]['temperature'] for r in all_results]
            agg[condition]['temperature_mean'] = float(np.mean(temps))
            agg[condition]['temperature_std'] = float(np.std(temps))
    
    # Key question: does temperature scaling recover accuracy after same-class BN replacement?
    acc_recovery = {
        'global_baseline_acc': agg['global_baseline']['accuracy_mean'],
        'same_class_acc_before_temp': agg['same_class_no_tempscale']['accuracy_mean'],
        'same_class_acc_after_temp': agg['same_class_with_tempscale']['accuracy_mean'],
        'accuracy_drop_from_bn_replacement': agg['global_baseline']['accuracy_mean'] - agg['same_class_no_tempscale']['accuracy_mean'],
        'accuracy_recovered_by_temp': agg['same_class_with_tempscale']['accuracy_mean'] - agg['same_class_no_tempscale']['accuracy_mean'],
        'note': 'Temperature scaling only rescales logits — it cannot change the argmax (and thus accuracy) if it uses a single scalar. Any accuracy change comes from the val/test split randomness.',
    }
    
    output = {
        'experiment': 'Temperature Scaling after BN Replacement',
        'config': {'model': 'SmallResNet', 'dataset': 'CIFAR-10', 'num_classes': 10,
                   'epochs': EPOCHS, 'seeds': SEEDS, 'lr': LR, 'batch_size': BATCH_SIZE,
                   'val_test_split': '50/50 of CIFAR-10 test set (5000/5000)',
                   'temperature_fitting': 'LBFGS on validation NLL'},
        'aggregate': agg,
        'accuracy_recovery_analysis': acc_recovery,
        'per_seed': all_results,
        'total_time_seconds': total_time,
    }
    
    out_path = OUTPUT_DIR / 'results_tempscaling.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Temperature Scaling Results saved to {out_path}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"{'='*60}")
    
    print("\n=== Summary ===")
    for cond in ['global_baseline', 'same_class_no_tempscale', 'same_class_with_tempscale',
                 'wrong_class_no_tempscale', 'wrong_class_with_tempscale']:
        a = agg[cond]
        temp_str = f" T={a.get('temperature_mean', 0):.2f}" if 'temperature_mean' in a else ""
        print(f"  {cond:30s} | acc={a['accuracy_mean']:.4f}+-{a['accuracy_std']:.4f} | "
              f"ECE={a['ece_mean']:.4f} | NLL={a['nll_mean']:.4f}{temp_str}")
    
    print(f"\n  Key finding: Temp scaling {'DOES' if abs(acc_recovery['accuracy_recovered_by_temp']) > 0.01 else 'does NOT'} recover accuracy.")
    print(f"  (Note: single-scalar temperature cannot change argmax predictions)")

if __name__ == '__main__':
    main()
