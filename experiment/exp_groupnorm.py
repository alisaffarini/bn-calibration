#!/usr/bin/env python3
"""Experiment 2: GroupNorm Control - SmallResNet-GN on CIFAR-10, 20 epochs, 3 seeds."""

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
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class SmallResNetGN(nn.Module):
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

# ── Data ──
def get_data():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_targets = np.array(testset.targets)
    class_loaders_test = {}
    for c in range(NUM_CLASSES):
        indices = np.where(test_targets == c)[0].tolist()
        class_loaders_test[c] = DataLoader(Subset(testset, indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader, class_loaders_test

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
def evaluate(model, class_loaders_test):
    model.eval()
    per_class_acc = {}
    per_class_conf = {}
    all_features = []
    all_labels = []
    for c in range(NUM_CLASSES):
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
    all_features = torch.cat(all_features, dim=0)
    all_labels = np.array(all_labels)
    return per_class_acc, per_class_conf, all_features, all_labels

def linear_probe(features, labels, num_classes=10):
    n = len(labels)
    perm = np.random.permutation(n)
    split = int(0.8 * n)
    X_train, y_train = features[perm[:split]].numpy(), labels[perm[:split]]
    X_test, y_test = features[perm[split:]].numpy(), labels[perm[split:]]
    Y_train = np.eye(num_classes)[y_train]
    XtX = X_train.T @ X_train + 1e-3 * np.eye(X_train.shape[1])
    W = np.linalg.solve(XtX, X_train.T @ Y_train)
    return float(np.mean((X_test @ W).argmax(axis=1) == y_test))

def main():
    t_start = time.time()
    print(f"Device: {DEVICE}")
    print(f"GroupNorm Control: SmallResNet-GN on CIFAR-10, {EPOCHS} epochs, seeds {SEEDS}")
    print()

    train_loader, test_loader, cl_test = get_data()
    all_results = []

    for i, seed in enumerate(SEEDS):
        t0 = time.time()
        print(f"\n--- GroupNorm Seed {seed} ({i+1}/{len(SEEDS)}) ---")
        torch.manual_seed(seed)
        model = SmallResNetGN(num_classes=NUM_CLASSES).to(DEVICE)
        model = train_model(model, train_loader, test_loader, seed, f"GN s{seed}")
        acc, conf, feats, labs = evaluate(model, cl_test)
        np.random.seed(seed + 2000)
        probe = linear_probe(feats, labs)
        result = {
            'seed': seed,
            'accuracy': float(np.mean(list(acc.values()))),
            'confidence': float(np.mean(list(conf.values()))),
            'linear_probe_acc': probe,
            'per_class_accuracy': {str(k): v for k, v in acc.items()},
            'per_class_confidence': {str(k): v for k, v in conf.items()},
        }
        all_results.append(result)
        elapsed = time.time() - t0
        print(f"  GN s{seed}: acc={result['accuracy']:.4f}, probe={result['linear_probe_acc']:.4f} ({elapsed:.0f}s)")
        del model; gc.collect()
        if DEVICE == "mps": torch.mps.empty_cache()

    accs = [r['accuracy'] for r in all_results]
    probes = [r['linear_probe_acc'] for r in all_results]
    confs = [r['confidence'] for r in all_results]
    total_time = time.time() - t_start

    output = {
        'experiment': 'GroupNorm Control',
        'config': {'model': 'SmallResNet-GN', 'dataset': 'CIFAR-10', 'num_classes': 10,
                   'epochs': EPOCHS, 'seeds': SEEDS, 'num_groups': 8,
                   'lr': LR, 'batch_size': BATCH_SIZE, 'weight_decay': WEIGHT_DECAY},
        'aggregate': {
            'accuracy_mean': float(np.mean(accs)), 'accuracy_std': float(np.std(accs)),
            'confidence_mean': float(np.mean(confs)), 'confidence_std': float(np.std(confs)),
            'linear_probe_mean': float(np.mean(probes)), 'linear_probe_std': float(np.std(probes)),
        },
        'per_seed': all_results,
        'note': 'GroupNorm has no running statistics. No BN replacement experiment possible. This is a control showing the BN stat replacement phenomenon is specific to BatchNorm.',
        'total_time_seconds': total_time,
    }

    out_path = OUTPUT_DIR / 'results_groupnorm.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Aggregate: acc={np.mean(accs):.4f}+-{np.std(accs):.4f}, probe={np.mean(probes):.4f}+-{np.std(probes):.4f}")

if __name__ == '__main__':
    main()
