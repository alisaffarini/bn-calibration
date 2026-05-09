"""Disentangle μ vs σ² contributions in same-class BN substitution.

Trains a single SmallResNet on CIFAR-10 (1 seed), computes class-conditional
BN buffers, then runs four conditions:
- global (baseline)
- both: load (μ_c, σ_c²) — the standard same-class condition
- mu_only: load μ_c, keep σ²_global
- sigma_only: load σ_c², keep μ_global

The algebraic argument predicts μ-substitution alone produces most of the effect
(it is the cancellation of a^(c) - μ_c that zeros the class signal).
"""
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
EPOCHS = 20
BATCH_SIZE = 128

OUTPUT_DIR = Path(__file__).parent
RESULTS_DIR = OUTPUT_DIR.parent / "results"
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


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
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = self.layer4(out)
        out = self.avgpool(out)
        return self.fc(out.view(out.size(0), -1))


def get_bn_layers(model):
    return [(name, mod) for name, mod in model.named_modules()
            if isinstance(mod, nn.BatchNorm2d)]


def save_bn_stats(model):
    return {name: {"mean": mod.running_mean.clone(),
                   "var":  mod.running_var.clone()}
            for name, mod in get_bn_layers(model)}


def load_bn_stats(model, stats, mode="both"):
    """mode: 'both', 'mu_only', 'sigma_only'."""
    for name, mod in get_bn_layers(model):
        if mode in ("both", "mu_only"):
            mod.running_mean.copy_(stats[name]["mean"])
        if mode in ("both", "sigma_only"):
            mod.running_var.copy_(stats[name]["var"])


def compute_class_conditional_stats(model, class_loaders, num_classes):
    class_stats = {}
    original = save_bn_stats(model)
    for c in range(num_classes):
        for _, mod in get_bn_layers(model):
            mod.running_mean.zero_()
            mod.running_var.fill_(1.0)
            mod.num_batches_tracked.zero_()
        model.train()
        with torch.no_grad():
            for images, _ in class_loaders[c]:
                _ = model(images.to(DEVICE))
        class_stats[c] = save_bn_stats(model)
    # restore globals
    for name, mod in get_bn_layers(model):
        mod.running_mean.copy_(original[name]["mean"])
        mod.running_var.copy_(original[name]["var"])
    model.eval()
    return class_stats, original


def get_data():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform_test)
    trainset_clean = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    targets = np.array(trainset_clean.targets)
    class_loaders = {c: DataLoader(Subset(trainset_clean, np.where(targets == c)[0].tolist()),
                                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
                     for c in range(10)}
    test_targets = np.array(testset.targets)
    class_test_loaders = {c: DataLoader(Subset(testset, np.where(test_targets == c)[0].tolist()),
                                         batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
                          for c in range(10)}
    return train_loader, test_loader, class_loaders, class_test_loaders


def train_model(model, train_loader, test_loader):
    torch.manual_seed(SEED); np.random.seed(SEED)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        if (epoch+1) % 5 == 0 or epoch == 0 or epoch == EPOCHS-1:
            model.eval()
            tc = tt = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    out = model(images.to(DEVICE, non_blocking=True))
                    tc += (out.argmax(1) == labels.to(DEVICE)).sum().item()
                    tt += labels.size(0)
            print(f"  Epoch {epoch+1}/{EPOCHS}  train={correct/total:.4f}  test={tc/tt:.4f}", flush=True)
    model.eval()
    return model


def eval_class_conditional(model, class_test_loaders, class_stats, original, mode):
    """Evaluate accuracy under class-c statistics in given mode (both/mu_only/sigma_only)."""
    correct = total = 0
    for c in range(10):
        # restore globals first
        for name, mod in get_bn_layers(model):
            mod.running_mean.copy_(original[name]["mean"])
            mod.running_var.copy_(original[name]["var"])
        # apply per-class
        load_bn_stats(model, class_stats[c], mode=mode)
        with torch.no_grad():
            for images, labels in class_test_loaders[c]:
                out = model(images.to(DEVICE))
                correct += (out.argmax(1) == labels.to(DEVICE)).sum().item()
                total += labels.size(0)
    # restore globals
    for name, mod in get_bn_layers(model):
        mod.running_mean.copy_(original[name]["mean"])
        mod.running_var.copy_(original[name]["var"])
    return correct / total


def main():
    print(f"Device: {DEVICE}, seed: {SEED}", flush=True)
    train_loader, test_loader, class_loaders, class_test_loaders = get_data()
    torch.manual_seed(SEED)
    model = SmallResNet().to(DEVICE)
    model = train_model(model, train_loader, test_loader)

    print("\nComputing class-conditional BN stats...", flush=True)
    class_stats, original = compute_class_conditional_stats(model, class_loaders, 10)

    # Global baseline
    global_acc = eval_class_conditional(model, class_test_loaders, {c: original for c in range(10)}, original, mode="both")
    print(f"global = {global_acc:.4f}", flush=True)

    # Both (= standard same-class)
    both_acc = eval_class_conditional(model, class_test_loaders, class_stats, original, mode="both")
    print(f"both (mu+sigma) = {both_acc:.4f}", flush=True)

    # mu_only
    mu_acc = eval_class_conditional(model, class_test_loaders, class_stats, original, mode="mu_only")
    print(f"mu_only         = {mu_acc:.4f}", flush=True)

    # sigma_only
    sigma_acc = eval_class_conditional(model, class_test_loaders, class_stats, original, mode="sigma_only")
    print(f"sigma_only      = {sigma_acc:.4f}", flush=True)

    out = {
        "config": {"seed": SEED, "epochs": EPOCHS, "model": "SmallResNet", "dataset": "CIFAR-10"},
        "global_acc": global_acc,
        "both_acc": both_acc,
        "mu_only_acc": mu_acc,
        "sigma_only_acc": sigma_acc,
    }
    out_path = RESULTS_DIR / f"results_mu_sigma_split_seed{SEED}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
