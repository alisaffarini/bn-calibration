"""Layer-wise BN-statistic replacement ablation.

Trains a single SmallResNet on CIFAR-10, then for each BN layer L individually
(and for the cumulative replacement L=1..k for k=1..L_total) substitutes the
class-conditional buffers and measures top-1 accuracy. Tests whether the
same-class collapse is dominated by deep BN layers, shallow ones, or compounds
uniformly.
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path(__file__).parent
RESULTS_DIR = OUTPUT_DIR.parent / "results"
DATA_DIR = OUTPUT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 128
EPOCHS = 20
import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42

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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        return self.fc(out.view(out.size(0), -1))


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


def load_bn_stats(model, stats, only_layers=None):
    for name, mod in get_bn_layers(model):
        if only_layers is not None and name not in only_layers:
            continue
        mod.running_mean.copy_(stats[name]['running_mean'])
        mod.running_var.copy_(stats[name]['running_var'])


def compute_class_conditional_stats(model, class_loaders_train, num_classes):
    class_stats = {}
    original = save_bn_stats(model)
    for c in range(num_classes):
        for _, mod in get_bn_layers(model):
            mod.running_mean.zero_()
            mod.running_var.fill_(1.0)
            mod.num_batches_tracked.zero_()
        model.train()
        with torch.no_grad():
            for images, _ in class_loaders_train[c]:
                images = images.to(DEVICE)
                _ = model(images)
        class_stats[c] = save_bn_stats(model)
    load_bn_stats(model, original)
    model.eval()
    return class_stats


def get_data():
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
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
    trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform_test)
    trainset_clean = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=False, transform=transform_test)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    targets = np.array(trainset_clean.targets)
    class_loaders = {}
    for c in range(10):
        idx = np.where(targets == c)[0].tolist()
        class_loaders[c] = DataLoader(Subset(trainset_clean, idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_targets = np.array(testset.targets)
    class_test_loaders = {}
    for c in range(10):
        idx = np.where(test_targets == c)[0].tolist()
        class_test_loaders[c] = DataLoader(Subset(testset, idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader, class_loaders, class_test_loaders


def train_model(model, train_loader, test_loader):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        model.train()
        correct = total = 0
        t0 = time.time()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        if (epoch+1) % 5 == 0 or epoch == 0 or epoch == EPOCHS-1:
            model.eval()
            tc = tt = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
                    out = model(images)
                    tc += (out.argmax(1) == labels).sum().item()
                    tt += labels.size(0)
            print(f"  Epoch {epoch+1}/{EPOCHS}  train={correct/total:.4f}  test={tc/tt:.4f}  ({time.time()-t0:.1f}s)")
    model.eval()
    return model


def eval_with_layer_replacement(model, class_test_loaders, class_stats, layers_to_replace, num_classes=10):
    """Replace `layers_to_replace` (list of BN layer names) with class-c stats; measure accuracy."""
    original = save_bn_stats(model)
    model.eval()
    correct = total = 0
    for c in range(num_classes):
        # Set the chosen layers to class-c stats; other layers remain at original
        load_bn_stats(model, original)
        if layers_to_replace:
            load_bn_stats(model, class_stats[c], only_layers=set(layers_to_replace))
        with torch.no_grad():
            for images, labels in class_test_loaders[c]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                out = model(images)
                correct += (out.argmax(1) == labels).sum().item()
                total += labels.size(0)
    load_bn_stats(model, original)
    return correct / total


def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"Training SmallResNet on CIFAR-10 ({EPOCHS} epochs, seed {SEED})")
    train_loader, test_loader, class_loaders, class_test_loaders = get_data()
    torch.manual_seed(SEED)
    model = SmallResNet().to(DEVICE)
    model = train_model(model, train_loader, test_loader)

    bn_layer_names = [name for name, _ in get_bn_layers(model)]
    print(f"Total BN layers: {len(bn_layer_names)}")
    for i, n in enumerate(bn_layer_names):
        print(f"  [{i}] {n}")

    print("\nComputing class-conditional BN stats...")
    class_stats = compute_class_conditional_stats(model, class_loaders, 10)

    print("\nGlobal baseline:")
    glob_acc = eval_with_layer_replacement(model, class_test_loaders, class_stats, [])
    print(f"  global = {glob_acc:.4f}")

    print("\nFull same-class replacement (all layers):")
    all_acc = eval_with_layer_replacement(model, class_test_loaders, class_stats, bn_layer_names)
    print(f"  all-layer same-class = {all_acc:.4f}")

    # 1) Single-layer replacement: replace only layer i
    single_layer = []
    print("\nSingle-layer replacement:")
    for i, name in enumerate(bn_layer_names):
        acc = eval_with_layer_replacement(model, class_test_loaders, class_stats, [name])
        print(f"  [{i:2d}] {name:30s}  acc={acc:.4f}  drop={glob_acc-acc:.4f}")
        single_layer.append({"index": i, "name": name, "acc": acc, "drop": glob_acc - acc})

    # 2) Cumulative shallow-to-deep: replace layers 0..k
    cum_shallow = []
    print("\nCumulative shallow-to-deep replacement:")
    for k in range(1, len(bn_layer_names) + 1):
        layers = bn_layer_names[:k]
        acc = eval_with_layer_replacement(model, class_test_loaders, class_stats, layers)
        print(f"  k={k:2d}  acc={acc:.4f}")
        cum_shallow.append({"k": k, "layers": layers, "acc": acc})

    # 3) Cumulative deep-to-shallow: replace layers L-k..L
    cum_deep = []
    print("\nCumulative deep-to-shallow replacement:")
    L = len(bn_layer_names)
    for k in range(1, L + 1):
        layers = bn_layer_names[L-k:]
        acc = eval_with_layer_replacement(model, class_test_loaders, class_stats, layers)
        print(f"  k={k:2d}  acc={acc:.4f}")
        cum_deep.append({"k": k, "layers": layers, "acc": acc})

    out = {
        "config": {"seed": SEED, "epochs": EPOCHS, "model": "SmallResNet", "dataset": "CIFAR-10"},
        "global_acc": glob_acc,
        "all_layer_same_class_acc": all_acc,
        "bn_layer_names": bn_layer_names,
        "single_layer": single_layer,
        "cum_shallow_to_deep": cum_shallow,
        "cum_deep_to_shallow": cum_deep,
    }
    out_path = RESULTS_DIR / f"results_layerwise_seed{SEED}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

