#!/usr/bin/env python3
"""ECE + Temperature Scaling experiment.
Trains SmallResNet on CIFAR-10 (3 seeds), computes ECE for all BN conditions,
and tests if temperature scaling can recover accuracy after BN replacement."""

import json, gc, time, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
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
EPOCHS = 20
NUM_BINS = 15
SEEDS = [42, 43, 44]

print(f"Device: {DEVICE}")

# ── Models ──
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
        self.layer1 = self._make_layer(32, 2, 1)
        self.layer2 = self._make_layer(64, 2, 2)
        self.layer3 = self._make_layer(128, 2, 2)
        self.layer4 = self._make_layer(256, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ── Data ──
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

# ── Helper functions ──
def compute_ece(logits, labels, num_bins=15):
    """Compute Expected Calibration Error."""
    softmax = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmax, dim=1)
    accuracies = predictions.eq(labels).float()
    
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    ece = 0.0
    for i in range(num_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += (mask.sum().float() / len(labels)) * abs(bin_acc - bin_conf)
    return ece.item()

def train_model(seed, num_classes=10):
    """Train SmallResNet on CIFAR-10."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    trainset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = SmallResNet(num_classes=num_classes).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(EPOCHS):
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Seed {seed}, Epoch {epoch+1}/{EPOCHS}")
    
    return model

def compute_class_conditional_stats(model, trainset, num_classes=10):
    """Compute per-class BN stats."""
    model.eval()
    class_stats = {}
    
    for c in range(num_classes):
        indices = [i for i, (_, label) in enumerate(trainset) if label == c]
        subset = Subset(trainset, indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        # Reset running stats
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.zero_()
                m.running_var.fill_(1)
                m.num_batches_tracked.zero_()
        
        model.train()
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(DEVICE)
                model(inputs)
        
        stats_c = {}
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                stats_c[name] = {
                    'mean': m.running_mean.clone(),
                    'var': m.running_var.clone()
                }
        class_stats[c] = stats_c
    
    return class_stats

def set_bn_stats(model, stats_dict):
    """Load BN stats into model."""
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name in stats_dict:
            m.running_mean.copy_(stats_dict[name]['mean'])
            m.running_var.copy_(stats_dict[name]['var'])

def save_global_stats(model):
    """Save current BN stats."""
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            stats[name] = {
                'mean': m.running_mean.clone(),
                'var': m.running_var.clone()
            }
    return stats

def collect_logits(model, loader):
    """Collect all logits and labels from a dataloader."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            all_logits.append(logits.cpu())
            all_labels.append(targets)
    return torch.cat(all_logits), torch.cat(all_labels)

def temperature_scale(logits, labels, lr=0.01, max_iter=200):
    """Learn optimal temperature on validation set."""
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

# ── Main experiment ──
results = {"ece": {}, "temp_scaling": {}}
start = time.time()

trainset_raw = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform_test)
testset = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True, transform=transform_test)

# Split test into val (for temp scaling) and test
val_size = 2000
test_size = len(testset) - val_size
valset, testset_final = random_split(testset, [val_size, test_size], generator=torch.Generator().manual_seed(0))
valloader = DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
testloader = DataLoader(testset_final, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

trainset_aug = torchvision.datasets.CIFAR10(root=str(DATA_DIR), train=True, download=True, transform=transform_train)

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    
    # Train
    model = train_model(seed)
    global_stats = save_global_stats(model)
    
    # Compute class-conditional stats
    print(f"  Computing class-conditional stats...")
    class_stats = compute_class_conditional_stats(model, trainset_raw, num_classes=10)
    
    # Restore global stats
    set_bn_stats(model, global_stats)
    
    seed_ece = {}
    seed_temp = {}
    
    # === GLOBAL condition ===
    print(f"  Evaluating GLOBAL condition...")
    set_bn_stats(model, global_stats)
    logits, labels = collect_logits(model, testloader)
    acc = (logits.argmax(1) == labels).float().mean().item()
    ece = compute_ece(logits, labels, NUM_BINS)
    seed_ece["global"] = {"accuracy": acc, "ece": ece}
    print(f"    Global: acc={acc:.4f}, ECE={ece:.4f}")
    
    # === SAME-CLASS condition ===
    print(f"  Evaluating SAME-CLASS condition...")
    all_logits_same = []
    all_labels_same = []
    for batch_inputs, batch_targets in testloader:
        batch_logits = []
        for inp, tgt in zip(batch_inputs, batch_targets):
            c = tgt.item()
            set_bn_stats(model, class_stats[c])
            model.eval()
            with torch.no_grad():
                logit = model(inp.unsqueeze(0).to(DEVICE)).cpu()
            batch_logits.append(logit)
        all_logits_same.append(torch.cat(batch_logits))
        all_labels_same.append(batch_targets)
    logits_same = torch.cat(all_logits_same)
    labels_same = torch.cat(all_labels_same)
    acc_same = (logits_same.argmax(1) == labels_same).float().mean().item()
    ece_same = compute_ece(logits_same, labels_same, NUM_BINS)
    seed_ece["same_class"] = {"accuracy": acc_same, "ece": ece_same}
    print(f"    Same-class: acc={acc_same:.4f}, ECE={ece_same:.4f}")
    
    # Temperature scaling on same-class condition
    print(f"  Fitting temperature scaling on same-class logits...")
    # Collect val logits under same-class condition
    all_logits_val = []
    all_labels_val = []
    for batch_inputs, batch_targets in valloader:
        batch_logits = []
        for inp, tgt in zip(batch_inputs, batch_targets):
            c = tgt.item()
            set_bn_stats(model, class_stats[c])
            model.eval()
            with torch.no_grad():
                logit = model(inp.unsqueeze(0).to(DEVICE)).cpu()
            batch_logits.append(logit)
        all_logits_val.append(torch.cat(batch_logits))
        all_labels_val.append(batch_targets)
    logits_val_same = torch.cat(all_logits_val)
    labels_val_same = torch.cat(all_labels_val)
    
    # Fit temperature
    T = temperature_scale(logits_val_same, labels_val_same)
    
    # Apply to test
    scaled_logits = logits_same / T
    acc_scaled = (scaled_logits.argmax(1) == labels_same).float().mean().item()
    ece_scaled = compute_ece(scaled_logits, labels_same, NUM_BINS)
    seed_temp["same_class"] = {
        "temperature": T,
        "accuracy_before": acc_same,
        "accuracy_after": acc_scaled,
        "ece_before": ece_same,
        "ece_after": ece_scaled
    }
    print(f"    Temp scaling (T={T:.3f}): acc {acc_same:.4f} -> {acc_scaled:.4f}, ECE {ece_same:.4f} -> {ece_scaled:.4f}")
    
    # === WRONG-CLASS condition ===
    print(f"  Evaluating WRONG-CLASS condition...")
    all_logits_wrong = []
    all_labels_wrong = []
    for batch_inputs, batch_targets in testloader:
        batch_logits = []
        for inp, tgt in zip(batch_inputs, batch_targets):
            c = tgt.item()
            wrong_c = (c + 5) % 10
            set_bn_stats(model, class_stats[wrong_c])
            model.eval()
            with torch.no_grad():
                logit = model(inp.unsqueeze(0).to(DEVICE)).cpu()
            batch_logits.append(logit)
        all_logits_wrong.append(torch.cat(batch_logits))
        all_labels_wrong.append(batch_targets)
    logits_wrong = torch.cat(all_logits_wrong)
    labels_wrong = torch.cat(all_labels_wrong)
    acc_wrong = (logits_wrong.argmax(1) == labels_wrong).float().mean().item()
    ece_wrong = compute_ece(logits_wrong, labels_wrong, NUM_BINS)
    seed_ece["wrong_class"] = {"accuracy": acc_wrong, "ece": ece_wrong}
    print(f"    Wrong-class: acc={acc_wrong:.4f}, ECE={ece_wrong:.4f}")
    
    results["ece"][f"seed_{seed}"] = seed_ece
    results["temp_scaling"][f"seed_{seed}"] = seed_temp
    
    del model
    gc.collect()
    if DEVICE == "mps":
        torch.mps.empty_cache()

# Aggregate
ece_agg = {}
for cond in ["global", "same_class", "wrong_class"]:
    accs = [results["ece"][f"seed_{s}"][cond]["accuracy"] for s in SEEDS]
    eces = [results["ece"][f"seed_{s}"][cond]["ece"] for s in SEEDS]
    ece_agg[cond] = {
        "accuracy_mean": np.mean(accs), "accuracy_std": np.std(accs),
        "ece_mean": np.mean(eces), "ece_std": np.std(eces)
    }
results["ece"]["aggregate"] = ece_agg

# Temp scaling aggregate
temp_accs_before = [results["temp_scaling"][f"seed_{s}"]["same_class"]["accuracy_before"] for s in SEEDS]
temp_accs_after = [results["temp_scaling"][f"seed_{s}"]["same_class"]["accuracy_after"] for s in SEEDS]
temp_eces_before = [results["temp_scaling"][f"seed_{s}"]["same_class"]["ece_before"] for s in SEEDS]
temp_eces_after = [results["temp_scaling"][f"seed_{s}"]["same_class"]["ece_after"] for s in SEEDS]
temps = [results["temp_scaling"][f"seed_{s}"]["same_class"]["temperature"] for s in SEEDS]
results["temp_scaling"]["aggregate"] = {
    "temperature_mean": np.mean(temps), "temperature_std": np.std(temps),
    "accuracy_before_mean": np.mean(temp_accs_before),
    "accuracy_after_mean": np.mean(temp_accs_after),
    "ece_before_mean": np.mean(temp_eces_before),
    "ece_after_mean": np.mean(temp_eces_after),
    "accuracy_recovery": np.mean(temp_accs_after) - np.mean(temp_accs_before)
}

results["total_time_seconds"] = time.time() - start

# Save
outpath = OUTPUT_DIR / "results_ece_tempscaling.json"
with open(outpath, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"DONE in {results['total_time_seconds']:.0f}s")
print(f"Results saved to {outpath}")
print(f"\nECE Summary:")
for cond, vals in ece_agg.items():
    print(f"  {cond}: acc={vals['accuracy_mean']:.4f}±{vals['accuracy_std']:.4f}, ECE={vals['ece_mean']:.4f}±{vals['ece_std']:.4f}")
print(f"\nTemp Scaling Summary:")
agg = results['temp_scaling']['aggregate']
print(f"  T={agg['temperature_mean']:.3f}, acc: {agg['accuracy_before_mean']:.4f} -> {agg['accuracy_after_mean']:.4f}")
print(f"  ECE: {agg['ece_before_mean']:.4f} -> {agg['ece_after_mean']:.4f}")
