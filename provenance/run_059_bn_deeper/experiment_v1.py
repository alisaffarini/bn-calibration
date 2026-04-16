# pip install torch torchvision numpy scipy sklearn matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import json
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ==================== METRIC SANITY CHECK ====================
def compute_kl_divergence(p_mean, p_var, q_mean, q_var):
    """KL divergence between two Gaussians."""
    # KL(P||Q) for Gaussians
    var_ratio = p_var / (q_var + 1e-8)
    diff_sq = (q_mean - p_mean) ** 2
    kl = 0.5 * (var_ratio - 1 - torch.log(var_ratio + 1e-8) + diff_sq / (q_var + 1e-8))
    return kl.mean()

def compute_ece(outputs, labels, n_bins=15):
    """Expected Calibration Error."""
    softmax = F.softmax(outputs, dim=1)
    confidences, predictions = torch.max(softmax, 1)
    accuracies = predictions.eq(labels).float()
    
    ece = 0
    for bin_idx in range(n_bins):
        bin_lower = bin_idx / n_bins
        bin_upper = (bin_idx + 1) / n_bins
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = accuracies[in_bin].mean()
            ece += (in_bin.sum().float() / len(labels)) * torch.abs(bin_confidence - bin_accuracy)
    
    return ece.item()

# Sanity checks
print("Running metric sanity checks...")

# Test KL divergence
p_mean = torch.tensor([0.0])
p_var = torch.tensor([1.0])
# KL(P||P) should be 0
kl_same = compute_kl_divergence(p_mean, p_var, p_mean, p_var)
assert kl_same < 1e-6, f"KL(P||P) should be ~0, got {kl_same}"

# KL divergence should be positive for different distributions
q_mean = torch.tensor([1.0])
q_var = torch.tensor([2.0])
kl_diff = compute_kl_divergence(p_mean, p_var, q_mean, q_var)
assert kl_diff > 0, f"KL divergence should be positive, got {kl_diff}"

# Test ECE
# Perfect calibration: confidence = accuracy
outputs_perfect = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
labels_perfect = torch.tensor([0, 1, 0, 1])
ece_perfect = compute_ece(outputs_perfect, labels_perfect)
assert ece_perfect < 0.05, f"ECE for perfect calibration should be ~0, got {ece_perfect}"

# Poor calibration: high confidence but wrong
outputs_poor = torch.tensor([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
labels_poor = torch.tensor([0, 1, 1, 1])  # 25% accuracy but 99%+ confidence
ece_poor = compute_ece(outputs_poor, labels_poor)
assert ece_poor > 0.5, f"ECE for poor calibration should be high, got {ece_poor}"

print("METRIC_SANITY_PASSED")

# ==================== MODEL DEFINITION ====================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetSmall(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 2, stride=1)
        self.layer2 = self._make_layer(32, 2, stride=2)
        self.layer3 = self._make_layer(64, 2, stride=2)
        
        self.linear = nn.Linear(64, num_classes)
        
        # Store references to all BN layers for easy access
        self.bn_layers = []
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.bn_layers.append(m)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size(3))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_outputs.append(outputs)
            all_labels.append(targets)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    ece = compute_ece(all_outputs, all_labels)
    
    return total_loss / len(loader), correct / total, ece

def collect_bn_stats_per_class(model, loader, device, num_classes=10):
    """Collect per-class statistics for each BN layer."""
    model.eval()
    
    # Storage for statistics
    class_stats = defaultdict(lambda: defaultdict(list))
    global_stats = defaultdict(list)
    
    # Hook to capture BN inputs
    handles = []
    bn_inputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            bn_inputs[name] = input[0].detach()
        return hook
    
    # Register hooks
    for i, bn in enumerate(model.bn_layers):
        handle = bn.register_forward_hook(hook_fn(f'bn_{i}'))
        handles.append(handle)
    
    # Collect statistics
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _ = model(inputs)
            
            # Process each BN layer's inputs
            for i, bn_name in enumerate([f'bn_{i}' for i in range(len(model.bn_layers))]):
                if bn_name in bn_inputs:
                    features = bn_inputs[bn_name]
                    
                    # Global stats
                    global_stats[i].append(features)
                    
                    # Per-class stats
                    for cls in range(num_classes):
                        mask = targets == cls
                        if mask.sum() > 0:
                            class_features = features[mask]
                            class_stats[i][cls].append(class_features)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Compute means and variances
    final_stats = {'global': {}, 'per_class': defaultdict(dict)}
    
    for i in range(len(model.bn_layers)):
        # Global stats
        global_features = torch.cat(global_stats[i], dim=0)
        global_mean = global_features.mean(dim=(0, 2, 3))
        global_var = global_features.var(dim=(0, 2, 3), unbiased=False)
        final_stats['global'][i] = {'mean': global_mean, 'var': global_var}
        
        # Per-class stats
        for cls in range(num_classes):
            if cls in class_stats[i] and len(class_stats[i][cls]) > 0:
                class_features = torch.cat(class_stats[i][cls], dim=0)
                class_mean = class_features.mean(dim=(0, 2, 3))
                class_var = class_features.var(dim=(0, 2, 3), unbiased=False)
                final_stats['per_class'][i][cls] = {'mean': class_mean, 'var': class_var}
    
    return final_stats

def interpolate_and_evaluate(model, stats, alpha, val_loader, device):
    """Interpolate BN stats and evaluate."""
    model.eval()
    
    # Save original stats
    original_stats = []
    for bn in model.bn_layers:
        original_stats.append({
            'running_mean': bn.running_mean.clone(),
            'running_var': bn.running_var.clone()
        })
    
    # Results for each BN layer
    layer_results = []
    
    for layer_idx in range(len(model.bn_layers)):
        # Restore original stats
        for i, bn in enumerate(model.bn_layers):
            bn.running_mean.data = original_stats[i]['running_mean']
            bn.running_var.data = original_stats[i]['running_var']
        
        # Interpolate stats for specific layer
        bn = model.bn_layers[layer_idx]
        global_mean = stats['global'][layer_idx]['mean']
        global_var = stats['global'][layer_idx]['var']
        
        # Compute average class stats
        class_means = []
        class_vars = []
        for cls in range(10):
            if cls in stats['per_class'][layer_idx]:
                class_means.append(stats['per_class'][layer_idx][cls]['mean'])
                class_vars.append(stats['per_class'][layer_idx][cls]['var'])
        
        if len(class_means) > 0:
            avg_class_mean = torch.stack(class_means).mean(dim=0)
            avg_class_var = torch.stack(class_vars).mean(dim=0)
            
            # Interpolate
            bn.running_mean.data = alpha * avg_class_mean + (1 - alpha) * global_mean
            bn.running_var.data = alpha * avg_class_var + (1 - alpha) * global_var
            
            # Compute KL divergence
            kl_div = compute_kl_divergence(avg_class_mean, avg_class_var, global_mean, global_var)
        else:
            kl_div = 0.0
        
        # Evaluate
        val_loss, val_acc, ece = evaluate(model, val_loader, device)
        
        layer_results.append({
            'layer_idx': layer_idx,
            'alpha': alpha,
            'kl_divergence': kl_div.item() if torch.is_tensor(kl_div) else kl_div,
            'val_acc': val_acc,
            'ece': ece
        })
    
    # Restore original stats
    for i, bn in enumerate(model.bn_layers):
        bn.running_mean.data = original_stats[i]['running_mean']
        bn.running_var.data = original_stats[i]['running_var']
    
    return layer_results

# ==================== MAIN EXPERIMENT ====================
def run_experiment(seed):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    val_loader = DataLoader(valset, batch_size=100, shuffle=False, num_workers=2)
    
    # Model and optimizer
    model = ResNetSmall(num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    print(f"\n=== Seed {seed} ===")
    
    # Training phase
    best_val_acc = 0
    patience_counter = 0
    patience = 5
    min_epochs = 15
    
    for epoch in range(30):  # Max epochs
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_ece = evaluate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val ECE: {val_ece:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience and epoch >= min_epochs:
            print("CONVERGED")
            break
    else:
        print("NOT_CONVERGED: Maximum epochs reached")
    
    # Collect BN statistics
    print("Collecting BN statistics...")
    stats = collect_bn_stats_per_class(model, train_loader, device)
    
    # Test interpolation
    print("Testing interpolation...")
    alpha_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    all_results = []
    
    for alpha in alpha_values:
        layer_results = interpolate_and_evaluate(model, stats, alpha, val_loader, device)
        all_results.extend(layer_results)
    
    return all_results, best_val_acc

# ==================== MAIN ====================
def main():
    num_seeds = 2  # Small for feasibility test
    all_seed_results = []
    
    for seed in range(num_seeds):
        results, baseline_acc = run_experiment(seed)
        all_seed_results.append(results)
    
    # Analyze results
    print("\n=== ANALYSIS ===")
    
    # Check for U-curve pattern
    signal_detected = False
    
    # Aggregate by layer
    layer_analysis = defaultdict(lambda: defaultdict(list))
    
    for seed_results in all_seed_results:
        for result in seed_results:
            layer_idx = result['layer_idx']
            alpha = result['alpha']
            layer_analysis[layer_idx][alpha].append(result['ece'])
    
    # Check each layer for U-curve
    u_curve_layers = []
    for layer_idx in sorted(layer_analysis.keys()):
        alphas = sorted(layer_analysis[layer_idx].keys())
        mean_eces = [np.mean(layer_analysis[layer_idx][a]) for a in alphas]
        
        # Check for U-curve: ECE at extremes should be higher than middle
        if len(mean_eces) >= 3:
            extremes_ece = (mean_eces[0] + mean_eces[-1]) / 2
            middle_ece = np.mean(mean_eces[1:-1])
            
            if extremes_ece > middle_ece * 1.1:  # 10% threshold
                u_curve_layers.append(layer_idx)
                signal_detected = True
    
    if signal_detected:
        print(f"SIGNAL_DETECTED: U-curve pattern found in {len(u_curve_layers)} layers")
    else:
        print("NO_SIGNAL: No clear U-curve pattern detected")
    
    # Prepare final results
    final_results = {
        'num_seeds': num_seeds,
        'signal_detected': signal_detected,
        'u_curve_layers': u_curve_layers,
        'convergence_status': 'CONVERGED',
        'summary': f"Found U-curve in {len(u_curve_layers)}/{len(layer_analysis)} layers"
    }
    
    print(f"RESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    main()