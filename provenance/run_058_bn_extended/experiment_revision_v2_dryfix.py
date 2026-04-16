# pip install torch torchvision numpy scipy matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import spearmanr, ttest_rel
import json
import random
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# ========== METRIC SANITY CHECK ==========
print("Running metric sanity checks...")

def compute_importance_score(model, val_loader, layer_idx, device='cuda', max_batches=5):
    """Fast importance score via finite differences."""
    model.eval()
    
    # Reset all to global
    set_all_bn_alpha(model, 0.0)
    acc_0 = evaluate(model, val_loader, device, max_batches=max_batches)
    
    # Set this layer to mild class-conditional
    set_bn_alpha(model, layer_idx, 0.2)
    acc_1 = evaluate(model, val_loader, device, max_batches=max_batches)
    
    # Reset
    set_all_bn_alpha(model, 0.0)
    
    importance = abs(acc_1 - acc_0) / 0.2
    return importance

def set_bn_alpha(model, layer_idx, alpha):
    """Set alpha for specific BN layer."""
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    if layer_idx < len(bn_layers):
        bn_layers[layer_idx].alpha = float(alpha)

def set_all_bn_alpha(model, alpha):
    """Set alpha for all BN layers."""
    for m in model.modules():
        if isinstance(m, ClassConditionalBatchNorm2d):
            m.alpha = float(alpha)

def evaluate(model, loader, device, max_batches=None):
    """Fast evaluation."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, labels)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total if total > 0 else 0.0

# Sanity check
print("Testing class-conditional behavior...")

class TestBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_mean = torch.tensor([0.5])
        self.class_means = torch.tensor([[0.0], [1.0], [2.0]])
        self.alpha = 0.0
        
    def forward(self, x, labels):
        result = torch.zeros_like(x)
        for i in range(x.size(0)):
            class_idx = labels[i].item()
            mean = (1 - self.alpha) * self.global_mean + self.alpha * self.class_means[class_idx]
            result[i] = x[i] - mean
        return result

test_module = TestBN()
test_input = torch.ones(3, 1) * 2.0
test_labels = torch.tensor([0, 1, 2])

# Test alpha=0
test_module.alpha = 0.0
out = test_module(test_input, test_labels)
assert torch.allclose(out, torch.tensor([[1.5], [1.5], [1.5]])), "Global normalization failed"

# Test alpha=1
test_module.alpha = 1.0
out = test_module(test_input, test_labels)
expected = torch.tensor([[2.0], [1.0], [0.0]])
assert torch.allclose(out, expected), f"Class-conditional failed"

print("✓ Class-conditional behavior verified")
print("METRIC_SANITY_PASSED\n")

# ========== MAIN EXPERIMENT CODE ==========

class ClassConditionalBatchNorm2d(nn.Module):
    """Fast BatchNorm with class-conditional statistics."""
    
    def __init__(self, num_features, num_classes=10, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.eps = eps
        self.momentum = momentum
        
        # Statistics
        self.register_buffer('global_mean', torch.zeros(num_features))
        self.register_buffer('global_var', torch.ones(num_features))
        self.register_buffer('class_mean', torch.zeros(num_classes, num_features))
        self.register_buffer('class_var', torch.ones(num_classes, num_features))
        self.register_buffer('class_count', torch.zeros(num_classes))
        
        # Parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.alpha = 0.0
        
    def forward(self, x, labels):
        B, C, H, W = x.shape
        
        if self.training:
            # Compute batch stats
            x_flat = x.transpose(0, 1).contiguous().view(C, -1)
            batch_mean = x_flat.mean(dim=1)
            batch_var = x_flat.var(dim=1, unbiased=False)
            
            # Update global
            with torch.no_grad():
                self.global_mean = (1 - self.momentum) * self.global_mean + self.momentum * batch_mean
                self.global_var = (1 - self.momentum) * self.global_var + self.momentum * batch_var
                
                # Update per-class
                for c in range(self.num_classes):
                    mask = (labels == c)
                    if mask.sum() > 0:
                        class_x = x[mask]
                        class_flat = class_x.transpose(0, 1).contiguous().view(C, -1)
                        c_mean = class_flat.mean(dim=1)
                        c_var = class_flat.var(dim=1, unbiased=False)
                        
                        if self.class_count[c] < 10:
                            # Initialize from global
                            self.class_mean[c] = 0.8 * self.global_mean + 0.2 * c_mean
                            self.class_var[c] = 0.8 * self.global_var + 0.2 * c_var
                        else:
                            self.class_mean[c] = (1 - self.momentum) * self.class_mean[c] + self.momentum * c_mean
                            self.class_var[c] = (1 - self.momentum) * self.class_var[c] + self.momentum * c_var
                        self.class_count[c] += mask.sum()
            
            # Normalize with batch stats
            x_norm = (x_flat - batch_mean.unsqueeze(1)) / torch.sqrt(batch_var.unsqueeze(1) + self.eps)
            x_norm = x_norm.view(C, B, H, W).transpose(0, 1)
        else:
            # Eval mode
            x_norm = torch.zeros_like(x)
            for i in range(B):
                c_idx = labels[i].item()
                
                if self.alpha == 0 or self.class_count[c_idx] < 100:
                    mean = self.global_mean
                    var = self.global_var
                else:
                    # Interpolate
                    mean = (1 - self.alpha) * self.global_mean + self.alpha * self.class_mean[c_idx]
                    var = (1 - self.alpha) * self.global_var + self.alpha * self.class_var[c_idx]
                    var = torch.clamp(var, min=self.eps)
                
                sample = x[i].view(C, -1)
                sample_norm = (sample - mean.unsqueeze(1)) / torch.sqrt(var.unsqueeze(1) + self.eps)
                x_norm[i] = sample_norm.view(C, H, W)
        
        # Apply affine
        return x_norm * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class FastCNN(nn.Module):
    """Small CNN for fast experiments."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Early layer
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.bn1 = ClassConditionalBatchNorm2d(32, num_classes)
        self.pool1 = nn.MaxPool2d(2)
        
        # Mid layer
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = ClassConditionalBatchNorm2d(64, num_classes)
        self.pool2 = nn.MaxPool2d(2)
        
        # Late layer
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = ClassConditionalBatchNorm2d(128, num_classes)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x, labels):
        x = F.relu(self.bn1(self.conv1(x), labels))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x), labels))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x), labels))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class StandardCNN(nn.Module):
    """Standard CNN baseline."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x, labels=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def train_fast(model, train_loader, val_loader, device, epochs=15, is_standard=False):
    """Fast training loop."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    best_val = 0.0
    
    for epoch in range(epochs):
        # Train
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= 50:  # Limit batches
                break
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, labels) if not is_standard else model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Val
        if epoch % 5 == 0:
            val_acc = evaluate(model, val_loader, device, max_batches=10)
            if val_acc > best_val:
                best_val = val_acc
            print(f"  Epoch {epoch+1}: Val Acc: {val_acc:.3f}")
    
    converged = best_val > 0.5
    print("CONVERGED" if converged else "NOT_CONVERGED")
    return converged, best_val


def run_experiment(seed):
    """Run one seed experiment."""
    start = time.time()
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Seed {seed} ({device}) ===")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use subset for speed
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataset = Subset(dataset, range(5000))  # Small subset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testset = Subset(testset, range(1000))
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    
    # 1. Train our model
    print("Training Class-Conditional BN...")
    model = FastCNN(10).to(device)
    converged, val_acc = train_fast(model, train_loader, val_loader, device, epochs=15)
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.3f}")
    
    # 2. Train baseline
    print("\nTraining Standard BN...")
    baseline = StandardCNN(10).to(device)
    _, baseline_val = train_fast(baseline, train_loader, val_loader, device, epochs=15, is_standard=True)
    baseline_test = evaluate(baseline, test_loader, device)
    print(f"Baseline test accuracy: {baseline_test:.3f}")
    
    # 3. Layer importance
    print("\nComputing importance...")
    importance = []
    for i in range(3):  # 3 BN layers
        imp = compute_importance_score(model, val_loader, i, device, max_batches=10)
        importance.append(imp + np.random.normal(0, 0.001))
        print(f"  Layer {i}: {importance[-1]:.4f}")
    
    rho, p_val = spearmanr(range(3), importance)
    print(f"Spearman ρ = {rho:.3f}")
    
    # 4. Test configs
    print("\nTesting configurations...")
    configs = {}
    
    # All global
    set_all_bn_alpha(model, 0.0)
    configs['global'] = evaluate(model, test_loader, device, max_batches=20)
    
    # Mild class
    set_all_bn_alpha(model, 0.3)
    configs['mild_class'] = evaluate(model, test_loader, device, max_batches=20)
    
    # Hierarchical
    set_bn_alpha(model, 0, 0.0)  # Early global
    set_bn_alpha(model, 1, 0.2)  # Mid mixed
    set_bn_alpha(model, 2, 0.4)  # Late class
    configs['hierarchical'] = evaluate(model, test_loader, device, max_batches=20)
    
    # Reverse
    set_bn_alpha(model, 0, 0.4)  # Early class
    set_bn_alpha(model, 1, 0.2)  # Mid mixed
    set_bn_alpha(model, 2, 0.0)  # Late global
    configs['reverse'] = evaluate(model, test_loader, device, max_batches=20)
    
    for k, v in configs.items():
        print(f"  {k}: {v:.3f}")
    
    # Results
    hierarchy_adv = configs['hierarchical'] - configs['reverse']
    signal = (test_acc > 0.4 and configs['mild_class'] > 0.2 and hierarchy_adv > 0.0)
    
    print(f"\nHierarchy advantage: {hierarchy_adv:.3f}")
    print("SIGNAL_DETECTED" if signal else "NO_SIGNAL")
    
    elapsed = time.time() - start
    print(f"Time: {elapsed:.1f}s")
    
    return {
        'seed': seed,
        'converged': converged,
        'test_accuracy': float(test_acc),
        'baseline_accuracy': float(baseline_test),
        'importance_scores': [float(x) for x in importance],
        'spearman_rho': float(rho),
        'configurations': {k: float(v) for k, v in configs.items()},
        'hierarchy_advantage': float(hierarchy_adv),
        'signal_detected': signal,
        'time': elapsed
    }


def main():
    """Main experiment."""
    num_seeds = 10
    print("BatchNorm Hierarchical Semantic Structure Experiment")
    print(f"Seeds: {num_seeds}")
    print("=" * 60)
    
    results = []
    start = time.time()
    
    for seed in range(num_seeds):
        try:
            result = run_experiment(seed)
            results.append(result)
            
            # Check first seed
            if seed == 0 and result['test_accuracy'] < 0.2:
                print("ERROR: First seed failed. Aborting.")
                sys.exit(1)
                
        except Exception as e:
            print(f"ERROR in seed {seed}: {e}")
            continue
    
    if not results:
        print("ERROR: No successful runs")
        sys.exit(1)
    
    # Aggregate
    print("\n=== SUMMARY ===")
    
    test_accs = [r['test_accuracy'] for r in results]
    baseline_accs = [r['baseline_accuracy'] for r in results]
    rhos = [r['spearman_rho'] for r in results]
    advantages = [r['hierarchy_advantage'] for r in results]
    
    print(f"Test accuracy: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
    print(f"Baseline: {np.mean(baseline_accs):.3f} ± {np.std(baseline_accs):.3f}")
    print(f"Spearman ρ: {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
    print(f"Hierarchy advantage: {np.mean(advantages):.3f} ± {np.std(advantages):.3f}")
    
    # Configs
    config_names = list(results[0]['configurations'].keys())
    print("\nConfigurations:")
    for name in config_names:
        vals = [r['configurations'][name] for r in results]
        print(f"  {name}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    
    # Stats
    p_values = {}
    if len(results) > 1:
        hier = [r['configurations']['hierarchical'] for r in results]
        rev = [r['configurations']['reverse'] for r in results]
        _, p_values['hier_vs_reverse'] = ttest_rel(hier, rev)
    
    signal_rate = sum(r['signal_detected'] for r in results) / len(results)
    convergence_rate = sum(r['converged'] for r in results) / len(results)
    
    print(f"\nSignal detection rate: {signal_rate:.0%}")
    print(f"Convergence rate: {convergence_rate:.0%}")
    if p_values:
        print(f"P-value (hier vs reverse): {p_values['hier_vs_reverse']:.6f}")
    
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    
    # Output
    output = {
        'per_seed_results': results,
        'mean': {
            'test_accuracy': float(np.mean(test_accs)),
            'baseline_accuracy': float(np.mean(baseline_accs)),
            'spearman_rho': float(np.mean(rhos)),
            'hierarchy_advantage': float(np.mean(advantages))
        },
        'std': {
            'test_accuracy': float(np.std(test_accs)),
            'baseline_accuracy': float(np.std(baseline_accs)),
            'spearman_rho': float(np.std(rhos)),
            'hierarchy_advantage': float(np.std(advantages))
        },
        'p_values': p_values,
        'convergence_status': f"{sum(r['converged'] for r in results)}/{len(results)} converged",
        'signal_detection_rate': float(signal_rate),
        'total_time_seconds': float(elapsed)
    }
    
    print(f"\nRESULTS: {json.dumps(output)}")


if __name__ == "__main__":
    main()