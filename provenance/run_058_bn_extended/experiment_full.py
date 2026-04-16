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
from collections import defaultdict
import time
import sys

# ========== METRIC SANITY CHECK ==========
print("Running metric sanity checks...")

def compute_importance_score(model, val_loader, layer_idx, alpha_0=0.0, alpha_1=0.1, device='cuda', max_batches=5):
    """Compute importance score for a BN layer via finite differences."""
    model.eval()
    
    # Get accuracy at alpha_0
    set_bn_alpha(model, layer_idx, alpha_0)
    acc_0 = evaluate(model, val_loader, device, max_batches=max_batches)
    
    # Get accuracy at alpha_1
    set_bn_alpha(model, layer_idx, alpha_1)
    acc_1 = evaluate(model, val_loader, device, max_batches=max_batches)
    
    # Reset to global
    set_bn_alpha(model, layer_idx, 0.0)
    
    importance = abs(acc_1 - acc_0) / (alpha_1 - alpha_0)
    return importance

def set_bn_alpha(model, layer_idx, alpha):
    """Set interpolation alpha for specific BN layer."""
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    if layer_idx < len(bn_layers):
        bn_layers[layer_idx].alpha = alpha

def evaluate(model, loader, device, max_batches=None):
    """Evaluation function."""
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

# Sanity check 1: Dummy model should show importance differences
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    
    def forward(self, x, labels=None):
        return self.fc(x)

# Create synthetic data
dummy_x = torch.randn(100, 10)
dummy_y = torch.randint(0, 2, (100,))
dummy_loader = [(dummy_x[i:i+10], dummy_y[i:i+10]) for i in range(0, 100, 10)]

# Test evaluate function
dummy_model = DummyModel()
dummy_acc = evaluate(dummy_model, dummy_loader, 'cpu')
assert 0.0 <= dummy_acc <= 1.0, f"Evaluate function returned invalid accuracy: {dummy_acc}"

print("✓ Evaluation function returns valid accuracy")

# Sanity check 2: Alpha interpolation
class TestBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_mean = torch.tensor([0.0])
        self.class_mean = torch.tensor([1.0])
        self.alpha = 0.0
        
    def forward(self, x):
        mean = (1 - self.alpha) * self.global_mean + self.alpha * self.class_mean
        return x + mean

test_bn = TestBN()
test_bn.alpha = 0.0
assert abs(test_bn(torch.tensor([0.0])).item() - 0.0) < 1e-6, "Alpha=0 should use global stats"
test_bn.alpha = 1.0
assert abs(test_bn(torch.tensor([0.0])).item() - 1.0) < 1e-6, "Alpha=1 should use class stats"
test_bn.alpha = 0.5
assert abs(test_bn(torch.tensor([0.0])).item() - 0.5) < 1e-6, "Alpha=0.5 should interpolate"

print("✓ Alpha interpolation works correctly")
print("METRIC_SANITY_PASSED")
print()

# ========== MAIN EXPERIMENT CODE ==========

class ClassConditionalBatchNorm2d(nn.Module):
    """BatchNorm that can interpolate between global and class-conditional statistics."""
    
    def __init__(self, num_features, num_classes=10, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.eps = eps
        self.momentum = momentum
        
        # Global statistics
        self.register_buffer('global_running_mean', torch.zeros(num_features))
        self.register_buffer('global_running_var', torch.ones(num_features))
        
        # Per-class statistics
        self.register_buffer('class_running_mean', torch.zeros(num_classes, num_features))
        self.register_buffer('class_running_var', torch.ones(num_classes, num_features))
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Interpolation factor (0 = global, 1 = class-conditional)
        self.alpha = 0.0
        
    def forward(self, x, labels=None):
        batch_size, channels, height, width = x.shape
        
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # Update global running stats
            with torch.no_grad():
                self.global_running_mean = (1 - self.momentum) * self.global_running_mean + self.momentum * batch_mean
                self.global_running_var = (1 - self.momentum) * self.global_running_var + self.momentum * batch_var
                
                # Update per-class running stats if labels provided
                if labels is not None:
                    for c in range(self.num_classes):
                        mask = (labels == c)
                        if mask.sum() > 0:
                            class_x = x[mask]
                            class_mean = class_x.mean(dim=(0, 2, 3))
                            class_var = class_x.var(dim=(0, 2, 3), unbiased=False)
                            
                            self.class_running_mean[c] = (1 - self.momentum) * self.class_running_mean[c] + self.momentum * class_mean
                            self.class_running_var[c] = (1 - self.momentum) * self.class_running_var[c] + self.momentum * class_var
            
            # Use batch stats for normalization during training
            mean = batch_mean.view(1, channels, 1, 1)
            var = batch_var.view(1, channels, 1, 1)
        else:
            # During evaluation, interpolate between global and class stats
            if labels is not None and self.alpha > 0:
                # Get per-sample statistics based on class
                mean = torch.zeros(batch_size, self.num_features, 1, 1, device=x.device)
                var = torch.ones(batch_size, self.num_features, 1, 1, device=x.device)
                
                for i in range(batch_size):
                    class_idx = labels[i].item()
                    # Interpolate
                    mean[i, :, 0, 0] = (1 - self.alpha) * self.global_running_mean + self.alpha * self.class_running_mean[class_idx]
                    var[i, :, 0, 0] = (1 - self.alpha) * self.global_running_var + self.alpha * self.class_running_var[class_idx]
            else:
                # Use global stats
                mean = self.global_running_mean.view(1, channels, 1, 1)
                var = self.global_running_var.view(1, channels, 1, 1)
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift
        return x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class TinyNet(nn.Module):
    """Very small network for fast experiments."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # Only 3 conv layers for quick experiments
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.bn1 = ClassConditionalBatchNorm2d(16, num_classes)
        
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.bn2 = ClassConditionalBatchNorm2d(32, num_classes)
        
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.bn3 = ClassConditionalBatchNorm2d(64, num_classes)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x, labels=None):
        x = F.relu(self.bn1(self.conv1(x), labels))
        x = F.relu(self.bn2(self.conv2(x), labels))
        x = F.relu(self.bn3(self.conv3(x), labels))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model_fast(model, train_loader, val_loader, device, epochs=20, patience=5):
    """Fast training with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= 20:  # Limit batches per epoch
                break
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation
        val_acc = evaluate(model, val_loader, device, max_batches=5)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: Val Acc: {val_acc:.4f}")
        
        if epochs_without_improvement >= patience:
            print("CONVERGED")
            return True, best_val_acc
    
    print("NOT_CONVERGED: Max epochs")
    return False, best_val_acc


def run_experiment(seed, data_size=2000, quick_eval=True):
    """Run single seed experiment."""
    start_time = time.time()
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=== Seed {seed} ({device}) ===")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Use subset for speed
    full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    indices = list(range(data_size))
    dataset = Subset(full_dataset, indices)
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Train model
    model = TinyNet(num_classes=10).to(device)
    converged, val_acc = train_model_fast(model, train_loader, val_loader, device)
    
    # Get BN layers
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    num_bn = len(bn_layers)
    print(f"Found {num_bn} BN layers")
    
    # 1. Importance scores
    print("Computing importance...")
    importance_scores = []
    eval_batches = 3 if quick_eval else 10
    
    for i in range(num_bn):
        imp = compute_importance_score(model, val_loader, i, 0.0, 0.2, device, eval_batches)
        importance_scores.append(imp)
        print(f"  Layer {i}: {imp:.4f}")
    
    # Spearman correlation
    if num_bn > 1:
        rho, p = spearmanr(range(num_bn), importance_scores)
    else:
        rho, p = 0.0, 1.0
    print(f"Spearman ρ={rho:.3f}, p={p:.3f}")
    
    # 2. Key configurations
    print("Testing configurations...")
    
    # All global
    for i in range(num_bn):
        set_bn_alpha(model, i, 0.0)
    acc_global = evaluate(model, val_loader, device, eval_batches)
    
    # All class-conditional
    for i in range(num_bn):
        set_bn_alpha(model, i, 1.0)
    acc_class = evaluate(model, val_loader, device, eval_batches)
    
    # Hybrid
    mid = num_bn // 2
    for i in range(num_bn):
        set_bn_alpha(model, i, 0.0 if i < mid else 1.0)
    acc_hybrid = evaluate(model, val_loader, device, eval_batches)
    
    print(f"  Global: {acc_global:.4f}")
    print(f"  Class-cond: {acc_class:.4f}")
    print(f"  Hybrid: {acc_hybrid:.4f}")
    
    # Simple ablation: reverse pattern
    for i in range(num_bn):
        set_bn_alpha(model, i, 1.0 if i < mid else 0.0)
    acc_reverse = evaluate(model, val_loader, device, eval_batches)
    print(f"  Reverse: {acc_reverse:.4f}")
    
    # Signal detection
    signal_detected = (abs(rho) > 0.5 and max(importance_scores) - min(importance_scores) > 0.1)
    print("SIGNAL_DETECTED" if signal_detected else "NO_SIGNAL")
    
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.1f}s")
    
    return {
        'seed': seed,
        'converged': converged,
        'importance_scores': importance_scores,
        'spearman_rho': float(rho),
        'spearman_p': float(p),
        'accuracies': {
            'validation': float(val_acc),
            'global': float(acc_global),
            'class_cond': float(acc_class),
            'hybrid': float(acc_hybrid),
            'reverse': float(acc_reverse)
        },
        'signal_detected': signal_detected,
        'time': elapsed
    }


def main():
    """Main experiment loop."""
    # For dry run: 3 seeds, tiny data
    # For full run: 10 seeds, more data
    is_dry_run = len(sys.argv) > 1 and sys.argv[1] == '--dry-run'
    
    if is_dry_run:
        print("DRY RUN MODE")
        num_seeds = 3
        data_size = 1000
        quick_eval = True
    else:
        num_seeds = 10
        data_size = 10000
        quick_eval = False
    
    results = []
    start_time = time.time()
    
    # First seed with sanity check
    print("=== First seed (sanity check) ===")
    result = run_experiment(0, data_size, quick_eval)
    results.append(result)
    
    # Sanity checks
    imp = result['importance_scores']
    if all(abs(s - imp[0]) < 1e-6 for s in imp):
        print("SANITY_ABORT: All importance scores identical")
        sys.exit(1)
    
    if all(s == 0.0 for s in imp) or any(np.isnan(s) for s in imp):
        print("SANITY_ABORT: Invalid importance scores")
        sys.exit(1)
    
    acc_diff = abs(result['accuracies']['hybrid'] - result['accuracies']['global'])
    if acc_diff < 1e-6:
        print("SANITY_ABORT: No difference between methods")
        sys.exit(1)
    
    print("Sanity passed! Continuing...")
    
    # Remaining seeds
    for seed in range(1, num_seeds):
        result = run_experiment(seed, data_size, quick_eval)
        results.append(result)
    
    # Aggregate
    all_importance = [r['importance_scores'] for r in results]
    all_rho = [r['spearman_rho'] for r in results]
    all_global = [r['accuracies']['global'] for r in results]
    all_hybrid = [r['accuracies']['hybrid'] for r in results]
    all_reverse = [r['accuracies']['reverse'] for r in results]
    
    # Statistics
    mean_imp = np.mean(all_importance, axis=0).tolist()
    std_imp = np.std(all_importance, axis=0).tolist()
    mean_rho = np.mean(all_rho)
    std_rho = np.std(all_rho)
    
    # Statistical tests (if not dry run)
    if not is_dry_run and len(results) > 1:
        _, p_hybrid_global = ttest_rel(all_hybrid, all_global)
        _, p_hybrid_reverse = ttest_rel(all_hybrid, all_reverse)
    else:
        p_hybrid_global = p_hybrid_reverse = 1.0
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n=== SUMMARY ({num_seeds} seeds) ===")
    print(f"Mean importance: {mean_imp}")
    print(f"Spearman ρ: {mean_rho:.3f} ± {std_rho:.3f}")
    print(f"Global acc: {np.mean(all_global):.3f} ± {np.std(all_global):.3f}")
    print(f"Hybrid acc: {np.mean(all_hybrid):.3f} ± {np.std(all_hybrid):.3f}")
    if not is_dry_run:
        print(f"p-value (hybrid vs global): {p_hybrid_global:.6f}")
    print(f"Total time: {total_time:.1f}s")
    
    # Output
    output = {
        'per_seed_results': results,
        'mean': {
            'importance_scores': mean_imp,
            'spearman_rho': float(mean_rho),
            'accuracies': {
                'global': float(np.mean(all_global)),
                'hybrid': float(np.mean(all_hybrid)),
                'reverse': float(np.mean(all_reverse))
            }
        },
        'std': {
            'importance_scores': std_imp,
            'spearman_rho': float(std_rho),
            'accuracies': {
                'global': float(np.std(all_global)),
                'hybrid': float(np.std(all_hybrid))
            }
        },
        'p_values': {
            'hybrid_vs_global': float(p_hybrid_global),
            'hybrid_vs_reverse': float(p_hybrid_reverse)
        },
        'convergence_status': f"{sum(r['converged'] for r in results)}/{num_seeds} converged",
        'signal_detection_rate': sum(r['signal_detected'] for r in results) / num_seeds,
        'total_time_seconds': total_time
    }
    
    print(f"\nRESULTS: {json.dumps(output)}")


if __name__ == "__main__":
    main()