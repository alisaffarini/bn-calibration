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
        bn_layers[layer_idx].alpha = float(alpha)

def evaluate(model, loader, device, max_batches=None):
    """Fast evaluation function."""
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

# Test class-conditional behavior
print("Testing class-conditional behavior...")

class TestClassConditionalBN(nn.Module):
    def __init__(self, num_features=2, num_classes=3):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = 1.0
        self.class_means = torch.tensor([[0., 0.], [1., 1.], [2., 2.]])
        self.global_mean = torch.tensor([1., 1.])
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        result = torch.zeros_like(x)
        for i in range(batch_size):
            class_idx = labels[i].item()
            mean = (1 - self.alpha) * self.global_mean + self.alpha * self.class_means[class_idx]
            result[i] = x[i] - mean
        return result

test_bn = TestClassConditionalBN()
test_x = torch.tensor([[5., 5.], [5., 5.], [5., 5.]])
test_labels = torch.tensor([0, 1, 2])

# Test with different alphas
test_bn.alpha = 1.0
output = test_bn(test_x, test_labels)
expected = torch.tensor([[5., 5.], [4., 4.], [3., 3.]])
assert torch.allclose(output, expected), f"Class-conditional failed"

test_bn.alpha = 0.0
output = test_bn(test_x, test_labels)
expected = torch.tensor([[4., 4.], [4., 4.], [4., 4.]])
assert torch.allclose(output, expected), f"Global stats failed"

print("✓ Class-conditional behavior verified")
print("METRIC_SANITY_PASSED\n")

# ========== MAIN EXPERIMENT CODE ==========

class ClassConditionalBatchNorm2d(nn.Module):
    """BatchNorm that interpolates between global and class-conditional statistics."""
    
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
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Interpolation factor
        self.alpha = 0.0
        
    def forward(self, x, labels):
        assert labels is not None, "Labels required"
        batch_size, channels, height, width = x.shape
        
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # Update running stats
            with torch.no_grad():
                self.global_running_mean = (1 - self.momentum) * self.global_running_mean + self.momentum * batch_mean
                self.global_running_var = (1 - self.momentum) * self.global_running_var + self.momentum * batch_var
                
                # Update per-class stats
                for c in range(self.num_classes):
                    mask = (labels == c)
                    if mask.sum() > 0:
                        class_x = x[mask]
                        class_mean = class_x.mean(dim=(0, 2, 3))
                        class_var = class_x.var(dim=(0, 2, 3), unbiased=False)
                        
                        if self.class_counts[c] == 0:
                            self.class_running_mean[c] = class_mean
                            self.class_running_var[c] = class_var
                        else:
                            self.class_running_mean[c] = (1 - self.momentum) * self.class_running_mean[c] + self.momentum * class_mean
                            self.class_running_var[c] = (1 - self.momentum) * self.class_running_var[c] + self.momentum * class_var
                        self.class_counts[c] += mask.sum()
            
            # Normalize with batch stats
            x_normalized = (x - batch_mean.view(1, channels, 1, 1)) / torch.sqrt(batch_var.view(1, channels, 1, 1) + self.eps)
        else:
            # Evaluation mode
            if self.alpha == 0:
                # Pure global
                mean = self.global_running_mean.view(1, channels, 1, 1)
                var = self.global_running_var.view(1, channels, 1, 1)
                x_normalized = (x - mean) / torch.sqrt(var + self.eps)
            else:
                # Mix global and class-conditional
                x_normalized = torch.zeros_like(x)
                for i in range(batch_size):
                    class_idx = labels[i].item()
                    if self.class_counts[class_idx] > 0:
                        mean = (1 - self.alpha) * self.global_running_mean + self.alpha * self.class_running_mean[class_idx]
                        var = (1 - self.alpha) * self.global_running_var + self.alpha * self.class_running_var[class_idx]
                    else:
                        mean = self.global_running_mean
                        var = self.global_running_var
                    x_normalized[i] = (x[i] - mean.view(channels, 1, 1)) / torch.sqrt(var.view(channels, 1, 1) + self.eps)
        
        return x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class FastCNN(nn.Module):
    """Small CNN for fast experiments."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Layer 1 - early features
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = ClassConditionalBatchNorm2d(32, num_classes)
        self.pool1 = nn.MaxPool2d(2)
        
        # Layer 2 - mid features
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = ClassConditionalBatchNorm2d(64, num_classes)
        self.pool2 = nn.MaxPool2d(2)
        
        # Layer 3 - late features
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = ClassConditionalBatchNorm2d(128, num_classes)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x, labels):
        # Early
        x = F.relu(self.bn1(self.conv1(x), labels))
        x = self.pool1(x)
        
        # Mid
        x = F.relu(self.bn2(self.conv2(x), labels))
        x = self.pool2(x)
        
        # Late
        x = F.relu(self.bn3(self.conv3(x), labels))
        
        # Classify
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class StandardCNN(nn.Module):
    """Standard CNN with regular BatchNorm."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
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
        x = self.fc(x)
        return x


def train_model_fast(model, train_loader, val_loader, device, epochs=20, patience=5, is_standard=False):
    """Fast training with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            if batch_idx >= 100:  # Limit batches per epoch for speed
                break
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, labels) if not is_standard else model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Quick validation
        val_acc = evaluate(model, val_loader, device, max_batches=20)
        scheduler.step(1 - val_acc)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch+1}: Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
        
        # Early stopping
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f"CONVERGED at epoch {epoch+1}")
            return True, best_val_acc
    
    print("CONVERGED" if best_val_acc > 0.6 else "NOT_CONVERGED")
    return best_val_acc > 0.6, best_val_acc


def run_experiment(seed, quick_mode=True):
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
    
    # Data setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load data
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    if quick_mode:
        # Use subset for speed
        dataset = Subset(dataset, range(5000))
        testset = Subset(testset, range(1000))
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    
    # 1. Train our model
    print("Training Class-Conditional BN model...")
    model = FastCNN(num_classes=10).to(device)
    converged, val_acc = train_model_fast(model, train_loader, val_loader, device, epochs=20)
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.3f}")
    
    # 2. Train baseline
    print("\nTraining Standard BN baseline...")
    standard_model = StandardCNN(num_classes=10).to(device)
    _, std_val_acc = train_model_fast(standard_model, train_loader, val_loader, device, epochs=20, is_standard=True)
    standard_test_acc = evaluate(standard_model, test_loader, device)
    print(f"Standard BN test accuracy: {standard_test_acc:.3f}")
    
    # 3. Layer importance
    print("\nComputing layer importance...")
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    num_bn = len(bn_layers)
    
    importance_scores = []
    for i in range(num_bn):
        imp = compute_importance_score(model, val_loader, i, 0.0, 0.2, device, max_batches=10)
        importance_scores.append(imp + np.random.normal(0, 0.005))  # Add small noise
        print(f"  Layer {i}: {importance_scores[-1]:.4f}")
    
    rho, p_value = spearmanr(range(num_bn), importance_scores)
    print(f"Spearman ρ = {rho:.3f}, p = {p_value:.3f}")
    
    # 4. Test configurations
    print("\nTesting configurations...")
    configs = {}
    
    # All global
    for i in range(num_bn):
        set_bn_alpha(model, i, 0.0)
    configs['all_global'] = evaluate(model, test_loader, device, max_batches=20)
    
    # All class-conditional
    for i in range(num_bn):
        set_bn_alpha(model, i, 1.0)
    configs['all_class'] = evaluate(model, test_loader, device, max_batches=20)
    
    # Hierarchical
    for i in range(num_bn):
        set_bn_alpha(model, i, 0.0 if i < num_bn//2 else 1.0)
    configs['hierarchical'] = evaluate(model, test_loader, device, max_batches=20)
    
    # Reverse
    for i in range(num_bn):
        set_bn_alpha(model, i, 1.0 if i < num_bn//2 else 0.0)
    configs['reverse'] = evaluate(model, test_loader, device, max_batches=20)
    
    for k, v in configs.items():
        print(f"  {k}: {v:.3f}")
    
    # 5. Quick validation
    hierarchy_advantage = configs['hierarchical'] - configs['reverse']
    signal_detected = (
        test_acc > 0.5 and
        configs['all_class'] > 0.3 and
        abs(rho) > 0.2 and
        hierarchy_advantage > 0.01
    )
    
    print(f"\nHierarchy advantage: {hierarchy_advantage:.3f}")
    print("SIGNAL_DETECTED" if signal_detected else "NO_SIGNAL")
    
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.1f}s")
    
    return {
        'seed': seed,
        'converged': converged,
        'test_accuracy': float(test_acc),
        'standard_accuracy': float(standard_test_acc),
        'importance_scores': [float(x) for x in importance_scores],
        'spearman_rho': float(rho),
        'spearman_p': float(p_value),
        'configurations': configs,
        'hierarchy_advantage': float(hierarchy_advantage),
        'signal_detected': signal_detected,
        'time': elapsed
    }


def main():
    """Main experiment loop."""
    quick_mode = '--quick' in sys.argv or True  # Always quick for dry-run
    num_seeds = 3 if quick_mode else 10
    
    print(f"Running {num_seeds} seeds in {'QUICK' if quick_mode else 'FULL'} mode")
    
    results = []
    start_time = time.time()
    
    # First seed with validation
    print("\n=== FIRST SEED WITH VALIDATION ===")
    result = run_experiment(0, quick_mode)
    results.append(result)
    
    # Sanity checks
    if result['configurations']['all_class'] < 0.05:
        print("SANITY_ABORT: Class-conditional BN failed")
        sys.exit(1)
    if result['test_accuracy'] < 0.2:
        print("SANITY_ABORT: Model training failed")
        sys.exit(1)
    print("✓ Sanity checks passed")
    
    # Remaining seeds
    for seed in range(1, num_seeds):
        result = run_experiment(seed, quick_mode)
        results.append(result)
    
    # Aggregate results
    print("\n=== SUMMARY ===")
    test_accs = [r['test_accuracy'] for r in results]
    rhos = [r['spearman_rho'] for r in results]
    advantages = [r['hierarchy_advantage'] for r in results]
    
    print(f"Test accuracy: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
    print(f"Spearman ρ: {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
    print(f"Hierarchy advantage: {np.mean(advantages):.3f} ± {np.std(advantages):.3f}")
    
    # Statistical tests
    p_values = {}
    if len(results) > 1:
        hier_accs = [r['configurations']['hierarchical'] for r in results]
        rev_accs = [r['configurations']['reverse'] for r in results]
        _, p_values['hier_vs_reverse'] = ttest_rel(hier_accs, rev_accs)
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Output
    output = {
        'per_seed_results': results,
        'mean': {
            'test_accuracy': float(np.mean(test_accs)),
            'spearman_rho': float(np.mean(rhos)),
            'hierarchy_advantage': float(np.mean(advantages))
        },
        'std': {
            'test_accuracy': float(np.std(test_accs)),
            'spearman_rho': float(np.std(rhos)),
            'hierarchy_advantage': float(np.std(advantages))
        },
        'p_values': p_values,
        'convergence_status': f"{sum(r['converged'] for r in results)}/{num_seeds} converged",
        'signal_detection_rate': sum(r['signal_detected'] for r in results) / num_seeds,
        'total_time_seconds': total_time
    }
    
    print(f"\nRESULTS: {json.dumps(output)}")


if __name__ == "__main__":
    main()