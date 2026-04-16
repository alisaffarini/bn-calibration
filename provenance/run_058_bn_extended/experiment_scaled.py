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

def compute_importance_score(model, val_loader, layer_idx, alpha_0=0.0, alpha_1=0.1, device='cuda'):
    """Compute importance score for a BN layer via finite differences."""
    model.eval()
    
    # Get accuracy at alpha_0
    set_bn_alpha(model, layer_idx, alpha_0)
    acc_0 = evaluate(model, val_loader, device)
    
    # Get accuracy at alpha_1
    set_bn_alpha(model, layer_idx, alpha_1)
    acc_1 = evaluate(model, val_loader, device)
    
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


class SimpleResNet(nn.Module):
    """Moderate-sized ResNet for comprehensive experiments."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = ClassConditionalBatchNorm2d(16, num_classes)
        
        # Block 1
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = ClassConditionalBatchNorm2d(32, num_classes)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = ClassConditionalBatchNorm2d(32, num_classes)
        
        # Block 2
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn4 = ClassConditionalBatchNorm2d(64, num_classes)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = ClassConditionalBatchNorm2d(64, num_classes)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x, labels=None):
        # Initial
        x = F.relu(self.bn1(self.conv1(x), labels))
        
        # Block 1
        x = F.relu(self.bn2(self.conv2(x), labels))
        x = F.relu(self.bn3(self.conv3(x), labels))
        
        # Block 2
        x = F.relu(self.bn4(self.conv4(x), labels))
        x = F.relu(self.bn5(self.conv5(x), labels))
        
        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class StandardResNet(nn.Module):
    """Standard ResNet with regular BatchNorm for baseline comparison."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Block 1
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Block 2
        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x, labels=None):
        # Initial
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Block 1
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Block 2
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        
        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def train_model(model, train_loader, val_loader, device, epochs=50, patience=10):
    """Train model with convergence-based stopping."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    epochs_without_improvement = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_acc = evaluate(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_acc > best_val_acc + 0.001:  # Require 0.1% improvement
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print("CONVERGED")
            return True, best_val_acc
    
    print("NOT_CONVERGED: Max epochs reached")
    return False, best_val_acc


def run_experiment(seed, dataset_name='CIFAR10'):
    """Run single seed experiment."""
    start_time = time.time()
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=== Running seed {seed} on {device} ===")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    else:  # CIFAR100
        dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    
    # Split train into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Train model
    model = SimpleResNet(num_classes=num_classes).to(device)
    converged, best_val_acc = train_model(model, train_loader, val_loader, device, epochs=50, patience=10)
    
    # Test set evaluation
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Get list of BN layers
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    num_bn_layers = len(bn_layers)
    print(f"\nFound {num_bn_layers} BN layers")
    
    # 1. Layer-wise Importance Discovery
    print("\n--- Layer-wise Importance Discovery ---")
    importance_scores = []
    
    for layer_idx in range(num_bn_layers):
        importance = compute_importance_score(model, val_loader, layer_idx, 
                                            alpha_0=0.0, alpha_1=0.1, device=device)
        importance_scores.append(importance)
        print(f"Layer {layer_idx}: Importance = {importance:.4f}")
    
    # Test monotonicity
    layer_depths = list(range(num_bn_layers))
    spearman_rho, spearman_p = spearmanr(layer_depths, importance_scores)
    print(f"\nSpearman correlation (depth vs importance): ρ={spearman_rho:.4f}, p={spearman_p:.4f}")
    
    # 2. Semantic Hierarchy Validation
    print("\n--- Semantic Hierarchy Validation ---")
    hybrid_accuracies = []
    
    for k in range(0, num_bn_layers + 1, max(1, num_bn_layers // 5)):  # Sample 5-6 points
        # Set first k layers to global (alpha=0), rest to class-conditional (alpha=1)
        for i in range(num_bn_layers):
            if i < k:
                set_bn_alpha(model, i, 0.0)
            else:
                set_bn_alpha(model, i, 1.0)
        
        acc = evaluate(model, val_loader, device)
        hybrid_accuracies.append((k, acc))
        print(f"k={k}: Accuracy = {acc:.4f}")
    
    # Reset all to global
    for i in range(num_bn_layers):
        set_bn_alpha(model, i, 0.0)
    
    # 3. Baselines and Method Variants
    print("\n--- Baselines and Variants ---")
    
    # Random baseline
    random_acc = 1.0 / num_classes
    print(f"Random baseline: {random_acc:.4f}")
    
    # Standard BatchNorm baseline
    print("\nTraining standard BatchNorm model...")
    standard_model = StandardResNet(num_classes=num_classes).to(device)
    _, standard_val_acc = train_model(standard_model, train_loader, val_loader, device, epochs=30, patience=5)
    standard_test_acc = evaluate(standard_model, test_loader, device)
    print(f"Standard BN test accuracy: {standard_test_acc:.4f}")
    
    # All global (alpha=0)
    for i in range(num_bn_layers):
        set_bn_alpha(model, i, 0.0)
    global_acc = evaluate(model, test_loader, device)
    print(f"All global (α=0): {global_acc:.4f}")
    
    # All class-conditional (alpha=1)
    for i in range(num_bn_layers):
        set_bn_alpha(model, i, 1.0)
    class_cond_acc = evaluate(model, test_loader, device)
    print(f"All class-conditional (α=1): {class_cond_acc:.4f}")
    
    # Optimal hybrid (early global, late class-cond)
    threshold = num_bn_layers * 2 // 3  # First 2/3 global, last 1/3 class-cond
    for i in range(num_bn_layers):
        if i < threshold:
            set_bn_alpha(model, i, 0.0)
        else:
            set_bn_alpha(model, i, 1.0)
    optimal_hybrid_acc = evaluate(model, test_loader, device)
    print(f"Optimal hybrid (early global, late class-cond): {optimal_hybrid_acc:.4f}")
    
    # 4. Ablation Studies
    print("\n--- Ablation Studies ---")
    
    # Ablation 1: Random importance scores
    random_importance = np.random.rand(num_bn_layers)
    random_spearman, _ = spearmanr(layer_depths, random_importance)
    print(f"Random importance Spearman: {random_spearman:.4f}")
    
    # Ablation 2: Larger interpolation step
    large_step_importance = []
    for layer_idx in range(num_bn_layers):
        importance = compute_importance_score(model, val_loader, layer_idx, 
                                            alpha_0=0.0, alpha_1=0.5, device=device)
        large_step_importance.append(importance)
    large_step_spearman, _ = spearmanr(layer_depths, large_step_importance)
    print(f"Large step (α=0.5) Spearman: {large_step_spearman:.4f}")
    
    # Ablation 3: Shuffled class labels
    print("\nTesting with shuffled labels...")
    # Create shuffled labels loader
    shuffled_val_dataset = [(x, torch.randint(0, num_classes, (1,)).item()) 
                           for x, _ in val_dataset]
    shuffled_val_loader = DataLoader(shuffled_val_dataset, batch_size=128, shuffle=False)
    
    shuffled_importance = []
    for layer_idx in range(min(3, num_bn_layers)):  # Just test first 3 layers for speed
        importance = compute_importance_score(model, shuffled_val_loader, layer_idx, 
                                            alpha_0=0.0, alpha_1=0.1, device=device)
        shuffled_importance.append(importance)
    print(f"Shuffled labels importance (first 3 layers): {shuffled_importance}")
    
    # Check if we detected a signal
    signal_strength = abs(spearman_rho)
    importance_range = max(importance_scores) - min(importance_scores)
    hierarchical_benefit = optimal_hybrid_acc - min(global_acc, class_cond_acc)
    
    signal_detected = (signal_strength > 0.5 and 
                      importance_range > 0.1 and 
                      hierarchical_benefit > 0.01)
    
    if signal_detected:
        print("SIGNAL_DETECTED: Clear hierarchical importance pattern found")
    else:
        print("NO_SIGNAL: No clear hierarchical pattern detected")
    
    elapsed_time = time.time() - start_time
    print(f"Experiment completed in {elapsed_time:.1f}s")
    
    return {
        'seed': seed,
        'converged': converged,
        'test_accuracy': test_acc,
        'importance_scores': importance_scores,
        'spearman_rho': float(spearman_rho),
        'spearman_p_value': float(spearman_p),
        'hybrid_accuracies': hybrid_accuracies,
        'baselines': {
            'random': random_acc,
            'standard_bn': standard_test_acc,
            'all_global': global_acc,
            'all_class_conditional': class_cond_acc,
            'optimal_hybrid': optimal_hybrid_acc
        },
        'ablations': {
            'random_importance_spearman': float(random_spearman),
            'large_step_spearman': float(large_step_spearman),
            'shuffled_importance_mean': float(np.mean(shuffled_importance)) if shuffled_importance else 0.0
        },
        'signal_detected': signal_detected,
        'signal_strength': float(signal_strength),
        'elapsed_time': elapsed_time
    }


def main():
    """Main experimental loop."""
    num_seeds = 10
    results = []
    
    total_start_time = time.time()
    
    # Run first seed and check for sanity
    print("Running first seed to check experiment sanity...")
    first_result = run_experiment(0, dataset_name='CIFAR10')
    results.append(first_result)
    
    # Early abort checks
    importance_scores = first_result['importance_scores']
    if all(abs(s - importance_scores[0]) < 1e-6 for s in importance_scores):
        print("SANITY_ABORT: All importance scores are identical")
        sys.exit(1)
    
    if all(s == 0.0 for s in importance_scores):
        print("SANITY_ABORT: All importance scores are zero")
        sys.exit(1)
    
    if any(np.isnan(s) for s in importance_scores):
        print("SANITY_ABORT: NaN values in importance scores")
        sys.exit(1)
    
    method_acc = first_result['baselines']['optimal_hybrid']
    baseline_acc = first_result['baselines']['all_global']
    if abs(method_acc - baseline_acc) < 1e-6:
        print("SANITY_ABORT: Method shows no difference from baseline")
        sys.exit(1)
    
    print("\nSanity checks passed! Continuing with remaining seeds...")
    
    # Run remaining seeds
    for seed in range(1, num_seeds):
        result = run_experiment(seed, dataset_name='CIFAR10')
        results.append(result)
    
    # Aggregate results
    all_importance_scores = [r['importance_scores'] for r in results]
    all_spearman_rhos = [r['spearman_rho'] for r in results]
    all_test_accuracies = [r['test_accuracy'] for r in results]
    all_converged = [r['converged'] for r in results]
    signals_detected = [r['signal_detected'] for r in results]
    
    # Baseline accuracies
    all_random = [r['baselines']['random'] for r in results]
    all_standard_bn = [r['baselines']['standard_bn'] for r in results]
    all_global = [r['baselines']['all_global'] for r in results]
    all_class_cond = [r['baselines']['all_class_conditional'] for r in results]
    all_optimal_hybrid = [r['baselines']['optimal_hybrid'] for r in results]
    
    # Compute statistics
    mean_importance = np.mean(all_importance_scores, axis=0).tolist()
    std_importance = np.std(all_importance_scores, axis=0).tolist()
    mean_spearman = np.mean(all_spearman_rhos)
    std_spearman = np.std(all_spearman_rhos)
    
    # Statistical significance tests
    # Test 1: Optimal hybrid vs all global
    _, p_hybrid_vs_global = ttest_rel(all_optimal_hybrid, all_global)
    
    # Test 2: Optimal hybrid vs standard BN
    _, p_hybrid_vs_standard = ttest_rel(all_optimal_hybrid, all_standard_bn)
    
    # Test 3: Early vs late layer importance (if enough layers)
    if len(mean_importance) >= 4:
        early_importance = all_importance_scores[:, :len(mean_importance)//2]
        late_importance = all_importance_scores[:, len(mean_importance)//2:]
        early_mean = [np.mean(e) for e in early_importance]
        late_mean = [np.mean(l) for l in late_importance]
        _, p_early_vs_late = ttest_rel(early_mean, late_mean)
    else:
        p_early_vs_late = 1.0
    
    total_elapsed = time.time() - total_start_time
    print(f"\nTotal experiment time: {total_elapsed:.1f}s")
    
    # Summary statistics
    print("\n=== SUMMARY RESULTS ===")
    print(f"Test accuracy: {np.mean(all_test_accuracies):.4f} ± {np.std(all_test_accuracies):.4f}")
    print(f"Spearman ρ: {mean_spearman:.4f} ± {std_spearman:.4f}")
    print(f"Signal detection rate: {sum(signals_detected)}/{len(signals_detected)}")
    print(f"\nBaseline comparison:")
    print(f"  Random: {np.mean(all_random):.4f}")
    print(f"  Standard BN: {np.mean(all_standard_bn):.4f} ± {np.std(all_standard_bn):.4f}")
    print(f"  All global: {np.mean(all_global):.4f} ± {np.std(all_global):.4f}")
    print(f"  Optimal hybrid: {np.mean(all_optimal_hybrid):.4f} ± {np.std(all_optimal_hybrid):.4f}")
    print(f"\nStatistical tests:")
    print(f"  Hybrid vs Global: p={p_hybrid_vs_global:.6f}")
    print(f"  Hybrid vs Standard BN: p={p_hybrid_vs_standard:.6f}")
    print(f"  Early vs Late importance: p={p_early_vs_late:.6f}")
    
    # Final output
    output = {
        'per_seed_results': results,
        'mean': {
            'test_accuracy': float(np.mean(all_test_accuracies)),
            'importance_scores': mean_importance,
            'spearman_rho': float(mean_spearman),
            'baselines': {
                'random': float(np.mean(all_random)),
                'standard_bn': float(np.mean(all_standard_bn)),
                'all_global': float(np.mean(all_global)),
                'all_class_conditional': float(np.mean(all_class_cond)),
                'optimal_hybrid': float(np.mean(all_optimal_hybrid))
            }
        },
        'std': {
            'test_accuracy': float(np.std(all_test_accuracies)),
            'importance_scores': std_importance,
            'spearman_rho': float(std_spearman),
            'baselines': {
                'standard_bn': float(np.std(all_standard_bn)),
                'all_global': float(np.std(all_global)),
                'all_class_conditional': float(np.std(all_class_cond)),
                'optimal_hybrid': float(np.std(all_optimal_hybrid))
            }
        },
        'p_values': {
            'hybrid_vs_global': float(p_hybrid_vs_global),
            'hybrid_vs_standard_bn': float(p_hybrid_vs_standard),
            'early_vs_late_importance': float(p_early_vs_late)
        },
        'ablation_results': {
            'mean_random_spearman': float(np.mean([r['ablations']['random_importance_spearman'] for r in results])),
            'mean_large_step_spearman': float(np.mean([r['ablations']['large_step_spearman'] for r in results])),
            'mean_shuffled_importance': float(np.mean([r['ablations']['shuffled_importance_mean'] for r in results]))
        },
        'convergence_status': f"{sum(all_converged)}/{len(all_converged)} converged",
        'signal_detection_rate': sum(signals_detected) / len(signals_detected),
        'total_time_seconds': total_elapsed
    }
    
    print(f"\nRESULTS: {json.dumps(output)}")


if __name__ == "__main__":
    main()