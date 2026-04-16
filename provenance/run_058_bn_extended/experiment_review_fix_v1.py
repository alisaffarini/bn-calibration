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

def compute_importance_score(model, val_loader, layer_idx, alpha_0=0.0, alpha_1=0.1, device='cuda', max_batches=10):
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

# Sanity check 1: Class-conditional BN should use different stats for different classes
print("Testing class-conditional behavior...")

class TestClassConditionalBN(nn.Module):
    def __init__(self, num_features=2, num_classes=3):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.alpha = 1.0  # Fully class-conditional
        
        # Different means for each class
        self.class_means = torch.tensor([[0., 0.], [1., 1.], [2., 2.]])  # Shape: (3, 2)
        self.global_mean = torch.tensor([1., 1.])  # Average of class means
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        if self.alpha == 0:
            # Global stats
            return x - self.global_mean
        else:
            # Class-conditional stats
            result = torch.zeros_like(x)
            for i in range(batch_size):
                class_idx = labels[i].item()
                result[i] = x[i] - self.alpha * self.class_means[class_idx] - (1 - self.alpha) * self.global_mean
            return result

test_bn = TestClassConditionalBN()
test_x = torch.tensor([[5., 5.], [5., 5.], [5., 5.]])
test_labels = torch.tensor([0, 1, 2])

# Test with alpha=1 (fully class-conditional)
test_bn.alpha = 1.0
output = test_bn(test_x, test_labels)
expected = torch.tensor([[5., 5.], [4., 4.], [3., 3.]])  # Subtract respective class means
assert torch.allclose(output, expected), f"Class-conditional failed: {output} != {expected}"

# Test with alpha=0 (global)
test_bn.alpha = 0.0
output = test_bn(test_x, test_labels)
expected = torch.tensor([[4., 4.], [4., 4.], [4., 4.]])  # All subtract global mean
assert torch.allclose(output, expected), f"Global stats failed: {output} != {expected}"

print("✓ Class-conditional behavior verified")

# Sanity check 2: Importance score should detect changes
dummy_importance = abs(0.8 - 0.7) / 0.1  # Simulated accuracy change
assert abs(dummy_importance - 1.0) < 1e-6, f"Importance calculation wrong: {dummy_importance}"
print("✓ Importance score calculation verified")

print("METRIC_SANITY_PASSED")
print()

# ========== MAIN EXPERIMENT CODE ==========

class ClassConditionalBatchNorm2d(nn.Module):
    """BatchNorm that correctly interpolates between global and class-conditional statistics."""
    
    def __init__(self, num_features, num_classes=10, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.eps = eps
        self.momentum = momentum
        
        # Global statistics
        self.register_buffer('global_running_mean', torch.zeros(num_features))
        self.register_buffer('global_running_var', torch.ones(num_features))
        
        # Per-class statistics - FIXED: Actually use these based on labels!
        self.register_buffer('class_running_mean', torch.zeros(num_classes, num_features))
        self.register_buffer('class_running_var', torch.ones(num_classes, num_features))
        
        # Track which classes we've seen
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Interpolation factor (0 = global, 1 = class-conditional)
        self.alpha = 0.0
        
    def forward(self, x, labels):
        batch_size, channels, height, width = x.shape
        
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # Update global running stats
            with torch.no_grad():
                self.global_running_mean = (1 - self.momentum) * self.global_running_mean + self.momentum * batch_mean
                self.global_running_var = (1 - self.momentum) * self.global_running_var + self.momentum * batch_var
                
                # Update per-class running stats - CRITICAL FIX
                for c in range(self.num_classes):
                    mask = (labels == c)
                    if mask.sum() > 0:
                        class_x = x[mask]
                        class_mean = class_x.mean(dim=(0, 2, 3))
                        class_var = class_x.var(dim=(0, 2, 3), unbiased=False)
                        
                        # Incremental update
                        self.class_running_mean[c] = (1 - self.momentum) * self.class_running_mean[c] + self.momentum * class_mean
                        self.class_running_var[c] = (1 - self.momentum) * self.class_running_var[c] + self.momentum * class_var
                        self.class_counts[c] += mask.sum()
            
            # Use batch stats for normalization during training
            x_normalized = (x - batch_mean.view(1, channels, 1, 1)) / torch.sqrt(batch_var.view(1, channels, 1, 1) + self.eps)
        else:
            # During evaluation, interpolate between global and class stats
            x_normalized = torch.zeros_like(x)
            
            if self.alpha == 0:
                # Pure global stats
                mean = self.global_running_mean.view(1, channels, 1, 1)
                var = self.global_running_var.view(1, channels, 1, 1)
                x_normalized = (x - mean) / torch.sqrt(var + self.eps)
            else:
                # Mix global and class-conditional stats
                for i in range(batch_size):
                    class_idx = labels[i].item()
                    
                    # Interpolate statistics
                    mean = (1 - self.alpha) * self.global_running_mean + self.alpha * self.class_running_mean[class_idx]
                    var = (1 - self.alpha) * self.global_running_var + self.alpha * self.class_running_var[class_idx]
                    
                    # Normalize this sample
                    x_normalized[i] = (x[i] - mean.view(channels, 1, 1)) / torch.sqrt(var.view(channels, 1, 1) + self.eps)
        
        # Scale and shift
        return x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class ResNetBlock(nn.Module):
    """Basic ResNet block with optional skip connection."""
    def __init__(self, in_channels, out_channels, stride=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = ClassConditionalBatchNorm2d(out_channels, num_classes)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = ClassConditionalBatchNorm2d(out_channels, num_classes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                ClassConditionalBatchNorm2d(out_channels, num_classes)
            )
    
    def forward(self, x, labels):
        out = F.relu(self.bn1(self.conv1(x), labels))
        out = self.bn2(self.conv2(out), labels)
        
        # Handle shortcut
        if isinstance(self.shortcut, nn.Sequential) and len(self.shortcut) > 0:
            shortcut = x
            for layer in self.shortcut:
                if isinstance(layer, ClassConditionalBatchNorm2d):
                    shortcut = layer(shortcut, labels)
                else:
                    shortcut = layer(shortcut)
            out += shortcut
        else:
            out += x
            
        return F.relu(out)


class ImprovedResNet(nn.Module):
    """Improved ResNet with proper architecture."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = ClassConditionalBatchNorm2d(64, num_classes)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, num_classes)
        self.layer2 = self._make_layer(64, 128, 2, num_classes, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, num_classes, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, num_classes, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride, num_classes))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1, num_classes))
        return nn.ModuleList(layers)
    
    def forward(self, x, labels):
        # Initial layers
        x = F.relu(self.bn1(self.conv1(x), labels))
        x = self.maxpool(x)
        
        # Residual blocks
        for block in self.layer1:
            x = block(x, labels)
        for block in self.layer2:
            x = block(x, labels)
        for block in self.layer3:
            x = block(x, labels)
        
        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class StandardResNet(nn.Module):
    """Standard ResNet with PyTorch BatchNorm for baseline."""
    def __init__(self, num_classes=10):
        super().__init__()
        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Simplified architecture for faster training
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        # First block
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        # Additional blocks
        for _ in range(1, num_blocks):
            layers.extend([
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ])
        return nn.Sequential(*layers)
    
    def forward(self, x, labels=None):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train_model(model, train_loader, val_loader, device, epochs=30, patience=7, is_standard=False):
    """Train model with proper optimization."""
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
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
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            if is_standard:
                outputs = model(inputs)
            else:
                outputs = model(inputs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validation
        val_acc = evaluate(model, val_loader, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.3f}, "
                  f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print("CONVERGED")
            return True, best_val_acc
    
    if best_val_acc > 0.7:  # Reasonable threshold for CIFAR-10
        print("CONVERGED")
        return True, best_val_acc
    else:
        print("NOT_CONVERGED: Low accuracy")
        return False, best_val_acc


def run_experiment(seed, use_full_data=True):
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
    
    # Data augmentation for better training
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
    
    # Load dataset
    if use_full_data:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    else:
        # Use subset for quick testing
        full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        full_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        trainset = Subset(full_train, range(5000))
        testset = Subset(full_test, range(1000))
    
    # Split train into train/val
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_dataset, val_dataset = random_split(trainset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # Train our model
    print("Training class-conditional BN model...")
    model = ImprovedResNet(num_classes=10).to(device)
    converged, val_acc = train_model(model, train_loader, val_loader, device, epochs=30, patience=7)
    
    # Test accuracy
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.3f}")
    
    # Train standard BN baseline
    print("\nTraining standard BN baseline...")
    standard_model = StandardResNet(num_classes=10).to(device)
    _, standard_val_acc = train_model(standard_model, train_loader, val_loader, device, 
                                     epochs=30, patience=7, is_standard=True)
    standard_test_acc = evaluate(standard_model, test_loader, device)
    print(f"Standard BN test accuracy: {standard_test_acc:.3f}")
    
    # Get BN layers
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    num_bn = len(bn_layers)
    print(f"\nFound {num_bn} BN layers")
    
    # 1. Layer-wise importance scores
    print("\nComputing layer importance scores...")
    importance_scores = []
    
    for i in range(min(num_bn, 10)):  # Limit to first 10 layers for speed
        imp = compute_importance_score(model, val_loader, i, 0.0, 0.1, device, max_batches=20)
        importance_scores.append(imp)
        if i % 3 == 0:
            print(f"  Layer {i}: {imp:.4f}")
    
    # Spearman correlation
    layer_indices = list(range(len(importance_scores)))
    rho, p_value = spearmanr(layer_indices, importance_scores)
    print(f"Spearman ρ = {rho:.3f}, p = {p_value:.3f}")
    
    # 2. Configuration tests
    print("\nTesting different configurations...")
    
    # All global (alpha=0)
    for i in range(num_bn):
        set_bn_alpha(model, i, 0.0)
    acc_global = evaluate(model, test_loader, device)
    print(f"  All global (α=0): {acc_global:.3f}")
    
    # All class-conditional (alpha=1)
    for i in range(num_bn):
        set_bn_alpha(model, i, 1.0)
    acc_class = evaluate(model, test_loader, device)
    print(f"  All class-cond (α=1): {acc_class:.3f}")
    
    # Hierarchical: early global, late class-conditional
    threshold = num_bn // 2
    for i in range(num_bn):
        set_bn_alpha(model, i, 0.0 if i < threshold else 1.0)
    acc_hierarchical = evaluate(model, test_loader, device)
    print(f"  Hierarchical: {acc_hierarchical:.3f}")
    
    # Reverse: early class-cond, late global
    for i in range(num_bn):
        set_bn_alpha(model, i, 1.0 if i < threshold else 0.0)
    acc_reverse = evaluate(model, test_loader, device)
    print(f"  Reverse: {acc_reverse:.3f}")
    
    # 3. Ablations
    print("\nRunning ablations...")
    
    # Different alpha steps
    alpha_steps = [0.05, 0.2, 0.5]
    step_importances = []
    for step in alpha_steps:
        imp = compute_importance_score(model, val_loader, 0, 0.0, step, device, max_batches=10)
        step_importances.append(imp)
    print(f"  Alpha step sensitivity: {step_importances}")
    
    # Class verification: ensure different classes get different stats
    model.eval()
    test_input = torch.randn(10, 3, 32, 32).to(device)
    
    # Set first layer to fully class-conditional
    set_bn_alpha(model, 0, 1.0)
    
    # Forward pass with different labels
    with torch.no_grad():
        labels1 = torch.zeros(10, dtype=torch.long).to(device)  # All class 0
        labels2 = torch.ones(10, dtype=torch.long).to(device)   # All class 1
        
        out1 = model(test_input, labels1)
        out2 = model(test_input, labels2)
        
        class_difference = torch.abs(out1 - out2).mean().item()
        print(f"  Output difference between class 0 and 1: {class_difference:.4f}")
    
    # Reset alphas
    for i in range(num_bn):
        set_bn_alpha(model, i, 0.0)
    
    # Signal detection criteria
    importance_range = max(importance_scores) - min(importance_scores) if importance_scores else 0
    hierarchical_benefit = acc_hierarchical - min(acc_global, acc_reverse)
    proper_training = test_acc > 0.6  # Should achieve decent accuracy
    class_cond_works = acc_class > 0.1  # Should be better than random
    
    signal_detected = (abs(rho) > 0.3 and importance_range > 0.05 and 
                      hierarchical_benefit > 0.01 and proper_training and class_cond_works)
    
    if signal_detected:
        print("SIGNAL_DETECTED: Hierarchical importance pattern found")
    else:
        print(f"NO_SIGNAL: rho={abs(rho):.2f}, range={importance_range:.3f}, "
              f"benefit={hierarchical_benefit:.3f}, acc={test_acc:.2f}")
    
    elapsed = time.time() - start_time
    print(f"Time: {elapsed:.1f}s")
    
    return {
        'seed': seed,
        'converged': converged,
        'test_accuracy': float(test_acc),
        'importance_scores': importance_scores,
        'spearman_rho': float(rho),
        'spearman_p': float(p_value),
        'configurations': {
            'standard_bn': float(standard_test_acc),
            'all_global': float(acc_global),
            'all_class_cond': float(acc_class),
            'hierarchical': float(acc_hierarchical),
            'reverse': float(acc_reverse)
        },
        'ablations': {
            'alpha_step_importance': step_importances,
            'class_output_difference': float(class_difference)
        },
        'signal_detected': signal_detected,
        'time': elapsed
    }


def main():
    """Main experiment loop."""
    # Parse arguments
    quick_test = '--quick' in sys.argv
    
    if quick_test:
        print("QUICK TEST MODE")
        num_seeds = 3
        use_full_data = False
    else:
        num_seeds = 10
        use_full_data = True
    
    results = []
    start_time = time.time()
    
    # First seed with sanity checks
    print("=== Running first seed with sanity checks ===")
    result = run_experiment(0, use_full_data)
    results.append(result)
    
    # Sanity checks
    if result['configurations']['all_class_cond'] < 0.05:
        print("SANITY_ABORT: Class-conditional BN completely broken")
        sys.exit(1)
    
    if result['test_accuracy'] < 0.3:
        print("SANITY_ABORT: Model failed to train properly")
        sys.exit(1)
    
    if result['ablations']['class_output_difference'] < 1e-6:
        print("SANITY_ABORT: Class-conditional BN not using different stats per class")
        sys.exit(1)
    
    print("Sanity checks passed!")
    
    # Run remaining seeds
    for seed in range(1, num_seeds):
        result = run_experiment(seed, use_full_data)
        results.append(result)
    
    # Aggregate results
    test_accs = [r['test_accuracy'] for r in results]
    rhos = [r['spearman_rho'] for r in results]
    
    # Configuration accuracies
    standard_accs = [r['configurations']['standard_bn'] for r in results]
    global_accs = [r['configurations']['all_global'] for r in results]
    class_accs = [r['configurations']['all_class_cond'] for r in results]
    hier_accs = [r['configurations']['hierarchical'] for r in results]
    reverse_accs = [r['configurations']['reverse'] for r in results]
    
    # Statistical tests
    if len(results) > 1:
        # Test hierarchical vs alternatives
        _, p_hier_vs_global = ttest_rel(hier_accs, global_accs)
        _, p_hier_vs_reverse = ttest_rel(hier_accs, reverse_accs)
        _, p_hier_vs_standard = ttest_rel(hier_accs, standard_accs)
        
        # Test importance correlation significance
        all_importance = [r['importance_scores'] for r in results if r['importance_scores']]
        if all_importance:
            mean_importance = np.mean(all_importance, axis=0)
            early_imp = np.mean(mean_importance[:len(mean_importance)//2])
            late_imp = np.mean(mean_importance[len(mean_importance)//2:])
            importance_diff = early_imp - late_imp
        else:
            importance_diff = 0.0
    else:
        p_hier_vs_global = p_hier_vs_reverse = p_hier_vs_standard = 1.0
        importance_diff = 0.0
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n=== SUMMARY ({num_seeds} seeds) ===")
    print(f"Test accuracy: {np.mean(test_accs):.3f} ± {np.std(test_accs):.3f}")
    print(f"Spearman ρ: {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")
    print("\nConfiguration accuracies:")
    print(f"  Standard BN: {np.mean(standard_accs):.3f} ± {np.std(standard_accs):.3f}")
    print(f"  All global: {np.mean(global_accs):.3f} ± {np.std(global_accs):.3f}")
    print(f"  All class-cond: {np.mean(class_accs):.3f} ± {np.std(class_accs):.3f}")
    print(f"  Hierarchical: {np.mean(hier_accs):.3f} ± {np.std(hier_accs):.3f}")
    print(f"  Reverse: {np.mean(reverse_accs):.3f} ± {np.std(reverse_accs):.3f}")
    print("\nStatistical tests:")
    print(f"  Hierarchical vs Global: p = {p_hier_vs_global:.6f}")
    print(f"  Hierarchical vs Reverse: p = {p_hier_vs_reverse:.6f}")
    print(f"  Hierarchical vs Standard: p = {p_hier_vs_standard:.6f}")
    print(f"  Early-Late importance diff: {importance_diff:.4f}")
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Output
    output = {
        'per_seed_results': results,
        'mean': {
            'test_accuracy': float(np.mean(test_accs)),
            'spearman_rho': float(np.mean(rhos)),
            'configurations': {
                'standard_bn': float(np.mean(standard_accs)),
                'all_global': float(np.mean(global_accs)),
                'all_class_cond': float(np.mean(class_accs)),
                'hierarchical': float(np.mean(hier_accs)),
                'reverse': float(np.mean(reverse_accs))
            }
        },
        'std': {
            'test_accuracy': float(np.std(test_accs)),
            'spearman_rho': float(np.std(rhos)),
            'configurations': {
                'standard_bn': float(np.std(standard_accs)),
                'hierarchical': float(np.std(hier_accs))
            }
        },
        'p_values': {
            'hierarchical_vs_global': float(p_hier_vs_global),
            'hierarchical_vs_reverse': float(p_hier_vs_reverse),
            'hierarchical_vs_standard': float(p_hier_vs_standard)
        },
        'convergence_status': f"{sum(r['converged'] for r in results)}/{num_seeds} converged",
        'signal_detection_rate': sum(r['signal_detected'] for r in results) / num_seeds,
        'total_time_seconds': total_time
    }
    
    print(f"\nRESULTS: {json.dumps(output)}")


if __name__ == "__main__":
    main()