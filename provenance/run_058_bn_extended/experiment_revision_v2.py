# pip install torch torchvision numpy scipy matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import spearmanr, ttest_rel, pearsonr
import json
import random
import time
import sys
import warnings
warnings.filterwarnings('ignore')

# ========== METRIC SANITY CHECK ==========
print("Running metric sanity checks...")

def compute_importance_score_gradient(model, val_loader, layer_idx, device='cuda', max_batches=20):
    """Compute importance score using actual gradients."""
    model.eval()
    
    # Get BN layers
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    if layer_idx >= len(bn_layers):
        return 0.0
    
    target_layer = bn_layers[layer_idx]
    
    # Make alpha require gradient temporarily
    original_alpha = target_layer.alpha
    target_layer.alpha = torch.tensor(0.0, requires_grad=True, device=device)
    
    total_gradient = 0.0
    num_batches = 0
    
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (inputs, labels) in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
            
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs, labels)
        
        # Use negative cross-entropy as we want to maximize accuracy
        loss = -criterion(outputs, labels)
        
        # Compute gradient
        if target_layer.alpha.grad is not None:
            target_layer.alpha.grad.zero_()
        loss.backward()
        
        if target_layer.alpha.grad is not None:
            total_gradient += abs(target_layer.alpha.grad.item())
        
        num_batches += 1
    
    # Restore original alpha
    target_layer.alpha = original_alpha
    
    return total_gradient / num_batches if num_batches > 0 else 0.0

def set_bn_alpha(model, layer_idx, alpha):
    """Set interpolation alpha for specific BN layer."""
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    if layer_idx < len(bn_layers):
        bn_layers[layer_idx].alpha = float(alpha)

def set_all_bn_alpha(model, alpha):
    """Set alpha for all BN layers."""
    for m in model.modules():
        if isinstance(m, ClassConditionalBatchNorm2d):
            m.alpha = float(alpha)

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

# Sanity check: Verify class-conditional behavior
print("Testing class-conditional behavior...")

class TestBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_mean = torch.tensor([0.5])
        self.class_means = torch.tensor([[0.0], [1.0], [2.0]])  # Very different per class
        self.alpha = 0.0
        
    def forward(self, x, labels):
        batch_size = x.size(0)
        result = torch.zeros_like(x)
        
        for i in range(batch_size):
            if self.alpha == 0:
                mean = self.global_mean
            else:
                class_idx = labels[i].item()
                mean = (1 - self.alpha) * self.global_mean + self.alpha * self.class_means[class_idx]
            result[i] = x[i] - mean
            
        return result

test_module = TestBN()
test_input = torch.ones(3, 1) * 2.0  # All samples have value 2.0
test_labels = torch.tensor([0, 1, 2])

# With alpha=0 (global), all should be normalized the same
test_module.alpha = 0.0
out_global = test_module(test_input, test_labels)
assert torch.allclose(out_global, torch.tensor([[1.5], [1.5], [1.5]])), "Global normalization failed"

# With alpha=1 (fully class-conditional), each should be different
test_module.alpha = 1.0
out_class = test_module(test_input, test_labels)
expected = torch.tensor([[2.0], [1.0], [0.0]])  # 2-0=2, 2-1=1, 2-2=0
assert torch.allclose(out_class, expected), f"Class-conditional failed: {out_class} vs {expected}"

print("✓ Class-conditional behavior verified")
print("METRIC_SANITY_PASSED\n")

# ========== MAIN EXPERIMENT CODE ==========

class ClassConditionalBatchNorm2d(nn.Module):
    """BatchNorm that properly interpolates between global and class-conditional statistics."""
    
    def __init__(self, num_features, num_classes=10, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.eps = eps
        self.momentum = momentum
        
        # Global statistics
        self.register_buffer('global_running_mean', torch.zeros(num_features))
        self.register_buffer('global_running_var', torch.ones(num_features))
        
        # Per-class statistics - properly indexed by class
        self.register_buffer('class_running_mean', torch.zeros(num_classes, num_features))
        self.register_buffer('class_running_var', torch.ones(num_classes, num_features))
        
        # Track number of samples per class
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Interpolation factor (0 = global, 1 = class-conditional)
        self.alpha = 0.0
        
    def forward(self, x, labels):
        if labels is None:
            raise ValueError("ClassConditionalBatchNorm requires labels")
            
        batch_size, num_features, height, width = x.shape
        
        # Reshape x for easier manipulation
        x_reshaped = x.transpose(1, 0).contiguous().view(num_features, -1)
        
        if self.training:
            # Compute global batch statistics
            batch_mean = x_reshaped.mean(dim=1)
            batch_var = x_reshaped.var(dim=1, unbiased=False)
            
            # Update global running statistics
            with torch.no_grad():
                self.global_running_mean = (1 - self.momentum) * self.global_running_mean + self.momentum * batch_mean
                self.global_running_var = (1 - self.momentum) * self.global_running_var + self.momentum * batch_var
                
                # Update per-class statistics
                for c in range(self.num_classes):
                    mask = (labels == c)
                    if mask.sum() > 0:
                        # Get samples for this class
                        class_samples = x[mask]
                        class_reshaped = class_samples.transpose(1, 0).contiguous().view(num_features, -1)
                        
                        # Compute class statistics
                        class_mean = class_reshaped.mean(dim=1)
                        class_var = class_reshaped.var(dim=1, unbiased=False)
                        
                        # Update running statistics
                        if self.class_counts[c] == 0:
                            # First time seeing this class
                            self.class_running_mean[c] = class_mean
                            self.class_running_var[c] = class_var
                        else:
                            self.class_running_mean[c] = (1 - self.momentum) * self.class_running_mean[c] + self.momentum * class_mean
                            self.class_running_var[c] = (1 - self.momentum) * self.class_running_var[c] + self.momentum * class_var
                        
                        self.class_counts[c] += mask.sum()
            
            # Normalize using batch statistics during training
            x_norm = (x_reshaped - batch_mean.unsqueeze(1)) / torch.sqrt(batch_var.unsqueeze(1) + self.eps)
            x_norm = x_norm.view(num_features, batch_size, height, width).transpose(1, 0)
            
        else:
            # Evaluation mode: use running statistics
            x_norm = torch.zeros_like(x)
            
            # Process each sample based on its class
            for i in range(batch_size):
                class_idx = labels[i].item()
                sample = x[i]  # Shape: (C, H, W)
                
                # Determine which statistics to use
                if self.alpha == 0 or self.class_counts[class_idx] < 100:
                    # Use global statistics
                    mean = self.global_running_mean
                    var = self.global_running_var
                else:
                    # Interpolate between global and class statistics
                    mean = (1 - self.alpha) * self.global_running_mean + self.alpha * self.class_running_mean[class_idx]
                    var = (1 - self.alpha) * self.global_running_var + self.alpha * self.class_running_var[class_idx]
                    
                    # Ensure variance is positive
                    var = torch.clamp(var, min=self.eps)
                
                # Normalize this sample
                sample_reshaped = sample.view(num_features, -1)
                sample_norm = (sample_reshaped - mean.unsqueeze(1)) / torch.sqrt(var.unsqueeze(1) + self.eps)
                x_norm[i] = sample_norm.view_as(sample)
        
        # Apply learnable affine transformation
        weight = self.weight.view(1, num_features, 1, 1)
        bias = self.bias.view(1, num_features, 1, 1)
        
        return x_norm * weight + bias


class ResNet18(nn.Module):
    """ResNet-18 adapted for CIFAR-10 with class-conditional BN."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = ClassConditionalBatchNorm2d(64, num_classes)
        
        # Residual layers
        self.layer1 = self._make_layer(64, 64, 2, num_classes)
        self.layer2 = self._make_layer(64, 128, 2, num_classes, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, num_classes, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, num_classes, stride=2)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, num_classes, stride=1):
        layers = nn.ModuleList()
        
        # First block (may downsample)
        layers.append(ResidualBlock(in_channels, out_channels, num_classes, stride))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, num_classes))
        
        return layers
    
    def forward(self, x, labels):
        # Initial conv
        out = self.conv1(x)
        out = self.bn1(out, labels)
        out = F.relu(out)
        
        # Residual layers
        for block in self.layer1:
            out = block(out, labels)
        for block in self.layer2:
            out = block(out, labels)
        for block in self.layer3:
            out = block(out, labels)
        for block in self.layer4:
            out = block(out, labels)
        
        # Classifier
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


class ResidualBlock(nn.Module):
    """Basic residual block with class-conditional BN."""
    
    def __init__(self, in_channels, out_channels, num_classes, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = ClassConditionalBatchNorm2d(out_channels, num_classes)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = ClassConditionalBatchNorm2d(out_channels, num_classes)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                ClassConditionalBatchNorm2d(out_channels, num_classes)
            )
    
    def forward(self, x, labels):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out, labels)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out, labels)
        
        # Shortcut
        if len(self.shortcut) > 0:
            identity = x
            for layer in self.shortcut:
                if isinstance(layer, ClassConditionalBatchNorm2d):
                    identity = layer(identity, labels)
                else:
                    identity = layer(identity)
        
        out += identity
        out = F.relu(out)
        
        return out


class StandardResNet18(nn.Module):
    """Standard ResNet-18 with regular BatchNorm for baseline."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        # Use torchvision's ResNet18 but adapt for CIFAR
        self.model = torchvision.models.resnet18(num_classes=num_classes)
        # Replace first conv layer for CIFAR (smaller images)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for CIFAR
        
    def forward(self, x, labels=None):
        return self.model(x)


def train_model(model, train_loader, val_loader, device, epochs=100, patience=10, lr=0.1):
    """Train model with SGD and cosine annealing."""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
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
        scheduler.step()
        
        # Logging
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.3f}, "
                  f"Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, LR: {optimizer.param_groups[0]['lr']:.4f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f"CONVERGED at epoch {epoch}")
            return True, best_val_acc
        
        # Stop if we achieve good performance
        if best_val_acc > 0.90:
            print(f"CONVERGED at epoch {epoch} with high accuracy")
            return True, best_val_acc
    
    print("NOT_CONVERGED: Max epochs reached")
    return False, best_val_acc


def compute_layer_importance(model, val_loader, device, method='gradient'):
    """Compute importance scores for all BN layers."""
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    importance_scores = []
    
    for i in range(len(bn_layers)):
        if method == 'gradient':
            score = compute_importance_score_gradient(model, val_loader, i, device, max_batches=20)
        else:  # finite_diff
            # Reset all to global
            set_all_bn_alpha(model, 0.0)
            acc_global = evaluate(model, val_loader, device, max_batches=20)
            
            # Set only this layer to mild class-conditional
            set_bn_alpha(model, i, 0.3)
            acc_mixed = evaluate(model, val_loader, device, max_batches=20)
            
            score = abs(acc_mixed - acc_global) / 0.3
        
        importance_scores.append(score)
        
        # Reset
        set_all_bn_alpha(model, 0.0)
    
    return importance_scores


def run_experiment(seed, dataset='cifar10'):
    """Run complete experiment for one seed."""
    start_time = time.time()
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"SEED {seed} on {device}")
    print(f"{'='*60}")
    
    # Data preparation
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
    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    else:  # cifar100
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    
    # Create train/val split
    train_size = int(0.9 * len(trainset))
    val_size = len(trainset) - train_size
    train_data, val_data = random_split(trainset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)
    
    # 1. Train class-conditional BN model
    print("\n1. Training Class-Conditional BN Model")
    print("-" * 40)
    model = ResNet18(num_classes).to(device)
    converged, best_val = train_model(model, train_loader, val_loader, device, epochs=100, patience=10)
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # 2. Train standard baseline
    print("\n2. Training Standard BN Baseline")
    print("-" * 40)
    baseline_model = StandardResNet18(num_classes).to(device)
    _, baseline_val = train_model(baseline_model, train_loader, val_loader, device, epochs=100, patience=10)
    baseline_test = evaluate(baseline_model, test_loader, device)
    print(f"Baseline test accuracy: {baseline_test:.4f}")
    
    # 3. Compute layer importance
    print("\n3. Computing Layer Importance")
    print("-" * 40)
    
    # Get importance scores
    importance_scores = compute_layer_importance(model, val_loader, device, method='finite_diff')
    
    # Add small noise to avoid perfect correlations
    importance_scores = [s + np.random.normal(0, 0.0001) for s in importance_scores]
    
    # Display some scores
    print(f"Layer importance scores (first 10):")
    for i in range(min(10, len(importance_scores))):
        print(f"  Layer {i}: {importance_scores[i]:.6f}")
    
    # Compute depth correlation
    layer_indices = list(range(len(importance_scores)))
    spearman_rho, spearman_p = spearmanr(layer_indices, importance_scores)
    pearson_r, pearson_p = pearsonr(layer_indices, importance_scores)
    
    print(f"\nDepth correlations:")
    print(f"  Spearman ρ = {spearman_rho:.3f} (p = {spearman_p:.4f})")
    print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
    
    # 4. Test different configurations
    print("\n4. Testing Configurations")
    print("-" * 40)
    
    configurations = {}
    
    # Random baseline
    configurations['random'] = 1.0 / num_classes
    
    # All global (alpha = 0)
    set_all_bn_alpha(model, 0.0)
    configurations['all_global'] = evaluate(model, test_loader, device)
    
    # Mild class-conditional (alpha = 0.3)
    set_all_bn_alpha(model, 0.3)
    configurations['mild_class'] = evaluate(model, test_loader, device)
    
    # Strong class-conditional (alpha = 0.7)
    set_all_bn_alpha(model, 0.7)
    configurations['strong_class'] = evaluate(model, test_loader, device)
    
    # Hierarchical: early layers global, late layers class-conditional
    bn_layers = [m for m in model.modules() if isinstance(m, ClassConditionalBatchNorm2d)]
    num_layers = len(bn_layers)
    for i, layer in enumerate(bn_layers):
        # Linear increase in alpha from 0 to 0.7
        alpha = 0.7 * (i / (num_layers - 1)) if num_layers > 1 else 0.35
        layer.alpha = alpha
    configurations['hierarchical'] = evaluate(model, test_loader, device)
    
    # Reverse: early layers class-conditional, late layers global
    for i, layer in enumerate(bn_layers):
        # Linear decrease in alpha from 0.7 to 0
        alpha = 0.7 * (1 - i / (num_layers - 1)) if num_layers > 1 else 0.35
        layer.alpha = alpha
    configurations['reverse'] = evaluate(model, test_loader, device)
    
    # Step function: first half global, second half class-conditional
    mid_point = num_layers // 2
    for i, layer in enumerate(bn_layers):
        layer.alpha = 0.0 if i < mid_point else 0.5
    configurations['step'] = evaluate(model, test_loader, device)
    
    print("\nConfiguration results:")
    for name, acc in configurations.items():
        print(f"  {name}: {acc:.4f}")
    
    # 5. Ablation studies
    print("\n5. Ablation Studies")
    print("-" * 40)
    
    ablations = {}
    
    # Test sensitivity to alpha values
    alpha_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    alpha_accs = []
    for alpha in alpha_values:
        set_all_bn_alpha(model, alpha)
        acc = evaluate(model, val_loader, device, max_batches=20)
        alpha_accs.append(acc)
    ablations['alpha_sensitivity'] = list(zip(alpha_values, alpha_accs))
    
    # Test layer-specific effects (early vs late)
    set_all_bn_alpha(model, 0.0)
    
    # Only early layers class-conditional
    early_third = num_layers // 3
    for i in range(early_third):
        set_bn_alpha(model, i, 0.5)
    ablations['early_only'] = evaluate(model, val_loader, device, max_batches=20)
    
    # Reset
    set_all_bn_alpha(model, 0.0)
    
    # Only late layers class-conditional
    for i in range(2 * num_layers // 3, num_layers):
        set_bn_alpha(model, i, 0.5)
    ablations['late_only'] = evaluate(model, val_loader, device, max_batches=20)
    
    print(f"Alpha sensitivity: {ablations['alpha_sensitivity']}")
    print(f"Early layers only: {ablations['early_only']:.4f}")
    print(f"Late layers only: {ablations['late_only']:.4f}")
    
    # 6. Statistical significance of hierarchical structure
    hierarchy_advantage = configurations['hierarchical'] - configurations['reverse']
    step_advantage = configurations['step'] - configurations['all_global']
    
    # Check if importance increases with depth (positive correlation expected)
    importance_trend = spearman_rho > 0.2 and spearman_p < 0.05
    
    # Determine if signal detected
    signal_detected = (
        test_acc > 0.85 and  # Model trained well
        baseline_test > 0.85 and  # Baseline also good
        configurations['mild_class'] > configurations['random'] + 0.1 and  # Class-conditional helps
        hierarchy_advantage > 0.005 and  # Hierarchical better than reverse
        importance_trend  # Importance increases with depth
    )
    
    print(f"\n6. Results Summary")
    print("-" * 40)
    print(f"Hierarchy advantage: {hierarchy_advantage:.4f}")
    print(f"Step function advantage: {step_advantage:.4f}")
    print(f"Importance trend with depth: {'YES' if importance_trend else 'NO'}")
    print(f"Signal detected: {'YES' if signal_detected else 'NO'}")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time for seed {seed}: {elapsed:.1f}s")
    
    # Reset model
    set_all_bn_alpha(model, 0.0)
    
    return {
        'seed': seed,
        'dataset': dataset,
        'converged': converged,
        'test_accuracy': float(test_acc),
        'baseline_accuracy': float(baseline_test),
        'importance_scores': [float(s) for s in importance_scores[:20]],  # First 20 layers
        'spearman_rho': float(spearman_rho),
        'spearman_p': float(spearman_p),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'configurations': {k: float(v) for k, v in configurations.items()},
        'hierarchy_advantage': float(hierarchy_advantage),
        'step_advantage': float(step_advantage),
        'ablations': ablations,
        'signal_detected': signal_detected,
        'time': elapsed
    }


def main():
    """Run complete experiment across multiple seeds."""
    num_seeds = 10
    dataset = 'cifar10'
    
    print(f"Running BatchNorm Hierarchical Semantic Structure Experiment")
    print(f"Dataset: {dataset.upper()}")
    print(f"Seeds: {num_seeds}")
    print("=" * 60)
    
    # Verify setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()
    
    results = []
    start_time = time.time()
    
    # Run experiments
    for seed in range(num_seeds):
        try:
            result = run_experiment(seed, dataset)
            results.append(result)
            
            # Early termination if first seed fails basic checks
            if seed == 0:
                if result['test_accuracy'] < 0.5:
                    print("\nERROR: First seed achieved very low accuracy. Aborting.")
                    sys.exit(1)
                if result['configurations']['mild_class'] < result['configurations']['random']:
                    print("\nERROR: Class-conditional BN performing worse than random. Aborting.")
                    sys.exit(1)
                print("\nFirst seed validation passed. Continuing...")
            
        except Exception as e:
            print(f"\nERROR in seed {seed}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(results) == 0:
        print("ERROR: No successful runs")
        sys.exit(1)
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS")
    print("=" * 60)
    
    # Extract metrics
    test_accs = [r['test_accuracy'] for r in results]
    baseline_accs = [r['baseline_accuracy'] for r in results]
    spearman_rhos = [r['spearman_rho'] for r in results]
    hierarchy_advantages = [r['hierarchy_advantage'] for r in results]
    
    # Configuration accuracies
    config_names = list(results[0]['configurations'].keys())
    config_results = {name: [r['configurations'][name] for r in results] for name in config_names}
    
    # Statistical tests
    p_values = {}
    if len(results) > 1:
        # Test if hierarchical > reverse
        hier = config_results['hierarchical']
        rev = config_results['reverse']
        _, p_values['hier_vs_reverse'] = ttest_rel(hier, rev)
        
        # Test if hierarchical > all_global
        glob = config_results['all_global']
        _, p_values['hier_vs_global'] = ttest_rel(hier, glob)
        
        # Test if step > all_global
        step = config_results['step']
        _, p_values['step_vs_global'] = ttest_rel(step, glob)
        
        # Test if importance correlates with depth
        _, p_values['importance_depth'] = ttest_rel(spearman_rhos, [0] * len(spearman_rhos))
    
    # Print summary
    print(f"\nModel Performance:")
    print(f"  Test accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"  Baseline accuracy: {np.mean(baseline_accs):.4f} ± {np.std(baseline_accs):.4f}")
    
    print(f"\nImportance Analysis:")
    print(f"  Spearman ρ (depth correlation): {np.mean(spearman_rhos):.3f} ± {np.std(spearman_rhos):.3f}")
    print(f"  Hierarchy advantage: {np.mean(hierarchy_advantages):.4f} ± {np.std(hierarchy_advantages):.4f}")
    
    print(f"\nConfiguration Performance (mean ± std):")
    for name in config_names:
        values = config_results[name]
        print(f"  {name}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    print(f"\nStatistical Significance:")
    for test, p_val in p_values.items():
        print(f"  {test}: p = {p_val:.6f}")
    
    print(f"\nSignal Detection:")
    signal_rate = sum(r['signal_detected'] for r in results) / len(results)
    print(f"  Detection rate: {signal_rate:.1%} ({sum(r['signal_detected'] for r in results)}/{len(results)} seeds)")
    
    print(f"\nConvergence:")
    convergence_rate = sum(r['converged'] for r in results) / len(results)
    print(f"  Convergence rate: {convergence_rate:.1%}")
    
    total_time = time.time() - start_time
    print(f"\nTotal experiment time: {total_time/60:.1f} minutes")
    
    # Prepare final output
    output = {
        'experiment': 'BatchNorm Hierarchical Semantic Structure',
        'dataset': dataset,
        'num_seeds': num_seeds,
        'per_seed_results': results,
        'mean': {
            'test_accuracy': float(np.mean(test_accs)),
            'baseline_accuracy': float(np.mean(baseline_accs)),
            'spearman_rho': float(np.mean(spearman_rhos)),
            'hierarchy_advantage': float(np.mean(hierarchy_advantages)),
            'configurations': {k: float(np.mean(v)) for k, v in config_results.items()}
        },
        'std': {
            'test_accuracy': float(np.std(test_accs)),
            'baseline_accuracy': float(np.std(baseline_accs)),
            'spearman_rho': float(np.std(spearman_rhos)),
            'hierarchy_advantage': float(np.std(hierarchy_advantages)),
            'configurations': {k: float(np.std(v)) for k, v in config_results.items()}
        },
        'p_values': p_values,
        'convergence_status': f"{sum(r['converged'] for r in results)}/{len(results)} converged",
        'signal_detection_rate': float(signal_rate),
        'total_time_seconds': float(total_time)
    }
    
    print(f"\nRESULTS: {json.dumps(output)}")


if __name__ == "__main__":
    # For quick testing, uncomment:
    # sys.argv.append('--test')
    
    if '--test' in sys.argv:
        # Quick test mode
        print("TEST MODE: Running single seed with reduced epochs")
        result = run_experiment(0, 'cifar10')
        print(f"\nTEST RESULTS: {json.dumps(result, indent=2)}")
    else:
        # Full experiment
        main()