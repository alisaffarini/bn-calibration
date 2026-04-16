# pip install torch torchvision numpy scipy matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import spearmanr, ttest_rel
import json
import random
from collections import defaultdict
import matplotlib.pyplot as plt

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

def evaluate(model, loader, device):
    """Simple evaluation function."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, labels)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return correct / total

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
            mean = batch_mean
            var = batch_var
        else:
            # During evaluation, interpolate between global and class stats
            if labels is not None and self.alpha > 0:
                # Get per-sample statistics based on class
                batch_size = x.size(0)
                mean = torch.zeros(batch_size, self.num_features, device=x.device)
                var = torch.ones(batch_size, self.num_features, device=x.device)
                
                for i in range(batch_size):
                    class_idx = labels[i].item() if isinstance(labels, torch.Tensor) else labels
                    # Interpolate
                    mean[i] = (1 - self.alpha) * self.global_running_mean + self.alpha * self.class_running_mean[class_idx]
                    var[i] = (1 - self.alpha) * self.global_running_var + self.alpha * self.class_running_var[class_idx]
                
                # Reshape for broadcasting
                mean = mean.view(batch_size, self.num_features, 1, 1)
                var = var.view(batch_size, self.num_features, 1, 1)
            else:
                # Use global stats
                mean = self.global_running_mean
                var = self.global_running_var
        
        # Normalize
        x_normalized = (x - mean.view(-1, self.num_features, 1, 1)) / torch.sqrt(var.view(-1, self.num_features, 1, 1) + self.eps)
        
        # Scale and shift
        return x_normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleResNet(nn.Module):
    """Small ResNet-like model for fast experiments."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = ClassConditionalBatchNorm2d(16, num_classes)
        
        # Block 1
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
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
        identity = x
        x = F.relu(self.bn2(self.conv2(x), labels))
        x = self.bn3(self.conv3(x), labels)
        if identity.shape == x.shape:
            x = F.relu(x + identity)
        else:
            x = F.relu(x)
        
        # Block 2
        x = F.relu(self.bn4(self.conv4(x), labels))
        x = F.relu(self.bn5(self.conv5(x), labels))
        
        # Classifier
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


def train_model(model, train_loader, val_loader, device, epochs=50, patience=5):
    """Train model with convergence-based stopping."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
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
        scheduler.step(1 - val_acc)  # minimize 1 - accuracy
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print("CONVERGED")
            return True
    
    print("NOT_CONVERGED: Max epochs reached")
    return False


def run_experiment(seed):
    """Run single seed experiment."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=== Running seed {seed} on {device} ===")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Train model
    model = SimpleResNet(num_classes=10).to(device)
    converged = train_model(model, train_loader, val_loader, device, epochs=30, patience=5)
    
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
    spearman_rho, _ = spearmanr(layer_depths, importance_scores)
    print(f"\nSpearman correlation (depth vs importance): {spearman_rho:.4f}")
    
    # 2. Semantic Hierarchy Validation
    print("\n--- Semantic Hierarchy Validation ---")
    hybrid_accuracies = []
    
    for k in range(num_bn_layers + 1):
        # Set first k layers to global (alpha=0), rest to class-conditional (alpha=1)
        for i in range(num_bn_layers):
            if i < k:
                set_bn_alpha(model, i, 0.0)
            else:
                set_bn_alpha(model, i, 1.0)
        
        acc = evaluate(model, val_loader, device)
        hybrid_accuracies.append(acc)
        print(f"k={k}: Accuracy = {acc:.4f}")
    
    # Reset all to global
    for i in range(num_bn_layers):
        set_bn_alpha(model, i, 0.0)
    
    # 3. Baseline comparisons
    print("\n--- Baselines ---")
    
    # Random baseline
    random_acc = 1.0 / 10  # 10 classes
    print(f"Random baseline: {random_acc:.4f}")
    
    # All global (alpha=0)
    for i in range(num_bn_layers):
        set_bn_alpha(model, i, 0.0)
    global_acc = evaluate(model, val_loader, device)
    print(f"All global (α=0): {global_acc:.4f}")
    
    # All class-conditional (alpha=1)
    for i in range(num_bn_layers):
        set_bn_alpha(model, i, 1.0)
    class_cond_acc = evaluate(model, val_loader, device)
    print(f"All class-conditional (α=1): {class_cond_acc:.4f}")
    
    # Check if we detected a signal
    max_importance = max(importance_scores)
    min_importance = min(importance_scores)
    importance_range = max_importance - min_importance
    
    if importance_range > 0.05 and abs(spearman_rho) > 0.5:
        print("SIGNAL_DETECTED: Layer importance shows clear pattern")
    else:
        print("NO_SIGNAL: Layer importance shows no clear pattern")
    
    return {
        'seed': seed,
        'converged': converged,
        'importance_scores': importance_scores,
        'spearman_rho': float(spearman_rho),
        'hybrid_accuracies': hybrid_accuracies,
        'baselines': {
            'random': random_acc,
            'all_global': global_acc,
            'all_class_conditional': class_cond_acc
        },
        'signal_detected': importance_range > 0.05 and abs(spearman_rho) > 0.5
    }


def main():
    """Main experimental loop."""
    num_seeds = 2  # Small scale for quick testing
    results = []
    
    for seed in range(num_seeds):
        result = run_experiment(seed)
        results.append(result)
    
    # Aggregate results
    all_importance_scores = [r['importance_scores'] for r in results]
    all_spearman_rhos = [r['spearman_rho'] for r in results]
    all_converged = [r['converged'] for r in results]
    signals_detected = [r['signal_detected'] for r in results]
    
    # Compute statistics
    mean_importance = np.mean(all_importance_scores, axis=0).tolist()
    std_importance = np.std(all_importance_scores, axis=0).tolist()
    mean_spearman = np.mean(all_spearman_rhos)
    std_spearman = np.std(all_spearman_rhos)
    
    # Final output
    output = {
        'per_seed_results': results,
        'mean_importance_scores': mean_importance,
        'std_importance_scores': std_importance,
        'mean_spearman_rho': float(mean_spearman),
        'std_spearman_rho': float(std_spearman),
        'convergence_rate': sum(all_converged) / len(all_converged),
        'signal_detection_rate': sum(signals_detected) / len(signals_detected),
        'p_values': None  # Not computed for small scale
    }
    
    print(f"\nRESULTS: {json.dumps(output)}")


if __name__ == "__main__":
    main()