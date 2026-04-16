# pip install numpy torch torchvision scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
import json
from collections import defaultdict
import math
from scipy import stats

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============== METRIC SANITY CHECKS ==============
def compute_ece(confidences, predictions, labels, n_bins=15):
    """
    Expected Calibration Error
    confidences: (N,) max probability
    predictions: (N,) predicted class  
    labels: (N,) true class
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.astype(float).mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).astype(float).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def sanity_check_ece():
    print("Running ECE sanity checks...")
    
    # Test 1: Perfect calibration (all predictions correct with 100% confidence)
    n = 1000
    confidences = np.ones(n)
    predictions = np.arange(n) % 10
    labels = predictions.copy()
    ece = compute_ece(confidences, predictions, labels)
    assert ece < 0.01, f"Perfect predictions should have ECE ≈ 0, got {ece}"
    
    # Test 2: Random predictions with uniform confidence
    confidences = np.ones(n) * 0.1  # 10% confidence (10 classes)
    predictions = np.random.randint(0, 10, n)
    labels = np.random.randint(0, 10, n)
    accuracy = (predictions == labels).mean()
    ece = compute_ece(confidences, predictions, labels)
    # With uniform 0.1 confidence and ~10% accuracy, ECE should be small
    assert ece < 0.05, f"Calibrated random should have low ECE, got {ece}"
    
    # Test 3: Overconfident predictions
    confidences = np.ones(n) * 0.9  # 90% confidence
    predictions = np.random.randint(0, 10, n)
    labels = np.random.randint(0, 10, n)
    ece = compute_ece(confidences, predictions, labels)
    assert ece > 0.7, f"Overconfident random should have high ECE, got {ece}"
    
    print("METRIC_SANITY_PASSED")

sanity_check_ece()

# ============== CALIBRATED NORM IMPLEMENTATION ==============
class CalibratedNorm(nn.Module):
    def __init__(self, num_features, num_groups=32):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        
        # Global stats
        self.bn_global = nn.BatchNorm2d(num_features)
        
        # Group stats  
        self.bn_groups = nn.ModuleList([
            nn.BatchNorm2d(num_features) for _ in range(num_groups)
        ])
        
        # Lightweight alpha predictor
        self.alpha_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, labels=None):
        batch_size = x.size(0)
        
        # Predict alpha from features
        alpha = self.alpha_net(x)  # (B, 1, 1, 1)
        
        # Global normalization
        x_global = self.bn_global(x)
        
        if self.training and labels is not None:
            # During training: use class-based group assignment
            x_groups = torch.zeros_like(x)
            for i in range(batch_size):
                group_idx = labels[i].item() % self.num_groups
                x_groups[i:i+1] = self.bn_groups[group_idx](x[i:i+1])
        else:
            # During inference: average all group stats
            x_groups = torch.stack([bn(x) for bn in self.bn_groups], dim=0).mean(dim=0)
        
        # Interpolate between global and group stats
        return (1 - alpha) * x_global + alpha * x_groups

# Fixed alpha version for ablation
class CalibratedNormFixedAlpha(nn.Module):
    def __init__(self, num_features, num_groups=32, alpha=0.5):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.alpha = alpha
        
        self.bn_global = nn.BatchNorm2d(num_features)
        self.bn_groups = nn.ModuleList([
            nn.BatchNorm2d(num_features) for _ in range(num_groups)
        ])
        
    def forward(self, x, labels=None):
        batch_size = x.size(0)
        x_global = self.bn_global(x)
        
        if self.training and labels is not None:
            x_groups = torch.zeros_like(x)
            for i in range(batch_size):
                group_idx = labels[i].item() % self.num_groups
                x_groups[i:i+1] = self.bn_groups[group_idx](x[i:i+1])
        else:
            x_groups = torch.stack([bn(x) for bn in self.bn_groups], dim=0).mean(dim=0)
        
        return (1 - self.alpha) * x_global + self.alpha * x_groups

# ============== SMALL RESNET FOR CIFAR ==============
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = lambda c: nn.BatchNorm2d(c)
            
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                norm_layer(out_channels)
            )
    
    def forward(self, x, labels=None):
        out = F.relu(self._apply_norm(self.bn1, self.conv1(x), labels))
        out = self._apply_norm(self.bn2, self.conv2(out), labels)
        out += self._apply_norm_shortcut(self.shortcut, x, labels)
        return F.relu(out)
    
    def _apply_norm(self, norm, x, labels):
        if hasattr(norm, 'forward') and 'labels' in norm.forward.__code__.co_varnames:
            return norm(x, labels)
        return norm(x)
    
    def _apply_norm_shortcut(self, shortcut, x, labels):
        if len(shortcut) == 0:
            return x
        conv = shortcut[0](x)
        if len(shortcut) > 1 and hasattr(shortcut[1], 'forward') and 'labels' in shortcut[1].forward.__code__.co_varnames:
            return shortcut[1](conv, labels)
        elif len(shortcut) > 1:
            return shortcut[1](conv)
        return conv

class ResNet(nn.Module):
    def __init__(self, num_classes=10, norm_type='standard'):
        super().__init__()
        self.norm_type = norm_type
        
        # Define norm layer
        if norm_type == 'standard':
            self.norm_layer = lambda c: nn.BatchNorm2d(c)
        elif norm_type == 'calibrated':
            self.norm_layer = lambda c: CalibratedNorm(c, num_groups=32)
        elif norm_type == 'calibrated_fixed':
            self.norm_layer = lambda c: CalibratedNormFixedAlpha(c, num_groups=32, alpha=0.5)
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = self.norm_layer(16)
        
        self.layer1 = self._make_layer(16, 16, 2, stride=1)
        self.layer2 = self._make_layer(16, 32, 2, stride=2)
        self.layer3 = self._make_layer(32, 64, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, self.norm_layer))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, self.norm_layer))
        return nn.Sequential(*layers)
    
    def forward(self, x, labels=None):
        out = F.relu(self._apply_norm(self.bn1, self.conv1(x), labels))
        
        # Pass through layers with labels
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                out = block(out, labels)
                
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
    
    def _apply_norm(self, norm, x, labels):
        if hasattr(norm, 'forward') and 'labels' in norm.forward.__code__.co_varnames:
            return norm(x, labels)
        return norm(x)

# ============== TEMPERATURE SCALING ==============
class TemperatureScaling:
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits, labels):
        """Find optimal temperature on validation set"""
        from scipy.optimize import minimize
        
        def nll_loss(t):
            scaled_logits = logits / t
            loss = F.cross_entropy(torch.from_numpy(scaled_logits), torch.from_numpy(labels))
            return loss.item()
        
        result = minimize(nll_loss, 1.0, method='L-BFGS-B', bounds=[(0.1, 10.0)])
        self.temperature = result.x[0]
        
    def transform(self, logits):
        return logits / self.temperature

# ============== TRAINING AND EVALUATION ==============
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, labels)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            
            total_loss += loss.item()
            all_logits.append(outputs.cpu())
            all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Compute accuracy
    _, predicted = all_logits.max(1)
    accuracy = (predicted == all_labels).float().mean().item()
    
    # Compute ECE
    probs = F.softmax(all_logits, dim=1)
    confidences, predictions = probs.max(1)
    ece = compute_ece(confidences.numpy(), predictions.numpy(), all_labels.numpy())
    
    return total_loss / len(loader), accuracy, ece, all_logits.numpy(), all_labels.numpy()

def run_experiment(seed, norm_type='standard'):
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Data loading
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
    
    # Split train into train/val
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create model
    model = ResNet(num_classes=10, norm_type=norm_type).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Training with early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    converged = False
    
    print(f"\nTraining {norm_type} model (seed {seed})...")
    
    for epoch in range(100):  # Max epochs
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_ece, _, _ = evaluate(model, val_loader, device)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.3f}, "
                  f"Val Loss={val_loss:.4f} Acc={val_acc:.3f} ECE={val_ece:.3f}")
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"CONVERGED at epoch {epoch}")
            converged = True
            break
    
    if not converged:
        print("NOT_CONVERGED: Reached max epochs")
    
    # Test evaluation
    test_loss, test_acc, test_ece_before, test_logits, test_labels = evaluate(model, test_loader, device)
    
    # Apply temperature scaling
    val_loss, val_acc, val_ece, val_logits, val_labels = evaluate(model, val_loader, device)
    temp_scaling = TemperatureScaling()
    temp_scaling.fit(val_logits, val_labels)
    
    # Evaluate with temperature scaling
    scaled_logits = temp_scaling.transform(test_logits)
    probs = F.softmax(torch.from_numpy(scaled_logits), dim=1)
    confidences, predictions = probs.max(1)
    test_ece_after = compute_ece(confidences.numpy(), predictions.numpy(), test_labels)
    
    return {
        'test_acc': test_acc,
        'test_ece_before_temp': test_ece_before,
        'test_ece_after_temp': test_ece_after,
        'temperature': temp_scaling.temperature,
        'converged': converged
    }

# ============== MAIN EXPERIMENT ==============
def main():
    print("\n========== CALIBRATEDNORM FEASIBILITY EXPERIMENT ==========")
    print("Testing: Standard BN vs CalibratedNorm vs CalibratedNorm-FixedAlpha")
    print("Dataset: CIFAR-10, Model: Small ResNet")
    
    num_seeds = 3  # Feasibility probe with 3 seeds
    results = defaultdict(list)
    
    for norm_type in ['standard', 'calibrated', 'calibrated_fixed']:
        print(f"\n\n===== Running {norm_type} =====")
        
        for seed in range(num_seeds):
            result = run_experiment(seed, norm_type)
            results[norm_type].append(result)
    
    # Compute statistics
    summary = {}
    for norm_type, runs in results.items():
        acc_values = [r['test_acc'] for r in runs]
        ece_before_values = [r['test_ece_before_temp'] for r in runs]
        ece_after_values = [r['test_ece_after_temp'] for r in runs]
        
        summary[norm_type] = {
            'test_acc_mean': np.mean(acc_values),
            'test_acc_std': np.std(acc_values),
            'ece_before_temp_mean': np.mean(ece_before_values),
            'ece_before_temp_std': np.std(ece_before_values),
            'ece_after_temp_mean': np.mean(ece_after_values),
            'ece_after_temp_std': np.std(ece_after_values),
            'per_seed_results': runs
        }
    
    # Statistical tests (calibrated vs standard, after temperature scaling)
    standard_ece = [r['test_ece_after_temp'] for r in results['standard']]
    calibrated_ece = [r['test_ece_after_temp'] for r in results['calibrated']]
    
    if len(standard_ece) >= 2 and len(calibrated_ece) >= 2:
        t_stat, p_value = stats.ttest_rel(standard_ece, calibrated_ece)
        improvement = (np.mean(standard_ece) - np.mean(calibrated_ece)) / np.mean(standard_ece) * 100
    else:
        p_value = None
        improvement = 0
    
    # Check for signal
    calibrated_mean = summary['calibrated']['ece_after_temp_mean']
    standard_mean = summary['standard']['ece_after_temp_mean']
    fixed_mean = summary['calibrated_fixed']['ece_after_temp_mean']
    
    if calibrated_mean < standard_mean and calibrated_mean < fixed_mean:
        print(f"\nSIGNAL_DETECTED: CalibratedNorm achieves {improvement:.1f}% lower ECE than standard BN")
    else:
        print("\nNO_SIGNAL: CalibratedNorm did not outperform baselines")
    
    # Final results
    final_results = {
        'summary': summary,
        'statistical_tests': {
            'calibrated_vs_standard_pvalue': p_value,
            'improvement_percentage': improvement
        },
        'convergence_status': all(r['converged'] for runs in results.values() for r in runs)
    }
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == '__main__':
    main()