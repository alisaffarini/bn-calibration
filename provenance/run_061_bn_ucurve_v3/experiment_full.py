# pip install scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import json
from scipy.stats import ttest_rel
from scipy import stats
from typing import Dict, List, Tuple
import warnings
import sys
import time
import os
warnings.filterwarnings('ignore')

# DRY RUN MODE - set to False for full publication run
DRY_RUN = True

def compute_ece(predictions, labels, n_bins=15):
    """
    Compute Expected Calibration Error (ECE)
    predictions: (N, C) probabilities
    labels: (N,) true labels
    """
    if predictions.dim() != 2:
        raise ValueError(f"Expected 2D predictions, got {predictions.dim()}D")
    
    confidences, predicted = torch.max(predictions, 1)
    accuracies = predicted.eq(labels)
    
    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += prop_in_bin * torch.abs(avg_confidence_in_bin - accuracy_in_bin)
    
    return ece.item()

def compute_mce(predictions, labels, n_bins=15):
    """Compute Maximum Calibration Error (MCE)"""
    confidences, predicted = torch.max(predictions, 1)
    accuracies = predicted.eq(labels)
    
    mce = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            mce = max(mce, torch.abs(avg_confidence_in_bin - accuracy_in_bin).item())
    
    return mce

def compute_brier_score(predictions, labels):
    """Compute Brier Score"""
    num_classes = predictions.size(1)
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
    return torch.mean(torch.sum((predictions - one_hot_labels) ** 2, dim=1)).item()

# Metric sanity check
print("=== METRIC SANITY CHECK ===")

# Test 1: Perfect calibration
perfect_probs = torch.tensor([[0.99, 0.01], [0.01, 0.99], [0.98, 0.02], [0.02, 0.98]])
perfect_labels = torch.tensor([0, 1, 0, 1])
ece_perfect = compute_ece(perfect_probs, perfect_labels)
mce_perfect = compute_mce(perfect_probs, perfect_labels)
brier_perfect = compute_brier_score(perfect_probs, perfect_labels)
print(f"Perfect calibration - ECE: {ece_perfect:.4f}, MCE: {mce_perfect:.4f}, Brier: {brier_perfect:.4f}")
assert ece_perfect < 0.1, f"ECE for perfect calibration should be near 0, got {ece_perfect}"

# Test 2: Bad calibration
overconf_probs = torch.tensor([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])
overconf_labels = torch.tensor([1, 0, 1, 0])
ece_overconf = compute_ece(overconf_probs, overconf_labels)
print(f"Overconfident ECE: {ece_overconf:.4f}")
assert ece_overconf > 0.8, f"ECE for terrible calibration should be high, got {ece_overconf}"

print("METRIC_SANITY_PASSED\n")

class ConfidenceAdaptiveBN2d(nn.Module):
    """Confidence-Adaptive BatchNorm for 2D inputs"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, threshold=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.threshold = threshold
        self.training_mode = True
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        self.register_buffer('running_mean_high', torch.zeros(num_features))
        self.register_buffer('running_var_high', torch.ones(num_features))
        self.register_buffer('running_mean_low', torch.zeros(num_features))
        self.register_buffer('running_var_low', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def forward(self, input, confidence=None):
        if self.training_mode and self.training:
            N, C, H, W = input.shape
            x_flat = input.transpose(0, 1).contiguous().view(C, -1)
            batch_mean = x_flat.mean(dim=1)
            batch_var = x_flat.var(dim=1, unbiased=False)
            
            if confidence is not None and confidence.numel() > 0:
                high_mask = confidence > self.threshold
                low_mask = ~high_mask
                
                if high_mask.sum() > 1:
                    high_input = input[high_mask]
                    high_flat = high_input.transpose(0, 1).contiguous().view(C, -1)
                    high_mean = high_flat.mean(dim=1)
                    high_var = high_flat.var(dim=1, unbiased=False)
                    
                    momentum = self.momentum
                    self.running_mean_high.mul_(1 - momentum).add_(high_mean, alpha=momentum)
                    self.running_var_high.mul_(1 - momentum).add_(high_var, alpha=momentum)
                
                if low_mask.sum() > 1:
                    low_input = input[low_mask]
                    low_flat = low_input.transpose(0, 1).contiguous().view(C, -1)
                    low_mean = low_flat.mean(dim=1)
                    low_var = low_flat.var(dim=1, unbiased=False)
                    
                    momentum = self.momentum
                    self.running_mean_low.mul_(1 - momentum).add_(low_mean, alpha=momentum)
                    self.running_var_low.mul_(1 - momentum).add_(low_var, alpha=momentum)
            else:
                momentum = self.momentum
                self.running_mean_high.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.running_var_high.mul_(1 - momentum).add_(batch_var, alpha=momentum)
                self.running_mean_low.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.running_var_low.mul_(1 - momentum).add_(batch_var, alpha=momentum)
            
            self.num_batches_tracked += 1
            mean = batch_mean.view(1, C, 1, 1)
            var = batch_var.view(1, C, 1, 1)
        else:
            mean = 0.5 * self.running_mean_low + 0.5 * self.running_mean_high
            var = 0.5 * self.running_var_low + 0.5 * self.running_var_high
            mean = mean.view(1, self.num_features, 1, 1)
            var = var.view(1, self.num_features, 1, 1)
        
        out = (input - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, self.num_features, 1, 1)
        bias = self.bias.view(1, self.num_features, 1, 1)
        
        return weight * out + bias

class SimpleNet(nn.Module):
    """Simple CNN for quick experiments"""
    def __init__(self, use_cabn=True, num_classes=10):
        super().__init__()
        self.use_cabn = use_cabn
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = ConfidenceAdaptiveBN2d(32) if use_cabn else nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = ConfidenceAdaptiveBN2d(64) if use_cabn else nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, update_stats=True):
        if self.use_cabn:
            for m in self.modules():
                if isinstance(m, ConfidenceAdaptiveBN2d):
                    m.training_mode = update_stats
        
        confidence = None
        if self.use_cabn and self.training and update_stats:
            with torch.no_grad():
                h = F.relu(self.bn1(self.conv1(x), None))
                h = self.pool(h)
                h = F.relu(self.bn2(self.conv2(h), None))
                h = self.pool(h)
                h = h.view(h.size(0), -1)
                h = F.relu(self.fc1(h))
                logits = self.fc2(h)
                probs = F.softmax(logits, dim=1)
                confidence, _ = probs.max(dim=1)
        
        out = self.conv1(x)
        out = self.bn1(out, confidence) if self.use_cabn else self.bn1(out)
        out = F.relu(out)
        out = self.pool(out)
        
        out = self.conv2(out)
        out = self.bn2(out, confidence) if self.use_cabn else self.bn2(out)
        out = F.relu(out)
        out = self.pool(out)
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, logits):
        return logits / self.temperature

def train_epoch(model, train_loader, optimizer, criterion, device, max_batches=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if max_batches and batch_idx >= max_batches:
            break
    
    return total_loss / (batch_idx + 1), 100. * correct / total

def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, update_stats=False)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_probs.append(probs.cpu())
            all_labels.append(targets.cpu())
            
            if max_batches and batch_idx >= max_batches:
                break
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    acc = 100. * correct / total
    ece = compute_ece(all_probs, all_labels)
    mce = compute_mce(all_probs, all_labels)
    brier = compute_brier_score(all_probs, all_labels)
    
    return acc, ece, mce, brier

def calibrate_temperature(model, val_loader, device, max_batches=None):
    """Fast temperature calibration"""
    model.eval()
    
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, update_stats=False)
            all_logits.append(outputs)
            all_labels.append(targets)
            
            if max_batches and batch_idx >= max_batches:
                break
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Simple grid search for temperature
    temps = torch.linspace(0.5, 3.0, 20)
    best_ece = float('inf')
    best_temp = 1.5
    
    for temp in temps:
        with torch.no_grad():
            scaled_logits = all_logits / temp
            probs = F.softmax(scaled_logits, dim=1)
            ece = compute_ece(probs, all_labels)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp.item()
    
    return best_temp

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"DRY RUN MODE: {DRY_RUN}")
    
    # Set parameters based on mode
    if DRY_RUN:
        num_seeds = 3
        max_epochs = 5
        patience = 2
        data_fraction = 0.05
        max_train_batches = 10
        max_eval_batches = 5
    else:
        num_seeds = 10
        max_epochs = 30
        patience = 5
        data_fraction = 1.0
        max_train_batches = None
        max_eval_batches = None
    
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
    
    # CIFAR-10 dataset
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    # Use subset for dry run
    if DRY_RUN:
        train_indices = list(range(0, int(len(train_data) * data_fraction)))
        train_data = Subset(train_data, train_indices)
        test_indices = list(range(0, int(len(test_data) * data_fraction)))
        test_data = Subset(test_data, test_indices)
    
    # Split train into train/val
    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_data, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0)
    
    # Results storage
    results = {
        'standard_bn': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'converged': []},
        'cabn': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'converged': []},
        'temp_scaling': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'temperature': []},
        'label_smoothing': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'converged': []},
        'random': {'acc': [], 'ece': [], 'mce': [], 'brier': []}
    }
    
    for seed in range(num_seeds):
        print(f"\n{'='*60}")
        print(f"SEED {seed}/{num_seeds-1}")
        print('='*60)
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Random baseline (only compute once)
        if seed == 0:
            print("\nRandom Baseline...")
            num_classes = 10
            random_acc = 100.0 / num_classes
            # Sample some test data for random baseline
            sample_size = min(1000, len(test_data))
            random_probs = torch.ones(sample_size, num_classes) / num_classes
            test_labels = []
            for i in range(sample_size):
                test_labels.append(test_data[i][1] if hasattr(test_data[i], '__getitem__') else test_data.dataset[test_data.indices[i]][1])
            test_labels = torch.tensor(test_labels)
            random_ece = compute_ece(random_probs, test_labels)
            random_mce = compute_mce(random_probs, test_labels)
            random_brier = compute_brier_score(random_probs, test_labels)
            
            for _ in range(num_seeds):
                results['random']['acc'].append(random_acc)
                results['random']['ece'].append(random_ece)
                results['random']['mce'].append(random_mce)
                results['random']['brier'].append(random_brier)
            print(f"Random: Acc: {random_acc:.2f}%, ECE: {random_ece:.4f}")
        
        # Standard BatchNorm
        print("\nTraining Standard BN...")
        model_standard = SimpleNet(use_cabn=False).to(device)
        optimizer = torch.optim.Adam(model_standard.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_ece = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model_standard, train_loader, optimizer, criterion, device, max_train_batches)
            val_acc, val_ece, val_mce, val_brier = evaluate(model_standard, val_loader, device, max_eval_batches)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
            
            if val_ece < best_ece - 0.001:
                best_ece = val_ece
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("CONVERGED")
                converged = True
                break
        
        if not converged:
            print("NOT_CONVERGED: Reached max epochs")
        
        # Test evaluation
        test_acc, test_ece, test_mce, test_brier = evaluate(model_standard, test_loader, device, max_eval_batches)
        results['standard_bn']['acc'].append(test_acc)
        results['standard_bn']['ece'].append(test_ece)
        results['standard_bn']['mce'].append(test_mce)
        results['standard_bn']['brier'].append(test_brier)
        results['standard_bn']['converged'].append(converged)
        
        # Temperature Scaling
        print("\nCalibrating temperature...")
        temp_value = calibrate_temperature(model_standard, val_loader, device, max_eval_batches)
        
        # Test with temperature scaling
        model_standard.eval()
        with torch.no_grad():
            all_probs = []
            all_labels = []
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_standard(inputs, update_stats=False)
                scaled_outputs = outputs / temp_value
                probs = F.softmax(scaled_outputs, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(targets.cpu())
                if max_eval_batches and batch_idx >= max_eval_batches:
                    break
            
            all_probs = torch.cat(all_probs)
            all_labels = torch.cat(all_labels)
            test_ece_temp = compute_ece(all_probs, all_labels)
            test_mce_temp = compute_mce(all_probs, all_labels)
            test_brier_temp = compute_brier_score(all_probs, all_labels)
            _, predicted = all_probs.max(1)
            test_acc_temp = 100. * predicted.eq(all_labels).sum().item() / all_labels.size(0)
        
        results['temp_scaling']['acc'].append(test_acc_temp)
        results['temp_scaling']['ece'].append(test_ece_temp)
        results['temp_scaling']['mce'].append(test_mce_temp)
        results['temp_scaling']['brier'].append(test_brier_temp)
        results['temp_scaling']['temperature'].append(temp_value)
        print(f"Temperature {temp_value:.3f}: ECE = {test_ece_temp:.4f}")
        
        # Label Smoothing
        print("\nTraining with Label Smoothing...")
        model_ls = SimpleNet(use_cabn=False).to(device)
        optimizer = torch.optim.Adam(model_ls.parameters(), lr=0.001)
        criterion_ls = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_ece = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model_ls, train_loader, optimizer, criterion_ls, device, max_train_batches)
            val_acc, val_ece, val_mce, val_brier = evaluate(model_ls, val_loader, device, max_eval_batches)
            
            if epoch == 0 or epoch == max_epochs - 1:
                print(f"Epoch {epoch+1}: Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
            
            if val_ece < best_ece - 0.001:
                best_ece = val_ece
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("CONVERGED")
                converged = True
                break
        
        if not converged:
            print("NOT_CONVERGED: Reached max epochs")
        
        test_acc, test_ece, test_mce, test_brier = evaluate(model_ls, test_loader, device, max_eval_batches)
        results['label_smoothing']['acc'].append(test_acc)
        results['label_smoothing']['ece'].append(test_ece)
        results['label_smoothing']['mce'].append(test_mce)
        results['label_smoothing']['brier'].append(test_brier)
        results['label_smoothing']['converged'].append(converged)
        
        # Confidence-Adaptive BN
        print("\nTraining CA-BN...")
        model_cabn = SimpleNet(use_cabn=True).to(device)
        optimizer = torch.optim.Adam(model_cabn.parameters(), lr=0.001)
        
        best_ece = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model_cabn, train_loader, optimizer, criterion, device, max_train_batches)
            val_acc, val_ece, val_mce, val_brier = evaluate(model_cabn, val_loader, device, max_eval_batches)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
            
            if val_ece < best_ece - 0.001:
                best_ece = val_ece
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("CONVERGED")
                converged = True
                break
        
        if not converged:
            print("NOT_CONVERGED: Reached max epochs")
        
        test_acc, test_ece, test_mce, test_brier = evaluate(model_cabn, test_loader, device, max_eval_batches)
        results['cabn']['acc'].append(test_acc)
        results['cabn']['ece'].append(test_ece)
        results['cabn']['mce'].append(test_mce)
        results['cabn']['brier'].append(test_brier)
        results['cabn']['converged'].append(converged)
        
        # Early abort check after first seed
        if seed == 0:
            print("\n=== EARLY ABORT CHECK ===")
            if abs(results['standard_bn']['ece'][0] - results['cabn']['ece'][0]) < 1e-6:
                print(f"SANITY_ABORT: Standard BN and CA-BN have identical ECE ({results['standard_bn']['ece'][0]:.4f})")
                sys.exit(1)
            
            if results['standard_bn']['ece'][0] == 0.0 or np.isnan(results['standard_bn']['ece'][0]):
                print(f"SANITY_ABORT: Standard BN ECE is degenerate ({results['standard_bn']['ece'][0]})")
                sys.exit(1)
            
            improvement = (results['standard_bn']['ece'][0] - results['cabn']['ece'][0]) / results['standard_bn']['ece'][0]
            print(f"First seed improvement: {improvement*100:.1f}%")
            print("Metrics look reasonable, continuing...")
    
    # Compute final statistics
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print('='*60)
    
    final_results = {}
    
    for method in results:
        if len(results[method]['acc']) == 0:
            continue
            
        acc_mean = np.mean(results[method]['acc'])
        acc_std = np.std(results[method]['acc'])
        ece_mean = np.mean(results[method]['ece'])
        ece_std = np.std(results[method]['ece'])
        mce_mean = np.mean(results[method]['mce'])
        mce_std = np.std(results[method]['mce'])
        brier_mean = np.mean(results[method]['brier'])
        brier_std = np.std(results[method]['brier'])
        
        print(f"\n{method.upper()}:")
        print(f"  Accuracy: {acc_mean:.2f} ± {acc_std:.2f}%")
        print(f"  ECE: {ece_mean:.4f} ± {ece_std:.4f}")
        print(f"  MCE: {mce_mean:.4f} ± {mce_std:.4f}")
        print(f"  Brier: {brier_mean:.4f} ± {brier_std:.4f}")
        
        # Convert to JSON-serializable format
        per_seed_results = {
            'acc': [float(x) for x in results[method]['acc']],
            'ece': [float(x) for x in results[method]['ece']],
            'mce': [float(x) for x in results[method]['mce']],
            'brier': [float(x) for x in results[method]['brier']]
        }
        
        if 'converged' in results[method]:
            per_seed_results['converged'] = [int(c) for c in results[method]['converged']]
        
        if 'temperature' in results[method]:
            per_seed_results['temperature'] = [float(t) for t in results[method]['temperature']]
        
        final_results[method] = {
            'mean_acc': float(acc_mean),
            'std_acc': float(acc_std),
            'mean_ece': float(ece_mean),
            'std_ece': float(ece_std),
            'mean_mce': float(mce_mean),
            'std_mce': float(mce_std),
            'mean_brier': float(brier_mean),
            'std_brier': float(brier_std),
            'per_seed_results': per_seed_results
        }
        
        if 'converged' in results[method] and len(results[method]['converged']) > 0:
            conv_rate = sum(results[method]['converged']) / len(results[method]['converged'])
            final_results[method]['convergence_rate'] = float(conv_rate)
    
    # Statistical tests
    print("\n=== STATISTICAL TESTS ===")
    
    if num_seeds >= 3:
        # CA-BN vs Standard BN
        _, p_value_ece = ttest_rel(results['cabn']['ece'], results['standard_bn']['ece'])
        _, p_value_acc = ttest_rel(results['cabn']['acc'], results['standard_bn']['acc'])
        final_results['p_value_cabn_vs_standard_ece'] = float(p_value_ece)
        final_results['p_value_cabn_vs_standard_acc'] = float(p_value_acc)
        print(f"CA-BN vs Standard BN - ECE p-value: {p_value_ece:.4f}, Acc p-value: {p_value_acc:.4f}")
        
        # CA-BN vs Temperature Scaling
        _, p_value_temp = ttest_rel(results['cabn']['ece'], results['temp_scaling']['ece'])
        final_results['p_value_cabn_vs_temp'] = float(p_value_temp)
        print(f"CA-BN vs Temperature Scaling - ECE p-value: {p_value_temp:.4f}")
    
    # Signal detection
    ece_improvement = 100 * (np.mean(results['standard_bn']['ece']) - np.mean(results['cabn']['ece'])) / np.mean(results['standard_bn']['ece'])
    
    signal_detected = False
    if num_seeds >= 3:
        signal_detected = p_value_ece < 0.05 and ece_improvement > 5
    else:
        signal_detected = ece_improvement > 5
        
    if signal_detected:
        print(f"\nSIGNAL_DETECTED: CA-BN improves ECE by {ece_improvement:.1f}%")
    else:
        print("\nNO_SIGNAL: CA-BN does not significantly improve calibration")
    
    # Quick ablation
    if not DRY_RUN:
        print(f"\n{'='*60}")
        print("ABLATION: Confidence Thresholds")
        print('='*60)
        
        thresholds = [0.8, 0.9, 0.95]
        ablation_results = {}
        
        torch.manual_seed(42)
        for thresh in thresholds:
            model_ablation = SimpleNet(use_cabn=True).to(device)
            for m in model_ablation.modules():
                if isinstance(m, ConfidenceAdaptiveBN2d):
                    m.threshold = thresh
            
            optimizer = torch.optim.Adam(model_ablation.parameters(), lr=0.001)
            for _ in range(3):
                train_epoch(model_ablation, train_loader, optimizer, criterion, device, 10)
            
            _, ece, _, _ = evaluate(model_ablation, val_loader, device, 5)
            ablation_results[f'threshold_{thresh}'] = float(ece)
            print(f"  Threshold {thresh}: ECE = {ece:.4f}")
        
        final_results['ablation_thresholds'] = ablation_results
    
    # Final summary
    final_results['signal_detected'] = int(signal_detected)
    final_results['ece_improvement_percent'] = float(ece_improvement)
    final_results['dry_run'] = DRY_RUN
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    start_time = time.time()
    run_experiment()
    end_time = time.time()
    print(f"\nTotal runtime: {(end_time - start_time) / 60:.2f} minutes")