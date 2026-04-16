# pip install scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json
from scipy.stats import ttest_rel
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

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

# Metric sanity check
print("=== METRIC SANITY CHECK ===")

# Test 1: Perfect calibration
perfect_probs = torch.tensor([[0.7, 0.3], [0.8, 0.2], [0.6, 0.4], [0.9, 0.1]])
perfect_labels = torch.tensor([0, 0, 0, 0])  
ece_perfect = compute_ece(perfect_probs, perfect_labels)
print(f"Perfect calibration ECE: {ece_perfect:.4f}")
assert ece_perfect < 0.1, f"ECE for perfect calibration should be near 0, got {ece_perfect}"

# Test 2: Overconfident predictions (high confidence, low accuracy)
overconf_probs = torch.tensor([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01], [0.01, 0.99]])
overconf_labels = torch.tensor([0, 0, 0, 0])  # 50% accuracy with 99% confidence
ece_overconf = compute_ece(overconf_probs, overconf_labels)
print(f"Overconfident ECE: {ece_overconf:.4f}")
assert ece_overconf > 0.4, f"ECE for overconfident predictions should be high, got {ece_overconf}"

# Test 3: Random baseline
random_probs = F.softmax(torch.randn(100, 10), dim=1)
random_labels = torch.randint(0, 10, (100,))
ece_random = compute_ece(random_probs, random_labels)
print(f"Random predictions ECE: {ece_random:.4f}")

print("METRIC_SANITY_PASSED\n")

class ConfidenceAdaptiveBN2d(nn.Module):
    """Confidence-Adaptive BatchNorm for 2D inputs (Conv layers)"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, threshold=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.threshold = threshold
        self.training_mode = True
        
        # Learnable affine parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics for high and low confidence
        self.register_buffer('running_mean_high', torch.zeros(num_features))
        self.register_buffer('running_var_high', torch.ones(num_features))
        self.register_buffer('running_mean_low', torch.zeros(num_features))
        self.register_buffer('running_var_low', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
        # Learnable interpolation
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, input, confidence=None):
        # input shape: (N, C, H, W)
        if self.training_mode and self.training:
            # Compute batch statistics
            N, C, H, W = input.shape
            x_flat = input.transpose(0, 1).contiguous().view(C, -1)
            batch_mean = x_flat.mean(dim=1)
            batch_var = x_flat.var(dim=1, unbiased=False)
            
            # Update running statistics based on confidence
            if confidence is not None:
                # Separate high/low confidence samples
                high_mask = confidence > self.threshold
                low_mask = ~high_mask
                
                # Update high confidence stats if we have samples
                if high_mask.sum() > 0:
                    high_input = input[high_mask]
                    high_flat = high_input.transpose(0, 1).contiguous().view(C, -1)
                    high_mean = high_flat.mean(dim=1)
                    high_var = high_flat.var(dim=1, unbiased=False)
                    
                    momentum = self.momentum if self.num_batches_tracked > 0 else 1.0
                    self.running_mean_high.mul_(1 - momentum).add_(high_mean, alpha=momentum)
                    self.running_var_high.mul_(1 - momentum).add_(high_var, alpha=momentum)
                
                # Update low confidence stats if we have samples
                if low_mask.sum() > 0:
                    low_input = input[low_mask]
                    low_flat = low_input.transpose(0, 1).contiguous().view(C, -1)
                    low_mean = low_flat.mean(dim=1)
                    low_var = low_flat.var(dim=1, unbiased=False)
                    
                    momentum = self.momentum if self.num_batches_tracked > 0 else 1.0
                    self.running_mean_low.mul_(1 - momentum).add_(low_mean, alpha=momentum)
                    self.running_var_low.mul_(1 - momentum).add_(low_var, alpha=momentum)
            else:
                # No confidence info - update both with batch stats
                momentum = self.momentum if self.num_batches_tracked > 0 else 1.0
                self.running_mean_high.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.running_var_high.mul_(1 - momentum).add_(batch_var, alpha=momentum)
                self.running_mean_low.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.running_var_low.mul_(1 - momentum).add_(batch_var, alpha=momentum)
            
            self.num_batches_tracked += 1
            
            # Use batch stats for normalization
            mean = batch_mean.view(1, C, 1, 1)
            var = batch_var.view(1, C, 1, 1)
        else:
            # Evaluation mode - interpolate between high/low stats
            if confidence is not None and confidence.numel() > 0:
                # Use mean confidence for batch
                avg_conf = confidence.mean().item()
                interp_weight = torch.sigmoid((avg_conf - self.threshold) * 5.0).item()
            else:
                interp_weight = self.alpha.sigmoid().item()
            
            mean = (1 - interp_weight) * self.running_mean_low + interp_weight * self.running_mean_high
            var = (1 - interp_weight) * self.running_var_low + interp_weight * self.running_var_high
            mean = mean.view(1, self.num_features, 1, 1)
            var = var.view(1, self.num_features, 1, 1)
        
        # Apply normalization
        out = (input - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        weight = self.weight.view(1, self.num_features, 1, 1)
        bias = self.bias.view(1, self.num_features, 1, 1)
        
        return weight * out + bias

class SimpleNet(nn.Module):
    """Simple CNN for CIFAR-10"""
    def __init__(self, use_cabn=True, num_classes=10):
        super().__init__()
        self.use_cabn = use_cabn
        
        # Simple architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = ConfidenceAdaptiveBN2d(32) if use_cabn else nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = ConfidenceAdaptiveBN2d(64) if use_cabn else nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, update_stats=True):
        # Store original training mode for CABN
        if self.use_cabn:
            for m in self.modules():
                if isinstance(m, ConfidenceAdaptiveBN2d):
                    m.training_mode = update_stats
        
        # First pass to get confidences if using CABN
        if self.use_cabn and self.training and update_stats:
            with torch.no_grad():
                # Quick forward to get confidence
                h = F.relu(self.bn1(self.conv1(x)))
                h = self.pool(h)
                h = F.relu(self.bn2(self.conv2(h)))
                h = self.pool(h)
                h = h.view(h.size(0), -1)
                h = F.relu(self.fc1(h))
                logits = self.fc2(h)
                probs = F.softmax(logits, dim=1)
                confidence, _ = probs.max(dim=1)
        else:
            confidence = None
        
        # Main forward pass
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

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / total, 100. * correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, update_stats=False)
            probs = F.softmax(outputs, dim=1)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_probs.append(probs.cpu())
            all_labels.append(targets.cpu())
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    acc = 100. * correct / total
    ece = compute_ece(all_probs, all_labels)
    
    return acc, ece

def calibrate_temperature(model, val_loader, device):
    """Calibrate temperature scaling on validation set"""
    model.eval()
    temp_model = TemperatureScaling().to(device)
    
    # Collect all logits
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, update_stats=False)
            all_logits.append(outputs)
            all_labels.append(targets)
    
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Optimize temperature
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
    
    def eval_temp():
        scaled_logits = temp_model(all_logits)
        loss = nll_criterion(scaled_logits, all_labels)
        return loss
    
    for _ in range(10):
        optimizer.step(eval_temp)
    
    # Evaluate calibrated model
    with torch.no_grad():
        scaled_logits = temp_model(all_logits)
        probs = F.softmax(scaled_logits, dim=1)
        _, predicted = scaled_logits.max(1)
        acc = 100. * predicted.eq(all_labels).sum().item() / all_labels.size(0)
        ece = compute_ece(probs, all_labels)
    
    return acc, ece, temp_model.temperature.item()

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    # Split train into train/val
    train_size = int(0.9 * len(train_data))
    val_size = len(train_data) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_data, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2)
    
    # Experiment parameters
    num_seeds = 3
    max_epochs = 20  # Reduced for feasibility
    patience = 5
    
    results = {
        'standard_bn': {'acc': [], 'ece': [], 'converged': []},
        'cabn': {'acc': [], 'ece': [], 'converged': []},
        'temp_scaling': {'acc': [], 'ece': [], 'temperature': []}
    }
    
    for seed in range(num_seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print('='*50)
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Standard BatchNorm
        print("\nTraining Standard BN...")
        model_standard = SimpleNet(use_cabn=False).to(device)
        optimizer = torch.optim.SGD(model_standard.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_ece = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model_standard, train_loader, optimizer, criterion, device)
            val_acc, val_ece = evaluate(model_standard, val_loader, device)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
            
            scheduler.step()
            
            if val_ece < best_ece:
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
            
        results['standard_bn']['acc'].append(val_acc)
        results['standard_bn']['ece'].append(val_ece)
        results['standard_bn']['converged'].append(converged)
        
        # Temperature Scaling
        print("\nCalibrating temperature...")
        temp_acc, temp_ece, temp_value = calibrate_temperature(model_standard, val_loader, device)
        print(f"Temperature Scaling - Acc: {temp_acc:.2f}%, ECE: {temp_ece:.4f}, T: {temp_value:.3f}")
        results['temp_scaling']['acc'].append(temp_acc)
        results['temp_scaling']['ece'].append(temp_ece)
        results['temp_scaling']['temperature'].append(temp_value)
        
        # Confidence-Adaptive BN
        print("\nTraining CA-BN...")
        model_cabn = SimpleNet(use_cabn=True).to(device)
        optimizer = torch.optim.SGD(model_cabn.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        best_ece = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model_cabn, train_loader, optimizer, criterion, device)
            val_acc, val_ece = evaluate(model_cabn, val_loader, device)
            
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
            
            scheduler.step()
            
            if val_ece < best_ece:
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
            
        results['cabn']['acc'].append(val_acc)
        results['cabn']['ece'].append(val_ece)
        results['cabn']['converged'].append(converged)
    
    # Compute final statistics
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print('='*50)
    
    final_results = {}
    
    for method in results:
        acc_mean = np.mean(results[method]['acc'])
        acc_std = np.std(results[method]['acc']) if len(results[method]['acc']) > 1 else 0.0
        ece_mean = np.mean(results[method]['ece'])
        ece_std = np.std(results[method]['ece']) if len(results[method]['ece']) > 1 else 0.0
        
        print(f"\n{method.upper()}:")
        print(f"  Accuracy: {acc_mean:.2f} ± {acc_std:.2f}%")
        print(f"  ECE: {ece_mean:.4f} ± {ece_std:.4f}")
        
        final_results[method] = {
            'mean_acc': float(acc_mean),
            'std_acc': float(acc_std),
            'mean_ece': float(ece_mean),
            'std_ece': float(ece_std),
            'per_seed_results': results[method]
        }
        
        if 'converged' in results[method]:
            conv_rate = sum(results[method]['converged']) / len(results[method]['converged'])
            final_results[method]['convergence_rate'] = float(conv_rate)
    
    # Statistical significance tests
    if num_seeds >= 3:
        # CA-BN vs Standard BN
        _, p_value = ttest_rel(results['cabn']['ece'], results['standard_bn']['ece'])
        final_results['p_value_cabn_vs_standard'] = float(p_value)
        print(f"\nCA-BN vs Standard BN (ECE): p = {p_value:.4f}")
        
        # Temp Scaling vs Standard BN
        _, p_value_temp = ttest_rel(results['temp_scaling']['ece'], results['standard_bn']['ece'])
        final_results['p_value_temp_vs_standard'] = float(p_value_temp)
    
    # Signal detection
    improvement = (results['standard_bn']['ece'][0] - results['cabn']['ece'][0]) / results['standard_bn']['ece'][0]
    
    if np.mean(results['cabn']['ece']) < np.mean(results['standard_bn']['ece']) and improvement > 0.05:
        print(f"\nSIGNAL_DETECTED: CA-BN improves ECE by {improvement*100:.1f}%")
    else:
        print("\nNO_SIGNAL: CA-BN does not significantly improve calibration")
    
    # Simple ablation: different thresholds
    print(f"\n{'='*50}")
    print("ABLATION: Confidence Thresholds")
    print('='*50)
    
    torch.manual_seed(42)
    thresholds = [0.8, 0.9, 0.95]
    ablation_results = {}
    
    for thresh in thresholds:
        model_ablation = SimpleNet(use_cabn=True).to(device)
        # Set threshold
        for m in model_ablation.modules():
            if isinstance(m, ConfidenceAdaptiveBN2d):
                m.threshold = thresh
        
        # Quick training
        optimizer = torch.optim.SGD(model_ablation.parameters(), lr=0.1, momentum=0.9)
        for epoch in range(5):  # Quick test
            train_epoch(model_ablation, train_loader, optimizer, criterion, device)
        
        _, ece = evaluate(model_ablation, val_loader, device)
        ablation_results[f'threshold_{thresh}'] = float(ece)
        print(f"Threshold {thresh}: ECE = {ece:.4f}")
    
    final_results['ablation_thresholds'] = ablation_results
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    run_experiment()