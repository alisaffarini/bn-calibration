# pip install scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import json
from scipy.stats import ttest_rel, bootstrap
from scipy import stats
from typing import Dict, List, Tuple
import warnings
import sys
import time
import os
warnings.filterwarnings('ignore')

# PUBLICATION MODE - NOT DRY RUN
DRY_RUN = False

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

def compute_nll(predictions, labels):
    """Compute Negative Log Likelihood"""
    return F.nll_loss(torch.log(predictions + 1e-8), labels).item()

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
overconf_labels = torch.tensor([1, 0, 1, 0])  # All wrong
ece_overconf = compute_ece(overconf_probs, overconf_labels)
print(f"Overconfident ECE: {ece_overconf:.4f}")
assert ece_overconf > 0.8, f"ECE for terrible calibration should be high, got {ece_overconf}"

print("METRIC_SANITY_PASSED\n")

class ConfidenceAdaptiveBN2d(nn.Module):
    """Confidence-Adaptive BatchNorm for 2D inputs (Conv layers)"""
    def __init__(self, num_features, eps=1e-5, momentum=0.1, threshold=0.9, 
                 interpolation='fixed', learnable_k=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.threshold = threshold
        self.interpolation = interpolation
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
        
        # Interpolation parameters
        if learnable_k:
            self.k = nn.Parameter(torch.tensor(5.0))
        else:
            self.register_buffer('k', torch.tensor(5.0))
        
        if interpolation == 'learnable':
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
            if confidence is not None and confidence.numel() > 0:
                # Separate high/low confidence samples
                high_mask = confidence > self.threshold
                low_mask = ~high_mask
                
                # Update high confidence stats if we have enough samples
                if high_mask.sum() > 1:
                    high_input = input[high_mask]
                    high_flat = high_input.transpose(0, 1).contiguous().view(C, -1)
                    high_mean = high_flat.mean(dim=1)
                    high_var = high_flat.var(dim=1, unbiased=False)
                    
                    momentum = self.momentum
                    self.running_mean_high.mul_(1 - momentum).add_(high_mean, alpha=momentum)
                    self.running_var_high.mul_(1 - momentum).add_(high_var, alpha=momentum)
                
                # Update low confidence stats if we have enough samples
                if low_mask.sum() > 1:
                    low_input = input[low_mask]
                    low_flat = low_input.transpose(0, 1).contiguous().view(C, -1)
                    low_mean = low_flat.mean(dim=1)
                    low_var = low_flat.var(dim=1, unbiased=False)
                    
                    momentum = self.momentum
                    self.running_mean_low.mul_(1 - momentum).add_(low_mean, alpha=momentum)
                    self.running_var_low.mul_(1 - momentum).add_(low_var, alpha=momentum)
            else:
                # No confidence info - update both with batch stats
                momentum = self.momentum
                self.running_mean_high.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.running_var_high.mul_(1 - momentum).add_(batch_var, alpha=momentum)
                self.running_mean_low.mul_(1 - momentum).add_(batch_mean, alpha=momentum)
                self.running_var_low.mul_(1 - momentum).add_(batch_var, alpha=momentum)
            
            self.num_batches_tracked += 1
            
            # Use batch stats for normalization during training
            mean = batch_mean.view(1, C, 1, 1)
            var = batch_var.view(1, C, 1, 1)
        else:
            # Evaluation mode - interpolate between high/low stats
            if self.interpolation == 'fixed':
                interp_weight = 0.5
            elif self.interpolation == 'sigmoid':
                if confidence is not None and confidence.numel() > 0:
                    avg_conf = confidence.mean().item()
                    interp_weight = torch.sigmoid((avg_conf - self.threshold) * self.k).item()
                else:
                    interp_weight = 0.5
            elif self.interpolation == 'learnable':
                interp_weight = self.alpha.sigmoid().item()
            else:
                interp_weight = 0.5
            
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

class ResNetBlock(nn.Module):
    """Basic ResNet block"""
    def __init__(self, in_channels, out_channels, stride=1, use_cabn=True, cabn_kwargs={}):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = ConfidenceAdaptiveBN2d(out_channels, **cabn_kwargs) if use_cabn else nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = ConfidenceAdaptiveBN2d(out_channels, **cabn_kwargs) if use_cabn else nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                ConfidenceAdaptiveBN2d(out_channels, **cabn_kwargs) if use_cabn else nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x, confidence=None):
        out = F.relu(self.bn1(self.conv1(x), confidence) if hasattr(self.bn1, 'threshold') else self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out), confidence) if hasattr(self.bn2, 'threshold') else self.bn2(self.conv2(out))
        
        # Handle shortcut
        shortcut = x
        for layer in self.shortcut:
            if hasattr(layer, 'threshold'):
                shortcut = layer(shortcut, confidence)
            else:
                shortcut = layer(shortcut)
        
        out += shortcut
        return F.relu(out)

class ResNet(nn.Module):
    """ResNet for CIFAR-10/100"""
    def __init__(self, num_blocks, use_cabn=True, num_classes=10, cabn_kwargs={}):
        super().__init__()
        self.use_cabn = use_cabn
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = ConfidenceAdaptiveBN2d(64, **cabn_kwargs) if use_cabn else nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1, use_cabn=use_cabn, cabn_kwargs=cabn_kwargs)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2, use_cabn=use_cabn, cabn_kwargs=cabn_kwargs)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2, use_cabn=use_cabn, cabn_kwargs=cabn_kwargs)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2, use_cabn=use_cabn, cabn_kwargs=cabn_kwargs)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, out_channels, num_blocks, stride, use_cabn, cabn_kwargs):
        layers = []
        layers.append(ResNetBlock(self.in_channels, out_channels, stride, use_cabn, cabn_kwargs))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, 1, use_cabn, cabn_kwargs))
        return nn.ModuleList(layers)
    
    def forward(self, x, update_stats=True):
        # Store original training mode for CABN
        if self.use_cabn:
            for m in self.modules():
                if isinstance(m, ConfidenceAdaptiveBN2d):
                    m.training_mode = update_stats
        
        # First pass to get confidences if using CABN during training
        confidence = None
        if self.use_cabn and self.training and update_stats:
            with torch.no_grad():
                # Forward pass to get confidence
                h = F.relu(self.bn1(self.conv1(x), None))
                for block in self.layer1:
                    h = block(h, None)
                for block in self.layer2:
                    h = block(h, None)
                for block in self.layer3:
                    h = block(h, None)
                for block in self.layer4:
                    h = block(h, None)
                h = self.avgpool(h)
                h = h.view(h.size(0), -1)
                logits = self.fc(h)
                probs = F.softmax(logits, dim=1)
                confidence, _ = probs.max(dim=1)
        
        # Main forward pass
        out = self.conv1(x)
        out = self.bn1(out, confidence) if self.use_cabn else self.bn1(out)
        out = F.relu(out)
        
        for block in self.layer1:
            out = block(out, confidence)
        for block in self.layer2:
            out = block(out, confidence)
        for block in self.layer3:
            out = block(out, confidence)
        for block in self.layer4:
            out = block(out, confidence)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out

def ResNet18(use_cabn=True, num_classes=10, **cabn_kwargs):
    return ResNet([2, 2, 2, 2], use_cabn, num_classes, cabn_kwargs)

class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration"""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, logits):
        return logits / self.temperature

def mixup_data(x, y, alpha=1.0):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_epoch(model, train_loader, optimizer, criterion, device, use_mixup=False, mixup_alpha=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, mixup_alpha)
            
        optimizer.zero_grad()
        outputs = model(inputs)
        
        if use_mixup:
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            loss = criterion(outputs, targets)
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if not use_mixup:
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        else:
            # Approximate accuracy for mixup
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (lam * predicted.eq(targets_a).sum().item() + 
                       (1 - lam) * predicted.eq(targets_b).sum().item())
    
    return total_loss / (batch_idx + 1), 100. * correct / total

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
    mce = compute_mce(all_probs, all_labels)
    brier = compute_brier_score(all_probs, all_labels)
    nll = compute_nll(all_probs, all_labels)
    
    return acc, ece, mce, brier, nll

def calibrate_temperature(model, val_loader, device):
    """Calibrate temperature scaling on validation set"""
    model.eval()
    temp_model = TemperatureScaling().to(device)
    
    # Collect all logits and labels
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
    
    # Optimize temperature using NLL
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temp_model.temperature], lr=0.01, max_iter=50)
    
    def eval_temp():
        optimizer.zero_grad()
        scaled_logits = temp_model(all_logits)
        loss = nll_criterion(scaled_logits, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_temp)
    
    # Ensure temperature is positive
    temp_model.temperature.data = torch.clamp(temp_model.temperature.data, min=0.1)
    
    return temp_model.temperature.item()

def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"PUBLICATION MODE: DRY_RUN = {DRY_RUN}")
    
    # Experiment parameters
    num_seeds = 10
    max_epochs = 100
    patience = 10
    initial_lr = 0.1
    
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
    
    # Split train into train/val (45k/5k)
    train_size = 45000
    val_size = 5000
    train_subset, val_subset = torch.utils.data.random_split(train_data, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=4)
    
    # Results storage
    results = {
        'standard_bn': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'nll': [], 'converged': []},
        'cabn': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'nll': [], 'converged': []},
        'temp_scaling': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'nll': [], 'temperature': []},
        'label_smoothing': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'nll': [], 'converged': []},
        'mixup': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'nll': [], 'converged': []},
        'random': {'acc': [], 'ece': [], 'mce': [], 'brier': [], 'nll': []}
    }
    
    for seed in range(num_seeds):
        print(f"\n{'='*70}")
        print(f"SEED {seed}/{num_seeds-1}")
        print('='*70)
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Random baseline (compute once)
        if seed == 0:
            print("\nRandom Baseline...")
            num_classes = 10
            random_acc = 100.0 / num_classes
            # Get all test labels
            test_labels = torch.tensor([test_data[i][1] for i in range(len(test_data))])
            random_probs = torch.ones(len(test_data), num_classes) / num_classes
            random_ece = compute_ece(random_probs, test_labels)
            random_mce = compute_mce(random_probs, test_labels)
            random_brier = compute_brier_score(random_probs, test_labels)
            random_nll = compute_nll(random_probs, test_labels)
            
            for _ in range(num_seeds):
                results['random']['acc'].append(random_acc)
                results['random']['ece'].append(random_ece)
                results['random']['mce'].append(random_mce)
                results['random']['brier'].append(random_brier)
                results['random']['nll'].append(random_nll)
            print(f"Random: Acc: {random_acc:.2f}%, ECE: {random_ece:.4f}")
        
        # 1. Standard BatchNorm
        print("\n[1/4] Training Standard BN...")
        model_standard = ResNet18(use_cabn=False).to(device)
        optimizer = torch.optim.SGD(model_standard.parameters(), lr=initial_lr, 
                                   momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model_standard, train_loader, optimizer, criterion, device)
            val_acc, val_ece, val_mce, val_brier, val_nll = evaluate(model_standard, val_loader, device)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
            
            scheduler.step()
            
            # Early stopping based on validation loss
            val_loss = val_nll
            if val_loss < best_val_loss - 0.001:
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
        test_acc, test_ece, test_mce, test_brier, test_nll = evaluate(model_standard, test_loader, device)
        results['standard_bn']['acc'].append(test_acc)
        results['standard_bn']['ece'].append(test_ece)
        results['standard_bn']['mce'].append(test_mce)
        results['standard_bn']['brier'].append(test_brier)
        results['standard_bn']['nll'].append(test_nll)
        results['standard_bn']['converged'].append(converged)
        print(f"Standard BN Test: Acc={test_acc:.2f}%, ECE={test_ece:.4f}")
        
        # 2. Temperature Scaling (post-hoc on standard model)
        print("\n[2/4] Calibrating Temperature Scaling...")
        temp_value = calibrate_temperature(model_standard, val_loader, device)
        print(f"Optimal temperature: {temp_value:.3f}")
        
        # Evaluate with temperature scaling
        model_standard.eval()
        temp_model = TemperatureScaling().to(device)
        temp_model.temperature.data = torch.tensor(temp_value)
        
        with torch.no_grad():
            all_probs = []
            all_labels = []
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model_standard(inputs, update_stats=False)
                scaled_outputs = temp_model(outputs)
                probs = F.softmax(scaled_outputs, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(targets.cpu())
            
            all_probs = torch.cat(all_probs)
            all_labels = torch.cat(all_labels)
            test_ece_temp = compute_ece(all_probs, all_labels)
            test_mce_temp = compute_mce(all_probs, all_labels)
            test_brier_temp = compute_brier_score(all_probs, all_labels)
            test_nll_temp = compute_nll(all_probs, all_labels)
            _, predicted = all_probs.max(1)
            test_acc_temp = 100. * predicted.eq(all_labels).sum().item() / all_labels.size(0)
        
        results['temp_scaling']['acc'].append(test_acc_temp)
        results['temp_scaling']['ece'].append(test_ece_temp)
        results['temp_scaling']['mce'].append(test_mce_temp)
        results['temp_scaling']['brier'].append(test_brier_temp)
        results['temp_scaling']['nll'].append(test_nll_temp)
        results['temp_scaling']['temperature'].append(temp_value)
        print(f"Temp Scaling Test: Acc={test_acc_temp:.2f}%, ECE={test_ece_temp:.4f}")
        
        # 3. Label Smoothing
        print("\n[3/4] Training with Label Smoothing...")
        model_ls = ResNet18(use_cabn=False).to(device)
        optimizer = torch.optim.SGD(model_ls.parameters(), lr=initial_lr, 
                                   momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        criterion_ls = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        best_val_loss = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model_ls, train_loader, optimizer, criterion_ls, device)
            val_acc, val_ece, val_mce, val_brier, val_nll = evaluate(model_ls, val_loader, device)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
            
            scheduler.step()
            
            val_loss = val_nll
            if val_loss < best_val_loss - 0.001:
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
        
        test_acc, test_ece, test_mce, test_brier, test_nll = evaluate(model_ls, test_loader, device)
        results['label_smoothing']['acc'].append(test_acc)
        results['label_smoothing']['ece'].append(test_ece)
        results['label_smoothing']['mce'].append(test_mce)
        results['label_smoothing']['brier'].append(test_brier)
        results['label_smoothing']['nll'].append(test_nll)
        results['label_smoothing']['converged'].append(converged)
        print(f"Label Smoothing Test: Acc={test_acc:.2f}%, ECE={test_ece:.4f}")
        
        # 4. Mixup
        print("\n[4/4] Training with Mixup...")
        model_mixup = ResNet18(use_cabn=False).to(device)
        optimizer = torch.optim.SGD(model_mixup.parameters(), lr=initial_lr, 
                                   momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model_mixup, train_loader, optimizer, 
                                              criterion, device, use_mixup=True, mixup_alpha=1.0)
            val_acc, val_ece, val_mce, val_brier, val_nll = evaluate(model_mixup, val_loader, device)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
            
            scheduler.step()
            
            val_loss = val_nll
            if val_loss < best_val_loss - 0.001:
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
        
        test_acc, test_ece, test_mce, test_brier, test_nll = evaluate(model_mixup, test_loader, device)
        results['mixup']['acc'].append(test_acc)
        results['mixup']['ece'].append(test_ece)
        results['mixup']['mce'].append(test_mce)
        results['mixup']['brier'].append(test_brier)
        results['mixup']['nll'].append(test_nll)
        results['mixup']['converged'].append(converged)
        print(f"Mixup Test: Acc={test_acc:.2f}%, ECE={test_ece:.4f}")
        
        # 5. Confidence-Adaptive BN (default: fixed interpolation)
        print("\n[5/5] Training CA-BN...")
        model_cabn = ResNet18(use_cabn=True, interpolation='fixed', threshold=0.9).to(device)
        optimizer = torch.optim.SGD(model_cabn.parameters(), lr=initial_lr, 
                                   momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model_cabn, train_loader, optimizer, criterion, device)
            val_acc, val_ece, val_mce, val_brier, val_nll = evaluate(model_cabn, val_loader, device)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val ECE: {val_ece:.4f}")
            
            scheduler.step()
            
            val_loss = val_nll
            if val_loss < best_val_loss - 0.001:
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
        
        test_acc, test_ece, test_mce, test_brier, test_nll = evaluate(model_cabn, test_loader, device)
        results['cabn']['acc'].append(test_acc)
        results['cabn']['ece'].append(test_ece)
        results['cabn']['mce'].append(test_mce)
        results['cabn']['brier'].append(test_brier)
        results['cabn']['nll'].append(test_nll)
        results['cabn']['converged'].append(converged)
        print(f"CA-BN Test: Acc={test_acc:.2f}%, ECE={test_ece:.4f}")
        
        # Early abort check after first seed
        if seed == 0:
            print("\n=== EARLY ABORT CHECK ===")
            # Check if metrics are reasonable
            if test_acc < 70:  # CIFAR-10 should achieve >70% easily
                print(f"SANITY_ABORT: Test accuracy too low ({test_acc:.2f}%)")
                sys.exit(1)
            
            if abs(results['standard_bn']['ece'][0] - results['cabn']['ece'][0]) < 1e-6:
                print(f"SANITY_ABORT: Standard BN and CA-BN have identical ECE")
                sys.exit(1)
            
            if results['standard_bn']['ece'][0] == 0.0 or np.isnan(results['standard_bn']['ece'][0]):
                print(f"SANITY_ABORT: Standard BN ECE is degenerate")
                sys.exit(1)
            
            improvement = (results['standard_bn']['ece'][0] - results['cabn']['ece'][0]) / results['standard_bn']['ece'][0]
            print(f"First seed ECE improvement: {improvement*100:.1f}%")
            print(f"Test accuracies: Standard={test_acc:.1f}%, CA-BN={results['cabn']['acc'][0]:.1f}%")
            print("Metrics look reasonable, continuing...")
    
    # Ablation Studies
    print(f"\n{'='*70}")
    print("ABLATION STUDIES")
    print('='*70)
    
    ablation_results = {}
    
    # Ablation 1: Confidence thresholds
    print("\n[Ablation 1] Confidence Thresholds")
    thresholds = [0.7, 0.8, 0.85, 0.9, 0.95]
    threshold_results = {'acc': [], 'ece': []}
    
    torch.manual_seed(42)
    for thresh in thresholds:
        print(f"\nTesting threshold={thresh}")
        model_thresh = ResNet18(use_cabn=True, threshold=thresh).to(device)
        optimizer = torch.optim.SGD(model_thresh.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        # Quick training for ablation
        for epoch in range(30):
            train_loss, train_acc = train_epoch(model_thresh, train_loader, optimizer, criterion, device)
            scheduler.step()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train Acc: {train_acc:.2f}%")
        
        test_acc, test_ece, _, _, _ = evaluate(model_thresh, test_loader, device)
        threshold_results['acc'].append(test_acc)
        threshold_results['ece'].append(test_ece)
        print(f"  Threshold {thresh}: Test Acc={test_acc:.2f}%, ECE={test_ece:.4f}")
    
    ablation_results['thresholds'] = {
        'values': thresholds,
        'acc': threshold_results['acc'],
        'ece': threshold_results['ece']
    }
    
    # Ablation 2: Interpolation strategies
    print("\n[Ablation 2] Interpolation Strategies")
    interpolations = ['fixed', 'sigmoid', 'learnable']
    interp_results = {'acc': [], 'ece': []}
    
    for interp in interpolations:
        print(f"\nTesting interpolation={interp}")
        model_interp = ResNet18(use_cabn=True, interpolation=interp, learnable_k=(interp=='sigmoid')).to(device)
        optimizer = torch.optim.SGD(model_interp.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        
        for epoch in range(30):
            train_loss, train_acc = train_epoch(model_interp, train_loader, optimizer, criterion, device)
            scheduler.step()
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}: Train Acc: {train_acc:.2f}%")
        
        test_acc, test_ece, _, _, _ = evaluate(model_interp, test_loader, device)
        interp_results['acc'].append(test_acc)
        interp_results['ece'].append(test_ece)
        print(f"  {interp}: Test Acc={test_acc:.2f}%, ECE={test_ece:.4f}")
    
    ablation_results['interpolations'] = {
        'strategies': interpolations,
        'acc': interp_results['acc'],
        'ece': interp_results['ece']
    }
    
    # Compute final statistics
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print('='*70)
    
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
        nll_mean = np.mean(results[method]['nll'])
        nll_std = np.std(results[method]['nll'])
        
        print(f"\n{method.upper()}:")
        print(f"  Accuracy: {acc_mean:.2f} ± {acc_std:.2f}%")
        print(f"  ECE: {ece_mean:.4f} ± {ece_std:.4f}")
        print(f"  MCE: {mce_mean:.4f} ± {mce_std:.4f}")
        print(f"  Brier: {brier_mean:.4f} ± {brier_std:.4f}")
        print(f"  NLL: {nll_mean:.4f} ± {nll_std:.4f}")
        
        # Convert to JSON-serializable format
        per_seed_results = {
            'acc': [float(x) for x in results[method]['acc']],
            'ece': [float(x) for x in results[method]['ece']],
            'mce': [float(x) for x in results[method]['mce']],
            'brier': [float(x) for x in results[method]['brier']],
            'nll': [float(x) for x in results[method]['nll']]
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
            'mean_nll': float(nll_mean),
            'std_nll': float(nll_std),
            'per_seed_results': per_seed_results
        }
        
        if 'converged' in results[method] and len(results[method]['converged']) > 0:
            conv_rate = sum(results[method]['converged']) / len(results[method]['converged'])
            final_results[method]['convergence_rate'] = float(conv_rate)
    
    # Statistical tests
    print("\n=== STATISTICAL TESTS ===")
    
    # Paired t-tests
    _, p_ece = ttest_rel(results['cabn']['ece'], results['standard_bn']['ece'])
    _, p_acc = ttest_rel(results['cabn']['acc'], results['standard_bn']['acc'])
    _, p_ece_temp = ttest_rel(results['cabn']['ece'], results['temp_scaling']['ece'])
    _, p_ece_mixup = ttest_rel(results['cabn']['ece'], results['mixup']['ece'])
    
    final_results['p_values'] = {
        'cabn_vs_standard_ece': float(p_ece),
        'cabn_vs_standard_acc': float(p_acc),
        'cabn_vs_temp_scaling_ece': float(p_ece_temp),
        'cabn_vs_mixup_ece': float(p_ece_mixup)
    }
    
    print(f"CA-BN vs Standard BN - ECE p-value: {p_ece:.4f}, Acc p-value: {p_acc:.4f}")
    print(f"CA-BN vs Temperature Scaling - ECE p-value: {p_ece_temp:.4f}")
    print(f"CA-BN vs Mixup - ECE p-value: {p_ece_mixup:.4f}")
    
    # Bootstrap confidence intervals
    def bootstrap_ci(data1, data2, n_bootstrap=1000):
        """Compute bootstrap CI for difference in means"""
        diffs = []
        data1, data2 = np.array(data1), np.array(data2)
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(data1), len(data1), replace=True)
            diff = np.mean(data1[idx] - data2[idx])
            diffs.append(diff)
        return np.percentile(diffs, [2.5, 97.5])
    
    ci_ece = bootstrap_ci(results['standard_bn']['ece'], results['cabn']['ece'])
    print(f"\n95% Bootstrap CI for ECE improvement: [{ci_ece[0]:.4f}, {ci_ece[1]:.4f}]")
    
    # Effect sizes
    ece_improvement = 100 * (np.mean(results['standard_bn']['ece']) - np.mean(results['cabn']['ece'])) / np.mean(results['standard_bn']['ece'])
    
    # Signal detection
    signal_detected = p_ece < 0.05 and ece_improvement > 10 and ci_ece[0] > 0
    
    if signal_detected:
        print(f"\nSIGNAL_DETECTED: CA-BN significantly improves ECE by {ece_improvement:.1f}% (p={p_ece:.4f})")
    else:
        print(f"\nNO_SIGNAL: CA-BN does not significantly improve calibration (improvement={ece_improvement:.1f}%, p={p_ece:.4f})")
    
    # Add summary to results
    final_results['summary'] = {
        'signal_detected': int(signal_detected),
        'ece_improvement_percent': float(ece_improvement),
        'bootstrap_ci_lower': float(ci_ece[0]),
        'bootstrap_ci_upper': float(ci_ece[1]),
        'dry_run': False
    }
    
    # Add ablation results
    final_results['ablations'] = ablation_results
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    start_time = time.time()
    run_experiment()
    end_time = time.time()
    print(f"\nTotal runtime: {(end_time - start_time) / 3600:.2f} hours")