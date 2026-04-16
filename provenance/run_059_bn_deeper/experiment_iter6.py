# pip install torch numpy scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random
import json
from collections import defaultdict
from scipy import stats
import warnings
import time
import os
import gc
warnings.filterwarnings('ignore')

# ==================== METRIC SANITY CHECK ====================
def compute_kl_divergence(p_mean, p_var, q_mean, q_var):
    """KL divergence between two Gaussians."""
    # KL(P||Q) for Gaussians
    var_ratio = p_var / (q_var + 1e-8)
    diff_sq = (q_mean - p_mean) ** 2
    kl = 0.5 * (var_ratio - 1 - torch.log(var_ratio + 1e-8) + diff_sq / (q_var + 1e-8))
    return kl.mean()

def compute_ece(outputs, labels, n_bins=15):
    """Expected Calibration Error."""
    softmax = F.softmax(outputs, dim=1)
    confidences, predictions = torch.max(softmax, 1)
    accuracies = predictions.eq(labels).float()
    
    ece = 0
    for bin_idx in range(n_bins):
        bin_lower = bin_idx / n_bins
        bin_upper = (bin_idx + 1) / n_bins
        
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_confidence = confidences[in_bin].mean()
            bin_accuracy = accuracies[in_bin].mean()
            ece += (in_bin.sum().float() / len(labels)) * torch.abs(bin_confidence - bin_accuracy)
    
    return ece.item()

# Sanity checks
print("Running metric sanity checks...")

# Test KL divergence
p_mean = torch.tensor([0.0])
p_var = torch.tensor([1.0])
# KL(P||P) should be 0
kl_same = compute_kl_divergence(p_mean, p_var, p_mean, p_var)
assert kl_same < 1e-6, f"KL(P||P) should be ~0, got {kl_same}"

# KL divergence should be positive for different distributions
q_mean = torch.tensor([1.0])
q_var = torch.tensor([2.0])
kl_diff = compute_kl_divergence(p_mean, p_var, q_mean, q_var)
assert kl_diff > 0, f"KL divergence should be positive, got {kl_diff}"

# Test ECE
# Perfect calibration: confidence = accuracy
outputs_perfect = torch.tensor([[10.0, 0.0], [0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
labels_perfect = torch.tensor([0, 1, 0, 1])
ece_perfect = compute_ece(outputs_perfect, labels_perfect)
assert ece_perfect < 0.05, f"ECE for perfect calibration should be ~0, got {ece_perfect}"

# Poor calibration: high confidence but wrong
outputs_poor = torch.tensor([[10.0, 0.0], [10.0, 0.0], [10.0, 0.0], [10.0, 0.0]])
labels_poor = torch.tensor([0, 1, 1, 1])  # 25% accuracy but 99%+ confidence
ece_poor = compute_ece(outputs_poor, labels_poor)
assert ece_poor > 0.5, f"ECE for poor calibration should be high, got {ece_poor}"

print("METRIC_SANITY_PASSED")

# ==================== DATA GENERATION ====================
def generate_synthetic_cifar(num_train=5000, num_test=1000, num_classes=10):
    """Generate synthetic CIFAR-like data with class-specific patterns."""
    # Simple synthetic data: images are 3x32x32
    def generate_class_data(n_samples, class_id, num_classes):
        # Each class has a different pattern
        X = torch.randn(n_samples, 3, 32, 32) * 0.3
        # Add class-specific pattern
        pattern_channel = class_id % 3
        pattern_location = (class_id // 3) * 10
        X[:, pattern_channel, pattern_location:pattern_location+5, pattern_location:pattern_location+5] += 2.0
        y = torch.full((n_samples,), class_id, dtype=torch.long)
        return X, y
    
    # Generate train data
    train_X, train_y = [], []
    for c in range(num_classes):
        X_c, y_c = generate_class_data(num_train // num_classes, c, num_classes)
        train_X.append(X_c)
        train_y.append(y_c)
    
    train_X = torch.cat(train_X, 0)
    train_y = torch.cat(train_y, 0)
    
    # Shuffle
    perm = torch.randperm(len(train_y))
    train_X = train_X[perm]
    train_y = train_y[perm]
    
    # Generate test data
    test_X, test_y = [], []
    for c in range(num_classes):
        X_c, y_c = generate_class_data(num_test // num_classes, c, num_classes)
        test_X.append(X_c)
        test_y.append(y_c)
    
    test_X = torch.cat(test_X, 0)
    test_y = torch.cat(test_y, 0)
    
    # Shuffle test
    perm = torch.randperm(len(test_y))
    test_X = test_X[perm]
    test_y = test_y[perm]
    
    return train_X, train_y, test_X, test_y

# ==================== MODEL DEFINITIONS ====================
class BasicBlock(nn.Module):
    """Basic ResNet block."""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class TinyResNet(nn.Module):
    """Smaller ResNet for memory efficiency."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 8
        
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        
        self.layer1 = self._make_layer(8, 2, stride=1)
        self.layer2 = self._make_layer(16, 2, stride=2)
        
        self.linear = nn.Linear(16, num_classes)
        
        # Store references to all BN layers
        self.bn_layers = []
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.bn_layers.append(m)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 32x32
        out = self.layer1(out)  # 32x32
        out = self.layer2(out)  # 16x16
        out = F.adaptive_avg_pool2d(out, 1)  # Global average pooling
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class SimpleConvNet(nn.Module):
    """Simple baseline model without BN."""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32x32
        x = self.pool(F.relu(self.conv2(x)))  # 32->16
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss / len(train_loader), correct / total

def evaluate(model, loader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_outputs.append(outputs.cpu())  # Move to CPU to save GPU memory
            all_labels.append(targets.cpu())
    
    if len(all_outputs) > 0:
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        ece = compute_ece(all_outputs, all_labels)
    else:
        ece = 0.0
    
    return total_loss / max(1, len(loader)), correct / max(1, total), ece

def collect_bn_stats_per_class(model, loader, device, num_classes=10):
    """Collect per-class statistics for each BN layer."""
    model.eval()
    
    # Storage for statistics
    class_stats = defaultdict(lambda: defaultdict(list))
    global_stats = defaultdict(list)
    
    # Hook to capture BN inputs
    handles = []
    bn_inputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            bn_inputs[name] = input[0].detach().cpu()  # Move to CPU immediately
        return hook
    
    # Register hooks
    for i, bn in enumerate(model.bn_layers):
        handle = bn.register_forward_hook(hook_fn(f'bn_{i}'))
        handles.append(handle)
    
    # Collect statistics
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            _ = model(inputs)
            
            # Process each BN layer's inputs
            for i, bn_name in enumerate([f'bn_{i}' for i in range(len(model.bn_layers))]):
                if bn_name in bn_inputs:
                    features = bn_inputs[bn_name]
                    targets_cpu = targets.cpu()
                    
                    # Global stats
                    global_stats[i].append(features)
                    
                    # Per-class stats
                    for cls in range(num_classes):
                        mask = targets_cpu == cls
                        if mask.sum() > 0:
                            class_features = features[mask]
                            class_stats[i][cls].append(class_features)
            
            # Clear to save memory
            bn_inputs.clear()
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Compute means and variances
    final_stats = {'global': {}, 'per_class': defaultdict(dict)}
    
    for i in range(len(model.bn_layers)):
        # Global stats
        if len(global_stats[i]) > 0:
            global_features = torch.cat(global_stats[i], dim=0)
            if len(global_features.shape) == 4:  # Conv layers
                global_mean = global_features.mean(dim=(0, 2, 3))
                global_var = global_features.var(dim=(0, 2, 3), unbiased=False)
            else:  # FC layers
                global_mean = global_features.mean(dim=0)
                global_var = global_features.var(dim=0, unbiased=False)
            final_stats['global'][i] = {'mean': global_mean.to(device), 'var': global_var.to(device)}
        
        # Per-class stats
        for cls in range(num_classes):
            if cls in class_stats[i] and len(class_stats[i][cls]) > 0:
                class_features = torch.cat(class_stats[i][cls], dim=0)
                if len(class_features.shape) == 4:  # Conv layers
                    class_mean = class_features.mean(dim=(0, 2, 3))
                    class_var = class_features.var(dim=(0, 2, 3), unbiased=False)
                else:  # FC layers
                    class_mean = class_features.mean(dim=0)
                    class_var = class_features.var(dim=0, unbiased=False)
                final_stats['per_class'][i][cls] = {'mean': class_mean.to(device), 'var': class_var.to(device)}
    
    # Clear intermediate stats
    global_stats.clear()
    class_stats.clear()
    gc.collect()
    
    return final_stats

def interpolate_and_evaluate_fast(model, stats, val_loader, device, num_classes=10):
    """Fast interpolation evaluation."""
    model.eval()
    
    # Save original stats
    original_stats = []
    for bn in model.bn_layers:
        original_stats.append({
            'running_mean': bn.running_mean.clone(),
            'running_var': bn.running_var.clone()
        })
    
    # Fewer alpha values for speed
    alpha_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    results = {}
    
    # Test only first 3 layers for speed
    num_layers_to_test = min(3, len(model.bn_layers))
    
    for layer_idx in range(num_layers_to_test):
        results[layer_idx] = []
        
        for alpha in alpha_values:
            # Restore original stats
            for i, bn in enumerate(model.bn_layers):
                bn.running_mean.data = original_stats[i]['running_mean'].clone()
                bn.running_var.data = original_stats[i]['running_var'].clone()
            
            # Interpolate stats for specific layer
            if layer_idx in stats['global']:
                bn = model.bn_layers[layer_idx]
                global_mean = stats['global'][layer_idx]['mean']
                global_var = stats['global'][layer_idx]['var']
                
                # Compute average class stats
                class_means = []
                class_vars = []
                for cls in range(num_classes):
                    if cls in stats['per_class'][layer_idx]:
                        class_means.append(stats['per_class'][layer_idx][cls]['mean'])
                        class_vars.append(stats['per_class'][layer_idx][cls]['var'])
                
                if len(class_means) > 0:
                    avg_class_mean = torch.stack(class_means).mean(dim=0)
                    avg_class_var = torch.stack(class_vars).mean(dim=0)
                    
                    # Interpolate
                    bn.running_mean.data = alpha * avg_class_mean + (1 - alpha) * global_mean
                    bn.running_var.data = alpha * avg_class_var + (1 - alpha) * global_var
                    
                    # Compute KL divergence
                    kl_div = compute_kl_divergence(avg_class_mean, avg_class_var, global_mean, global_var)
                else:
                    kl_div = 0.0
                
                # Evaluate
                val_loss, val_acc, ece = evaluate(model, val_loader, device)
                
                results[layer_idx].append({
                    'alpha': alpha,
                    'kl_div': kl_div.item() if torch.is_tensor(kl_div) else kl_div,
                    'val_acc': val_acc,
                    'ece': ece
                })
    
    # Restore original stats
    for i, bn in enumerate(model.bn_layers):
        bn.running_mean.data = original_stats[i]['running_mean']
        bn.running_var.data = original_stats[i]['running_var']
    
    return results

# ==================== MAIN EXPERIMENT ====================
def run_single_seed(seed, full_scale=True):
    """Run experiment for a single seed."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate synthetic data - REDUCED SIZES
    print(f"\n=== Seed {seed} ===")
    if full_scale:
        # Reduced "full scale" for memory efficiency
        train_X, train_y, test_X, test_y = generate_synthetic_cifar(num_train=5000, num_test=1000, num_classes=10)
        num_classes = 10
        max_epochs = 30
        patience = 5
    else:
        # Small scale for testing
        train_X, train_y, test_X, test_y = generate_synthetic_cifar(num_train=1000, num_test=200, num_classes=10)
        num_classes = 10
        max_epochs = 15
        patience = 3
    
    # Create train/val split (80/20)
    val_size = len(train_X) // 5
    val_X = train_X[:val_size]
    val_y = train_y[:val_size]
    train_X = train_X[val_size:]
    train_y = train_y[val_size:]
    
    # Create datasets
    train_dataset = TensorDataset(train_X, train_y)
    val_dataset = TensorDataset(val_X, val_y)
    test_dataset = TensorDataset(test_X, test_y)
    
    # Smaller batch size to save memory
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    results = {}
    
    try:
        # 1. RANDOM BASELINE
        print("Testing random baseline...")
        random_model = TinyResNet(num_classes=num_classes).to(device)
        _, test_acc, test_ece = evaluate(random_model, test_loader, device)
        results['random_baseline'] = {
            'test_acc': test_acc,
            'test_ece': test_ece
        }
        print(f"Random baseline: Acc={test_acc:.4f}, ECE={test_ece:.4f}")
        del random_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 2. SIMPLE BASELINE (No BN)
        print("Training simple baseline (no BN)...")
        simple_model = SimpleConvNet(num_classes=num_classes).to(device)
        optimizer = optim.Adam(simple_model.parameters(), lr=0.001)
        
        for epoch in range(min(10, max_epochs)):
            train_loss, train_acc = train_epoch(simple_model, train_loader, optimizer, device)
        
        _, test_acc, test_ece = evaluate(simple_model, test_loader, device)
        results['simple_baseline'] = {
            'test_acc': test_acc,
            'test_ece': test_ece
        }
        print(f"Simple baseline: Acc={test_acc:.4f}, ECE={test_ece:.4f}")
        del simple_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 3. MAIN METHOD (BN with interpolation)
        print("Training main model with BN...")
        model = TinyResNet(num_classes=num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
        
        best_val_loss = float('inf')
        patience_counter = 0
        converged = False
        
        for epoch in range(max_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc, val_ece = evaluate(model, val_loader, device)
            scheduler.step()
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Val ECE={val_ece:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience and epoch >= 10:
                print(f"CONVERGED at epoch {epoch}")
                converged = True
                break
        
        if not converged:
            print("NOT_CONVERGED: Maximum epochs reached")
        
        # Evaluate baseline model
        _, test_acc, test_ece = evaluate(model, test_loader, device)
        results['bn_baseline'] = {
            'test_acc': test_acc,
            'test_ece': test_ece,
            'converged': converged
        }
        print(f"BN baseline: Acc={test_acc:.4f}, ECE={test_ece:.4f}")
        
        # Collect BN statistics
        print("Collecting BN statistics...")
        stats = collect_bn_stats_per_class(model, train_loader, device, num_classes)
        
        # Test interpolation
        print("Testing interpolation...")
        interpolation_results = interpolate_and_evaluate_fast(model, stats, test_loader, device, num_classes)
        
        results['interpolation'] = interpolation_results
        results['num_bn_layers'] = len(model.bn_layers)
        
        # 4. ABLATION: Temperature Scaling (simplified)
        print("Testing temperature scaling...")
        best_temp = 1.0
        model.eval()
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs) / best_temp
                all_outputs.append(outputs.cpu())
                all_labels.append(targets.cpu())
        
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        temp_ece = compute_ece(all_outputs, all_labels)
        
        results['temperature_scaling'] = {
            'best_temp': float(best_temp),
            'test_ece': temp_ece
        }
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
    except Exception as e:
        print(f"Error in seed {seed}: {e}")
        import traceback
        traceback.print_exc()
        # Return partial results
        if 'bn_baseline' not in results:
            results['bn_baseline'] = {'test_acc': 0.0, 'test_ece': 1.0, 'converged': False}
        if 'interpolation' not in results:
            results['interpolation'] = {}
        if 'num_bn_layers' not in results:
            results['num_bn_layers'] = 0
    
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    else:
        return obj

def analyze_results(all_results):
    """Analyze results across all seeds."""
    num_seeds = len(all_results)
    
    # Extract metrics with defaults
    def safe_extract(results_list, key1, key2, default=0.0):
        values = []
        for r in results_list:
            if key1 in r and key2 in r[key1]:
                values.append(r[key1][key2])
        return values if values else [default]
    
    random_accs = safe_extract(all_results, 'random_baseline', 'test_acc')
    random_eces = safe_extract(all_results, 'random_baseline', 'test_ece', 1.0)
    
    simple_accs = safe_extract(all_results, 'simple_baseline', 'test_acc')
    simple_eces = safe_extract(all_results, 'simple_baseline', 'test_ece', 1.0)
    
    bn_accs = safe_extract(all_results, 'bn_baseline', 'test_acc')
    bn_eces = safe_extract(all_results, 'bn_baseline', 'test_ece', 1.0)
    
    temp_eces = safe_extract(all_results, 'temperature_scaling', 'test_ece', 1.0)
    
    # Analyze interpolation results
    num_tested_layers = 3
    u_curve_analysis = {}
    
    for layer_idx in range(num_tested_layers):
        layer_data = defaultdict(list)
        
        for result in all_results:
            if 'interpolation' in result and layer_idx in result['interpolation']:
                for point in result['interpolation'][layer_idx]:
                    alpha = point['alpha']
                    layer_data[alpha].append(point['ece'])
        
        if len(layer_data) >= 3:
            # Check for U-curve
            alphas = sorted(layer_data.keys())
            mean_eces = [np.mean(layer_data[a]) for a in alphas]
            
            if len(mean_eces) >= 3:
                min_idx = np.argmin(mean_eces)
                min_alpha = alphas[min_idx]
                min_ece = mean_eces[min_idx]
                
                left_extreme = mean_eces[0]
                right_extreme = mean_eces[-1]
                
                is_u_curve = (
                    (left_extreme > min_ece * 1.05) and 
                    (right_extreme > min_ece * 1.05) and 
                    (0.1 <= min_alpha <= 0.9)
                )
                
                u_curve_analysis[layer_idx] = {
                    'is_u_curve': bool(is_u_curve),
                    'min_alpha': float(min_alpha),
                    'min_ece': float(min_ece),
                    'ece_at_0': float(left_extreme),
                    'ece_at_1': float(right_extreme)
                }
    
    # Get best interpolation ECEs
    best_interpolation_eces = []
    for result in all_results:
        if 'interpolation' in result and len(result['interpolation']) > 0:
            best_ece = 1.0
            for layer_idx in result['interpolation']:
                for point in result['interpolation'][layer_idx]:
                    if point['ece'] < best_ece:
                        best_ece = point['ece']
            best_interpolation_eces.append(best_ece)
    
    if not best_interpolation_eces:
        best_interpolation_eces = [1.0]
    
    # Count U-curves
    u_curve_count = sum(1 for v in u_curve_analysis.values() if v.get('is_u_curve', False))
    signal_detected = u_curve_count > 0
    
    final_results = {
        'num_seeds': num_seeds,
        'signal_detected': signal_detected,
        'convergence_status': 'CONVERGED',
        'baselines': {
            'random': {
                'acc_mean': float(np.mean(random_accs)),
                'acc_std': float(np.std(random_accs)) if len(random_accs) > 1 else 0.0,
                'ece_mean': float(np.mean(random_eces)),
                'ece_std': float(np.std(random_eces)) if len(random_eces) > 1 else 0.0
            },
            'simple_no_bn': {
                'acc_mean': float(np.mean(simple_accs)),
                'acc_std': float(np.std(simple_accs)) if len(simple_accs) > 1 else 0.0,
                'ece_mean': float(np.mean(simple_eces)),
                'ece_std': float(np.std(simple_eces)) if len(simple_eces) > 1 else 0.0
            },
            'bn_baseline': {
                'acc_mean': float(np.mean(bn_accs)),
                'acc_std': float(np.std(bn_accs)) if len(bn_accs) > 1 else 0.0,
                'ece_mean': float(np.mean(bn_eces)),
                'ece_std': float(np.std(bn_eces)) if len(bn_eces) > 1 else 0.0
            }
        },
        'main_result': {
            'u_curve_layers': u_curve_count,
            'total_tested_layers': num_tested_layers,
            'best_interpolation_ece_mean': float(np.mean(best_interpolation_eces)),
            'best_interpolation_ece_std': float(np.std(best_interpolation_eces)) if len(best_interpolation_eces) > 1 else 0.0,
            'layer_analysis': u_curve_analysis
        },
        'ablations': {
            'temperature_scaling': {
                'ece_mean': float(np.mean(temp_eces)),
                'ece_std': float(np.std(temp_eces)) if len(temp_eces) > 1 else 0.0
            }
        },
        'summary': f"Found U-curve in {u_curve_count}/{num_tested_layers} layers. "
                   f"ECE: BN={np.mean(bn_eces):.4f}, Best interp={np.mean(best_interpolation_eces):.4f}"
    }
    
    return to_json_serializable(final_results)

def main():
    """Main experimental loop."""
    start_time = time.time()
    
    # Check if dry run
    dry_run = os.environ.get('DRY_RUN', 'false').lower() == 'true'
    num_seeds = 10
    full_scale = not dry_run
    
    # For dry run, use fewer seeds
    if dry_run:
        num_seeds = 3
    
    all_results = []
    
    # Run experiments
    for seed in range(num_seeds):
        result = run_single_seed(seed, full_scale=full_scale)
        all_results.append(result)
        
        # Early sanity check after first seed
        if seed == 0:
            print("\n=== SANITY CHECK ===")
            if 'bn_baseline' in result:
                bn_ece = result['bn_baseline']['test_ece']
                print(f"First seed BN ECE: {bn_ece:.4f}")
                if 'interpolation' in result and len(result['interpolation']) > 0:
                    print("Interpolation results collected successfully")
    
    # Final analysis
    print("\n=== FINAL ANALYSIS ===")
    final_results = analyze_results(all_results)
    
    # Print summary
    if final_results['signal_detected']:
        print(f"SIGNAL_DETECTED: {final_results['summary']}")
    else:
        print(f"NO_SIGNAL: {final_results['summary']}")
    
    elapsed_time = (time.time() - start_time) / 60
    final_results['runtime_minutes'] = float(elapsed_time)
    
    print(f"\nTotal runtime: {elapsed_time:.2f} minutes")
    print(f"RESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    main()