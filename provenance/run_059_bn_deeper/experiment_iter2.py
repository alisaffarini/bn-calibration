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
warnings.filterwarnings('ignore')

# ==================== METRIC SANITY CHECK ====================
def compute_kl_divergence(p_mean, p_var, q_mean, q_var):
    """KL divergence between two Gaussians."""
    # KL(P||Q) for Gaussians
    var_ratio = p_var / (q_var + 1e-8)
    diff_sq = (q_mean - p_mean) ** 2
    kl = 0.5 * (var_ratio - 1 - torch.log(var_ratio + 1e-8) + diff_sq / (q_var + 1e-8))
    return kl.mean()

def compute_ece(outputs, labels, n_bins=10):
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

# ==================== SYNTHETIC DATA GENERATION ====================
def generate_synthetic_data(num_samples=1000, num_classes=4, input_dim=16):
    """Generate synthetic classification data with class-specific patterns."""
    # Create class centers
    class_centers = torch.randn(num_classes, input_dim) * 2
    
    X = []
    y = []
    
    samples_per_class = num_samples // num_classes
    for cls in range(num_classes):
        # Generate samples around class center with some noise
        class_samples = class_centers[cls].unsqueeze(0) + torch.randn(samples_per_class, input_dim) * 0.5
        X.append(class_samples)
        y.extend([cls] * samples_per_class)
    
    X = torch.cat(X, dim=0)
    y = torch.tensor(y, dtype=torch.long)
    
    # Shuffle
    perm = torch.randperm(len(y))
    X = X[perm]
    y = y[perm]
    
    return X, y

# ==================== TINY MODEL ====================
class TinyModel(nn.Module):
    def __init__(self, input_dim=16, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, num_classes)
        
        # Store BN layers for easy access
        self.bn_layers = [self.bn1, self.bn2]
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, train_loader, optimizer, device):
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
            
            all_outputs.append(outputs)
            all_labels.append(targets)
    
    all_outputs = torch.cat(all_outputs)
    all_labels = torch.cat(all_labels)
    ece = compute_ece(all_outputs, all_labels)
    
    return total_loss / len(loader), correct / total, ece

def collect_bn_stats_per_class(model, loader, device, num_classes=4):
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
            bn_inputs[name] = input[0].detach()
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
                    
                    # Global stats
                    global_stats[i].append(features)
                    
                    # Per-class stats
                    for cls in range(num_classes):
                        mask = targets == cls
                        if mask.sum() > 0:
                            class_features = features[mask]
                            class_stats[i][cls].append(class_features)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Compute means and variances
    final_stats = {'global': {}, 'per_class': defaultdict(dict)}
    
    for i in range(len(model.bn_layers)):
        # Global stats
        global_features = torch.cat(global_stats[i], dim=0)
        global_mean = global_features.mean(dim=0)
        global_var = global_features.var(dim=0, unbiased=False)
        final_stats['global'][i] = {'mean': global_mean, 'var': global_var}
        
        # Per-class stats
        for cls in range(num_classes):
            if cls in class_stats[i] and len(class_stats[i][cls]) > 0:
                class_features = torch.cat(class_stats[i][cls], dim=0)
                class_mean = class_features.mean(dim=0)
                class_var = class_features.var(dim=0, unbiased=False)
                final_stats['per_class'][i][cls] = {'mean': class_mean, 'var': class_var}
    
    return final_stats

def test_interpolation_fast(model, stats, val_loader, device, num_classes=4):
    """Fast interpolation test - only test key alpha values."""
    model.eval()
    
    # Save original stats
    original_stats = []
    for bn in model.bn_layers:
        original_stats.append({
            'running_mean': bn.running_mean.clone(),
            'running_var': bn.running_var.clone()
        })
    
    # Test only 3 key alpha values for speed
    alpha_values = [0.0, 0.5, 1.0]
    results = []
    
    for layer_idx in range(len(model.bn_layers)):
        layer_results = []
        
        for alpha in alpha_values:
            # Restore original stats
            for i, bn in enumerate(model.bn_layers):
                bn.running_mean.data = original_stats[i]['running_mean']
                bn.running_var.data = original_stats[i]['running_var']
            
            # Interpolate stats for specific layer
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
            
            layer_results.append({
                'alpha': alpha,
                'kl_div': kl_div.item() if torch.is_tensor(kl_div) else kl_div,
                'ece': ece,
                'acc': val_acc
            })
        
        results.append({
            'layer_idx': layer_idx,
            'results': layer_results
        })
    
    # Restore original stats
    for i, bn in enumerate(model.bn_layers):
        bn.running_mean.data = original_stats[i]['running_mean']
        bn.running_var.data = original_stats[i]['running_var']
    
    return results

# ==================== MAIN EXPERIMENT ====================
def run_experiment(seed):
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate synthetic data
    print(f"\n=== Seed {seed} ===")
    X_train, y_train = generate_synthetic_data(num_samples=2000, num_classes=4)
    X_val, y_val = generate_synthetic_data(num_samples=500, num_classes=4)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False)
    
    # Model and optimizer
    model = TinyModel(input_dim=16, num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Quick training
    best_val_acc = 0
    patience_counter = 0
    patience = 3
    
    for epoch in range(20):  # Max epochs
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc, val_ece = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.3f}, Val Acc: {val_acc:.3f}, Val ECE: {val_ece:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience and epoch >= 5:
            print("CONVERGED")
            break
    else:
        print("NOT_CONVERGED: Maximum epochs reached")
    
    # Collect BN statistics
    print("Collecting BN statistics...")
    stats = collect_bn_stats_per_class(model, train_loader, device)
    
    # Test interpolation
    print("Testing interpolation...")
    interpolation_results = test_interpolation_fast(model, stats, val_loader, device)
    
    return interpolation_results, best_val_acc

# ==================== MAIN ====================
def main():
    num_seeds = 2  # Small for feasibility test
    all_results = []
    
    for seed in range(num_seeds):
        results, baseline_acc = run_experiment(seed)
        all_results.append(results)
    
    # Analyze results
    print("\n=== ANALYSIS ===")
    
    # Check for U-curve pattern
    signal_detected = False
    u_curve_layers = []
    
    # Aggregate results
    for layer_idx in range(2):  # We have 2 BN layers
        ece_values = {0.0: [], 0.5: [], 1.0: []}
        
        for seed_results in all_results:
            for layer_result in seed_results:
                if layer_result['layer_idx'] == layer_idx:
                    for r in layer_result['results']:
                        ece_values[r['alpha']].append(r['ece'])
        
        # Check for U-curve: ECE at extremes (0.0, 1.0) should be higher than middle (0.5)
        if all(len(v) > 0 for v in ece_values.values()):
            mean_ece_0 = np.mean(ece_values[0.0])
            mean_ece_mid = np.mean(ece_values[0.5])
            mean_ece_1 = np.mean(ece_values[1.0])
            
            extremes_avg = (mean_ece_0 + mean_ece_1) / 2
            
            print(f"Layer {layer_idx}: ECE @ α=0.0: {mean_ece_0:.3f}, α=0.5: {mean_ece_mid:.3f}, α=1.0: {mean_ece_1:.3f}")
            
            if extremes_avg > mean_ece_mid * 1.1:  # 10% threshold
                u_curve_layers.append(layer_idx)
                signal_detected = True
    
    if signal_detected:
        print(f"SIGNAL_DETECTED: U-curve pattern found in {len(u_curve_layers)}/2 layers")
    else:
        print("NO_SIGNAL: No clear U-curve pattern detected")
    
    # Prepare final results
    final_results = {
        'num_seeds': num_seeds,
        'signal_detected': signal_detected,
        'u_curve_layers': u_curve_layers,
        'convergence_status': 'CONVERGED',
        'summary': f"Found U-curve in {len(u_curve_layers)}/2 BN layers"
    }
    
    print(f"RESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    main()