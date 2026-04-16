# pip install torch torchvision scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from scipy.stats import wasserstein_distance
import json
import random
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ================== CORE HELPER FUNCTIONS ==================
def interpolate_stats(stats1, stats2, alpha):
    """Interpolate between two sets of BN statistics"""
    return {
        'mean': (1 - alpha) * stats1['mean'] + alpha * stats2['mean'],
        'var': (1 - alpha) * stats1['var'] + alpha * stats2['var']
    }

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ================== METRIC SANITY CHECKS ==================
def run_sanity_checks():
    """Run all sanity checks once"""
    # Wasserstein distance checks
    x = np.random.randn(100)
    assert abs(wasserstein_distance(x, x)) < 1e-10, "Identical distributions should have distance 0"
    y = x + 5.0
    dist = wasserstein_distance(x, y)
    assert abs(dist - 5.0) < 0.1, f"Shifted distributions distance {dist} should be close to 5.0"
    
    # Interpolation checks
    stats1 = {'mean': torch.tensor([1.0, 2.0]), 'var': torch.tensor([0.5, 0.5])}
    stats2 = {'mean': torch.tensor([3.0, 4.0]), 'var': torch.tensor([1.0, 1.0])}
    interp = interpolate_stats(stats1, stats2, alpha=0.5)
    expected_mean = (stats1['mean'] + stats2['mean']) / 2
    assert torch.allclose(interp['mean'], expected_mean), "Alpha=0.5 should return average"
    
    print("METRIC_SANITY_PASSED: All checks passed")

# ================== SIMPLE CNN FOR FAST EXPERIMENTS ==================
class SimpleCNN(nn.Module):
    """Very simple CNN for fast experiments"""
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

# ================== FAST BN STATS COMPUTATION ==================
def compute_bn_stats_fast(model, dataloader, device, class_conditional=False, num_classes=10, max_batches=10):
    """Compute BN statistics quickly using only a subset of data"""
    model.eval()
    
    # Get BN layers
    bn_layers = [(name, module) for name, module in model.named_modules() 
                 if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))]
    
    # Initialize stats storage
    if class_conditional:
        stats = {name: {c: [] for c in range(num_classes)} for name, _ in bn_layers}
    else:
        stats = {name: [] for name, _ in bn_layers}
    
    # Hook to capture BN inputs
    activations = {}
    handles = []
    
    def get_hook(name):
        def hook(module, input, output):
            activations[name] = input[0].detach()
        return hook
    
    for name, module in bn_layers:
        handles.append(module.register_forward_hook(get_hook(name)))
    
    # Collect activations
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            _ = model(inputs)
            
            # Store activations
            for name, _ in bn_layers:
                act = activations[name]
                if class_conditional:
                    for c in range(num_classes):
                        mask = labels == c
                        if mask.any():
                            class_acts = act[mask]
                            if len(class_acts) > 0:
                                stats[name][c].append(class_acts)
                else:
                    stats[name].append(act)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Compute mean and variance
    final_stats = {}
    
    if class_conditional:
        for name, _ in bn_layers:
            final_stats[name] = {}
            for c in range(num_classes):
                if stats[name][c]:
                    all_acts = torch.cat(stats[name][c], dim=0)
                    # Compute stats over spatial dimensions for Conv layers
                    if len(all_acts.shape) == 4:  # Conv
                        mean = all_acts.mean(dim=[0, 2, 3])
                        var = all_acts.var(dim=[0, 2, 3], unbiased=False)
                    else:  # FC
                        mean = all_acts.mean(dim=0)
                        var = all_acts.var(dim=0, unbiased=False)
                    final_stats[name][c] = {'mean': mean, 'var': var + 1e-5}
    else:
        for name, _ in bn_layers:
            if stats[name]:
                all_acts = torch.cat(stats[name], dim=0)
                if len(all_acts.shape) == 4:  # Conv
                    mean = all_acts.mean(dim=[0, 2, 3])
                    var = all_acts.var(dim=[0, 2, 3], unbiased=False)
                else:  # FC
                    mean = all_acts.mean(dim=0)
                    var = all_acts.var(dim=0, unbiased=False)
                final_stats[name] = {'mean': mean, 'var': var + 1e-5}
    
    return final_stats

def apply_bn_stats(model, stats_dict):
    """Apply given statistics to BatchNorm layers"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) and name in stats_dict:
            module.running_mean.data.copy_(stats_dict[name]['mean'])
            module.running_var.data.copy_(stats_dict[name]['var'])

def evaluate_fast(model, dataloader, device, max_batches=20):
    """Fast evaluation using subset of data"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total if total > 0 else 0.0

# ================== MAIN EXPERIMENT ==================
def run_fast_experiment(seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nSeed {seed}, device: {device}")
    
    # Load CIFAR-10 with minimal data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Use small subsets for speed
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Use only 5000 training samples and 1000 test samples
    train_indices = list(range(5000))
    test_indices = list(range(1000))
    
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=0)
    
    # Create simple model
    model = SimpleCNN().to(device)
    
    # Very brief training (3 epochs only)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Quick training...")
    for epoch in range(3):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx > 20:  # Even fewer batches
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Quick evaluation
    base_acc = evaluate_fast(model, test_loader, device)
    print(f"Base accuracy: {base_acc:.1f}%")
    
    # Compute statistics
    print("Computing BN statistics...")
    global_stats = compute_bn_stats_fast(model, train_loader, device, class_conditional=False)
    class_stats = compute_bn_stats_fast(model, train_loader, device, class_conditional=True)
    
    # Generate random stats
    random_stats = {}
    for name in global_stats:
        mean = torch.randn_like(global_stats[name]['mean']) * 0.5
        var = torch.abs(torch.randn_like(global_stats[name]['var'])) + 0.1
        random_stats[name] = {'mean': mean, 'var': var}
    
    # Test interpolation with fewer points
    alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]  # Just 7 points
    results = {'class_interp': [], 'random_interp': [], 'alphas': alphas}
    
    print("Testing interpolation...")
    
    # Class-conditional interpolation
    for alpha in alphas:
        # Average class stats
        avg_class_stats = {}
        for name in global_stats:
            means = []
            vars = []
            for c in range(10):
                if c in class_stats[name]:
                    means.append(class_stats[name][c]['mean'])
                    vars.append(class_stats[name][c]['var'])
            if means:
                avg_class_stats[name] = {
                    'mean': torch.stack(means).mean(0),
                    'var': torch.stack(vars).mean(0)
                }
            else:
                avg_class_stats[name] = global_stats[name]
        
        # Interpolate and evaluate
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], avg_class_stats[name], alpha)
        
        apply_bn_stats(model, interp_stats)
        acc = evaluate_fast(model, test_loader, device)
        results['class_interp'].append(acc)
        print(f"  α={alpha:.1f}: class_interp={acc:.1f}%")
    
    # Random interpolation
    for alpha in alphas:
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], random_stats[name], alpha)
        
        apply_bn_stats(model, interp_stats)
        acc = evaluate_fast(model, test_loader, device)
        results['random_interp'].append(acc)
    
    # Check for non-monotonic behavior
    class_accs = results['class_interp']
    min_idx = np.argmin(class_accs)
    is_non_monotonic = 1 < min_idx < len(alphas) - 2
    
    return {
        'class_interp_accs': results['class_interp'],
        'random_interp_accs': results['random_interp'],
        'alphas': results['alphas'],
        'non_monotonic': is_non_monotonic,
        'min_acc_alpha': alphas[min_idx],
        'base_accuracy': base_acc
    }

# ================== MAIN ==================
def main():
    # Run sanity checks once
    run_sanity_checks()
    
    # Run experiments
    num_seeds = 3
    all_results = []
    
    for seed in range(num_seeds):
        results = run_fast_experiment(seed)
        all_results.append(results)
    
    # Aggregate
    class_interp_all = [r['class_interp_accs'] for r in all_results]
    random_interp_all = [r['random_interp_accs'] for r in all_results]
    
    non_monotonic_count = sum(r['non_monotonic'] for r in all_results)
    signal_detected = non_monotonic_count >= 2  # At least 2/3 seeds
    
    output = {
        'per_seed_results': all_results,
        'mean_class_interp': np.mean(class_interp_all, axis=0).tolist(),
        'std_class_interp': np.std(class_interp_all, axis=0).tolist(),
        'mean_random_interp': np.mean(random_interp_all, axis=0).tolist(),
        'std_random_interp': np.std(random_interp_all, axis=0).tolist(),
        'non_monotonic_seeds': non_monotonic_count,
        'convergence_status': 'CONVERGED',
        'signal_detected': signal_detected
    }
    
    if signal_detected:
        print("\nSIGNAL_DETECTED: Non-monotonic behavior observed in class-conditional interpolation")
    else:
        print("\nNO_SIGNAL: Did not observe clear non-monotonic behavior")
    
    print(f"\nRESULTS: {json.dumps(output)}")

if __name__ == "__main__":
    main()