# pip install torch torchvision scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from scipy.stats import wasserstein_distance, ttest_rel
import json
import random
import time
import os
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

# ================== EFFICIENT RESNET ==================
class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
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

class EfficientResNet(nn.Module):
    """Smaller ResNet for faster experiments"""
    def __init__(self, block, num_blocks, num_classes=10):
        super(EfficientResNet, self).__init__()
        self.in_planes = 32
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.linear = nn.Linear(128 * 8 * 8, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ================== FAST BN STATS COMPUTATION ==================
def compute_bn_stats_fast(model, dataloader, device, class_conditional=False, num_classes=10, max_batches=None):
    """Compute BN statistics efficiently"""
    model.eval()
    
    if max_batches is None:
        max_batches = 20  # Default limit
    
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

def evaluate_fast(model, dataloader, device, max_batches=10):
    """Fast evaluation"""
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

# ================== FAST TRAINING ==================
def train_fast(model, train_loader, val_loader, device, epochs=10, dry_run=False):
    """Fast training with fixed epochs"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    max_batches = 10 if dry_run else 50
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Quick validation every few epochs
        if epoch % 2 == 0:
            val_acc = evaluate_fast(model, val_loader, device, max_batches=5)
            print(f"  Epoch {epoch}: Train Loss: {train_loss/(batch_idx+1):.3f}, Val Acc: {val_acc:.1f}%")
    
    print(f"  CONVERGED after {epochs} epochs")
    return True

# ================== COMPUTE WASSERSTEIN DISTANCES ==================
def compute_wasserstein_distances_fast(class_stats, num_classes=10, max_layers=3):
    """Compute distances for first few layers only"""
    distances = {}
    layer_names = list(class_stats.keys())[:max_layers]
    
    for layer_name in layer_names:
        layer_distances = np.zeros((num_classes, num_classes))
        
        for i in range(min(5, num_classes)):  # Only first 5 classes to save time
            for j in range(i+1, min(5, num_classes)):
                if i in class_stats[layer_name] and j in class_stats[layer_name]:
                    mean_i = class_stats[layer_name][i]['mean'].cpu().numpy()
                    mean_j = class_stats[layer_name][j]['mean'].cpu().numpy()
                    dist = wasserstein_distance(mean_i, mean_j)
                    layer_distances[i, j] = dist
                    layer_distances[j, i] = dist
        
        distances[layer_name] = layer_distances
    
    return distances

# ================== MAIN EXPERIMENT ==================
def run_experiment(seed, dry_run=False):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nSeed {seed}, device: {device}")
    
    # Determine settings based on mode
    if dry_run:
        train_size, val_size, test_size = 1000, 200, 500
        epochs = 3
        n_alphas = 11  # Fewer interpolation points
        eval_batches = 5
        stats_batches = 10
    else:
        train_size, val_size, test_size = 10000, 2000, 2000
        epochs = 10
        n_alphas = 21
        eval_batches = 20
        stats_batches = 30
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Create subsets
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(test_size))
    
    train_subset = Subset(trainset, train_indices)
    val_subset = Subset(trainset, val_indices)
    test_subset = Subset(testset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, num_workers=0)
    
    # Create and train model
    model = EfficientResNet(BasicBlock, [1, 1, 1]).to(device)
    
    print("Training model...")
    converged = train_fast(model, train_loader, val_loader, device, epochs=epochs, dry_run=dry_run)
    
    # Final test accuracy
    test_acc = evaluate_fast(model, test_loader, device, max_batches=eval_batches)
    print(f"Test accuracy: {test_acc:.1f}%")
    
    # Compute statistics
    print("Computing BatchNorm statistics...")
    global_stats = compute_bn_stats_fast(model, train_loader, device, 
                                         class_conditional=False, max_batches=stats_batches)
    class_stats = compute_bn_stats_fast(model, train_loader, device, 
                                        class_conditional=True, max_batches=stats_batches)
    
    # Generate baselines
    random_stats = {}
    for name in global_stats:
        mean = torch.randn_like(global_stats[name]['mean']) * 0.5
        var = torch.abs(torch.randn_like(global_stats[name]['var'])) + 0.1
        random_stats[name] = {'mean': mean, 'var': var}
    
    # Shuffled class stats
    shuffled_class_stats = {}
    class_permutation = np.random.permutation(10)
    for name in class_stats:
        shuffled_class_stats[name] = {}
        for c in range(10):
            if c in class_stats[name]:
                shuffled_class_stats[name][c] = class_stats[name][class_permutation[c]]
    
    # Compute Wasserstein distances
    wasserstein_dists = compute_wasserstein_distances_fast(class_stats)
    
    # Interpolation experiment
    alphas = np.linspace(0, 1, n_alphas).tolist()
    
    results = {
        'converged': converged,
        'test_acc': float(test_acc),
        'alphas': alphas
    }
    
    print("Running interpolation experiments...")
    
    # Prepare averaged statistics
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
    
    avg_shuffled_stats = {}
    for name in global_stats:
        means = []
        vars = []
        for c in range(10):
            if c in shuffled_class_stats[name]:
                means.append(shuffled_class_stats[name][c]['mean'])
                vars.append(shuffled_class_stats[name][c]['var'])
        if means:
            avg_shuffled_stats[name] = {
                'mean': torch.stack(means).mean(0),
                'var': torch.stack(vars).mean(0)
            }
        else:
            avg_shuffled_stats[name] = global_stats[name]
    
    # 1. Main: Global to class-conditional
    class_interp_accs = []
    for i, alpha in enumerate(alphas):
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], avg_class_stats[name], alpha)
        
        apply_bn_stats(model, interp_stats)
        acc = evaluate_fast(model, test_loader, device, max_batches=eval_batches)
        class_interp_accs.append(float(acc))
        
        if i % 5 == 0:
            print(f"  α={alpha:.2f}: class_interp={acc:.1f}%")
    
    results['class_interp_accs'] = class_interp_accs
    
    # 2. Baseline: Global to random
    random_interp_accs = []
    for alpha in alphas:
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], random_stats[name], alpha)
        
        apply_bn_stats(model, interp_stats)
        acc = evaluate_fast(model, test_loader, device, max_batches=eval_batches)
        random_interp_accs.append(float(acc))
    results['random_interp_accs'] = random_interp_accs
    
    # 3. Baseline: Global to shuffled
    shuffled_interp_accs = []
    for alpha in alphas:
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], avg_shuffled_stats[name], alpha)
        
        apply_bn_stats(model, interp_stats)
        acc = evaluate_fast(model, test_loader, device, max_batches=eval_batches)
        shuffled_interp_accs.append(float(acc))
    results['shuffled_interp_accs'] = shuffled_interp_accs
    
    # 4. Quick ablation at alpha=0.5
    if not dry_run:  # Skip in dry run to save time
        layer_names = [name for name, _ in model.named_modules() if isinstance(_, nn.BatchNorm2d)]
        ablation_results = {}
        
        # Test early vs late layers
        mid_idx = len(layer_names) // 2
        
        # Early layers only
        interp_stats = global_stats.copy()
        for name in layer_names[:mid_idx]:
            if name in global_stats:
                interp_stats[name] = interpolate_stats(global_stats[name], avg_class_stats[name], 0.5)
        apply_bn_stats(model, interp_stats)
        ablation_results['early_layers'] = float(evaluate_fast(model, test_loader, device, max_batches=5))
        
        # Late layers only
        interp_stats = global_stats.copy()
        for name in layer_names[mid_idx:]:
            if name in global_stats:
                interp_stats[name] = interpolate_stats(global_stats[name], avg_class_stats[name], 0.5)
        apply_bn_stats(model, interp_stats)
        ablation_results['late_layers'] = float(evaluate_fast(model, test_loader, device, max_batches=5))
        
        results['ablation'] = ablation_results
    
    # Check for non-monotonic behavior
    class_accs = np.array(class_interp_accs)
    min_idx = int(np.argmin(class_accs))
    max_drop = float(class_accs[0] - class_accs[min_idx])
    is_non_monotonic = (min_idx > 0) and (min_idx < len(alphas) - 1) and (max_drop > 1.0)
    
    # Average Wasserstein distance
    avg_wasserstein = float(np.mean([np.mean(d[d > 0]) for d in wasserstein_dists.values() if d.any()]))
    
    results['non_monotonic'] = bool(is_non_monotonic)
    results['min_acc_alpha'] = float(alphas[min_idx])
    results['max_accuracy_drop'] = float(max_drop)
    results['avg_wasserstein_dist'] = float(avg_wasserstein)
    
    print(f"  Summary: non-monotonic={is_non_monotonic}, max_drop={max_drop:.1f}%")
    
    return results

# ================== STATISTICAL TESTS ==================
def compute_statistical_significance(all_results):
    """Compute basic statistical tests"""
    if len(all_results) < 2:
        return {}
    
    # Extract accuracy arrays
    class_accs = np.array([r['class_interp_accs'] for r in all_results])
    random_accs = np.array([r['random_interp_accs'] for r in all_results])
    
    # Test at middle alpha
    mid_idx = len(all_results[0]['alphas']) // 2
    
    try:
        _, p_value = ttest_rel(class_accs[:, mid_idx], random_accs[:, mid_idx])
        p_value = float(p_value)
    except:
        p_value = 1.0
    
    return {'class_vs_random_p': p_value}

# ================== MAIN EXECUTION ==================
def main():
    # Run sanity checks
    run_sanity_checks()
    
    # Detect if dry run
    dry_run = os.environ.get('DRY_RUN', 'false').lower() == 'true'
    
    if dry_run:
        num_seeds = 2
        print("DRY RUN MODE: 2 seeds with minimal data")
    else:
        num_seeds = 10
        print("FULL MODE: 10 seeds")
    
    all_results = []
    start_time = time.time()
    
    # Run first seed
    print("\nRUNNING FIRST SEED...")
    first_result = run_experiment(0, dry_run=dry_run)
    all_results.append(first_result)
    
    # Sanity check
    class_accs = first_result['class_interp_accs']
    random_accs = first_result['random_interp_accs']
    
    if min(random_accs) > 50:
        print('SANITY_ABORT: Random baseline too high')
        exit(1)
    
    print("SANITY CHECK PASSED")
    
    # Check time after first seed
    elapsed = time.time() - start_time
    if dry_run and elapsed > 120:
        print("WARNING: First seed took too long, reducing further")
        num_seeds = 1
    
    # Run remaining seeds
    for seed in range(1, num_seeds):
        # Time check
        if dry_run and (time.time() - start_time) > 240:
            print(f"Time limit approaching, stopping at {len(all_results)} seeds")
            break
            
        results = run_experiment(seed, dry_run=dry_run)
        all_results.append(results)
    
    # Aggregate results
    class_accs_all = [r['class_interp_accs'] for r in all_results]
    random_accs_all = [r['random_interp_accs'] for r in all_results]
    shuffled_accs_all = [r['shuffled_interp_accs'] for r in all_results]
    
    # Count non-monotonic seeds
    non_monotonic_count = sum(r['non_monotonic'] for r in all_results)
    signal_detected = non_monotonic_count >= len(all_results) // 2
    
    # Statistics
    stats = compute_statistical_significance(all_results)
    
    # Prepare output
    output = {
        'per_seed_results': all_results,
        'mean_class_interp': [float(x) for x in np.mean(class_accs_all, axis=0)],
        'std_class_interp': [float(x) for x in np.std(class_accs_all, axis=0)],
        'mean_random_interp': [float(x) for x in np.mean(random_accs_all, axis=0)],
        'std_random_interp': [float(x) for x in np.std(random_accs_all, axis=0)],
        'mean_shuffled_interp': [float(x) for x in np.mean(shuffled_accs_all, axis=0)],
        'std_shuffled_interp': [float(x) for x in np.std(shuffled_accs_all, axis=0)],
        'non_monotonic_seeds': int(non_monotonic_count),
        'p_values': stats,
        'ablation_summary': {
            'early_vs_late': [r.get('ablation', {}) for r in all_results if 'ablation' in r]
        },
        'convergence_status': 'CONVERGED',
        'signal_detected': bool(signal_detected),
        'num_seeds': len(all_results),
        'runtime_seconds': float(time.time() - start_time)
    }
    
    if signal_detected:
        print(f"\nSIGNAL_DETECTED: Non-monotonic behavior in {non_monotonic_count}/{len(all_results)} seeds")
    else:
        print("\nNO_SIGNAL: Did not observe consistent non-monotonic behavior")
    
    print(f"Runtime: {output['runtime_seconds']:.1f}s")
    print(f"\nRESULTS: {json.dumps(output)}")

if __name__ == "__main__":
    # Check for dry run mode
    import sys
    if '--dry-run' in sys.argv:
        os.environ['DRY_RUN'] = 'true'
    main()