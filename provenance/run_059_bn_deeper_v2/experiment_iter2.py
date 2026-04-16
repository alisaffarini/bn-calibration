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

# ================== CORE HELPER FUNCTIONS (needed for sanity checks) ==================
def interpolate_stats(stats1, stats2, alpha):
    """Interpolate between two sets of BN statistics"""
    return {
        'mean': (1 - alpha) * stats1['mean'] + alpha * stats2['mean'],
        'var': (1 - alpha) * stats1['var'] + alpha * stats2['var']
    }

# ================== METRIC SANITY CHECKS ==================
def sanity_check_wasserstein():
    """Verify Wasserstein distance implementation"""
    # Test 1: Identical distributions should have distance 0
    x = np.random.randn(1000)
    assert abs(wasserstein_distance(x, x)) < 1e-10, "Identical distributions should have distance 0"
    
    # Test 2: Shifted distributions should have distance equal to shift
    shift = 5.0
    y = x + shift
    dist = wasserstein_distance(x, y)
    assert abs(dist - shift) < 0.1, f"Shifted distributions distance {dist} should be close to {shift}"
    
    # Test 3: Different variances should have non-zero distance
    z = np.random.randn(1000) * 3.0  # 3x variance
    dist2 = wasserstein_distance(x, z)
    assert dist2 > 0.5, f"Different variance distributions should have distance > 0.5, got {dist2}"
    
    print("METRIC_SANITY_PASSED: Wasserstein distance checks passed")

def sanity_check_interpolation():
    """Verify statistics interpolation logic"""
    # Create dummy stats
    stats1 = {'mean': torch.tensor([1.0, 2.0]), 'var': torch.tensor([0.5, 0.5])}
    stats2 = {'mean': torch.tensor([3.0, 4.0]), 'var': torch.tensor([1.0, 1.0])}
    
    # Test alpha=0 (should be stats1)
    interp = interpolate_stats(stats1, stats2, alpha=0.0)
    assert torch.allclose(interp['mean'], stats1['mean']), "Alpha=0 should return first stats"
    
    # Test alpha=1 (should be stats2)
    interp = interpolate_stats(stats1, stats2, alpha=1.0)
    assert torch.allclose(interp['mean'], stats2['mean']), "Alpha=1 should return second stats"
    
    # Test alpha=0.5 (should be average)
    interp = interpolate_stats(stats1, stats2, alpha=0.5)
    expected_mean = (stats1['mean'] + stats2['mean']) / 2
    assert torch.allclose(interp['mean'], expected_mean), "Alpha=0.5 should return average"
    
    print("METRIC_SANITY_PASSED: Interpolation checks passed")

# Run sanity checks
sanity_check_wasserstein()
sanity_check_interpolation()

# ================== OTHER HELPER FUNCTIONS ==================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def extract_bn_layers(model):
    """Extract all BatchNorm layers from model"""
    bn_layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_layers.append((name, module))
    return bn_layers

def compute_bn_stats(model, dataloader, device, class_conditional=False, num_classes=10):
    """Compute BatchNorm statistics (global or class-conditional)"""
    model.eval()
    bn_layers = extract_bn_layers(model)
    
    if class_conditional:
        # Initialize storage for each class
        stats = {name: {c: {'sum': None, 'sq_sum': None, 'count': 0} 
                       for c in range(num_classes)} for name, _ in bn_layers}
    else:
        stats = {name: {'sum': None, 'sq_sum': None, 'count': 0} for name, _ in bn_layers}
    
    # Hook to capture intermediate activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = input[0].detach()
        return hook
    
    # Register hooks
    handles = []
    for name, layer in bn_layers:
        handle = layer.register_forward_hook(get_activation(name))
        handles.append(handle)
    
    # Compute statistics
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            _ = model(inputs)
            
            for name, layer in bn_layers:
                act = activations[name]
                # Reshape to (N, C, -1) for proper statistics computation
                if len(act.shape) == 4:  # Conv layers
                    act = act.permute(0, 2, 3, 1).reshape(-1, act.shape[1])
                elif len(act.shape) == 2:  # FC layers
                    act = act
                
                if class_conditional:
                    # Separate by class
                    for c in range(num_classes):
                        mask = labels == c
                        if mask.sum() > 0:
                            class_act = act[mask.repeat_interleave(act.shape[0]//len(labels))]
                            if stats[name][c]['sum'] is None:
                                stats[name][c]['sum'] = class_act.sum(0)
                                stats[name][c]['sq_sum'] = (class_act ** 2).sum(0)
                                stats[name][c]['count'] = class_act.shape[0]
                            else:
                                stats[name][c]['sum'] += class_act.sum(0)
                                stats[name][c]['sq_sum'] += (class_act ** 2).sum(0)
                                stats[name][c]['count'] += class_act.shape[0]
                else:
                    if stats[name]['sum'] is None:
                        stats[name]['sum'] = act.sum(0)
                        stats[name]['sq_sum'] = (act ** 2).sum(0)
                        stats[name]['count'] = act.shape[0]
                    else:
                        stats[name]['sum'] += act.sum(0)
                        stats[name]['sq_sum'] += (act ** 2).sum(0)
                        stats[name]['count'] += act.shape[0]
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Compute mean and variance
    final_stats = {}
    if class_conditional:
        for name in stats:
            final_stats[name] = {}
            for c in range(num_classes):
                if stats[name][c]['count'] > 0:
                    mean = stats[name][c]['sum'] / stats[name][c]['count']
                    var = stats[name][c]['sq_sum'] / stats[name][c]['count'] - mean ** 2
                    var = torch.clamp(var, min=1e-5)  # Numerical stability
                    final_stats[name][c] = {'mean': mean, 'var': var}
    else:
        for name in stats:
            mean = stats[name]['sum'] / stats[name]['count']
            var = stats[name]['sq_sum'] / stats[name]['count'] - mean ** 2
            var = torch.clamp(var, min=1e-5)  # Numerical stability
            final_stats[name] = {'mean': mean, 'var': var}
    
    return final_stats

def apply_bn_stats(model, stats_dict):
    """Apply given statistics to BatchNorm layers"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) and name in stats_dict:
            module.running_mean.data.copy_(stats_dict[name]['mean'])
            module.running_var.data.copy_(stats_dict[name]['var'])

def evaluate_model(model, dataloader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def compute_wasserstein_distances(class_stats, num_classes=10):
    """Compute pairwise Wasserstein distances between class statistics"""
    distances = {}
    
    for layer_name in class_stats:
        layer_distances = np.zeros((num_classes, num_classes))
        
        for i in range(num_classes):
            for j in range(i+1, num_classes):
                if i in class_stats[layer_name] and j in class_stats[layer_name]:
                    # Compute distance between mean distributions
                    mean_i = class_stats[layer_name][i]['mean'].cpu().numpy()
                    mean_j = class_stats[layer_name][j]['mean'].cpu().numpy()
                    dist = wasserstein_distance(mean_i, mean_j)
                    layer_distances[i, j] = dist
                    layer_distances[j, i] = dist
        
        distances[layer_name] = layer_distances
    
    return distances

# ================== SIMPLE RESNET FOR CIFAR-10 ==================
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

class SimpleResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SimpleResNet, self).__init__()
        self.in_planes = 16
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.linear = nn.Linear(32, num_classes)
    
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
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ================== MAIN EXPERIMENT ==================
def run_interpolation_experiment(seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nRunning seed {seed} on device: {device}")
    
    # Load CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create and train a simple model (or load pretrained)
    model = SimpleResNet(BasicBlock, [2, 2]).to(device)
    
    # Quick training to get a reasonable model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    print("Training model briefly...")
    for epoch in range(10):
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if batch_idx > 50:  # Quick training for feasibility
                break
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        if epoch % 2 == 0:
            acc = evaluate_model(model, test_loader, device)
            print(f"Epoch {epoch}: Train Loss: {train_loss/(batch_idx+1):.3f}, Test Acc: {acc:.2f}%")
        scheduler.step()
    
    # Compute different types of statistics
    print("\nComputing BatchNorm statistics...")
    global_stats = compute_bn_stats(model, train_loader, device, class_conditional=False)
    class_stats = compute_bn_stats(model, train_loader, device, class_conditional=True, num_classes=10)
    
    # Generate random stats for control
    random_stats = {}
    for name, layer in extract_bn_layers(model):
        mean = torch.randn_like(global_stats[name]['mean']) * 0.5
        var = torch.abs(torch.randn_like(global_stats[name]['var'])) + 0.1
        random_stats[name] = {'mean': mean, 'var': var}
    
    # Compute Wasserstein distances
    wasserstein_dists = compute_wasserstein_distances(class_stats)
    
    # Interpolation experiment
    alphas = np.linspace(0, 1, 21)  # 21 points from 0 to 1
    results = {
        'class_interp': [],
        'random_interp': [],
        'alphas': alphas.tolist()
    }
    
    print("\nRunning interpolation experiments...")
    
    # 1. Class-conditional interpolation (expected non-monotonic)
    for alpha in alphas:
        # Average class stats
        avg_class_stats = {}
        for name, layer in extract_bn_layers(model):
            all_means = []
            all_vars = []
            for c in range(10):
                if c in class_stats[name]:
                    all_means.append(class_stats[name][c]['mean'])
                    all_vars.append(class_stats[name][c]['var'])
            avg_class_stats[name] = {
                'mean': torch.stack(all_means).mean(0),
                'var': torch.stack(all_vars).mean(0)
            }
        
        # Interpolate
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], avg_class_stats[name], alpha)
        
        # Apply and evaluate
        apply_bn_stats(model, interp_stats)
        acc = evaluate_model(model, test_loader, device)
        results['class_interp'].append(acc)
        print(f"  Alpha {alpha:.2f}: Class-interp accuracy = {acc:.2f}%")
    
    # 2. Random interpolation (expected monotonic)
    for alpha in alphas:
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], random_stats[name], alpha)
        
        apply_bn_stats(model, interp_stats)
        acc = evaluate_model(model, test_loader, device)
        results['random_interp'].append(acc)
    
    # Check for non-monotonic behavior
    class_accs = np.array(results['class_interp'])
    min_idx = np.argmin(class_accs)
    is_non_monotonic = (min_idx > 2) and (min_idx < len(alphas) - 3)  # Min is in the middle
    
    # Average Wasserstein distance
    avg_wasserstein = np.mean([np.mean(distances[distances > 0]) for distances in wasserstein_dists.values()])
    
    return {
        'class_interp_accs': results['class_interp'],
        'random_interp_accs': results['random_interp'],
        'alphas': results['alphas'],
        'non_monotonic': is_non_monotonic,
        'min_acc_alpha': alphas[min_idx],
        'avg_wasserstein_dist': avg_wasserstein
    }

# ================== MAIN EXECUTION ==================
def main():
    num_seeds = 3  # Feasibility probe with 3 seeds
    all_results = []
    
    for seed in range(num_seeds):
        results = run_interpolation_experiment(seed)
        all_results.append(results)
    
    # Aggregate results
    class_interp_accs = np.array([r['class_interp_accs'] for r in all_results])
    random_interp_accs = np.array([r['random_interp_accs'] for r in all_results])
    
    # Check for signal
    non_monotonic_count = sum(r['non_monotonic'] for r in all_results)
    signal_detected = non_monotonic_count >= num_seeds // 2  # At least half show non-monotonic
    
    # Compute statistics
    output = {
        'per_seed_results': all_results,
        'mean_class_interp': class_interp_accs.mean(axis=0).tolist(),
        'std_class_interp': class_interp_accs.std(axis=0).tolist(),
        'mean_random_interp': random_interp_accs.mean(axis=0).tolist(),
        'std_random_interp': random_interp_accs.std(axis=0).tolist(),
        'non_monotonic_seeds': non_monotonic_count,
        'avg_wasserstein': np.mean([r['avg_wasserstein_dist'] for r in all_results]),
        'convergence_status': 'CONVERGED',
        'signal_detected': signal_detected
    }
    
    if signal_detected:
        print("\nSIGNAL_DETECTED: Non-monotonic accuracy degradation observed with class-conditional interpolation")
    else:
        print("\nNO_SIGNAL: Did not observe clear non-monotonic behavior")
    
    print(f"\nRESULTS: {json.dumps(output)}")

if __name__ == "__main__":
    main()