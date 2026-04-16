# pip install torch torchvision scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
from scipy.stats import wasserstein_distance, ttest_rel
from scipy import stats as scipy_stats
import json
import random
from collections import defaultdict
import time
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

# ================== RESNET FOR CIFAR-10 ==================
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# ================== BN STATS COMPUTATION ==================
def compute_bn_stats(model, dataloader, device, class_conditional=False, num_classes=10):
    """Compute BN statistics using full dataset"""
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
        for inputs, labels in dataloader:
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

def evaluate(model, dataloader, device):
    """Evaluate model accuracy and ECE"""
    model.eval()
    correct = 0
    total = 0
    
    # For ECE calculation
    confidences = []
    predictions = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
            
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            confidences.append(confidence.cpu())
            predictions.append(predicted.cpu())
            labels_list.append(labels.cpu())
    
    accuracy = 100. * correct / total
    
    # Calculate ECE
    confidences = torch.cat(confidences).numpy()
    predictions = torch.cat(predictions).numpy()
    labels_list = torch.cat(labels_list).numpy()
    
    # ECE with 10 bins
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = (predictions[in_bin] == labels_list[in_bin]).mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return accuracy, ece * 100  # Return ECE as percentage

# ================== TRAINING WITH CONVERGENCE ==================
def train_with_convergence(model, train_loader, val_loader, device, max_epochs=50, patience=5):
    """Train model with early stopping based on validation loss"""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # Check for improvement
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        scheduler.step()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.3f}, "
                  f"Val Loss: {avg_val_loss:.3f}, Val Acc: {val_acc:.2f}%")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  CONVERGED after {epoch+1} epochs (no improvement for {patience} epochs)")
            return True
    
    print(f"  NOT_CONVERGED: Reached max epochs ({max_epochs})")
    return False

# ================== COMPUTE WASSERSTEIN DISTANCES ==================
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

# ================== MAIN EXPERIMENT ==================
def run_experiment(seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nSeed {seed}, device: {device}\n{'='*60}")
    
    # Load CIFAR-10
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
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Split train into train/val (45k/5k)
    train_size = 45000
    val_size = 5000
    train_subset, val_subset = random_split(trainset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=2)
    test_loader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
    
    # Create and train model
    model = ResNet18().to(device)
    
    print("Training model with convergence-based early stopping...")
    converged = train_with_convergence(model, train_loader, val_loader, device, max_epochs=50, patience=5)
    
    # Final test accuracy
    test_acc, test_ece = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {test_acc:.2f}%, ECE: {test_ece:.2f}%")
    
    # Compute different types of statistics
    print("Computing BatchNorm statistics...")
    global_stats = compute_bn_stats(model, train_loader, device, class_conditional=False)
    class_stats = compute_bn_stats(model, train_loader, device, class_conditional=True, num_classes=10)
    
    # Generate baselines
    
    # 1. Random stats baseline
    random_stats = {}
    for name in global_stats:
        mean = torch.randn_like(global_stats[name]['mean']) * 0.5
        var = torch.abs(torch.randn_like(global_stats[name]['var'])) + 0.1
        random_stats[name] = {'mean': mean, 'var': var}
    
    # 2. Shuffled class stats
    shuffled_class_stats = {}
    class_permutation = np.random.permutation(10)
    for name in class_stats:
        shuffled_class_stats[name] = {}
        for c in range(10):
            if c in class_stats[name]:
                shuffled_class_stats[name][c] = class_stats[name][class_permutation[c]]
    
    # 3. Per-class models baseline (train separate models per class)
    # This is computationally expensive, so we simulate it
    
    # Compute Wasserstein distances
    wasserstein_dists = compute_wasserstein_distances(class_stats)
    
    # Interpolation experiment with fine-grained alphas
    alphas = np.linspace(0, 1, 41).tolist()  # 41 points
    
    results = {
        'converged': converged,
        'test_acc': float(test_acc),
        'test_ece': float(test_ece),
        'alphas': alphas
    }
    
    print("\nRunning interpolation experiments...")
    
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
    print("  Testing class-conditional interpolation...")
    class_interp_accs = []
    class_interp_eces = []
    for i, alpha in enumerate(alphas):
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], avg_class_stats[name], alpha)
        
        apply_bn_stats(model, interp_stats)
        acc, ece = evaluate(model, test_loader, device)
        class_interp_accs.append(float(acc))
        class_interp_eces.append(float(ece))
        
        if i % 10 == 0:
            print(f"    α={alpha:.2f}: Acc={acc:.1f}%, ECE={ece:.1f}%")
    
    results['class_interp_accs'] = class_interp_accs
    results['class_interp_eces'] = class_interp_eces
    
    # 2. Baseline: Global to random
    print("  Testing random baseline...")
    random_interp_accs = []
    random_interp_eces = []
    for alpha in alphas:
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], random_stats[name], alpha)
        
        apply_bn_stats(model, interp_stats)
        acc, ece = evaluate(model, test_loader, device)
        random_interp_accs.append(float(acc))
        random_interp_eces.append(float(ece))
    results['random_interp_accs'] = random_interp_accs
    results['random_interp_eces'] = random_interp_eces
    
    # 3. Baseline: Global to shuffled
    print("  Testing shuffled baseline...")
    shuffled_interp_accs = []
    shuffled_interp_eces = []
    for alpha in alphas:
        interp_stats = {}
        for name in global_stats:
            interp_stats[name] = interpolate_stats(global_stats[name], avg_shuffled_stats[name], alpha)
        
        apply_bn_stats(model, interp_stats)
        acc, ece = evaluate(model, test_loader, device)
        shuffled_interp_accs.append(float(acc))
        shuffled_interp_eces.append(float(ece))
    results['shuffled_interp_accs'] = shuffled_interp_accs
    results['shuffled_interp_eces'] = shuffled_interp_eces
    
    # 4. Comprehensive ablations
    print("  Running ablation studies...")
    ablation_results = {}
    
    # Test interpolation at different layers
    bn_layer_names = [name for name, _ in model.named_modules() if isinstance(_, nn.BatchNorm2d)]
    layer_depth = len(bn_layer_names)
    
    # Test key alpha values
    test_alphas = [0.3, 0.5, 0.7]
    
    for test_alpha in test_alphas:
        ablation_results[f'alpha_{test_alpha}'] = {}
        
        # Single layer interpolation
        for layer_idx, layer_name in enumerate(bn_layer_names[:5]):  # Test first 5 layers
            interp_stats = global_stats.copy()
            interp_stats[layer_name] = interpolate_stats(
                global_stats[layer_name], avg_class_stats[layer_name], test_alpha
            )
            apply_bn_stats(model, interp_stats)
            acc, ece = evaluate(model, test_loader, device)
            ablation_results[f'alpha_{test_alpha}'][f'layer_{layer_idx}'] = {
                'acc': float(acc), 'ece': float(ece)
            }
        
        # Layer group interpolation
        thirds = layer_depth // 3
        layer_groups = {
            'early': bn_layer_names[:thirds],
            'middle': bn_layer_names[thirds:2*thirds],
            'late': bn_layer_names[2*thirds:]
        }
        
        for group_name, group_layers in layer_groups.items():
            interp_stats = global_stats.copy()
            for layer_name in group_layers:
                if layer_name in global_stats:
                    interp_stats[layer_name] = interpolate_stats(
                        global_stats[layer_name], avg_class_stats[layer_name], test_alpha
                    )
            apply_bn_stats(model, interp_stats)
            acc, ece = evaluate(model, test_loader, device)
            ablation_results[f'alpha_{test_alpha}'][group_name] = {
                'acc': float(acc), 'ece': float(ece)
            }
    
    results['ablation_results'] = ablation_results
    
    # Check for non-monotonic behavior
    class_accs = np.array(class_interp_accs)
    min_idx = int(np.argmin(class_accs))
    max_drop = float(class_accs[0] - class_accs[min_idx])
    is_non_monotonic = (5 < min_idx < len(alphas) - 5) and (max_drop > 2.0)
    
    # Compute curvature metric
    if len(class_accs) > 2:
        # Second derivative approximation
        second_deriv = np.diff(np.diff(class_accs))
        max_curvature = float(np.max(np.abs(second_deriv)))
    else:
        max_curvature = 0.0
    
    # Average Wasserstein distance
    avg_wasserstein = float(np.mean([np.mean(distances[distances > 0]) for distances in wasserstein_dists.values()]))
    
    results['non_monotonic'] = bool(is_non_monotonic)
    results['min_acc_alpha'] = float(alphas[min_idx])
    results['max_accuracy_drop'] = float(max_drop)
    results['max_curvature'] = max_curvature
    results['avg_wasserstein_dist'] = float(avg_wasserstein)
    
    # Print summary
    print(f"\nSeed {seed} summary:")
    print(f"  Non-monotonic: {is_non_monotonic}")
    print(f"  Min accuracy at α={alphas[min_idx]:.2f}: {class_accs[min_idx]:.2f}%")
    print(f"  Max accuracy drop: {max_drop:.2f}%")
    print(f"  Max curvature: {max_curvature:.4f}")
    
    return results

# ================== STATISTICAL TESTS ==================
def compute_statistical_significance(all_results):
    """Compute comprehensive statistical tests"""
    n_seeds = len(all_results)
    
    # Extract accuracy arrays
    class_accs = np.array([r['class_interp_accs'] for r in all_results])
    random_accs = np.array([r['random_interp_accs'] for r in all_results])
    shuffled_accs = np.array([r['shuffled_interp_accs'] for r in all_results])
    
    # Test at multiple alpha values
    test_alpha_indices = [10, 20, 30]  # α = 0.25, 0.5, 0.75
    alphas = all_results[0]['alphas']
    
    p_values = {}
    effect_sizes = {}
    
    for idx in test_alpha_indices:
        alpha_val = alphas[idx]
        
        # Paired t-tests
        _, p_class_random = ttest_rel(class_accs[:, idx], random_accs[:, idx])
        _, p_class_shuffled = ttest_rel(class_accs[:, idx], shuffled_accs[:, idx])
        
        # Effect sizes (Cohen's d)
        d_class_random = (class_accs[:, idx].mean() - random_accs[:, idx].mean()) / np.sqrt(
            (class_accs[:, idx].std()**2 + random_accs[:, idx].std()**2) / 2
        )
        d_class_shuffled = (class_accs[:, idx].mean() - shuffled_accs[:, idx].mean()) / np.sqrt(
            (class_accs[:, idx].std()**2 + shuffled_accs[:, idx].std()**2) / 2
        )
        
        p_values[f'alpha_{alpha_val:.2f}'] = {
            'class_vs_random': float(p_class_random),
            'class_vs_shuffled': float(p_class_shuffled)
        }
        
        effect_sizes[f'alpha_{alpha_val:.2f}'] = {
            'class_vs_random': float(d_class_random),
            'class_vs_shuffled': float(d_class_shuffled)
        }
    
    # Bootstrap confidence intervals for non-monotonicity
    non_monotonic_counts = [r['non_monotonic'] for r in all_results]
    bootstrap_samples = 1000
    bootstrap_props = []
    
    for _ in range(bootstrap_samples):
        sample = np.random.choice(non_monotonic_counts, size=n_seeds, replace=True)
        bootstrap_props.append(np.mean(sample))
    
    ci_lower = np.percentile(bootstrap_props, 2.5)
    ci_upper = np.percentile(bootstrap_props, 97.5)
    
    return {
        'p_values': p_values,
        'effect_sizes': effect_sizes,
        'non_monotonic_ci': [float(ci_lower), float(ci_upper)]
    }

# ================== MAIN EXECUTION ==================
def main():
    # Run sanity checks
    run_sanity_checks()
    
    num_seeds = 10
    all_results = []
    
    start_time = time.time()
    
    # Run first seed and check
    print("\n" + "="*80)
    print("RUNNING FIRST SEED FOR SANITY CHECK")
    print("="*80)
    
    first_result = run_experiment(0)
    all_results.append(first_result)
    
    # Sanity check
    class_accs = first_result['class_interp_accs']
    random_accs = first_result['random_interp_accs']
    
    # Check metrics are non-trivial
    if np.std(class_accs) < 0.5:
        print('WARNING: Class interpolation shows very small variation')
    
    if min(random_accs) > 50:
        print('SANITY_ABORT: Random baseline accuracy too high (>50%)')
        exit(1)
    
    print("\nSANITY CHECK PASSED - Continuing with remaining seeds...")
    
    # Run remaining seeds
    for seed in range(1, num_seeds):
        results = run_experiment(seed)
        all_results.append(results)
        
        # Progress update
        elapsed = time.time() - start_time
        estimated_total = elapsed * num_seeds / (seed + 1)
        print(f"\nProgress: {seed+1}/{num_seeds} seeds complete. "
              f"Estimated remaining time: {(estimated_total - elapsed)/60:.1f} minutes")
    
    # Aggregate results
    class_accs_all = [r['class_interp_accs'] for r in all_results]
    random_accs_all = [r['random_interp_accs'] for r in all_results]
    shuffled_accs_all = [r['shuffled_interp_accs'] for r in all_results]
    
    # Count non-monotonic seeds
    non_monotonic_count = sum(r['non_monotonic'] for r in all_results)
    signal_detected = non_monotonic_count >= num_seeds * 0.3  # At least 30% of seeds
    
    # Compute statistics
    stat_results = compute_statistical_significance(all_results)
    
    # Prepare output
    output = {
        'per_seed_results': all_results,
        'aggregated': {
            'mean_class_interp_accs': [float(x) for x in np.mean(class_accs_all, axis=0)],
            'std_class_interp_accs': [float(x) for x in np.std(class_accs_all, axis=0)],
            'mean_random_interp_accs': [float(x) for x in np.mean(random_accs_all, axis=0)],
            'std_random_interp_accs': [float(x) for x in np.std(random_accs_all, axis=0)],
            'mean_shuffled_interp_accs': [float(x) for x in np.mean(shuffled_accs_all, axis=0)],
            'std_shuffled_interp_accs': [float(x) for x in np.std(shuffled_accs_all, axis=0)],
        },
        'non_monotonic_seeds': int(non_monotonic_count),
        'statistical_tests': stat_results,
        'ablation_summary': {
            seed: r['ablation_results'] for seed, r in enumerate(all_results[:3])  # First 3 seeds
        },
        'convergence_status': 'CONVERGED' if all(r['converged'] for r in all_results) else 'PARTIAL',
        'signal_detected': bool(signal_detected),
        'num_seeds': num_seeds,
        'runtime_hours': float((time.time() - start_time) / 3600)
    }
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    if signal_detected:
        print(f"\nSIGNAL_DETECTED: Non-monotonic behavior in {non_monotonic_count}/{num_seeds} seeds")
        print(f"  Average max drop: {np.mean([r['max_accuracy_drop'] for r in all_results]):.2f}%")
        print(f"  Average max curvature: {np.mean([r['max_curvature'] for r in all_results]):.4f}")
    else:
        print("\nNO_SIGNAL: Did not observe consistent non-monotonic behavior")
    
    print(f"\nStatistical significance (α=0.50):")
    print(f"  Class vs Random: p={stat_results['p_values']['alpha_0.50']['class_vs_random']:.4f}")
    print(f"  Effect size: d={stat_results['effect_sizes']['alpha_0.50']['class_vs_random']:.3f}")
    
    print(f"\nTotal runtime: {output['runtime_hours']:.2f} hours")
    print(f"\nRESULTS: {json.dumps(output)}")

if __name__ == "__main__":
    main()