# pip install scikit-learn scipy matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import ttest_ind
import json
import random
import time
from collections import defaultdict

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################
# METRIC SANITY CHECK SECTION
################################################################################

def compute_mutual_information(features, labels, n_bins=10):
    """Compute mutual information between features and labels using binning."""
    # Flatten features if needed
    if len(features.shape) > 1:
        features = features.mean(axis=tuple(range(1, len(features.shape))))
    
    # Discretize continuous features
    features_binned = np.digitize(features, np.linspace(features.min(), features.max(), n_bins))
    
    # Compute MI
    mi = mutual_info_score(features_binned, labels)
    return mi

def sanity_check_mutual_information():
    """Verify mutual information metric works correctly."""
    print("Running metric sanity checks...")
    
    # Test 1: Identical features should have high MI with labels
    n_samples = 1000
    n_classes = 10
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Perfect class separation
    perfect_features = labels.astype(float) + np.random.normal(0, 0.1, len(labels))
    mi_perfect = compute_mutual_information(perfect_features, labels)
    
    # Random features should have low MI
    random_features = np.random.normal(0, 1, len(labels))
    mi_random = compute_mutual_information(random_features, labels)
    
    # Class-correlated features
    correlated_features = labels.astype(float) + np.random.normal(0, 0.5, len(labels))
    mi_correlated = compute_mutual_information(correlated_features, labels)
    
    # Checks
    assert mi_perfect > 1.0, f"Perfect MI too low: {mi_perfect}"
    assert mi_random < 0.1, f"Random MI too high: {mi_random}"
    assert mi_random < mi_correlated < mi_perfect, f"MI ordering wrong: {mi_random}, {mi_correlated}, {mi_perfect}"
    
    print(f"Sanity checks passed: Random MI={mi_random:.3f}, Correlated MI={mi_correlated:.3f}, Perfect MI={mi_perfect:.3f}")
    print("METRIC_SANITY_PASSED")
    return True

# Run sanity check
sanity_check_mutual_information()

################################################################################
# MODEL DEFINITION
################################################################################

class SimpleCNN(nn.Module):
    """Simple CNN with batch normalization for MNIST."""
    def __init__(self, track_bn_stats=True):
        super().__init__()
        self.track_bn_stats = track_bn_stats
        self.bn_stats = defaultdict(list)  # Store BN stats per layer
        
        # Conv layers with BN
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x, track_stats=False):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        if track_stats and self.track_bn_stats:
            self.bn_stats['layer1_mean'].append(self.bn1.running_mean.clone().cpu())
            self.bn_stats['layer1_var'].append(self.bn1.running_var.clone().cpu())
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        if track_stats and self.track_bn_stats:
            self.bn_stats['layer2_mean'].append(self.bn2.running_mean.clone().cpu())
            self.bn_stats['layer2_var'].append(self.bn2.running_var.clone().cpu())
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        if track_stats and self.track_bn_stats:
            self.bn_stats['layer3_mean'].append(self.bn3.running_mean.clone().cpu())
            self.bn_stats['layer3_var'].append(self.bn3.running_var.clone().cpu())
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        # FC layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.bn4(x)
        if track_stats and self.track_bn_stats:
            self.bn_stats['layer4_mean'].append(self.bn4.running_mean.clone().cpu())
            self.bn_stats['layer4_var'].append(self.bn4.running_var.clone().cpu())
        x = F.relu(x)
        x = self.fc2(x)
        
        return x

################################################################################
# DATA LOADERS
################################################################################

def get_mnist_loaders(batch_size=128, sampling_strategy='random'):
    """Get MNIST data loaders with different sampling strategies."""
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    if sampling_strategy == 'random':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    elif sampling_strategy == 'balanced':
        # Create balanced sampler
        from torch.utils.data import WeightedRandomSampler
        targets = train_dataset.targets
        class_counts = torch.bincount(targets)
        weights = 1.0 / class_counts[targets]
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    elif sampling_strategy == 'single_class':
        # For extreme case - cycle through single class batches
        indices_per_class = defaultdict(list)
        for idx, (_, label) in enumerate(train_dataset):
            indices_per_class[label].append(idx)
        
        # Create custom batch sampler
        class SingleClassSampler:
            def __init__(self, indices_per_class, batch_size):
                self.indices_per_class = indices_per_class
                self.batch_size = batch_size
                
            def __iter__(self):
                # Cycle through classes
                all_indices = []
                for class_idx in range(10):
                    class_indices = self.indices_per_class[class_idx]
                    np.random.shuffle(class_indices)
                    # Create batches from single class
                    for i in range(0, len(class_indices), self.batch_size):
                        batch = class_indices[i:i+self.batch_size]
                        if len(batch) == self.batch_size:
                            all_indices.extend(batch)
                
                return iter(all_indices)
            
            def __len__(self):
                return sum(len(indices) // self.batch_size * self.batch_size for indices in self.indices_per_class.values())
        
        sampler = SingleClassSampler(indices_per_class, batch_size)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

################################################################################
# TRAINING AND EVALUATION
################################################################################

def collect_bn_stats_per_class(model, loader, device):
    """Collect BN statistics grouped by class."""
    model.eval()
    stats_per_class = defaultdict(lambda: defaultdict(list))
    
    # Use hooks to capture BN stats
    bn_stats = {}
    
    def get_bn_hook(name):
        def hook(module, input, output):
            if hasattr(module, 'running_mean'):
                bn_stats[name] = {
                    'mean': module.running_mean.clone().cpu().numpy(),
                    'var': module.running_var.clone().cpu().numpy()
                }
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(get_bn_hook(name)))
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            if batch_idx > 50:  # Limit samples for speed
                break
                
            data = data.to(device)
            _ = model(data)
            
            # Group stats by class
            for i, label in enumerate(target):
                label = label.item()
                for layer_name, stats in bn_stats.items():
                    stats_per_class[label][f"{layer_name}_mean"].append(stats['mean'])
                    stats_per_class[label][f"{layer_name}_var"].append(stats['var'])
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Average stats per class
    avg_stats_per_class = defaultdict(dict)
    for class_idx, class_stats in stats_per_class.items():
        for stat_name, stat_list in class_stats.items():
            avg_stats_per_class[class_idx][stat_name] = np.mean(stat_list, axis=0)
    
    return avg_stats_per_class

def compute_class_specific_mi(stats_per_class):
    """Compute mutual information between BN stats and class labels."""
    mi_results = {}
    
    # For each layer
    layer_names = set()
    for class_stats in stats_per_class.values():
        layer_names.update([k.split('_')[0] for k in class_stats.keys()])
    
    for layer_name in layer_names:
        # Collect all stats and labels
        all_means = []
        all_vars = []
        all_labels = []
        
        for class_idx, class_stats in stats_per_class.items():
            mean_key = f"{layer_name}_mean"
            var_key = f"{layer_name}_var"
            
            if mean_key in class_stats:
                # Add multiple samples per class for better MI estimation
                for _ in range(10):
                    # Add small noise for MI computation
                    mean_with_noise = class_stats[mean_key] + np.random.normal(0, 0.01, class_stats[mean_key].shape)
                    var_with_noise = class_stats[var_key] + np.random.normal(0, 0.01, class_stats[var_key].shape)
                    
                    all_means.append(mean_with_noise)
                    all_vars.append(var_with_noise)
                    all_labels.append(class_idx)
        
        if all_means:
            all_means = np.array(all_means)
            all_vars = np.array(all_vars)
            all_labels = np.array(all_labels)
            
            # Compute MI
            mi_mean = compute_mutual_information(all_means, all_labels)
            mi_var = compute_mutual_information(all_vars, all_labels)
            
            mi_results[layer_name] = {
                'mi_mean': mi_mean,
                'mi_var': mi_var,
                'mi_combined': (mi_mean + mi_var) / 2
            }
    
    return mi_results

def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), correct / total

################################################################################
# MAIN EXPERIMENT
################################################################################

def run_experiment(seed, sampling_strategy='random', max_epochs=20):
    """Run single experiment with given seed and sampling strategy."""
    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Create model and data
    model = SimpleCNN(track_bn_stats=True).to(device)
    train_loader, val_loader = get_mnist_loaders(batch_size=128, sampling_strategy=sampling_strategy)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    patience = 0
    max_patience = 5
    
    print(f"\n[Seed {seed}] Training with {sampling_strategy} sampling...")
    
    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
        else:
            patience += 1
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}")
        
        if patience >= max_patience:
            print("CONVERGED")
            break
    else:
        print("NOT_CONVERGED: Max epochs reached")
    
    # Collect BN statistics per class
    stats_per_class = collect_bn_stats_per_class(model, val_loader, device)
    
    # Compute mutual information
    mi_results = compute_class_specific_mi(stats_per_class)
    
    # Compute variance decomposition
    variance_results = {}
    for layer_name in mi_results.keys():
        # Collect means across classes
        class_means = []
        for class_idx in range(10):
            if class_idx in stats_per_class:
                mean_key = f"{layer_name}_mean"
                if mean_key in stats_per_class[class_idx]:
                    class_means.append(stats_per_class[class_idx][mean_key])
        
        if class_means:
            class_means = np.array(class_means)
            total_var = np.var(class_means)
            between_class_var = np.var(np.mean(class_means, axis=0))
            within_class_var = total_var - between_class_var
            
            variance_results[layer_name] = {
                'total_variance': float(total_var),
                'between_class_variance': float(between_class_var),
                'within_class_variance': float(within_class_var),
                'class_specific_ratio': float(between_class_var / (total_var + 1e-8))
            }
    
    return {
        'final_val_acc': val_acc,
        'best_val_acc': best_val_acc,
        'mi_results': mi_results,
        'variance_results': variance_results,
        'converged': patience < max_patience
    }

def main():
    """Main experimental loop."""
    print("Starting BN Class-Specificity Experiment (Feasibility Probe)")
    print("=" * 60)
    
    # Experimental conditions
    strategies = ['random', 'balanced', 'single_class']
    n_seeds = 3  # Small for feasibility
    
    # Store results
    all_results = defaultdict(list)
    
    # Run experiments
    start_time = time.time()
    
    for strategy in strategies:
        print(f"\n\nTesting {strategy} sampling strategy...")
        print("-" * 40)
        
        for seed in range(n_seeds):
            results = run_experiment(seed, sampling_strategy=strategy, max_epochs=10)
            all_results[strategy].append(results)
    
    # Analyze results
    print("\n\nAnalyzing Results...")
    print("=" * 60)
    
    # Compute statistics
    summary = {}
    for strategy, results_list in all_results.items():
        # Average MI across layers
        avg_mi_per_seed = []
        for result in results_list:
            layer_mis = [layer_data['mi_combined'] for layer_data in result['mi_results'].values()]
            avg_mi_per_seed.append(np.mean(layer_mis))
        
        # Class-specific variance ratio
        avg_class_ratio_per_seed = []
        for result in results_list:
            ratios = [layer_data['class_specific_ratio'] for layer_data in result['variance_results'].values()]
            avg_class_ratio_per_seed.append(np.mean(ratios))
        
        # Val accuracies
        val_accs = [r['best_val_acc'] for r in results_list]
        
        summary[strategy] = {
            'avg_mi_mean': np.mean(avg_mi_per_seed),
            'avg_mi_std': np.std(avg_mi_per_seed),
            'class_variance_ratio_mean': np.mean(avg_class_ratio_per_seed),
            'class_variance_ratio_std': np.std(avg_class_ratio_per_seed),
            'val_acc_mean': np.mean(val_accs),
            'val_acc_std': np.std(val_accs),
            'converged_ratio': sum(r['converged'] for r in results_list) / len(results_list)
        }
    
    # Statistical tests
    p_values = {}
    if len(all_results['random']) >= 2 and len(all_results['balanced']) >= 2:
        # Compare MI between random and balanced
        random_mi = [np.mean([layer_data['mi_combined'] for layer_data in r['mi_results'].values()]) 
                     for r in all_results['random']]
        balanced_mi = [np.mean([layer_data['mi_combined'] for layer_data in r['mi_results'].values()]) 
                       for r in all_results['balanced']]
        
        _, p_val = ttest_ind(balanced_mi, random_mi)
        p_values['balanced_vs_random_mi'] = p_val
    
    # Check if signal detected
    signal_detected = False
    if summary['balanced']['avg_mi_mean'] > summary['random']['avg_mi_mean'] * 1.2:
        print("SIGNAL_DETECTED: Balanced sampling shows higher class-specific MI in BN stats")
        signal_detected = True
    elif summary['single_class']['avg_mi_mean'] > summary['random']['avg_mi_mean'] * 1.5:
        print("SIGNAL_DETECTED: Single-class sampling shows much higher class-specific MI")
        signal_detected = True
    else:
        print("NO_SIGNAL: No significant difference in class-specific information across sampling strategies")
    
    # Print summary
    print("\nSummary Statistics:")
    for strategy, stats in summary.items():
        print(f"\n{strategy.upper()} sampling:")
        print(f"  MI (BN stats ↔ class): {stats['avg_mi_mean']:.3f} ± {stats['avg_mi_std']:.3f}")
        print(f"  Class variance ratio: {stats['class_variance_ratio_mean']:.3f} ± {stats['class_variance_ratio_std']:.3f}")
        print(f"  Val accuracy: {stats['val_acc_mean']:.3f} ± {stats['val_acc_std']:.3f}")
    
    # Final results JSON
    final_results = {
        'experiment': 'bn_class_specificity_feasibility',
        'runtime_minutes': (time.time() - start_time) / 60,
        'signal_detected': signal_detected,
        'summary': summary,
        'p_values': p_values,
        'hypothesis_1_supported': summary['balanced']['avg_mi_mean'] > 0.1,  # Adjusted threshold for MNIST
        'hypothesis_2_check': 'Layer-wise analysis needed for full test',
        'hypothesis_3_supported': summary['balanced']['avg_mi_mean'] > summary['random']['avg_mi_mean'],
        'convergence_status': f"Converged in {sum(s['converged_ratio'] for s in summary.values())/len(summary)*100:.0f}% of runs"
    }
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    main()