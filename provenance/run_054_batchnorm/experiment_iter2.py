# pip install scikit-learn scipy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
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
# SYNTHETIC DATASET
################################################################################

def create_synthetic_dataset(n_samples=5000, n_features=16, n_classes=10, class_separation=2.0):
    """Create synthetic dataset with controllable class separation."""
    # Generate class centers
    class_centers = np.random.randn(n_classes, n_features) * class_separation
    
    # Generate samples
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        # Generate samples around class center
        class_samples = class_centers[class_idx] + np.random.randn(samples_per_class, n_features) * 0.5
        X.extend(class_samples)
        y.extend([class_idx] * samples_per_class)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

################################################################################
# SIMPLE MODEL
################################################################################

class TinyNet(nn.Module):
    """Tiny network for fast experiments."""
    def __init__(self, input_dim=16, hidden_dim=64, n_classes=10):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, n_classes)
        
        # Storage for BN stats during training
        self.bn_stats_history = defaultdict(list)
        self.track_stats = False
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        if self.track_stats and self.training:
            self.bn_stats_history['bn1_mean'].append(self.bn1.running_mean.clone().detach().cpu().numpy())
            self.bn_stats_history['bn1_var'].append(self.bn1.running_var.clone().detach().cpu().numpy())
        x = F.relu(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        if self.track_stats and self.training:
            self.bn_stats_history['bn2_mean'].append(self.bn2.running_mean.clone().detach().cpu().numpy())
            self.bn_stats_history['bn2_var'].append(self.bn2.running_var.clone().detach().cpu().numpy())
        x = F.relu(x)
        
        # Output
        x = self.fc3(x)
        
        return x

################################################################################
# CUSTOM SAMPLERS
################################################################################

class BalancedBatchSampler:
    """Ensures each batch has equal representation from each class."""
    def __init__(self, labels, batch_size, n_classes=10):
        self.labels = labels
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.samples_per_class = batch_size // n_classes
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
        
        # Ensure we have enough samples
        for c in range(n_classes):
            if len(self.class_indices[c]) < self.samples_per_class:
                raise ValueError(f"Not enough samples for class {c}")
    
    def __iter__(self):
        # Shuffle indices within each class
        for c in range(self.n_classes):
            random.shuffle(self.class_indices[c])
        
        # Generate batches
        n_batches = min(len(self.class_indices[c]) // self.samples_per_class for c in range(self.n_classes))
        
        for batch_idx in range(n_batches):
            batch = []
            for c in range(self.n_classes):
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class
                batch.extend(self.class_indices[c][start_idx:end_idx])
            
            random.shuffle(batch)  # Shuffle within batch
            yield batch
    
    def __len__(self):
        return min(len(self.class_indices[c]) // self.samples_per_class for c in range(self.n_classes))

class SingleClassBatchSampler:
    """Each batch contains samples from only one class."""
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_indices[label].append(idx)
    
    def __iter__(self):
        all_batches = []
        
        # Create single-class batches
        for class_idx, indices in self.class_indices.items():
            random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Only full batches
                    all_batches.append(batch)
        
        # Shuffle order of batches
        random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        count = 0
        for indices in self.class_indices.values():
            count += len(indices) // self.batch_size
        return count

################################################################################
# FAST TRAINING
################################################################################

def train_model_fast(model, train_loader, val_loader, max_epochs=5, device='cpu'):
    """Fast training with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Training
        model.train()
        model.track_stats = (epoch == max_epochs - 1)  # Only track stats in last epoch
        
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 50:  # Limit batches for speed
                break
                
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx > 20:  # Limit validation batches
                    break
                    
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 2:
            print("CONVERGED")
            break
    else:
        print("NOT_CONVERGED: Max epochs")
    
    return best_val_acc

################################################################################
# BN STATS ANALYSIS
################################################################################

def analyze_bn_stats(model, train_loader, device):
    """Analyze class-specificity of BN statistics."""
    model.eval()
    
    # Collect BN stats per class
    stats_per_class = defaultdict(lambda: defaultdict(list))
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx > 30:  # Limit for speed
                break
                
            data = data.to(device)
            
            # Forward pass to update BN running stats
            _ = model(data)
            
            # Record current BN stats for each sample's class
            for i, label in enumerate(target):
                label = label.item()
                stats_per_class[label]['bn1_mean'].append(model.bn1.running_mean.clone().cpu().numpy())
                stats_per_class[label]['bn1_var'].append(model.bn1.running_var.clone().cpu().numpy())
                stats_per_class[label]['bn2_mean'].append(model.bn2.running_mean.clone().cpu().numpy())
                stats_per_class[label]['bn2_var'].append(model.bn2.running_var.clone().cpu().numpy())
    
    # Compute average stats per class
    avg_stats_per_class = {}
    for class_idx in stats_per_class:
        avg_stats_per_class[class_idx] = {}
        for stat_name in stats_per_class[class_idx]:
            avg_stats_per_class[class_idx][stat_name] = np.mean(
                stats_per_class[class_idx][stat_name], axis=0
            )
    
    # Compute mutual information
    mi_results = {}
    for stat_type in ['bn1_mean', 'bn1_var', 'bn2_mean', 'bn2_var']:
        # Collect stats and labels
        all_stats = []
        all_labels = []
        
        for class_idx in range(10):
            if class_idx in avg_stats_per_class:
                # Add the average stat for this class multiple times with small noise
                avg_stat = avg_stats_per_class[class_idx][stat_type]
                for _ in range(5):  # Fewer samples for speed
                    noisy_stat = avg_stat + np.random.normal(0, 0.01, avg_stat.shape)
                    all_stats.append(noisy_stat)
                    all_labels.append(class_idx)
        
        if all_stats:
            all_stats = np.array(all_stats)
            all_labels = np.array(all_labels)
            
            mi = compute_mutual_information(all_stats, all_labels)
            mi_results[stat_type] = mi
    
    return mi_results, avg_stats_per_class

################################################################################
# MAIN EXPERIMENT
################################################################################

def run_single_experiment(seed, sampling_strategy='random'):
    """Run one experiment with given seed and strategy."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n[Seed {seed}] Running {sampling_strategy} sampling...")
    
    # Create synthetic dataset
    X_train, y_train = create_synthetic_dataset(n_samples=2000, n_features=16, n_classes=10, class_separation=2.0)
    X_val, y_val = create_synthetic_dataset(n_samples=500, n_features=16, n_classes=10, class_separation=2.0)
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    # Create data loaders based on sampling strategy
    batch_size = 100
    
    if sampling_strategy == 'random':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    elif sampling_strategy == 'balanced':
        sampler = BalancedBatchSampler(y_train, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    elif sampling_strategy == 'single_class':
        sampler = SingleClassBatchSampler(y_train, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create and train model
    model = TinyNet(input_dim=16, hidden_dim=64, n_classes=10).to(device)
    val_acc = train_model_fast(model, train_loader, val_loader, max_epochs=3, device=device)
    
    # Analyze BN statistics
    mi_results, stats_per_class = analyze_bn_stats(model, train_loader, device)
    
    # Compute average MI
    avg_mi = np.mean(list(mi_results.values()))
    
    return {
        'val_acc': val_acc,
        'avg_mi': avg_mi,
        'mi_results': mi_results
    }

def main():
    """Main experiment."""
    print("Starting BN Class-Specificity Experiment (Fast Feasibility Probe)")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run experiments
    strategies = ['random', 'balanced', 'single_class']
    n_seeds = 3
    
    results = defaultdict(list)
    
    for strategy in strategies:
        print(f"\n\nTesting {strategy} sampling...")
        print("-" * 40)
        
        for seed in range(n_seeds):
            result = run_single_experiment(seed, sampling_strategy=strategy)
            results[strategy].append(result)
    
    # Analyze results
    print("\n\nAnalyzing Results...")
    print("=" * 60)
    
    # Compute summary statistics
    summary = {}
    for strategy, strategy_results in results.items():
        mi_values = [r['avg_mi'] for r in strategy_results]
        acc_values = [r['val_acc'] for r in strategy_results]
        
        summary[strategy] = {
            'mi_mean': np.mean(mi_values),
            'mi_std': np.std(mi_values),
            'acc_mean': np.mean(acc_values),
            'acc_std': np.std(acc_values)
        }
        
        print(f"\n{strategy.upper()}:")
        print(f"  MI: {summary[strategy]['mi_mean']:.3f} ± {summary[strategy]['mi_std']:.3f}")
        print(f"  Acc: {summary[strategy]['acc_mean']:.3f} ± {summary[strategy]['acc_std']:.3f}")
    
    # Statistical test
    p_values = {}
    if len(results['random']) >= 2 and len(results['balanced']) >= 2:
        random_mi = [r['avg_mi'] for r in results['random']]
        balanced_mi = [r['avg_mi'] for r in results['balanced']]
        _, p_val = ttest_ind(balanced_mi, random_mi)
        p_values['balanced_vs_random'] = p_val
    
    # Check for signal
    signal_detected = False
    if summary['balanced']['mi_mean'] > summary['random']['mi_mean'] * 1.2:
        print("\nSIGNAL_DETECTED: Balanced sampling increases class-specific MI in BN stats")
        signal_detected = True
    elif summary['single_class']['mi_mean'] > summary['random']['mi_mean'] * 1.5:
        print("\nSIGNAL_DETECTED: Single-class sampling strongly increases class-specific MI")
        signal_detected = True
    else:
        print("\nNO_SIGNAL: No clear difference in class-specific MI across strategies")
    
    # Final results
    runtime = (time.time() - start_time) / 60
    
    final_results = {
        'experiment': 'bn_class_specificity_fast',
        'runtime_minutes': runtime,
        'signal_detected': signal_detected,
        'summary': summary,
        'p_values': p_values,
        'hypothesis_supported': summary['balanced']['mi_mean'] > 0.05
    }
    
    print(f"\nRESULTS: {json.dumps(final_results)}")

if __name__ == "__main__":
    main()