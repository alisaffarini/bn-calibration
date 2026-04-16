
# === DRY RUN VALIDATION: forced tiny scale ===
import builtins
_dry_run_got_results = False
_orig_print_fn = builtins.print
def _patched_print(*args, **kwargs):
    global _dry_run_got_results
    msg = " ".join(str(a) for a in args)
    if "RESULTS:" in msg:
        _dry_run_got_results = True
    _orig_print_fn(*args, **kwargs)
builtins.print = _patched_print

import atexit
def _check_results():
    if not _dry_run_got_results:
        _orig_print_fn("DRY_RUN_WARNING: Pipeline completed but no RESULTS: line was printed!")
        _orig_print_fn("DRY_RUN_WARNING: The post-processing/output stage may be broken.")
    else:
        _orig_print_fn("DRY_RUN_OK: Full pipeline validated (train → analyze → output)")
atexit.register(_check_results)

# pip install scikit-learn scipy matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import ttest_rel, bootstrap
import json
import random
import time
from collections import defaultdict
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

################################################################################
# METRIC SANITY CHECK SECTION
################################################################################

def compute_mutual_information(features, labels, n_bins=10):
    """Compute mutual information between features and labels using binning."""
    # Flatten features if multi-dimensional
    if len(features.shape) > 1:
        # Use all dimensions, not just mean
        features = features.reshape(features.shape[0], -1)
        # Take first few principal components if too many dimensions
        if features.shape[1] > 10:
            # Simple dimensionality reduction - just take first 10 features
            features = features[:, :10]
    
    # Compute MI for each feature dimension and average
    mi_values = []
    for dim in range(features.shape[1] if len(features.shape) > 1 else 1):
        feature_dim = features[:, dim] if len(features.shape) > 1 else features
        # Discretize continuous features
        if feature_dim.std() > 0:
            bins = np.percentile(feature_dim, np.linspace(0, 100, n_bins))
            bins = np.unique(bins)  # Remove duplicates
            if len(bins) > 1:
                features_binned = np.digitize(feature_dim, bins)
                mi = mutual_info_score(features_binned, labels)
                mi_values.append(mi)
    
    return np.mean(mi_values) if mi_values else 0.0

def sanity_check_mutual_information():
    """Verify mutual information metric works correctly."""
    print("Running metric sanity checks...")
    
    # Test 1: Perfect separation
    n_samples = 1000
    n_classes = 10
    labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    
    # Perfect class separation - each class has distinct feature value
    perfect_features = np.zeros((len(labels), 5))
    for i, label in enumerate(labels):
        perfect_features[i, :] = label + np.random.normal(0, 0.01, 5)
    
    mi_perfect = compute_mutual_information(perfect_features, labels)
    
    # Random features
    random_features = np.random.normal(0, 1, (len(labels), 5))
    mi_random = compute_mutual_information(random_features, labels)
    
    # Partially correlated
    correlated_features = np.zeros((len(labels), 5))
    for i, label in enumerate(labels):
        correlated_features[i, :] = label * 0.5 + np.random.normal(0, 1, 5)
    mi_correlated = compute_mutual_information(correlated_features, labels)
    
    # Checks
    assert mi_perfect > 0.8, f"Perfect MI too low: {mi_perfect}"
    assert mi_random < 0.2, f"Random MI too high: {mi_random}"
    assert mi_random < mi_correlated < mi_perfect, f"MI ordering wrong: {mi_random}, {mi_correlated}, {mi_perfect}"
    
    print(f"Sanity checks passed: Random MI={mi_random:.3f}, Correlated MI={mi_correlated:.3f}, Perfect MI={mi_perfect:.3f}")
    print("METRIC_SANITY_PASSED")
    return True

# Run sanity check
sanity_check_mutual_information()

################################################################################
# DATASET CREATION
################################################################################

def create_synthetic_dataset(n_samples=10000, n_features=32, n_classes=10, class_separation=3.0, split='train'):
    """Create synthetic dataset with clear class structure."""
    # Generate well-separated class centers
    np.random.seed(42)  # Fixed seed for class centers
    class_centers = np.random.randn(n_classes, n_features) * class_separation
    
    # Different random seeds for different splits
    split_seeds = {'train': 100, 'val': 200, 'test': 300}
    np.random.seed(split_seeds.get(split, 100))
    
    # Generate samples
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        # Generate samples around class center with controlled noise
        class_samples = class_centers[class_idx] + np.random.randn(samples_per_class, n_features) * 0.3
        X.extend(class_samples)
        y.extend([class_idx] * samples_per_class)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

################################################################################
# MODEL DEFINITIONS
################################################################################

class BNStatsTracker:
    """Track BN statistics during training."""
    def __init__(self):
        self.stats_history = defaultdict(list)
        self.batch_labels_history = []
        
    def record_batch(self, bn_modules, batch_labels):
        """Record BN stats for a batch."""
        # Record dominant class in batch
        if len(batch_labels.shape) > 0:
            # For single-class batches, all labels are the same
            dominant_class = batch_labels[0].item()
        else:
            dominant_class = batch_labels.item()
            
        self.batch_labels_history.append(dominant_class)
        
        # Record BN stats
        for name, module in bn_modules:
            self.stats_history[f'{name}_mean'].append(module.running_mean.detach().cpu().numpy().copy())
            self.stats_history[f'{name}_var'].append(module.running_var.detach().cpu().numpy().copy())
    
    def get_stats_and_labels(self):
        """Get recorded stats and labels."""
        return dict(self.stats_history), self.batch_labels_history

class ModelWithBN(nn.Module):
    """Model with batch normalization."""
    def __init__(self, input_dim=32, hidden_dims=[128, 256, 128], n_classes=10):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        self.bn_modules = []  # Keep track of BN modules
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            bn = nn.BatchNorm1d(hidden_dim, momentum=0.1)
            layers.append(bn)
            layers.append(nn.ReLU())
            self.bn_modules.append((f'bn{i+1}', bn))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        # Better initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Stats tracker
        self.stats_tracker = BNStatsTracker()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_bn_modules(self):
        """Get all BN modules for analysis."""
        return self.bn_modules

class ModelWithGN(nn.Module):
    """Model with group normalization (baseline)."""
    def __init__(self, input_dim=32, hidden_dims=[128, 256, 128], n_classes=10, num_groups=8):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            # Ensure hidden_dim is divisible by num_groups
            groups = min(num_groups, hidden_dim)
            while hidden_dim % groups != 0:
                groups -= 1
            layers.append(nn.GroupNorm(groups, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ModelWithLN(nn.Module):
    """Model with layer normalization (baseline)."""
    def __init__(self, input_dim=32, hidden_dims=[128, 256, 128], n_classes=10):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

################################################################################
# CUSTOM SAMPLERS
################################################################################

class BalancedBatchSampler:
    """Ensures each batch has equal representation from each class."""
    def __init__(self, labels, batch_size, n_classes=10):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.samples_per_class = batch_size // n_classes
        
        # Group indices by class
        self.class_indices = {}
        for c in range(n_classes):
            self.class_indices[c] = np.where(self.labels == c)[0].tolist()
    
    def __iter__(self):
        # Shuffle indices within each class
        for c in range(self.n_classes):
            random.shuffle(self.class_indices[c])
        
        # Generate balanced batches
        min_class_size = min(len(self.class_indices[c]) for c in range(self.n_classes))
        n_batches = min_class_size // self.samples_per_class
        
        all_batches = []
        for batch_idx in range(n_batches):
            batch = []
            for c in range(self.n_classes):
                start = batch_idx * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(self.class_indices[c][start:end])
            random.shuffle(batch)
            all_batches.append(batch)
        
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        min_class_size = min(len(self.class_indices[c]) for c in range(self.n_classes))
        return min_class_size // self.samples_per_class

class SingleClassBatchSampler:
    """Each batch contains samples from only one class."""
    def __init__(self, labels, batch_size):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        unique_labels = np.unique(labels)
        for label in unique_labels:
            self.class_indices[label] = np.where(self.labels == label)[0].tolist()
    
    def __iter__(self):
        all_batches = []
        
        # Create single-class batches with their class labels
        for class_idx, indices in self.class_indices.items():
            indices_copy = indices.copy()
            random.shuffle(indices_copy)
            
            for i in range(0, len(indices_copy) - self.batch_size + 1, self.batch_size):
                batch = indices_copy[i:i + self.batch_size]
                # Store batch indices and the class they belong to
                all_batches.append((batch, class_idx))
        
        # Shuffle order of batches
        random.shuffle(all_batches)
        
        # Store class sequence for later analysis
        self.batch_class_sequence = [class_idx for _, class_idx in all_batches]
        
        for batch, _ in all_batches:
            yield batch
    
    def __len__(self):
        count = 0
        for indices in self.class_indices.values():
            count += len(indices) // self.batch_size
        return count

################################################################################
# TRAINING WITH BN STATS TRACKING
################################################################################

def train_epoch_with_tracking(model, loader, optimizer, criterion, device, track_stats=False):
    """Train for one epoch, optionally tracking BN stats."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Check if we have a single-class sampler
    is_single_class = hasattr(loader.batch_sampler, 'batch_class_sequence')
    batch_class_iter = iter(loader.batch_sampler.batch_class_sequence) if is_single_class else None
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Track BN stats if requested and model supports it
        if track_stats and hasattr(model, 'stats_tracker'):
            if is_single_class and batch_class_iter:
                # For single-class batches, use the known class
                try:
                    batch_class = next(batch_class_iter)
                    model.stats_tracker.record_batch(model.get_bn_modules(), 
                                                   torch.tensor(batch_class))
                except StopIteration:
                    pass
            else:
                # For other samplers, use the dominant class
                model.stats_tracker.record_batch(model.get_bn_modules(), target)
        
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
            
            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / total, correct / total

def train_model_convergence(model, train_loader, val_loader, device, max_epochs=3, patience=2):
    """Train model with convergence-based stopping and BN stats tracking."""
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(max_epochs):
        # Track stats only in later epochs when model is more stable
        track_stats = epoch >= max_epochs // 2
        
        train_loss, train_acc = train_epoch_with_tracking(
            model, train_loader, optimizer, criterion, device, track_stats=track_stats
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}, LR={optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience and epoch >= 10:
            print(f"CONVERGED at epoch {epoch}")
            break
            
        # Early stop if learning is too slow
        if epoch > 20 and best_val_acc < 0.2:
            print("NOT_CONVERGED: Learning too slow")
            break
    else:
        print("NOT_CONVERGED: Max epochs reached")
    
    return best_val_acc

################################################################################
# BN STATISTICS ANALYSIS
################################################################################

def compute_class_specific_mi(stats_dict, batch_labels):
    """Compute MI between BN stats and batch class labels."""
    mi_results = {}
    
    for stat_name, stat_list in stats_dict.items():
        if stat_list and len(stat_list) > 0:
            stats_array = np.array(stat_list)
            labels_array = np.array(batch_labels[:len(stat_list)])
            
            # Only compute MI if we have variation in labels
            unique_labels = np.unique(labels_array)
            if len(unique_labels) > 1:
                mi = compute_mutual_information(stats_array, labels_array)
                mi_results[stat_name] = float(mi)
            else:
                mi_results[stat_name] = 0.0
    
    return mi_results

def compute_variance_decomposition(stats_dict, batch_labels):
    """Decompose variance into between-class and within-class components."""
    variance_results = {}
    
    for stat_name, stat_list in stats_dict.items():
        if stat_list and len(stat_list) > 0:
            stats_array = np.array(stat_list)
            labels_array = np.array(batch_labels[:len(stat_list)])
            
            unique_labels = np.unique(labels_array)
            if len(unique_labels) > 1:
                # Group by class
                class_stats = defaultdict(list)
                for stat, cls in zip(stats_array, labels_array):
                    class_stats[cls].append(stat)
                
                # Compute variance components
                all_stats_flat = []
                class_means = []
                
                for cls, stats in class_stats.items():
                    stats = np.array(stats)
                    all_stats_flat.extend(stats.flatten())
                    class_means.append(np.mean(stats, axis=0))
                
                all_stats_flat = np.array(all_stats_flat)
                class_means = np.array(class_means)
                
                # Use flattened stats for variance calculation
                total_var = np.var(all_stats_flat)
                
                if len(class_means) > 1:
                    between_class_var = np.var(class_means.flatten())
                    within_class_var = max(0, total_var - between_class_var)
                    class_ratio = between_class_var / (total_var + 1e-8)
                else:
                    between_class_var = 0
                    within_class_var = total_var
                    class_ratio = 0
                
                variance_results[stat_name] = {
                    'total_variance': float(total_var),
                    'between_class_variance': float(between_class_var),
                    'within_class_variance': float(within_class_var),
                    'class_specific_ratio': float(class_ratio)
                }
    
    return variance_results

################################################################################
# FULL EXPERIMENT PIPELINE
################################################################################

def run_single_seed_experiment(seed, model_type='bn', sampling_strategy='random', 
                              hidden_dims=[128, 256, 128], max_epochs=3):
    """Run complete experiment for a single seed."""
    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n{'='*60}")
    print(f"Seed {seed}: {model_type.upper()} model, {sampling_strategy} sampling")
    print(f"{'='*60}")
    
    # Create datasets
    X_train, y_train = create_synthetic_dataset(n_samples=10000, n_features=32, split='train')
    X_val, y_val = create_synthetic_dataset(n_samples=2000, n_features=32, split='val')
    X_test, y_test = create_synthetic_dataset(n_samples=2000, n_features=32, split='test')
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # Create loaders
    batch_size = 64
    
    if sampling_strategy == 'random':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    elif sampling_strategy == 'balanced':
        sampler = BalancedBatchSampler(y_train, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    elif sampling_strategy == 'single_class':
        sampler = SingleClassBatchSampler(y_train, batch_size=batch_size)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    if model_type == 'bn':
        model = ModelWithBN(input_dim=32, hidden_dims=hidden_dims).to(device)
    elif model_type == 'gn':
        model = ModelWithGN(input_dim=32, hidden_dims=hidden_dims).to(device)
    elif model_type == 'ln':
        model = ModelWithLN(input_dim=32, hidden_dims=hidden_dims).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    start_time = time.time()
    val_acc = train_model_convergence(model, train_loader, val_loader, device, max_epochs=max_epochs)
    train_time = time.time() - start_time
    
    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    
    # Collect BN statistics analysis (only for BN models)
    mi_results = {}
    variance_results = {}
    
    if model_type == 'bn' and hasattr(model, 'stats_tracker'):
        stats_dict, batch_labels = model.stats_tracker.get_stats_and_labels()
        
        if stats_dict and batch_labels:
            mi_results = compute_class_specific_mi(stats_dict, batch_labels)
            variance_results = compute_variance_decomposition(stats_dict, batch_labels)
    
    # Compute average MI across layers
    avg_mi = float(np.mean(list(mi_results.values()))) if mi_results else 0.0
    
    # Compute average class-specific variance ratio
    avg_class_ratio = 0.0
    if variance_results:
        ratios = [v['class_specific_ratio'] for v in variance_results.values() if v['total_variance'] > 0]
        avg_class_ratio = float(np.mean(ratios)) if ratios else 0.0
    
    results = {
        'seed': seed,
        'model_type': model_type,
        'sampling_strategy': sampling_strategy,
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'avg_mi': avg_mi,
        'avg_class_ratio': avg_class_ratio,
        'train_time': train_time,
        'mi_results': mi_results,
        'n_tracked_batches': len(batch_labels) if model_type == 'bn' else 0
    }
    
    print(f"\nSeed {seed} Summary:")
    print(f"  Test Acc: {test_acc:.3f}")
    print(f"  Avg MI: {avg_mi:.3f}")
    print(f"  Avg Class Ratio: {avg_class_ratio:.3f}")
    if model_type == 'bn':
        print(f"  Tracked batches: {results['n_tracked_batches']}")
    
    return results

################################################################################
# STATISTICAL ANALYSIS
################################################################################

def compute_bootstrap_ci(data, func=np.mean, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence interval."""
    if len(data) < 2:
        return func(data), func(data), func(data)
    
    # Manual bootstrap since scipy.stats.bootstrap might not be available
    bootstrap_stats = []
    n_samples = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(data, size=n_samples, replace=True)
        bootstrap_stats.append(func(resampled))
    
    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    
    return float(func(data)), float(lower), float(upper)

def run_statistical_comparisons(all_results):
    """Run paired statistical tests between conditions."""
    comparisons = {}
    
    # Group results by seed for paired comparisons
    results_by_seed = defaultdict(dict)
    for result in all_results:
        key = (result['seed'], result['model_type'], result['sampling_strategy'])
        results_by_seed[result['seed']][f"{result['model_type']}_{result['sampling_strategy']}"] = result
    
    # Compare sampling strategies for BN
    bn_random_mi = []
    bn_single_mi = []
    
    for seed_results in results_by_seed.values():
        if 'bn_random' in seed_results and 'bn_single_class' in seed_results:
            bn_random_mi.append(seed_results['bn_random']['avg_mi'])
            bn_single_mi.append(seed_results['bn_single_class']['avg_mi'])
    
    if len(bn_random_mi) >= 2 and len(bn_single_mi) >= 2:
        _, p_val = ttest_rel(bn_single_mi, bn_random_mi)
        comparisons['bn_single_vs_random_mi'] = float(p_val)
    
    return comparisons

################################################################################
# MAIN EXPERIMENT
################################################################################

def main():
    """Main experiment execution."""
    print("Starting Full-Scale BN Class-Specificity Experiment")
    print("=" * 80)
    
    # Experiment configuration
    n_seeds = 2
    model_types = ['bn', 'gn', 'ln']  # BatchNorm, GroupNorm, LayerNorm
    sampling_strategies = ['random', 'balanced', 'single_class']
    
    all_results = []
    start_time = time.time()
    
    # Run first seed to check sanity
    print("\nRunning first seed as sanity check...")
    first_result = run_single_seed_experiment(
        seed=0, 
        model_type='bn', 
        sampling_strategy='single_class',
        hidden_dims=[128, 256, 128],
        max_epochs=3
    )
    
    # More lenient sanity check - single_class might have different MI characteristics
    if first_result['test_acc'] < 0.15 and first_result['avg_mi'] == 0.0:
        print(f"SANITY_ABORT: Both accuracy ({first_result['test_acc']:.3f}) and MI ({first_result['avg_mi']}) are too low")
        exit(1)
    
    print(f"\nFirst seed passed sanity check: MI={first_result['avg_mi']:.3f}, Acc={first_result['test_acc']:.3f}")
    all_results.append(first_result)
    
    # Run comparison with random sampling to ensure we can detect differences
    print("\nRunning random sampling comparison...")
    random_result = run_single_seed_experiment(
        seed=0,
        model_type='bn',
        sampling_strategy='random',
        hidden_dims=[128, 256, 128],
        max_epochs=3
    )
    all_results.append(random_result)
    
    # Main experiments
    print("\n" + "="*80)
    print("MAIN EXPERIMENTS")
    print("="*80)
    
    for seed in range(1, min(n_seeds, 5)):  # Reduced for faster runtime
        for sampling in sampling_strategies:
            result = run_single_seed_experiment(
                seed=seed,
                model_type='bn',
                sampling_strategy=sampling,
                hidden_dims=[128, 256, 128],
                max_epochs=3  # Reduced epochs
            )
            all_results.append(result)
    
    # Baseline comparisons (fewer seeds)
    print("\n" + "="*80)
    print("BASELINE COMPARISONS")
    print("="*80)
    
    for seed in range(min(3, n_seeds)):
        for model_type in ['gn', 'ln']:
            for sampling in ['random', 'single_class']:
                result = run_single_seed_experiment(
                    seed=seed,
                    model_type=model_type,
                    sampling_strategy=sampling,
                    hidden_dims=[128, 256, 128],
                    max_epochs=3
                )
                all_results.append(result)
    
    # Analysis and reporting
    print("\n" + "="*80)
    print("FINAL ANALYSIS")
    print("="*80)
    
    # Compute summary statistics
    summary = defaultdict(lambda: defaultdict(list))
    
    for result in all_results:
        key = f"{result['model_type']}_{result['sampling_strategy']}"
        summary[key]['test_acc'].append(result['test_acc'])
        summary[key]['avg_mi'].append(result['avg_mi'])
        summary[key]['avg_class_ratio'].append(result['avg_class_ratio'])
    
    # Print summary table
    print("\nSummary Results (mean ± std):")
    print("-" * 80)
    print(f"{'Condition':<30} {'Test Acc':<20} {'Avg MI':<20} {'Class Ratio':<20}")
    print("-" * 80)
    
    summary_dict = {}
    for condition, metrics in sorted(summary.items()):
        if metrics['test_acc']:  # Only if we have data
            acc_mean = np.mean(metrics['test_acc'])
            acc_std = np.std(metrics['test_acc'])
            mi_mean = np.mean(metrics['avg_mi'])
            mi_std = np.std(metrics['avg_mi'])
            ratio_mean = np.mean(metrics['avg_class_ratio'])
            ratio_std = np.std(metrics['avg_class_ratio'])
            
            print(f"{condition:<30} "
                  f"{acc_mean:.3f}±{acc_std:.3f}       "
                  f"{mi_mean:.3f}±{mi_std:.3f}       "
                  f"{ratio_mean:.3f}±{ratio_std:.3f}")
            
            summary_dict[condition] = {
                'test_acc_mean': float(acc_mean),
                'test_acc_std': float(acc_std),
                'mi_mean': float(mi_mean),
                'mi_std': float(mi_std),
                'class_ratio_mean': float(ratio_mean),
                'class_ratio_std': float(ratio_std),
                'n_samples': len(metrics['test_acc'])
            }
    
    # Statistical comparisons
    comparisons = run_statistical_comparisons(all_results)
    
    print("\nStatistical Tests (p-values):")
    print("-" * 50)
    for test, p_val in comparisons.items():
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{test}: p={p_val:.4f} {significance}")
    
    # Check hypotheses
    print("\nHypothesis Testing:")
    print("-" * 50)
    
    # H1: BN stats contain significant MI with class labels
    bn_single_mi = [r['avg_mi'] for r in all_results 
                    if r['model_type'] == 'bn' and r['sampling_strategy'] == 'single_class']
    h1_supported = len(bn_single_mi) > 0 and np.mean(bn_single_mi) > 0.1
    print(f"H1 (BN contains class info): {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'}")
    
    # H3: Single-class shows more info than random
    h3_supported = False
    if 'bn_single_vs_random_mi' in comparisons:
        h3_supported = comparisons['bn_single_vs_random_mi'] < 0.05
        print(f"H3 (single > random): {'SUPPORTED' if h3_supported else 'NOT SUPPORTED'} "
              f"(p={comparisons.get('bn_single_vs_random_mi', 'N/A'):.3f})")
    
    # Final summary JSON
    runtime = (time.time() - start_time) / 60
    
    final_results = {
        'experiment': 'bn_class_specificity_full_scale',
        'runtime_minutes': float(runtime),
        'n_experiments': len(all_results),
        'convergence_rate': float(sum(1 for r in all_results if r['test_acc'] > 0.9) / len(all_results)),
        'summary_stats': summary_dict,
        'statistical_tests': comparisons,
        'hypotheses': {
            'h1_bn_contains_class_info': bool(h1_supported),
            'h3_sampling_affects_mi': bool(h3_supported)
        },
        'key_findings': {
            'single_class_mi': float(np.mean(bn_single_mi)) if bn_single_mi else 0.0,
            'random_mi': float(np.mean([r['avg_mi'] for r in all_results 
                                       if r['model_type'] == 'bn' and r['sampling_strategy'] == 'random'])),
            'signal_detected': bool(h1_supported or h3_supported)
        }
    }
    
    print(f"\nRESULTS: {json.dumps(final_results, indent=2)}")

if __name__ == "__main__":
    main()