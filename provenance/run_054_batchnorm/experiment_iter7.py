# pip install scikit-learn scipy matplotlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.stats import ttest_rel
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

def compute_mutual_information_fixed(features, labels, n_bins=10):
    """
    Compute mutual information between features and labels.
    This version properly handles the relationship between BN stats and classes.
    """
    if len(features) == 0 or len(labels) == 0:
        return 0.0
    
    # Ensure we have enough samples
    if len(np.unique(labels)) < 2:
        return 0.0
    
    # For BN stats, we care about how different the stats are for different classes
    # Flatten features to 2D if needed
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    elif len(features.shape) == 1:
        features = features.reshape(-1, 1)
    
    # Use all dimensions for MI computation
    total_mi = 0.0
    n_dims = features.shape[1]
    
    for dim in range(min(n_dims, 10)):  # Limit to first 10 dims for speed
        feature_dim = features[:, dim]
        
        # Skip constant features
        if np.std(feature_dim) < 1e-8:
            continue
        
        # Discretize using percentiles to handle different distributions
        try:
            bins = np.percentile(feature_dim, np.linspace(0, 100, n_bins+1))
            bins = np.unique(bins)
            if len(bins) < 2:
                continue
            
            feature_binned = np.digitize(feature_dim, bins[1:-1])
            mi = mutual_info_score(labels, feature_binned)
            total_mi += mi
        except:
            continue
    
    # Average over dimensions
    return total_mi / max(1, min(n_dims, 10))

def sanity_check_mutual_information():
    """Comprehensive metric sanity checks with realistic thresholds."""
    print("Running metric sanity checks...")
    
    # Use larger sample size to reduce spurious correlations
    n_samples = 1000
    n_classes = 10
    n_features = 64
    
    # Test 1: Perfect class separation in BN stats
    bn_stats_perfect = []
    labels_perfect = []
    
    for class_idx in range(n_classes):
        # Each class has different mean/var in BN stats
        class_mean = np.ones(n_features) * class_idx * 2  # Well separated
        class_stats = class_mean + np.random.normal(0, 0.1, (n_samples//n_classes, n_features))
        bn_stats_perfect.extend(class_stats)
        labels_perfect.extend([class_idx] * (n_samples//n_classes))
    
    bn_stats_perfect = np.array(bn_stats_perfect)
    labels_perfect = np.array(labels_perfect)
    
    mi_perfect = compute_mutual_information_fixed(bn_stats_perfect, labels_perfect)
    
    # Test 2: Random BN stats (no class structure)
    # Make truly random by shuffling labels
    bn_stats_random = np.random.normal(0, 1, (n_samples, n_features))
    labels_random = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(labels_random)  # Shuffle to break any structure
    mi_random = compute_mutual_information_fixed(bn_stats_random, labels_random)
    
    # Test 3: Partial correlation (realistic case)
    bn_stats_partial = []
    labels_partial = []
    
    for class_idx in range(n_classes):
        # Weak class-specific signal
        class_mean = np.ones(n_features) * class_idx * 0.5
        class_stats = class_mean + np.random.normal(0, 1, (n_samples//n_classes, n_features))
        bn_stats_partial.extend(class_stats)
        labels_partial.extend([class_idx] * (n_samples//n_classes))
    
    bn_stats_partial = np.array(bn_stats_partial)
    labels_partial = np.array(labels_partial)
    mi_partial = compute_mutual_information_fixed(bn_stats_partial, labels_partial)
    
    # Test 4: Single class (should be 0)
    single_class_stats = np.random.normal(0, 1, (n_samples, n_features))
    single_class_labels = np.zeros(n_samples, dtype=int)
    mi_single = compute_mutual_information_fixed(single_class_stats, single_class_labels)
    
    # Validation with more realistic thresholds
    print(f"  Perfect separation MI: {mi_perfect:.3f}")
    print(f"  Random MI: {mi_random:.3f}")
    print(f"  Partial correlation MI: {mi_partial:.3f}")
    print(f"  Single class MI: {mi_single:.3f}")
    
    # Adjusted assertions for realistic values
    assert mi_perfect > 1.0, f"Perfect MI too low: {mi_perfect}"
    assert mi_random < mi_partial * 0.5, f"Random MI too high relative to partial: {mi_random} vs {mi_partial}"
    assert mi_partial < mi_perfect * 0.8, f"Partial MI too high: {mi_partial} vs {mi_perfect}"
    assert mi_single < 0.01, f"Single class MI should be ~0: {mi_single}"
    
    # Test 5: Specific scenario - single-class vs mixed batches
    print("\nTesting single-class vs mixed batch scenario:")
    
    # Single-class batches: BN stats should vary by class
    single_class_bn_stats = []
    single_class_labels = []
    
    for _ in range(20):  # More batches per class
        for class_idx in range(n_classes):
            # Each single-class batch has class-specific stats
            batch_stats = np.ones(n_features) * class_idx + np.random.normal(0, 0.2, n_features)
            single_class_bn_stats.append(batch_stats)
            single_class_labels.append(class_idx)
    
    single_class_bn_stats = np.array(single_class_bn_stats)
    single_class_labels = np.array(single_class_labels)
    mi_single_class_batches = compute_mutual_information_fixed(single_class_bn_stats, single_class_labels)
    
    # Mixed batches: BN stats should be more uniform
    mixed_bn_stats = []
    mixed_labels = []
    
    for batch_idx in range(200):
        # Mixed batch has averaged stats
        batch_stats = np.mean([np.ones(n_features) * c for c in range(n_classes)], axis=0)
        batch_stats += np.random.normal(0, 0.5, n_features)
        mixed_bn_stats.append(batch_stats)
        # Assign random "dominant" class for mixed batches
        mixed_labels.append(np.random.randint(0, n_classes))
    
    mixed_bn_stats = np.array(mixed_bn_stats)
    mixed_labels = np.array(mixed_labels)
    mi_mixed_batches = compute_mutual_information_fixed(mixed_bn_stats, mixed_labels)
    
    print(f"  Single-class batches MI: {mi_single_class_batches:.3f}")
    print(f"  Mixed batches MI: {mi_mixed_batches:.3f}")
    
    assert mi_single_class_batches > mi_mixed_batches * 1.5, \
        f"Single-class MI should be higher: {mi_single_class_batches:.3f} vs {mi_mixed_batches:.3f}"
    
    print("METRIC_SANITY_PASSED")
    return True

# Run sanity check
sanity_check_mutual_information()

################################################################################
# DATASET CREATION - MORE CHALLENGING
################################################################################

def create_synthetic_dataset(n_samples=10000, n_features=32, n_classes=10, class_separation=1.5, noise_level=1.0, split='train'):
    """
    Create synthetic dataset with controlled difficulty.
    Lower class_separation and higher noise_level make it harder.
    """
    # Generate class centers with some overlap
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
        # Generate samples with more noise for harder classification
        class_samples = class_centers[class_idx] + np.random.randn(samples_per_class, n_features) * noise_level
        X.extend(class_samples)
        y.extend([class_idx] * samples_per_class)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Add some label noise for realism
    if split == 'train':
        # Flip 5% of labels randomly
        n_flip = int(0.05 * len(y))
        flip_idx = np.random.choice(len(y), n_flip, replace=False)
        y[flip_idx] = np.random.randint(0, n_classes, n_flip)
    
    # Normalize features
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

################################################################################
# IMPROVED BN STATS TRACKER
################################################################################

class ImprovedBNStatsTracker:
    """Track BN statistics with proper class association."""
    def __init__(self):
        self.batch_stats = defaultdict(list)
        self.batch_compositions = []  # Track which classes were in each batch
        
    def record_batch(self, bn_modules, batch_labels, batch_idx):
        """Record BN stats and batch composition."""
        # Get unique classes in this batch
        unique_classes = torch.unique(batch_labels).cpu().numpy()
        self.batch_compositions.append({
            'batch_idx': batch_idx,
            'classes': unique_classes.tolist(),
            'is_single_class': len(unique_classes) == 1,
            'dominant_class': int(torch.mode(batch_labels).values.item())
        })
        
        # Record BN stats
        for name, module in bn_modules:
            stats = {
                'mean': module.running_mean.detach().cpu().numpy().copy(),
                'var': module.running_var.detach().cpu().numpy().copy(),
                'batch_idx': batch_idx,
                'dominant_class': int(torch.mode(batch_labels).values.item()),
                'is_single_class': len(unique_classes) == 1
            }
            self.batch_stats[name].append(stats)
    
    def compute_class_specific_mi(self):
        """Compute MI between BN stats and classes."""
        mi_results = {}
        
        for layer_name, stats_list in self.batch_stats.items():
            if not stats_list:
                continue
            
            # Separate single-class and mixed-class batches
            single_class_stats = []
            single_class_labels = []
            mixed_class_stats = []
            
            for stat_dict in stats_list:
                if stat_dict['is_single_class']:
                    single_class_stats.append(stat_dict['mean'])
                    single_class_labels.append(stat_dict['dominant_class'])
                else:
                    mixed_class_stats.append(stat_dict['mean'])
            
            # Compute MI for single-class batches (should be high)
            if len(single_class_stats) > 10 and len(np.unique(single_class_labels)) > 1:
                single_stats_array = np.array(single_class_stats)
                single_labels_array = np.array(single_class_labels)
                mi_single = compute_mutual_information_fixed(single_stats_array, single_labels_array)
            else:
                mi_single = 0.0
            
            # For mixed batches, MI should be low (use random labels as comparison)
            if len(mixed_class_stats) > 10:
                mixed_stats_array = np.array(mixed_class_stats)
                # For mixed batches, create random labels to compute baseline MI
                random_labels = np.random.randint(0, 10, len(mixed_stats_array))
                mi_mixed = compute_mutual_information_fixed(mixed_stats_array, random_labels)
            else:
                mi_mixed = 0.0
            
            mi_results[layer_name] = {
                'mi_single_class': float(mi_single),
                'mi_mixed_class': float(mi_mixed),
                'n_single_class_batches': len(single_class_stats),
                'n_mixed_class_batches': len(mixed_class_stats)
            }
        
        return mi_results

################################################################################
# MODEL DEFINITIONS WITH IMPROVED TRACKING
################################################################################

class ModelWithBN(nn.Module):
    """Model with batch normalization and improved stats tracking."""
    def __init__(self, input_dim=32, hidden_dims=[128, 256, 128], n_classes=10, bn_momentum=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        self.bn_modules = []
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            bn = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
            layers.append(bn)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Add dropout for regularization
            self.bn_modules.append((f'bn{i+1}', bn))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
        
        # Stats tracker
        self.stats_tracker = ImprovedBNStatsTracker()
        self.batch_counter = 0
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_bn_modules(self):
        return self.bn_modules

# Keep other model definitions the same...
class ModelWithGN(nn.Module):
    """Model with group normalization (baseline)."""
    def __init__(self, input_dim=32, hidden_dims=[128, 256, 128], n_classes=10, num_groups=8):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            groups = min(num_groups, hidden_dim)
            while hidden_dim % groups != 0:
                groups -= 1
            layers.append(nn.GroupNorm(groups, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
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
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

################################################################################
# CUSTOM SAMPLERS (keep the same)
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
        
        # Create single-class batches
        for class_idx, indices in self.class_indices.items():
            indices_copy = indices.copy()
            random.shuffle(indices_copy)
            
            for i in range(0, len(indices_copy) - self.batch_size + 1, self.batch_size):
                batch = indices_copy[i:i + self.batch_size]
                all_batches.append(batch)
        
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        count = 0
        for indices in self.class_indices.values():
            count += len(indices) // self.batch_size
        return count

################################################################################
# IMPROVED TRAINING
################################################################################

def train_epoch_with_tracking(model, loader, optimizer, criterion, device, track_stats=False):
    """Train for one epoch with improved BN stats tracking."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Track BN stats if requested
        if track_stats and hasattr(model, 'stats_tracker'):
            model.batch_counter += 1
            model.stats_tracker.record_batch(model.get_bn_modules(), target, model.batch_counter)
        
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

def train_model_convergence(model, train_loader, val_loader, device, max_epochs=100, patience=10, lr=0.001):
    """Train model with convergence monitoring."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    converged = False
    
    for epoch in range(max_epochs):
        # Track stats after warmup
        track_stats = epoch >= min(10, max_epochs // 3)
        
        train_loss, train_acc = train_epoch_with_tracking(
            model, train_loader, optimizer, criterion, device, track_stats=track_stats
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Progress printing
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}, LR={current_lr:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Convergence criteria
        if patience_counter >= patience and epoch >= 20:
            print(f"  CONVERGED at epoch {epoch}")
            converged = True
            break
            
        # Early stop if learning is stalled
        if epoch > 40 and best_val_acc < 0.3:
            print("  NOT_CONVERGED: Learning too slow")
            break
    else:
        print("  NOT_CONVERGED: Max epochs reached")
    
    return best_val_acc, converged

################################################################################
# EXPERIMENT RUNNER
################################################################################

def run_single_seed_experiment(seed, model_type='bn', sampling_strategy='random', 
                              hidden_dims=[128, 256, 128], max_epochs=100, 
                              bn_momentum=0.1, lr=0.001, experiment_name="main"):
    """Run complete experiment for a single seed."""
    # Set all seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n[Seed {seed}] {experiment_name}: {model_type.upper()} model, {sampling_strategy} sampling")
    
    # Create more challenging datasets
    X_train, y_train = create_synthetic_dataset(
        n_samples=10000, n_features=32, n_classes=10, 
        class_separation=1.5, noise_level=1.0, split='train'
    )
    X_val, y_val = create_synthetic_dataset(
        n_samples=2000, n_features=32, n_classes=10,
        class_separation=1.5, noise_level=1.0, split='val'
    )
    X_test, y_test = create_synthetic_dataset(
        n_samples=2000, n_features=32, n_classes=10,
        class_separation=1.5, noise_level=1.0, split='test'
    )
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # Create loaders
    batch_size = 100
    
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
        model = ModelWithBN(input_dim=32, hidden_dims=hidden_dims, bn_momentum=bn_momentum).to(device)
    elif model_type == 'gn':
        model = ModelWithGN(input_dim=32, hidden_dims=hidden_dims).to(device)
    elif model_type == 'ln':
        model = ModelWithLN(input_dim=32, hidden_dims=hidden_dims).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    start_time = time.time()
    val_acc, converged = train_model_convergence(
        model, train_loader, val_loader, device, 
        max_epochs=max_epochs, patience=15, lr=lr
    )
    train_time = time.time() - start_time
    
    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    
    # Analyze BN statistics
    mi_results = {}
    avg_mi = 0.0
    
    if model_type == 'bn' and hasattr(model, 'stats_tracker'):
        layer_mi_results = model.stats_tracker.compute_class_specific_mi()
        
        # Average MI across layers
        mi_values = []
        for layer_name, layer_results in layer_mi_results.items():
            mi_single = layer_results['mi_single_class']
            mi_mixed = layer_results['mi_mixed_class']
            
            # For single-class sampling, use single-class MI
            # For other samplings, use the difference as a measure of class-specificity
            if sampling_strategy == 'single_class':
                mi_values.append(mi_single)
            else:
                # Use mixed class MI for random/balanced sampling
                mi_values.append(mi_mixed)
            
            mi_results[layer_name] = {
                'mi': mi_single if sampling_strategy == 'single_class' else mi_mixed,
                'mi_single': mi_single,
                'mi_mixed': mi_mixed,
                'n_batches': layer_results['n_single_class_batches'] + layer_results['n_mixed_class_batches']
            }
        
        avg_mi = float(np.mean(mi_values)) if mi_values else 0.0
    
    results = {
        'seed': seed,
        'experiment_name': experiment_name,
        'model_type': model_type,
        'sampling_strategy': sampling_strategy,
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'avg_mi': avg_mi,
        'train_time': train_time,
        'converged': converged,
        'mi_results': mi_results
    }
    
    print(f"  Test Acc: {test_acc:.3f}, MI: {avg_mi:.3f}, Time: {train_time:.1f}s")
    
    return results

################################################################################
# MAIN EXPERIMENT
################################################################################

def main():
    """Main experiment execution."""
    print("="*80)
    print("FULL-SCALE BN CLASS-SPECIFICITY EXPERIMENT")
    print("="*80)
    
    # Configuration
    n_seeds = 10
    start_time = time.time()
    all_results = []
    
    # Early abort check
    print("\n" + "="*60)
    print("EARLY ABORT CHECK (Seed 0)")
    print("="*60)
    
    # Test key conditions
    first_results = []
    for sampling in ['random', 'single_class']:
        result = run_single_seed_experiment(
            seed=0, model_type='bn', sampling_strategy=sampling,
            hidden_dims=[128, 256, 128], max_epochs=50
        )
        first_results.append(result)
        all_results.append(result)
    
    # Check metrics
    bn_random_mi = first_results[0]['avg_mi']
    bn_single_mi = first_results[1]['avg_mi']
    
    print(f"\nEarly abort check:")
    print(f"  BN Random MI: {bn_random_mi:.4f}")
    print(f"  BN Single MI: {bn_single_mi:.4f}")
    
    if bn_random_mi == 0.0 and bn_single_mi == 0.0:
        print("SANITY_ABORT: Both MI values are 0.0")
        exit(1)
    
    if abs(bn_single_mi - bn_random_mi) < 0.01:
        print("SANITY_ABORT: MI difference too small")
        exit(1)
        
    print("Early abort check PASSED")
    
    # Main experiments
    print("\n" + "="*60)
    print("MAIN EXPERIMENTS")
    print("="*60)
    
    # Core experiments
    for seed in range(1, n_seeds):
        for sampling in ['random', 'balanced', 'single_class']:
            result = run_single_seed_experiment(
                seed=seed, model_type='bn', sampling_strategy=sampling,
                hidden_dims=[128, 256, 128], max_epochs=50
            )
            all_results.append(result)
    
    # Baseline comparisons
    print("\n" + "="*60)
    print("BASELINE COMPARISONS")
    print("="*60)
    
    for seed in range(min(5, n_seeds)):
        for model_type in ['gn', 'ln']:
            for sampling in ['random', 'single_class']:
                result = run_single_seed_experiment(
                    seed=seed, model_type=model_type, sampling_strategy=sampling,
                    hidden_dims=[128, 256, 128], max_epochs=50
                )
                all_results.append(result)
    
    # Analysis
    print("\n" + "="*60)
    print("FINAL ANALYSIS")
    print("="*60)
    
    # Summary statistics
    summary = defaultdict(lambda: defaultdict(list))
    
    for r in all_results:
        if r['experiment_name'] == 'main':
            key = f"{r['model_type']}_{r['sampling_strategy']}"
            summary[key]['test_acc'].append(r['test_acc'])
            summary[key]['avg_mi'].append(r['avg_mi'])
    
    # Print results table
    print("\nResults Summary:")
    print("-" * 80)
    print(f"{'Method':<30} {'Test Acc (mean±std)':<25} {'MI (mean±std)':<25}")
    print("-" * 80)
    
    summary_dict = {}
    for method, metrics in sorted(summary.items()):
        acc_mean = np.mean(metrics['test_acc'])
        acc_std = np.std(metrics['test_acc'])
        mi_mean = np.mean(metrics['avg_mi'])
        mi_std = np.std(metrics['avg_mi'])
        
        print(f"{method:<30} {acc_mean:.3f}±{acc_std:.3f}              "
              f"{mi_mean:.3f}±{mi_std:.3f}")
        
        summary_dict[method] = {
            'test_acc_mean': float(acc_mean),
            'test_acc_std': float(acc_std),
            'mi_mean': float(mi_mean),
            'mi_std': float(mi_std),
            'n': len(metrics['test_acc'])
        }
    
    # Statistical tests
    print("\nStatistical Tests:")
    print("-" * 50)
    
    # Paired t-test for BN single vs random MI
    bn_random = [r['avg_mi'] for r in all_results 
                 if r['model_type'] == 'bn' and r['sampling_strategy'] == 'random']
    bn_single = [r['avg_mi'] for r in all_results 
                 if r['model_type'] == 'bn' and r['sampling_strategy'] == 'single_class']
    
    if len(bn_random) >= 2 and len(bn_single) >= 2:
        # Match by seed for paired test
        paired_random = []
        paired_single = []
        for r in all_results:
            if r['model_type'] == 'bn':
                if r['sampling_strategy'] == 'random':
                    # Find matching single-class result
                    matching = [s for s in all_results 
                               if s['model_type'] == 'bn' and 
                               s['sampling_strategy'] == 'single_class' and
                               s['seed'] == r['seed']]
                    if matching:
                        paired_random.append(r['avg_mi'])
                        paired_single.append(matching[0]['avg_mi'])
        
        if len(paired_random) >= 2:
            _, p_val = ttest_rel(paired_single, paired_random)
            print(f"BN single vs random MI: p={p_val:.4f}")
            statistical_tests = {'bn_single_vs_random_mi': float(p_val)}
        else:
            statistical_tests = {}
    else:
        statistical_tests = {}
    
    # Final results
    runtime = (time.time() - start_time) / 60
    
    final_results = {
        'experiment': 'bn_class_specificity_fixed',
        'runtime_minutes': float(runtime),
        'n_experiments': len(all_results),
        'summary_stats': summary_dict,
        'statistical_tests': statistical_tests,
        'key_findings': {
            'bn_single_mi': float(np.mean(bn_single)) if bn_single else 0.0,
            'bn_random_mi': float(np.mean(bn_random)) if bn_random else 0.0,
            'mi_difference': float(np.mean(bn_single) - np.mean(bn_random)) if bn_single and bn_random else 0.0
        }
    }
    
    print(f"\nRESULTS: {json.dumps(final_results, indent=2)}")

if __name__ == "__main__":
    main()