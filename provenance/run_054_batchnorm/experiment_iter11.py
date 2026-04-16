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

def compute_mutual_information_fixed(features, labels, n_bins=15):
    """
    Compute mutual information between features and labels.
    Higher n_bins for better resolution.
    """
    if len(features) == 0 or len(labels) == 0:
        return 0.0
    
    # Ensure we have enough unique labels
    if len(np.unique(labels)) < 2:
        return 0.0
    
    # Handle different feature shapes
    if len(features.shape) > 2:
        features = features.reshape(features.shape[0], -1)
    elif len(features.shape) == 1:
        features = features.reshape(-1, 1)
    
    # Compute MI for each dimension and average
    total_mi = 0.0
    valid_dims = 0
    n_dims = min(features.shape[1], 20)  # Limit dimensions for speed
    
    for dim in range(n_dims):
        feature_dim = features[:, dim]
        
        # Skip constant features
        if np.std(feature_dim) < 1e-8:
            continue
        
        try:
            # Use adaptive binning based on data distribution
            percentiles = np.linspace(0, 100, n_bins+1)
            bins = np.percentile(feature_dim, percentiles)
            bins = np.unique(bins)
            
            if len(bins) >= 2:
                # Add small noise to avoid identical values
                feature_dim_noisy = feature_dim + np.random.normal(0, 1e-10, len(feature_dim))
                feature_binned = np.digitize(feature_dim_noisy, bins[1:-1])
                
                # Compute MI
                mi = mutual_info_score(labels, feature_binned)
                total_mi += mi
                valid_dims += 1
        except Exception as e:
            continue
    
    # Average over valid dimensions
    return total_mi / max(1, valid_dims)

def sanity_check_mutual_information():
    """Comprehensive metric sanity checks."""
    print("Running metric sanity checks...")
    
    # Use larger sample size
    n_samples = 2000
    n_classes = 10
    n_features = 64
    
    # Test 1: Perfect class separation
    print("\nTest 1: Perfect class separation")
    perfect_stats = []
    perfect_labels = []
    
    for class_idx in range(n_classes):
        # Each class has very different statistics
        class_mean = np.ones(n_features) * (class_idx * 5)  # Large separation
        class_stats = class_mean + np.random.normal(0, 0.1, (n_samples//n_classes, n_features))
        perfect_stats.extend(class_stats)
        perfect_labels.extend([class_idx] * (n_samples//n_classes))
    
    perfect_stats = np.array(perfect_stats)
    perfect_labels = np.array(perfect_labels)
    mi_perfect = compute_mutual_information_fixed(perfect_stats, perfect_labels)
    print(f"  Perfect separation MI: {mi_perfect:.3f}")
    
    # Test 2: Random (shuffled labels)
    print("\nTest 2: Random baseline")
    random_stats = np.random.normal(0, 1, (n_samples, n_features))
    random_labels = np.repeat(np.arange(n_classes), n_samples // n_classes)
    np.random.shuffle(random_labels)  # Shuffle to break structure
    mi_random = compute_mutual_information_fixed(random_stats, random_labels)
    print(f"  Random MI: {mi_random:.3f}")
    
    # Test 3: Realistic BN scenario - class-specific stats
    print("\nTest 3: Realistic BN stats")
    bn_stats = []
    bn_labels = []
    
    # Simulate BN stats from single-class batches
    for batch_idx in range(100):
        # Each batch comes from a single class
        class_idx = batch_idx % n_classes
        
        # BN stats reflect the class
        batch_mean = np.ones(n_features) * class_idx + np.random.normal(0, 0.3, n_features)
        batch_var = np.ones(n_features) * (1 + class_idx * 0.1) + np.random.normal(0, 0.1, n_features)
        
        # Concatenate mean and var
        batch_stats = np.concatenate([batch_mean, batch_var])
        
        bn_stats.append(batch_stats)
        bn_labels.append(class_idx)
    
    bn_stats = np.array(bn_stats)
    bn_labels = np.array(bn_labels)
    mi_bn = compute_mutual_information_fixed(bn_stats, bn_labels)
    print(f"  BN single-class MI: {mi_bn:.3f}")
    
    # Test 4: Mixed batches
    print("\nTest 4: Mixed batch BN stats")
    mixed_stats = []
    mixed_labels = []
    
    for batch_idx in range(100):
        # Mixed batches have averaged statistics
        batch_mean = np.mean([np.ones(n_features) * c for c in range(n_classes)], axis=0)
        batch_mean += np.random.normal(0, 0.5, n_features)
        batch_var = np.ones(n_features) + np.random.normal(0, 0.2, n_features)
        
        batch_stats = np.concatenate([batch_mean, batch_var])
        mixed_stats.append(batch_stats)
        # Assign random class label
        mixed_labels.append(np.random.randint(0, n_classes))
    
    mixed_stats = np.array(mixed_stats)
    mixed_labels = np.array(mixed_labels)
    mi_mixed = compute_mutual_information_fixed(mixed_stats, mixed_labels)
    print(f"  BN mixed-batch MI: {mi_mixed:.3f}")
    
    # Validate results
    assert mi_perfect > 1.5, f"Perfect MI too low: {mi_perfect}"
    assert mi_random < mi_bn * 0.3, f"Random MI too high: {mi_random} vs {mi_bn}"
    assert mi_bn > mi_mixed * 2, f"Single-class MI not higher than mixed: {mi_bn} vs {mi_mixed}"
    
    print("\nMETRIC_SANITY_PASSED")
    return True

# Run sanity check
sanity_check_mutual_information()

################################################################################
# IMPROVED DATASET CREATION
################################################################################

def create_challenging_dataset(n_samples=10000, n_features=32, n_classes=10, difficulty='medium', split='train'):
    """
    Create a synthetic dataset with controlled difficulty.
    Changed to 'medium' difficulty for better single-class training.
    """
    # Fixed class structure across splits
    np.random.seed(42)
    
    if difficulty == 'hard':
        # Overlapping classes
        class_centers = np.random.randn(n_classes, n_features) * 0.8
        noise_scale = 1.2
    elif difficulty == 'medium':
        # Moderate separation - better for single-class training
        class_centers = np.random.randn(n_classes, n_features) * 1.5
        noise_scale = 0.8
    else:
        # Well-separated classes
        class_centers = np.random.randn(n_classes, n_features) * 2.0
        noise_scale = 0.5
    
    # Different random seeds for different splits
    split_seeds = {'train': 100, 'val': 200, 'test': 300}
    np.random.seed(split_seeds.get(split, 100))
    
    # Generate samples
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        # Add class-specific samples with noise
        class_samples = class_centers[class_idx] + np.random.randn(samples_per_class, n_features) * noise_scale
        
        # Add fewer outliers for single-class training
        n_outliers = int(0.05 * samples_per_class)  # Reduced from 0.1
        outlier_idx = np.random.choice(samples_per_class, n_outliers, replace=False)
        class_samples[outlier_idx] += np.random.randn(n_outliers, n_features) * 1.5
        
        X.extend(class_samples)
        y.extend([class_idx] * samples_per_class)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Less label noise for clearer signal
    if split == 'train':
        n_flip = int(0.05 * len(y))  # Reduced from 0.1
        flip_idx = np.random.choice(len(y), n_flip, replace=False)
        y[flip_idx] = np.random.randint(0, n_classes, n_flip)
    
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

################################################################################
# ENHANCED BN STATS TRACKER
################################################################################

class ImprovedBNStatsTracker:
    """Track actual batch statistics during training."""
    def __init__(self):
        self.batch_stats_history = []
        self.batch_labels_history = []
        self.layer_wise_stats = defaultdict(list)
        
    def record_batch_stats(self, model, data, labels):
        """Record the actual batch statistics during forward pass."""
        batch_stats = {}
        hooks = []
        
        def get_batch_stats_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'running_mean'):
                    # Compute batch stats from input
                    x = input[0]
                    if len(x.shape) == 2:  # FC layer
                        batch_mean = x.mean(dim=0)
                        batch_var = x.var(dim=0, unbiased=False)
                    else:
                        batch_mean = module.running_mean
                        batch_var = module.running_var
                    
                    batch_stats[name] = {
                        'mean': batch_mean.detach().cpu().numpy(),
                        'var': batch_var.detach().cpu().numpy()
                    }
                    
                    # Store layer-wise stats
                    self.layer_wise_stats[name].append({
                        'mean': batch_mean.detach().cpu().numpy(),
                        'var': batch_var.detach().cpu().numpy(),
                        'dominant_class': int(torch.mode(labels).values.item()),
                        'is_single_class': len(torch.unique(labels)) == 1
                    })
            return hook
        
        # Register hooks
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm1d):
                hooks.append(module.register_forward_hook(get_batch_stats_hook(name)))
        
        # Forward pass to trigger hooks
        model.eval()  # Don't update running stats
        with torch.no_grad():
            _ = model(data)
        model.train()
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Record stats
        if batch_stats:
            # Concatenate all BN stats into one vector
            all_stats = []
            for layer_name in sorted(batch_stats.keys()):
                stats = batch_stats[layer_name]
                all_stats.append(stats['mean'])
                all_stats.append(stats['var'])
            
            combined_stats = np.concatenate(all_stats)
            
            # Get batch composition
            unique_classes = torch.unique(labels).cpu().numpy()
            is_single_class = len(unique_classes) == 1
            dominant_class = int(torch.mode(labels).values.item())
            
            self.batch_stats_history.append({
                'stats': combined_stats,
                'is_single_class': is_single_class,
                'dominant_class': dominant_class,
                'unique_classes': unique_classes.tolist()
            })
            self.batch_labels_history.append(dominant_class)
    
    def compute_mi(self):
        """Compute MI for different batch types."""
        if len(self.batch_stats_history) < 10:
            return 0.0, {}
        
        # Separate single-class and mixed batches
        single_class_stats = []
        single_class_labels = []
        mixed_class_stats = []
        mixed_class_labels = []
        
        for batch_info in self.batch_stats_history:
            if batch_info['is_single_class']:
                single_class_stats.append(batch_info['stats'])
                single_class_labels.append(batch_info['dominant_class'])
            else:
                mixed_class_stats.append(batch_info['stats'])
                mixed_class_labels.append(batch_info['dominant_class'])
        
        # Compute MI for single-class batches
        mi_single = 0.0
        if len(single_class_stats) > 20 and len(np.unique(single_class_labels)) > 1:
            stats_array = np.array(single_class_stats)
            labels_array = np.array(single_class_labels)
            mi_single = compute_mutual_information_fixed(stats_array, labels_array)
        
        # For mixed batches, compute MI with actual dominant class
        mi_mixed = 0.0
        if len(mixed_class_stats) > 20 and len(np.unique(mixed_class_labels)) > 1:
            stats_array = np.array(mixed_class_stats)
            labels_array = np.array(mixed_class_labels)
            mi_mixed = compute_mutual_information_fixed(stats_array, labels_array)
        
        info = {
            'mi_single': mi_single,
            'mi_mixed': mi_mixed,
            'n_single_batches': len(single_class_stats),
            'n_mixed_batches': len(mixed_class_stats),
            'n_unique_classes_seen': len(np.unique(single_class_labels)) if single_class_stats else 0
        }
        
        # Return appropriate MI based on what we have
        if len(single_class_stats) > len(mixed_class_stats):
            return mi_single, info
        else:
            return mi_mixed, info
    
    def compute_layer_wise_mi(self):
        """Compute MI for each layer separately."""
        layer_mi_results = {}
        
        for layer_name, layer_stats in self.layer_wise_stats.items():
            # Only use single-class batches for clearer signal
            single_class_stats = []
            single_class_labels = []
            
            for stat_dict in layer_stats:
                if stat_dict['is_single_class']:
                    combined = np.concatenate([stat_dict['mean'], stat_dict['var']])
                    single_class_stats.append(combined)
                    single_class_labels.append(stat_dict['dominant_class'])
            
            if len(single_class_stats) > 20 and len(np.unique(single_class_labels)) > 1:
                stats_array = np.array(single_class_stats)
                labels_array = np.array(single_class_labels)
                mi = compute_mutual_information_fixed(stats_array, labels_array)
                layer_mi_results[layer_name] = mi
        
        return layer_mi_results

################################################################################
# MODEL DEFINITIONS WITH IMPROVED ARCHITECTURE
################################################################################

class ModelWithBN(nn.Module):
    """Model with batch normalization and batch stats tracking."""
    def __init__(self, input_dim=32, hidden_dims=[128, 256, 128], n_classes=10, bn_momentum=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        self.bn_layers = []
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            bn = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum, track_running_stats=True)
            layers.append(bn)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Reduced dropout for single-class training
            self.bn_layers.append(bn)
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        # Better initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Stats tracker
        self.stats_tracker = ImprovedBNStatsTracker()
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ModelWithGN(nn.Module):
    """Model with group normalization (baseline)."""
    def __init__(self, input_dim=32, hidden_dims=[128, 256, 128], n_classes=10, num_groups=16):
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class ModelNoBN(nn.Module):
    """Model without normalization."""
    def __init__(self, input_dim=32, hidden_dims=[128, 256, 128], n_classes=10):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            prev_dim = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

################################################################################
# IMPROVED SAMPLERS
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

class ImprovedSingleClassBatchSampler:
    """
    IMPROVED: Cycles through classes to ensure each class is seen regularly.
    This fixes the poor single-class training performance.
    """
    def __init__(self, labels, batch_size, n_classes=10):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_classes = n_classes
        
        # Group indices by class
        self.class_indices = {}
        for c in range(n_classes):
            self.class_indices[c] = np.where(self.labels == c)[0].tolist()
    
    def __iter__(self):
        # Create batches for each class
        class_batches = defaultdict(list)
        
        for class_idx, indices in self.class_indices.items():
            indices_copy = indices.copy()
            random.shuffle(indices_copy)
            
            # Create batches for this class
            for i in range(0, len(indices_copy) - self.batch_size + 1, self.batch_size):
                batch = indices_copy[i:i + self.batch_size]
                class_batches[class_idx].append(batch)
        
        # Interleave batches from different classes
        all_batches = []
        max_batches = max(len(batches) for batches in class_batches.values())
        
        for batch_idx in range(max_batches):
            # Cycle through classes
            for class_idx in range(self.n_classes):
                if batch_idx < len(class_batches[class_idx]):
                    all_batches.append(class_batches[class_idx][batch_idx])
        
        # Return batches
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
    """Train for one epoch with batch stats tracking."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # Track batch stats BEFORE optimizer step, sample more frequently
        if track_stats and hasattr(model, 'stats_tracker') and batch_idx % 3 == 0:  # More frequent
            model.stats_tracker.record_batch_stats(model, data, target)
        
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
    
    return total_loss / total, correct / total

def train_model_convergence(model, train_loader, val_loader, device, sampling_strategy, 
                           max_epochs=100, patience=15):
    """Train with convergence-based stopping."""
    # IMPROVED: Adjusted hyperparameters for single-class training
    if sampling_strategy == 'single_class':
        lr = 0.005  # Lower than before
        weight_decay = 5e-4
        min_epochs = 30  # Ensure enough training
    else:
        lr = 0.001
        weight_decay = 1e-4
        min_epochs = 20
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Use cosine annealing for single-class to help convergence
    if sampling_strategy == 'single_class':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    converged = False
    
    for epoch in range(max_epochs):
        # Track stats after warmup
        track_stats = epoch >= 10 and epoch < max_epochs - 5
        
        train_loss, train_acc = train_epoch_with_tracking(
            model, train_loader, optimizer, criterion, device, track_stats=track_stats
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Step scheduler
        if sampling_strategy == 'single_class':
            scheduler.step()
        else:
            scheduler.step(val_acc)
        
        # Progress every 10 epochs
        if epoch % 10 == 0 or (epoch < 10 and epoch % 2 == 0):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch}: Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}, LR={current_lr:.6f}")
        
        # Check convergence
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
        
        # Early stopping (but not too early for single-class)
        if patience_counter >= patience and epoch >= min_epochs:
            print(f"    CONVERGED at epoch {epoch}")
            converged = True
            break
    else:
        if epoch >= max_epochs - 1:
            print("    NOT_CONVERGED: Max epochs reached")
    
    return best_val_acc, converged

################################################################################
# EXPERIMENT RUNNER
################################################################################

def run_single_seed_experiment(seed, model_type='bn', sampling_strategy='random', 
                              hidden_dims=[256, 512, 256], max_epochs=100, 
                              bn_momentum=0.1, experiment_name="main"):
    """Run single seed experiment."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n[Seed {seed}] {experiment_name}: {model_type.upper()}, {sampling_strategy} sampling")
    
    # Create dataset with medium difficulty
    X_train, y_train = create_challenging_dataset(
        n_samples=10000, n_features=32, n_classes=10, 
        difficulty='medium', split='train'  # Changed from 'hard'
    )
    X_val, y_val = create_challenging_dataset(
        n_samples=2000, n_features=32, n_classes=10,
        difficulty='medium', split='val'
    )
    X_test, y_test = create_challenging_dataset(
        n_samples=2000, n_features=32, n_classes=10,
        difficulty='medium', split='test'
    )
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    # Create loaders
    batch_size = 128
    
    if sampling_strategy == 'random':
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    elif sampling_strategy == 'balanced':
        sampler = BalancedBatchSampler(y_train, batch_size=120)  # Divisible by 10
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    elif sampling_strategy == 'single_class':
        sampler = ImprovedSingleClassBatchSampler(y_train, batch_size=batch_size)  # Use improved sampler
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
    elif model_type == 'no_norm':
        model = ModelNoBN(input_dim=32, hidden_dims=hidden_dims).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train
    start_time = time.time()
    val_acc, converged = train_model_convergence(
        model, train_loader, val_loader, device, 
        sampling_strategy, max_epochs=max_epochs
    )
    train_time = time.time() - start_time
    
    # Test
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    
    # Analyze BN statistics
    avg_mi = 0.0
    mi_info = {}
    layer_mi = {}
    
    if model_type == 'bn' and hasattr(model, 'stats_tracker'):
        avg_mi, mi_info = model.stats_tracker.compute_mi()
        layer_mi = model.stats_tracker.compute_layer_wise_mi()
    
    results = {
        'seed': seed,
        'experiment_name': experiment_name,
        'model_type': model_type,
        'sampling_strategy': sampling_strategy,
        'val_acc': float(val_acc),
        'test_acc': float(test_acc),
        'avg_mi': float(avg_mi),
        'train_time': train_time,
        'converged': converged,
        'mi_info': mi_info,
        'layer_mi': layer_mi,
        'hidden_dims': hidden_dims,
        'bn_momentum': bn_momentum if model_type == 'bn' else None
    }
    
    print(f"  Test Acc: {test_acc:.3f}, MI: {avg_mi:.3f}, Time: {train_time:.1f}s")
    
    return results

################################################################################
# STATISTICAL ANALYSIS
################################################################################

def compute_bootstrap_ci(data, n_bootstrap=1000, confidence=0.95):
    """Compute bootstrap confidence intervals."""
    if len(data) < 2:
        return np.mean(data), np.mean(data), np.mean(data)
    
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(resampled))
    
    # Compute percentiles
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return np.mean(data), lower, upper

################################################################################
# MAIN EXPERIMENT
################################################################################

def main():
    """Main experiment execution."""
    print("="*80)
    print("IMPROVED BN CLASS-SPECIFICITY EXPERIMENT")
    print("="*80)
    
    # Configuration
    n_seeds = 10
    start_time = time.time()
    all_results = []
    
    # Early abort check
    print("\n" + "="*60)
    print("EARLY ABORT CHECK (Seed 0)")
    print("="*60)
    
    # Test critical conditions first
    first_results = []
    for sampling in ['random', 'single_class']:
        result = run_single_seed_experiment(
            seed=0, model_type='bn', sampling_strategy=sampling,
            hidden_dims=[256, 512, 256], max_epochs=60  # More epochs for single-class
        )
        first_results.append(result)
        all_results.append(result)
    
    # Validate metrics
    bn_random_mi = first_results[0]['avg_mi']
    bn_single_mi = first_results[1]['avg_mi']
    bn_random_acc = first_results[0]['test_acc']
    bn_single_acc = first_results[1]['test_acc']
    
    print(f"\nEarly abort check:")
    print(f"  BN Random: MI={bn_random_mi:.3f}, Acc={bn_random_acc:.3f}")
    print(f"  BN Single: MI={bn_single_mi:.3f}, Acc={bn_single_acc:.3f}")
    print(f"  MI Lift: {bn_single_mi / (bn_random_mi + 1e-8):.2f}x")
    
    if bn_random_mi == 0.0 and bn_single_mi == 0.0:
        print("SANITY_ABORT: Both MI values are 0.0")
        exit(1)
    
    if abs(bn_single_mi - bn_random_mi) < 0.01:
        print("SANITY_ABORT: MI difference too small")
        exit(1)
    
    print("Early abort check PASSED\n")
    
    # Main experiments
    print("="*60)
    print("MAIN EXPERIMENTS")
    print("="*60)
    
    # 1. Core experiments with all sampling strategies
    print("\n1. Core experiments (10 seeds)")
    for seed in range(1, n_seeds):  # Already did seed 0
        for sampling in ['random', 'balanced', 'single_class']:
            result = run_single_seed_experiment(
                seed=seed, model_type='bn', sampling_strategy=sampling,
                hidden_dims=[256, 512, 256], max_epochs=60
            )
            all_results.append(result)
    
    # 2. Baseline comparisons (fewer seeds)
    print("\n2. Baseline comparisons (5 seeds)")
    for seed in range(min(5, n_seeds)):
        for model_type in ['gn', 'ln', 'no_norm']:
            for sampling in ['random', 'single_class']:
                result = run_single_seed_experiment(
                    seed=seed, model_type=model_type, sampling_strategy=sampling,
                    hidden_dims=[256, 512, 256], max_epochs=50
                )
                all_results.append(result)
    
    # 3. Ablation studies
    print("\n" + "="*60)
    print("ABLATION STUDIES")
    print("="*60)
    
    # Ablation 1: Architecture depth
    print("\n3a. Architecture depth ablation (5 seeds)")
    for seed in range(5):
        # Shallow
        result = run_single_seed_experiment(
            seed=seed, model_type='bn', sampling_strategy='single_class',
            hidden_dims=[512], max_epochs=40,
            experiment_name="ablation_shallow"
        )
        all_results.append(result)
        
        # Deep
        result = run_single_seed_experiment(
            seed=seed, model_type='bn', sampling_strategy='single_class',
            hidden_dims=[128, 256, 512, 256, 128], max_epochs=40,
            experiment_name="ablation_deep"
        )
        all_results.append(result)
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS AND RESULTS")
    print("="*60)
    
    # Extract main results
    main_results = [r for r in all_results if r['experiment_name'] == 'main']
    
    # Summary statistics
    summary = defaultdict(lambda: defaultdict(list))
    
    for r in main_results:
        key = f"{r['model_type']}_{r['sampling_strategy']}"
        summary[key]['test_acc'].append(r['test_acc'])
        summary[key]['avg_mi'].append(r['avg_mi'])
        summary[key]['converged'].append(r['converged'])
    
    # Print main results table
    print("\nMain Results (mean [95% CI]):")
    print("-" * 100)
    print(f"{'Method':<30} {'Test Acc':<35} {'MI':<35} {'Conv%'}")
    print("-" * 100)
    
    summary_dict = {}
    for method, metrics in sorted(summary.items()):
        # Bootstrap CIs
        acc_mean, acc_lower, acc_upper = compute_bootstrap_ci(metrics['test_acc'])
        mi_mean, mi_lower, mi_upper = compute_bootstrap_ci(metrics['avg_mi'])
        conv_rate = sum(metrics['converged']) / len(metrics['converged']) * 100
        
        print(f"{method:<30} {acc_mean:.3f} [{acc_lower:.3f}, {acc_upper:.3f}]      "
              f"{mi_mean:.3f} [{mi_lower:.3f}, {mi_upper:.3f}]      {conv_rate:.0f}%")
        
        summary_dict[method] = {
            'test_acc_mean': float(acc_mean),
            'test_acc_ci': [float(acc_lower), float(acc_upper)],
            'mi_mean': float(mi_mean),
            'mi_ci': [float(mi_lower), float(mi_upper)],
            'convergence_rate': float(conv_rate / 100),
            'n': len(metrics['test_acc'])
        }
    
    # Statistical tests
    print("\nStatistical Tests:")
    print("-" * 60)
    
    # Initialize variables
    p_val = None
    t_stat = None
    cohens_d = None
    mi_lift = None
    
    # Paired t-test for key comparison
    bn_random_mi = []
    bn_single_mi = []
    
    for seed in range(n_seeds):
        random_results = [r for r in main_results if r['seed'] == seed and 
                         r['model_type'] == 'bn' and r['sampling_strategy'] == 'random']
        single_results = [r for r in main_results if r['seed'] == seed and 
                         r['model_type'] == 'bn' and r['sampling_strategy'] == 'single_class']
        
        if random_results and single_results:
            bn_random_mi.append(random_results[0]['avg_mi'])
            bn_single_mi.append(single_results[0]['avg_mi'])
    
    if len(bn_random_mi) >= 2:
        t_stat, p_val = ttest_rel(bn_single_mi, bn_random_mi)
        print(f"BN single vs random MI (paired t-test): t={t_stat:.3f}, p={p_val:.6f}")
        
        # Effect size (Cohen's d)
        diff = np.array(bn_single_mi) - np.array(bn_random_mi)
        cohens_d = np.mean(diff) / (np.std(diff) + 1e-8)
        print(f"Effect size (Cohen's d): {cohens_d:.3f}")
        
        # MI lift
        mi_lift = np.mean(bn_single_mi) / (np.mean(bn_random_mi) + 1e-8)
        print(f"MI lift (single/random): {mi_lift:.2f}x")
    
    # Layer-wise analysis
    print("\nLayer-wise MI Analysis (BN single-class):")
    print("-" * 60)
    
    layer_mi_all = defaultdict(list)
    for r in main_results:
        if r['model_type'] == 'bn' and r['sampling_strategy'] == 'single_class':
            for layer, mi in r.get('layer_mi', {}).items():
                layer_mi_all[layer].append(mi)
    
    layer_mis_ordered = []
    for layer in sorted(layer_mi_all.keys()):
        mi_values = layer_mi_all[layer]
        if mi_values:
            mean_mi = np.mean(mi_values)
            std_mi = np.std(mi_values)
            print(f"{layer:<30} MI: {mean_mi:.3f} ± {std_mi:.3f}")
            layer_mis_ordered.append(mean_mi)
    
    # Hypothesis testing
    print("\nHypothesis Testing:")
    print("-" * 60)
    
    h1_supported = summary_dict.get('bn_single_class', {}).get('mi_mean', 0) > 0.3
    print(f"H1 (BN contains class info): {'SUPPORTED' if h1_supported else 'NOT SUPPORTED'} "
          f"(MI = {summary_dict.get('bn_single_class', {}).get('mi_mean', 0):.3f})")
    
    # Check layer-wise pattern
    if len(layer_mis_ordered) >= 3:
        # Check if early layers have higher MI than later layers (expected pattern)
        h2_early_higher = layer_mis_ordered[0] > layer_mis_ordered[-1]
        print(f"H2 (early layers encode more class info): {'SUPPORTED' if h2_early_higher else 'NOT SUPPORTED'} "
              f"(Early: {layer_mis_ordered[0]:.3f}, Late: {layer_mis_ordered[-1]:.3f})")
    else:
        h2_early_higher = False
        print("H2: INSUFFICIENT DATA")
    
    h3_supported = p_val is not None and p_val < 0.001
    p_val_str = f"{p_val:.6f}" if p_val is not None else "N/A"
    print(f"H3 (sampling affects MI): {'SUPPORTED' if h3_supported else 'NOT SUPPORTED'} "
          f"(p = {p_val_str})")
    
    # Generate final JSON
    runtime = (time.time() - start_time) / 60
    
    final_results = {
        'experiment': 'bn_class_specificity_improved',
        'runtime_minutes': float(runtime),
        'n_experiments': len(all_results),
        'n_seeds': n_seeds,
        'summary_stats': summary_dict,
        'statistical_tests': {
            'bn_single_vs_random_mi_pval': float(p_val) if p_val is not None else None,
            'effect_size_cohens_d': float(cohens_d) if cohens_d is not None else None,
            'mi_lift_single_vs_random': float(mi_lift) if mi_lift is not None else None
        },
        'hypotheses': {
            'h1_bn_contains_class_info': bool(h1_supported),
            'h2_early_layers_higher_mi': bool(h2_early_higher),
            'h3_sampling_affects_mi': bool(h3_supported)
        },
        'key_findings': {
            'bn_single_mi_mean': float(np.mean(bn_single_mi)) if bn_single_mi else 0,
            'bn_random_mi_mean': float(np.mean(bn_random_mi)) if bn_random_mi else 0,
            'mi_lift': float(mi_lift) if mi_lift is not None else 0,
            'single_class_acc_improved': summary_dict.get('bn_single_class', {}).get('test_acc_mean', 0) > 0.5
        }
    }
    
    print(f"\nRESULTS: {json.dumps(final_results, indent=2)}")

if __name__ == "__main__":
    main()