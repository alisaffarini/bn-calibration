# pip install scikit-learn scipy matplotlib seaborn

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
import seaborn as sns

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
# HARDER DATASET CREATION
################################################################################

def create_harder_dataset(n_samples=10000, n_features=32, n_classes=10, difficulty='hard', split='train'):
    """
    Create a much harder synthetic dataset to show accuracy differences.
    """
    # Fixed class structure across splits
    np.random.seed(42)
    
    if difficulty == 'very_hard':
        # Highly overlapping classes - realistic scenario
        class_centers = np.random.randn(n_classes, n_features) * 0.5  # Much less separation
        noise_scale = 1.5  # Much more noise
        label_noise = 0.15  # More label noise
    elif difficulty == 'hard':
        # Overlapping classes
        class_centers = np.random.randn(n_classes, n_features) * 0.8
        noise_scale = 1.2
        label_noise = 0.1
    else:
        # Moderate difficulty
        class_centers = np.random.randn(n_classes, n_features) * 1.2
        noise_scale = 1.0
        label_noise = 0.05
    
    # Different random seeds for different splits
    split_seeds = {'train': 100, 'val': 200, 'test': 300}
    np.random.seed(split_seeds.get(split, 100))
    
    # Generate samples
    X = []
    y = []
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        # Add class-specific samples with high noise
        class_samples = class_centers[class_idx] + np.random.randn(samples_per_class, n_features) * noise_scale
        
        # Add outliers
        n_outliers = int(0.15 * samples_per_class)  # More outliers
        outlier_idx = np.random.choice(samples_per_class, n_outliers, replace=False)
        class_samples[outlier_idx] += np.random.randn(n_outliers, n_features) * 2.5
        
        X.extend(class_samples)
        y.extend([class_idx] * samples_per_class)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    # Add label noise
    if split == 'train':
        n_flip = int(label_noise * len(y))
        flip_idx = np.random.choice(len(y), n_flip, replace=False)
        y[flip_idx] = np.random.randint(0, n_classes, n_flip)
    
    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # Add some correlated noise features to make it harder
    n_noise_features = n_features // 4
    noise_features = np.random.randn(len(X), n_noise_features)
    X = np.hstack([X, noise_features])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y

################################################################################
# ENHANCED BN STATS TRACKER WITH VISUALIZATION
################################################################################

class EnhancedBNStatsTracker:
    """Track and analyze batch statistics with visualization support."""
    def __init__(self):
        self.batch_stats_history = []
        self.batch_labels_history = []
        self.layer_wise_stats = defaultdict(list)
        self.batch_accuracies = []  # Track per-batch accuracy
        
    def record_batch_stats(self, model, data, labels, predictions=None):
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
            outputs = model(data)
            if predictions is None:
                predictions = outputs.argmax(dim=1)
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
            
            # Calculate batch accuracy if predictions available
            batch_acc = (predictions == labels).float().mean().item()
            
            self.batch_stats_history.append({
                'stats': combined_stats,
                'is_single_class': is_single_class,
                'dominant_class': dominant_class,
                'unique_classes': unique_classes.tolist(),
                'batch_accuracy': batch_acc
            })
            self.batch_labels_history.append(dominant_class)
            self.batch_accuracies.append(batch_acc)
    
    def compute_mi(self):
        """Compute MI for different batch types."""
        if len(self.batch_stats_history) < 10:
            return 0.0, {}
        
        # Separate single-class and mixed batches
        single_class_stats = []
        single_class_labels = []
        single_class_accs = []
        mixed_class_stats = []
        mixed_class_labels = []
        mixed_class_accs = []
        
        for batch_info in self.batch_stats_history:
            if batch_info['is_single_class']:
                single_class_stats.append(batch_info['stats'])
                single_class_labels.append(batch_info['dominant_class'])
                single_class_accs.append(batch_info['batch_accuracy'])
            else:
                mixed_class_stats.append(batch_info['stats'])
                mixed_class_labels.append(batch_info['dominant_class'])
                mixed_class_accs.append(batch_info['batch_accuracy'])
        
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
            'n_unique_classes_seen': len(np.unique(single_class_labels)) if single_class_stats else 0,
            'avg_single_acc': np.mean(single_class_accs) if single_class_accs else 0,
            'avg_mixed_acc': np.mean(mixed_class_accs) if mixed_class_accs else 0
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
    
    def visualize_stats_distribution(self):
        """Create visualization of BN stats distribution."""
        if len(self.batch_stats_history) < 50:
            return None
        
        # Extract stats for visualization
        single_class_data = []
        mixed_class_data = []
        
        for batch_info in self.batch_stats_history:
            if batch_info['is_single_class']:
                single_class_data.append({
                    'stats_norm': np.linalg.norm(batch_info['stats'][:20]),  # First 20 dims
                    'class': batch_info['dominant_class'],
                    'accuracy': batch_info['batch_accuracy']
                })
            else:
                mixed_class_data.append({
                    'stats_norm': np.linalg.norm(batch_info['stats'][:20]),
                    'class': batch_info['dominant_class'],
                    'accuracy': batch_info['batch_accuracy']
                })
        
        return single_class_data, mixed_class_data

################################################################################
# MODEL DEFINITIONS WITH SINGLE-CLASS ADAPTATIONS
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
            layers.append(nn.Dropout(0.2))  # Increased dropout
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
        self.stats_tracker = EnhancedBNStatsTracker()
    
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
            layers.append(nn.Dropout(0.2))
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
            layers.append(nn.Dropout(0.2))
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
    Single-class batches with curriculum learning to improve training.
    """
    def __init__(self, labels, batch_size, n_classes=10, curriculum=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.curriculum = curriculum
        
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
        
        # Generate batch order
        all_batches = []
        
        if self.curriculum:
            # Start with easier classes (0-4), then harder (5-9)
            easy_classes = list(range(0, self.n_classes // 2))
            hard_classes = list(range(self.n_classes // 2, self.n_classes))
            
            # First epoch: mostly easy classes
            max_batches = max(len(class_batches[c]) for c in range(self.n_classes))
            
            for batch_idx in range(max_batches):
                # 70% easy, 30% hard initially
                for class_idx in easy_classes + easy_classes + hard_classes:
                    if class_idx < self.n_classes and batch_idx < len(class_batches[class_idx]):
                        all_batches.append(class_batches[class_idx][batch_idx])
        else:
            # Standard interleaving
            max_batches = max(len(batches) for batches in class_batches.values())
            
            for batch_idx in range(max_batches):
                for class_idx in range(self.n_classes):
                    if batch_idx < len(class_batches[class_idx]):
                        all_batches.append(class_batches[class_idx][batch_idx])
        
        # Shuffle with some structure preserved
        if not self.curriculum:
            random.shuffle(all_batches)
        
        for batch in all_batches:
            yield batch
    
    def __len__(self):
        count = 0
        for indices in self.class_indices.values():
            count += len(indices) // self.batch_size
        return count

################################################################################
# SPECIALIZED TRAINING FOR SINGLE-CLASS
################################################################################

def train_epoch_with_tracking(model, loader, optimizer, criterion, device, track_stats=False):
    """Train for one epoch with batch stats tracking."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Add regularization for single-class batches
        if hasattr(loader, 'batch_sampler') and hasattr(loader.batch_sampler, '__class__'):
            if 'Single' in loader.batch_sampler.__class__.__name__:
                # Add entropy regularization to prevent collapse
                probs = F.softmax(output, dim=1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
                loss = loss - 0.01 * entropy  # Encourage diverse predictions
        
        loss.backward()
        optimizer.step()
        
        # Track batch stats AFTER training update
        if track_stats and hasattr(model, 'stats_tracker') and batch_idx % 2 == 0:
            with torch.no_grad():
                predictions = output.argmax(dim=1)
                model.stats_tracker.record_batch_stats(model, data, target, predictions)
        
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
    # Different hyperparameters for single-class
    if sampling_strategy == 'single_class':
        lr = 0.002  # Adjusted
        weight_decay = 1e-3
        min_epochs = 40  # Need more epochs
    else:
        lr = 0.001
        weight_decay = 5e-4
        min_epochs = 20
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Different scheduler for single-class
    if sampling_strategy == 'single_class':
        # Use OneCycleLR for single-class training
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr * 10,
            epochs=max_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        use_one_cycle = True
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
        use_one_cycle = False
    
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_val_loss = float('inf')
    patience_counter = 0
    converged = False
    
    for epoch in range(max_epochs):
        # Track stats after warmup
        track_stats = epoch >= 15 and epoch < max_epochs - 5
        
        train_loss, train_acc = train_epoch_with_tracking(
            model, train_loader, optimizer, criterion, device, track_stats=track_stats
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Step scheduler
        if use_one_cycle:
            # OneCycleLR steps per batch, so we don't step here
            pass
        else:
            scheduler.step(val_acc)
        
        # Progress every 10 epochs
        if epoch % 10 == 0 or epoch < 5:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.3f}, LR={current_lr:.6f}")
        
        # Check convergence
        if val_acc > best_val_acc + 0.001:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if val_loss < best_val_loss - 0.001:
            best_val_loss = val_loss
        
        # Early stopping
        if patience_counter >= patience and epoch >= min_epochs:
            print(f"  CONVERGED at epoch {epoch}")
            converged = True
            break
    else:
        if epoch >= max_epochs - 1:
            print("  NOT_CONVERGED: Max epochs reached")
    
    return best_val_acc, converged

################################################################################
# EXPERIMENT RUNNER WITH PRACTICAL DEMONSTRATION
################################################################################

def run_single_seed_experiment(seed, model_type='bn', sampling_strategy='random', 
                              hidden_dims=[256, 512, 256], max_epochs=100, 
                              bn_momentum=0.1, experiment_name="main", difficulty='very_hard'):
    """Run single seed experiment."""
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"\n[Seed {seed}] {experiment_name}: {model_type.upper()}, {sampling_strategy} sampling")
    
    # Create harder dataset
    input_dim = 40  # 32 + 8 noise features
    X_train, y_train = create_harder_dataset(
        n_samples=10000, n_features=32, n_classes=10, 
        difficulty=difficulty, split='train'
    )
    X_val, y_val = create_harder_dataset(
        n_samples=2000, n_features=32, n_classes=10,
        difficulty=difficulty, split='val'
    )
    X_test, y_test = create_harder_dataset(
        n_samples=2000, n_features=32, n_classes=10,
        difficulty=difficulty, split='test'
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
        sampler = ImprovedSingleClassBatchSampler(y_train, batch_size=batch_size, curriculum=True)
        train_loader = DataLoader(train_dataset, batch_sampler=sampler)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    if model_type == 'bn':
        model = ModelWithBN(input_dim=input_dim, hidden_dims=hidden_dims, bn_momentum=bn_momentum).to(device)
    elif model_type == 'gn':
        model = ModelWithGN(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
    elif model_type == 'ln':
        model = ModelWithLN(input_dim=input_dim, hidden_dims=hidden_dims).to(device)
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
    viz_data = None
    
    if model_type == 'bn' and hasattr(model, 'stats_tracker'):
        avg_mi, mi_info = model.stats_tracker.compute_mi()
        layer_mi = model.stats_tracker.compute_layer_wise_mi()
        viz_data = model.stats_tracker.visualize_stats_distribution()
    
    # Test practical implication: Can we identify batch type from BN stats?
    batch_type_acc = 0.0
    if model_type == 'bn' and mi_info.get('n_single_batches', 0) > 10:
        # Simple classifier: predict single-class if stats norm is high
        single_norms = [np.linalg.norm(b['stats'][:20]) for b in model.stats_tracker.batch_stats_history if b['is_single_class']]
        mixed_norms = [np.linalg.norm(b['stats'][:20]) for b in model.stats_tracker.batch_stats_history if not b['is_single_class']]
        
        if single_norms and mixed_norms:
            threshold = (np.mean(single_norms) + np.mean(mixed_norms)) / 2
            
            # Test accuracy of batch type prediction
            correct_predictions = 0
            total_predictions = 0
            
            for b in model.stats_tracker.batch_stats_history[-100:]:  # Last 100 batches
                pred_single = np.linalg.norm(b['stats'][:20]) > threshold
                actual_single = b['is_single_class']
                if pred_single == actual_single:
                    correct_predictions += 1
                total_predictions += 1
            
            batch_type_acc = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
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
        'batch_type_prediction_acc': float(batch_type_acc),
        'hidden_dims': hidden_dims,
        'bn_momentum': bn_momentum if model_type == 'bn' else None
    }
    
    print(f"  Test Acc: {test_acc:.3f}, MI: {avg_mi:.3f}, Batch Type Pred: {batch_type_acc:.3f}, Time: {train_time:.1f}s")
    
    return results

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
    
    # Early abort check with harder dataset
    print("\n" + "="*60)
    print("EARLY ABORT CHECK (Seed 0)")
    print("="*60)
    
    # Test critical conditions first
    first_results = []
    for sampling in ['random', 'single_class']:
        result = run_single_seed_experiment(
            seed=0, model_type='bn', sampling_strategy=sampling,
            hidden_dims=[256, 512, 256], max_epochs=80,  # More epochs for harder task
            difficulty='very_hard'
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
    print(f"  Accuracy difference: {(bn_random_acc - bn_single_acc):.1%}")
    
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
    
    # 1. Core experiments
    print("\n1. Core experiments (10 seeds)")
    for seed in range(1, n_seeds):
        for sampling in ['random', 'balanced', 'single_class']:
            result = run_single_seed_experiment(
                seed=seed, model_type='bn', sampling_strategy=sampling,
                hidden_dims=[256, 512, 256], max_epochs=80,
                difficulty='very_hard'
            )
            all_results.append(result)
    
    # 2. Baselines
    print("\n2. Baseline comparisons (5 seeds)")
    for seed in range(min(5, n_seeds)):
        for model_type in ['gn', 'ln']:
            for sampling in ['random', 'single_class']:
                result = run_single_seed_experiment(
                    seed=seed, model_type=model_type, sampling_strategy=sampling,
                    hidden_dims=[256, 512, 256], max_epochs=60,
                    difficulty='very_hard'
                )
                all_results.append(result)
    
    # 3. Practical demonstration: Different difficulties
    print("\n3. Difficulty analysis (3 seeds)")
    for seed in range(3):
        for difficulty in ['hard', 'very_hard']:
            result = run_single_seed_experiment(
                seed=seed, model_type='bn', sampling_strategy='single_class',
                hidden_dims=[256, 512, 256], max_epochs=60,
                difficulty=difficulty,
                experiment_name=f'difficulty_{difficulty}'
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
        summary[key]['batch_type_acc'].append(r['batch_type_prediction_acc'])
        summary[key]['converged'].append(r['converged'])
    
    # Print main results table
    print("\nMain Results (mean ± std):")
    print("-" * 120)
    print(f"{'Method':<30} {'Test Acc':<25} {'MI':<25} {'Batch Type Pred':<25} {'Conv%'}")
    print("-" * 120)
    
    summary_dict = {}
    for method, metrics in sorted(summary.items()):
        if metrics['test_acc']:
            acc_mean = np.mean(metrics['test_acc'])
            acc_std = np.std(metrics['test_acc'])
            mi_mean = np.mean(metrics['avg_mi'])
            mi_std = np.std(metrics['avg_mi'])
            batch_pred_mean = np.mean(metrics['batch_type_acc'])
            conv_rate = sum(metrics['converged']) / len(metrics['converged']) * 100
            
            print(f"{method:<30} {acc_mean:.3f}±{acc_std:.3f}          "
                  f"{mi_mean:.3f}±{mi_std:.3f}          "
                  f"{batch_pred_mean:.3f}          {conv_rate:.0f}%")
            
            summary_dict[method] = {
                'test_acc_mean': float(acc_mean),
                'test_acc_std': float(acc_std),
                'mi_mean': float(mi_mean),
                'mi_std': float(mi_std),
                'batch_type_pred_mean': float(batch_pred_mean),
                'convergence_rate': float(conv_rate / 100),
                'n': len(metrics['test_acc'])
            }
    
    # Key comparisons
    print("\nKey Findings:")
    print("-" * 80)
    
    if 'bn_random' in summary_dict and 'bn_single_class' in summary_dict:
        acc_diff = summary_dict['bn_random']['test_acc_mean'] - summary_dict['bn_single_class']['test_acc_mean']
        mi_lift = summary_dict['bn_single_class']['mi_mean'] / (summary_dict['bn_random']['mi_mean'] + 1e-8)
        
        print(f"Accuracy cost of single-class training: {acc_diff:.1%}")
        print(f"MI increase from single-class training: {mi_lift:.2f}x")
        print(f"Batch type predictability: {summary_dict['bn_single_class']['batch_type_pred_mean']:.1%}")
    
    # Statistical tests
    print("\nStatistical Tests:")
    print("-" * 60)
    
    # Paired comparisons
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
        print(f"BN single vs random MI: t={t_stat:.3f}, p={p_val:.6f}")
    
    # Final summary
    runtime = (time.time() - start_time) / 60
    
    final_results = {
        'experiment': 'bn_class_specificity_improved',
        'runtime_minutes': float(runtime),
        'n_experiments': len(all_results),
        'summary_stats': summary_dict,
        'key_findings': {
            'accuracy_cost': float(acc_diff) if 'acc_diff' in locals() else 0,
            'mi_lift': float(mi_lift) if 'mi_lift' in locals() else 0,
            'batch_type_predictable': summary_dict.get('bn_single_class', {}).get('batch_type_pred_mean', 0) > 0.7
        }
    }
    
    print(f"\nRESULTS: {json.dumps(final_results, indent=2)}")

if __name__ == "__main__":
    main()