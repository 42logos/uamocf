"""
Training utilities for PyTorch classifiers.

Provides unified training loops for both synthetic 2D experiments and MNIST,
including ensemble training with data resampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from tqdm import tqdm

from .models import SimpleNN, MNISTClassifier

Array = np.ndarray


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Configuration for model training."""
    
    batch_size: int = 32
    val_batch_size: int = 256
    epochs: int = 250
    lr: float = 1e-3
    val_split: float = 0.2
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    shuffle: bool = True
    seed: Optional[int] = None
    progress: bool = True
    num_workers: int = 0
    
    def resolve_device(self) -> torch.device:
        """Get the actual device to use."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class MNISTTrainConfig(TrainConfig):
    """Configuration specifically for MNIST training."""
    
    batch_size: int = 64
    epochs: int = 5  # MNIST converges quickly
    img_size: int = 16


# =============================================================================
# Training Results
# =============================================================================

@dataclass
class TrainResult:
    """Container for training results."""
    
    model: nn.Module
    train_loss: float
    val_loss: float
    val_accuracy: float
    history: List[float] = field(default_factory=list)
    X_train: Optional[Array] = None
    y_train: Optional[Array] = None


# =============================================================================
# Utility Functions
# =============================================================================

def _seed_everything(seed: Optional[int]) -> None:
    """Set seeds for reproducibility."""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    X: Array,
    y: Array,
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        X: Features, shape (n, d)
        y: Labels, shape (n,)
        cfg: Training configuration
        
    Returns:
        train_loader, val_loader tuple
    """
    _seed_everything(cfg.seed)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    
    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    
    return train_loader, val_loader


def make_image_loaders(
    images: Array,
    labels: Array,
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation loaders for image data.
    
    Args:
        images: Image array, shape (n, c, h, w) or (n, h, w)
        labels: Labels, shape (n,)
        cfg: Training configuration
        
    Returns:
        train_loader, val_loader tuple
    """
    _seed_everything(cfg.seed)
    
    images_tensor = torch.tensor(images, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(images_tensor, labels_tensor)
    
    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    
    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size])
    else:
        train_ds = dataset
        val_ds = dataset  # Use full data as validation too
    
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    
    return train_loader, val_loader


# =============================================================================
# Training Functions
# =============================================================================

def train_model(
    model: nn.Module,
    X: Array,
    y: Array,
    cfg: TrainConfig,
    callback: Optional[Callable[[int, float], None]] = None,
) -> TrainResult:
    """
    Train a model on 2D feature data.
    
    Args:
        model: PyTorch model to train
        X: Features, shape (n, d)
        y: Labels, shape (n,)
        cfg: Training configuration
        callback: Optional callback called with (epoch, loss) each epoch
        
    Returns:
        TrainResult containing the trained model and metrics
    """
    device = cfg.resolve_device()
    model = model.to(device)
    train_loader, val_loader = make_loaders(X, y, cfg)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    history: List[float] = []
    progress_iter = tqdm(
        range(cfg.epochs),
        desc="Training",
        ncols=80,
        unit="epoch",
        disable=not cfg.progress,
    )
    
    for epoch in progress_iter:
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history.append(epoch_loss)
        progress_iter.set_postfix({"Train Loss": f"{epoch_loss:.4f}"})
        
        if callback:
            callback(epoch, epoch_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    
    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)
    
    return TrainResult(
        model=model,
        train_loss=history[-1] if history else 0.0,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        history=history,
        X_train=X,
        y_train=y,
    )


def train_image_model(
    model: nn.Module,
    images: Array,
    labels: Array,
    cfg: TrainConfig,
    test_loader: Optional[DataLoader] = None,
    callback: Optional[Callable[[int, float], None]] = None,
) -> TrainResult:
    """
    Train a model on image data (e.g., MNIST).
    
    Args:
        model: PyTorch model to train (e.g., MNISTClassifier)
        images: Image array, shape (n, c, h, w) or (n, h, w)
        labels: Labels, shape (n,)
        cfg: Training configuration
        test_loader: Optional separate test loader for final evaluation
        callback: Optional callback called with (epoch, loss) each epoch
        
    Returns:
        TrainResult containing the trained model and metrics
    """
    device = cfg.resolve_device()
    model = model.to(device)
    train_loader, val_loader = make_image_loaders(images, labels, cfg)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    history: List[float] = []
    progress_iter = tqdm(
        range(cfg.epochs),
        desc="Training",
        ncols=100,
        unit="epoch",
        colour="green",
        disable=not cfg.progress,
    )
    
    for epoch in progress_iter:
        model.train()
        running_loss = 0.0
        
        for batch_images, batch_labels in train_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        history.append(epoch_loss)
        progress_iter.set_postfix({"Loss": f"{epoch_loss:.4f}"})
        
        if callback:
            callback(epoch, epoch_loss)
    
    # Evaluation on test or validation set
    eval_loader = test_loader if test_loader is not None else val_loader
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_images, batch_labels in eval_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item() * batch_images.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    val_loss /= total
    val_accuracy = correct / total
    
    return TrainResult(
        model=model,
        train_loss=history[-1] if history else 0.0,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        history=history,
    )


# =============================================================================
# Ensemble Training
# =============================================================================

def train_ensemble(
    num_models: int,
    X: Array,
    y: Array,
    cfg: TrainConfig,
    model_factory: Callable[[], nn.Module] = lambda: SimpleNN(),
    resample_fn: Optional[Callable[[int], Tuple[Array, Array]]] = None,
    callback: Optional[Callable[[int, int, float], None]] = None,
) -> List[TrainResult]:
    """
    Train an ensemble of models on (possibly resampled) data.
    
    Each model gets:
    - Different random initialization (via seed offset)
    - Optionally different data sample (via resample_fn)
    
    Args:
        num_models: Number of models in the ensemble
        X: Features, shape (n, d)
        y: Labels, shape (n,)
        cfg: Training configuration
        model_factory: Callable that creates a new model instance
        resample_fn: Optional function(idx) -> (X, y) for bootstrapping
        callback: Optional callback(model_idx, epoch, loss)
        
    Returns:
        List of TrainResult for each model
    """
    results: List[TrainResult] = []
    
    for idx in range(num_models):
        print(f"\n--- Training Model {idx + 1}/{num_models} ---")
        
        model = model_factory()
        
        # Use different seed for each model
        run_cfg = TrainConfig(**vars(cfg))
        if cfg.seed is not None:
            run_cfg.seed = cfg.seed + idx * 1000
        
        # Get data (possibly resampled)
        Xi, yi = (X, y) if resample_fn is None else resample_fn(idx)
        
        def model_cb(epoch, loss):
            if callback:
                callback(idx, epoch, loss)
        
        result = train_model(model, Xi, yi, run_cfg, callback=model_cb)
        results.append(result)
        
        print(f"  Val Accuracy: {result.val_accuracy:.2%}")
    
    return results


def train_mnist_ensemble(
    num_models: int,
    mnist_dpg_fn: Callable[[int, int], Tuple[Array, Array, Array, Array]],
    test_loader: DataLoader,
    cfg: MNISTTrainConfig,
    n_samples_per_model: int = 5000,
    callback: Optional[Callable[[int, int, float], None]] = None,
) -> List[TrainResult]:
    """
    Train an ensemble of MNIST classifiers with different data samples.
    
    Each model is trained on a different random subset of the training data,
    providing both data and model diversity for uncertainty estimation.
    
    Args:
        num_models: Number of models in the ensemble
        mnist_dpg_fn: Function(n, seed) -> (X_idx, Y, probs, images)
        test_loader: Test data loader for evaluation
        cfg: Training configuration
        n_samples_per_model: Number of training samples per model
        callback: Optional callback(model_idx, epoch, loss)
        
    Returns:
        List of TrainResult for each model
    """
    results: List[TrainResult] = []
    device = cfg.resolve_device()
    
    print(f"{'='*60}")
    print(f"Training Deep Ensemble with {num_models} models")
    print(f"Each model trained on {n_samples_per_model} images")
    print(f"{'='*60}")
    
    for idx in range(num_models):
        print(f"\n--- Training Model {idx + 1}/{num_models} ---")
        
        # Generate different data sample for each model
        data_seed = (cfg.seed or 42) + idx * 123
        _, Y_sampled, _, images_sampled = mnist_dpg_fn(n_samples_per_model, data_seed)
        
        print(f"  Data sampling seed: {data_seed}")
        
        # Different model initialization
        model_seed = 1000 + idx * 1000
        torch.manual_seed(model_seed)
        print(f"  Model init seed: {model_seed}")
        
        model = MNISTClassifier(img_size=cfg.img_size)
        
        # Configure training for this model
        model_cfg = MNISTTrainConfig(**vars(cfg))
        model_cfg.val_split = 0.0  # Use all sampled data for training
        
        def model_cb(epoch, loss):
            if callback:
                callback(idx, epoch, loss)
        
        result = train_image_model(
            model=model,
            images=images_sampled,
            labels=Y_sampled,
            cfg=model_cfg,
            test_loader=test_loader,
            callback=model_cb,
        )
        
        results.append(result)
        print(f"  Test Accuracy: {result.val_accuracy:.2%}")
    
    print(f"\n{'='*60}")
    print(f"Ensemble Training Complete!")
    print(f"{'='*60}")
    
    return results


# =============================================================================
# Utility: Extract Models from Results
# =============================================================================

def extract_models(results: List[TrainResult]) -> List[nn.Module]:
    """Extract trained models from a list of TrainResults."""
    return [r.model for r in results]
