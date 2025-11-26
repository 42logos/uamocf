"""
Neural network model definitions.

Provides unified model architectures for both synthetic 2D experiments and 
MNIST classification, including ensemble wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin


# =============================================================================
# Device Configuration
# =============================================================================

@dataclass
class DeviceConfig:
    """Configuration for PyTorch device selection."""
    
    prefer_cuda: bool = True
    
    def pick(self) -> torch.device:
        """Select the best available device."""
        if self.prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Get the best available device."""
    return DeviceConfig(prefer_cuda).pick()


def to_device(
    models: Iterable[nn.Module],
    device: torch.device
) -> List[nn.Module]:
    """Move a sequence of models to the given device."""
    return [m.to(device) for m in models]


# =============================================================================
# MLP Building Blocks
# =============================================================================

def _mlp_layers(
    input_dim: int,
    hidden_dim: int,
    depth: int,
    output_dim: int,
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    Build a multi-layer perceptron with ReLU activations.
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        depth: Number of hidden layers
        output_dim: Output dimension
        dropout: Dropout probability (applied after each hidden layer)
        
    Returns:
        nn.Sequential containing the MLP layers
    """
    layers = []
    in_dim = input_dim
    
    for _ in range(depth):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = hidden_dim
    
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


# =============================================================================
# Simple Neural Network for 2D Synthetic Data
# =============================================================================

class SimpleNN(nn.Module):
    """
    Fully-connected network for 2D classification experiments.
    
    Default architecture: 2 -> 16 -> 16 -> 16 -> 16 -> n_classes
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        depth: int = 4,
        output_dim: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.output_dim = output_dim
        
        self.net = _mlp_layers(input_dim, hidden_dim, depth, output_dim, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# CNN for MNIST Classification
# =============================================================================

class MNISTClassifier(nn.Module):
    """
    Lightweight CNN classifier for MNIST.
    
    Architecture:
    - Conv1: 1 -> 16 channels, 3x3 kernel, MaxPool
    - Conv2: 16 -> 32 channels, 3x3 kernel, MaxPool
    - FC1: flattened -> 64
    - FC2: 64 -> 10
    
    Supports configurable input size (16x16 for reduced search space).
    """
    
    def __init__(self, img_size: int = 16):
        super().__init__()
        self.img_size = img_size
        
        # After 2 pooling layers: img_size -> img_size/2 -> img_size/4
        final_size = img_size // 4
        
        self.conv = nn.Sequential(
            # Conv layer 1
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv layer 2
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Flatten and FC
            nn.Flatten(),
            nn.Linear(32 * final_size * final_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 10),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# =============================================================================
# Ensemble Model
# =============================================================================

class EnsembleModel(nn.Module):
    """
    Ensemble wrapper that averages predictions from multiple models.
    
    For uncertainty estimation, use the individual models directly.
    This wrapper is for combined predictions.
    """
    
    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average the outputs (logits) of all models."""
        outputs = [m(x) for m in self.models]
        return torch.mean(torch.stack(outputs), dim=0)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get averaged probability predictions from all models.
        
        Args:
            x: Input tensor, shape (batch, features)
            
        Returns:
            Averaged probabilities, shape (batch, classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)
    
    def eval(self) -> "EnsembleModel":
        """Set all models to evaluation mode."""
        super().eval()
        for m in self.models:
            m.eval()
        return self
    
    def train(self, mode: bool = True) -> "EnsembleModel":
        """Set all models to training mode."""
        super().train(mode)
        for m in self.models:
            m.train(mode)
        return self
    
    def __len__(self) -> int:
        return len(self.models)
    
    def __iter__(self):
        return iter(self.models)
    
    def __getitem__(self, idx: int) -> nn.Module:
        return self.models[idx]


# =============================================================================
# sklearn-compatible Probability Estimator
# =============================================================================

class TorchProbaEstimator(ClassifierMixin, BaseEstimator):
    """
    sklearn-compatible wrapper for PyTorch classifiers.
    
    Enables use of sklearn utilities like DecisionBoundaryDisplay
    with PyTorch models.
    """
    
    _estimator_type = "classifier"
    
    def __init__(
        self,
        torch_model: nn.Module,
        device: Optional[torch.device] = None,
        already_prob: bool = False,
        batch_size: int = 8192,
    ):
        """
        Args:
            torch_model: Trained PyTorch classifier
            device: Device to run inference on
            already_prob: If True, model outputs probabilities; else logits
            batch_size: Batch size for inference
        """
        self.torch_model = torch_model
        self.device = device
        self.already_prob = already_prob
        self.batch_size = batch_size
        
        self.classes_ = None
        self.is_fitted_ = False
        
        self.torch_model.eval()
        if self.device is None:
            self.device = next(torch_model.parameters()).device
    
    def fit(self, X, y=None):
        """
        Dummy fit: does NOT train the model.
        
        Only sets fitted attributes for sklearn compatibility.
        """
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        
        if y is not None:
            self.classes_ = np.unique(y)
        else:
            p = self.predict_proba(X[:4])
            self.classes_ = np.arange(p.shape[1])
        
        self.is_fitted_ = True
        return self
    
    @torch.no_grad()
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features, shape (n, d)
            
        Returns:
            probs: Class probabilities, shape (n, n_classes)
        """
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        X_tensor = torch.from_numpy(X).float().to(self.device)
        
        probs_list = []
        for i in range(0, len(X_tensor), self.batch_size):
            xb = X_tensor[i:i + self.batch_size]
            out = self.torch_model(xb)
            
            if out.ndim == 1:
                out = out.unsqueeze(1)
            
            if self.already_prob:
                prob = out
            else:
                # Binary: single logit -> sigmoid
                if out.shape[1] == 1:
                    p1 = torch.sigmoid(out)
                    prob = torch.cat([1 - p1, p1], dim=1)
                # Multi-class: logits -> softmax
                else:
                    prob = torch.softmax(out, dim=1)
            
            probs_list.append(prob.cpu())
        
        return torch.cat(probs_list, dim=0).numpy()
    
    def predict(self, X) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features, shape (n, d)
            
        Returns:
            predictions: Class labels, shape (n,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =============================================================================
# Model Factory Functions
# =============================================================================

def create_simple_nn(
    n_classes: int = 2,
    hidden_dim: int = 16,
    depth: int = 4,
) -> SimpleNN:
    """Create a SimpleNN for synthetic 2D data."""
    return SimpleNN(
        input_dim=2,
        hidden_dim=hidden_dim,
        depth=depth,
        output_dim=n_classes,
    )


def create_mnist_classifier(img_size: int = 16) -> MNISTClassifier:
    """Create a CNN for MNIST classification."""
    return MNISTClassifier(img_size=img_size)


def create_ensemble(
    model_factory,
    n_models: int,
    device: Optional[torch.device] = None,
) -> EnsembleModel:
    """
    Create an ensemble of models.
    
    Args:
        model_factory: Callable that creates a new model instance
        n_models: Number of models in the ensemble
        device: Device to move models to
        
    Returns:
        EnsembleModel wrapping the created models
    """
    models = [model_factory() for _ in range(n_models)]
    if device is not None:
        models = to_device(models, device)
    return EnsembleModel(models)
