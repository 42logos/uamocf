"""
Data generation utilities and probability surfaces.

Supports both binary and multi-class classification problems with configurable
uncertainty profiles for synthetic experiments and MNIST dataset handling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

Array = np.ndarray


 
# Probability Surfaces (P(Y|X) for synthetic data)
 

def boundary_focus_prob_multiclass(
    X: Array,
    n_classes: int = 2,
    r_inner: float = np.sqrt(2),
    r_outer: float = 3.0,
    sigma: float = 0.85,
    max_uncertainty: float = 0.89,
) -> Array:
    """
    Multi-class probability distribution with uncertainty concentrated near 
    circular decision boundaries.
    
    Creates a synthetic conditional probability P(Y|X) where:
    - High certainty (low entropy) far from boundaries
    - High uncertainty (high entropy) near the circular boundaries at r_inner and r_outer
    
    Args:
        X: Feature array of shape (n, d) where d >= 2
        n_classes: Number of classes (default 2 for binary)
        r_inner: Inner circle radius for decision boundary
        r_outer: Outer circle radius for decision boundary
        sigma: Controls decay rate of uncertainty away from boundary
        max_uncertainty: Maximum entropy level near boundaries
        
    Returns:
        probs: Probability distribution, shape (n, n_classes)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    r = np.linalg.norm(X, axis=-1)
    
    # Base assignment: class 0 between boundaries, class 1 outside
    base = ((r <= r_inner) | (r >= r_outer)).astype(float)
    
    # Distance to nearest boundary
    dist_to_boundary = np.minimum(np.abs(r - r_inner), np.abs(r - r_outer))
    
    # Uncertainty peaks at boundary, decays with Gaussian profile
    alpha = max_uncertainty * np.exp(-(dist_to_boundary / sigma) ** 2)
    
    # Blend between certain prediction and 0.5 (maximum uncertainty)
    p1 = base * (1 - alpha) + 0.5 * alpha
    p1 = np.clip(p1, 0.0, max_uncertainty)
    
    # Build probability distribution
    probs = np.zeros((X.shape[0], n_classes))
    probs[:, 0] = 1 - p1  # class 0
    probs[:, 1] = p1       # class 1
    
    # For more than 2 classes, distribute remaining probability
    if n_classes > 2:
        remaining = 1.0 - probs[:, 0] - probs[:, 1]
        probs[:, 2:] = remaining[:, None] / (n_classes - 2)
        probs = probs / probs.sum(axis=-1, keepdims=True)
    
    return probs


def moon_focus_prob_multiclass(
    X: Array,
    n_classes: int = 2,
    sigma: float = 0.4,
    max_uncertainty: float = 0.95,
    min_uncertainty: float = 0.05,
) -> Array:
    """
    Multi-class probability distribution with uncertainty near a sinusoidal
    "moon-shaped" decision boundary.
    
    Creates a synthetic conditional probability P(Y|X) where:
    - Decision boundary follows sin(π * x1) / 2
    - Uncertainty is highest near this boundary
    
    Args:
        X: Feature array of shape (n, 2)
        n_classes: Number of classes
        sigma: Controls width of high-uncertainty region
        max_uncertainty: Maximum probability value near boundary
        min_uncertainty: Minimum probability value (prevents 0/1 extremes)
        
    Returns:
        probs: Probability distribution, shape (n, n_classes)
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    x1, x2 = X[:, 0], X[:, 1]
    
    # Sinusoidal decision boundary
    decision_boundary = np.sin(np.pi * x1) / 2
    dist_to_boundary = np.abs(x2 - decision_boundary)
    
    # Uncertainty profile (Gaussian centered at boundary)
    alpha = max_uncertainty * np.exp(-(dist_to_boundary / sigma) ** 2)
    
    # Class 1 probability: blend between boundary-based and certain
    p1 = 0.5 * alpha + (1 - alpha) * (x2 > decision_boundary).astype(float)
    p1 = np.clip(p1, min_uncertainty, max_uncertainty)
    
    # Build probability distribution
    probs = np.zeros((X.shape[0], n_classes))
    probs[:, 0] = 1 - p1
    probs[:, 1] = p1
    
    # Extended multi-class: add angular sectors
    if n_classes > 2:
        r = np.linalg.norm(X, axis=-1)
        for c in range(2, n_classes):
            angle = np.arctan2(X[:, 1], X[:, 0])
            sector_prob = 0.5 * (1 + np.cos(angle - 2 * np.pi * c / n_classes))
            probs[:, c] = sector_prob * alpha * 2
        probs = probs / probs.sum(axis=-1, keepdims=True)
    
    return probs


# Legacy wrappers for backward compatibility (return p1 for binary case)
def boundary_focus_prob(X: Array, **kwargs) -> Array:
    """Returns P(Y=1|X) for binary classification using circular boundaries."""
    probs = boundary_focus_prob_multiclass(X, n_classes=2, **kwargs)
    return probs[:, 1] if probs.ndim > 1 else probs[1]


def moon_focus_prob(X: Array, **kwargs) -> Array:
    """Returns P(Y=1|X) for binary classification using moon boundary."""
    probs = moon_focus_prob_multiclass(X, n_classes=2, **kwargs)
    return probs[:, 1] if probs.ndim > 1 else probs[1]


def make_aleatoric_uncertainty(
    p_fn: Callable[[Array], Array]
) -> Callable[[Array], Array]:
    """
    Wrap a probability function to return Shannon entropy (aleatoric uncertainty).
    
    Works with both binary (returns p1) and multi-class (returns full distribution)
    probability functions.
    
    Args:
        p_fn: Probability function returning either p1 or (n, n_classes)
        
    Returns:
        Function that returns entropy values
    """
    def wrapped_fn(X: Array) -> Array:
        probs = np.asarray(p_fn(X))
        
        # Handle binary case: convert p1 to [p0, p1]
        if probs.ndim == 1:
            p1 = probs
            probs = np.stack([1.0 - p1, p1], axis=-1)
        
        # Shannon entropy: H = -sum(p * log(p))
        return -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    
    return wrapped_fn


 
# Data Configurations
 

@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    
    n: int = 1500
    d: int = 2
    low: float = -1.0
    high: float = 1.0
    seed: Optional[int] = 42
    n_classes: int = 2
    p_fn_name: str = "moon"  # "moon" or "boundary"
    
    # Probability function parameters
    sigma: float = 0.4
    max_uncertainty: float = 0.95
    min_uncertainty: float = 0.05
    
    def get_p_fn(self) -> Callable[[Array], Array]:
        """Get the probability function based on configuration."""
        if self.p_fn_name == "moon":
            return partial(
                moon_focus_prob_multiclass,
                n_classes=self.n_classes,
                sigma=self.sigma,
                max_uncertainty=self.max_uncertainty,
                min_uncertainty=self.min_uncertainty,
            )
        elif self.p_fn_name == "boundary":
            return partial(
                boundary_focus_prob_multiclass,
                n_classes=self.n_classes,
                sigma=self.sigma,
                max_uncertainty=self.max_uncertainty,
            )
        else:
            raise ValueError(f"Unknown p_fn_name: {self.p_fn_name}")
    
    def make_sampler(self) -> Callable[..., Array]:
        """Create a uniform sampler with this config's bounds and seed."""
        rng = np.random.default_rng(self.seed)
        return partial(rng.uniform, low=self.low, high=self.high)


@dataclass
class MNISTConfig:
    """Configuration for MNIST experiment."""
    
    img_size: int = 16  # Reduced from 28 for faster optimization
    n_train_samples: int = 5000
    n_ensemble: int = 5
    num_epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    seed: Optional[int] = 42
    
    @property
    def n_pixels(self) -> int:
        return self.img_size * self.img_size


 
# Data Generating Process
 

def dpg(
    n: int,
    x_sampler: Callable[..., Array],
    p_fn: Callable[[Array], Array],
    d: int = 2,
    rng: Optional[np.random.Generator] = None,
    x_sampler_kwargs: Optional[dict] = None,
) -> Tuple[Array, Array, Array]:
    """
    Data Generating Process for classification.
    
    Samples X from P_X (via x_sampler) and Y|X from the conditional distribution
    defined by p_fn.
    
    Supports both binary (p_fn returns p1) and multi-class (p_fn returns full dist).
    
    Args:
        n: Number of samples
        x_sampler: Callable that generates X samples (e.g., rng.uniform)
        p_fn: Conditional probability function P(Y|X)
        d: Number of features
        rng: Random number generator for Y sampling
        x_sampler_kwargs: Additional kwargs for x_sampler
        
    Returns:
        X: Features, shape (n, d)
        Y: Labels, shape (n,)
        probs: Probability values/distributions
    """
    rng = np.random.default_rng() if rng is None else rng
    x_sampler_kwargs = {} if x_sampler_kwargs is None else dict(x_sampler_kwargs)
    
    if "size" not in x_sampler_kwargs:
        x_sampler_kwargs["size"] = (n, d)
    
    X = np.asarray(x_sampler(**x_sampler_kwargs))
    probs = np.asarray(p_fn(X))
    
    # Handle binary case: p_fn returns only p1
    if probs.ndim == 1:
        p1 = np.clip(probs, 0.0, 1.0)
        Y = rng.binomial(1, p1).astype(np.int64)
        return X, Y, p1
    
    # Multi-class case: p_fn returns full distribution
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)  # Normalize
    
    n_classes = probs.shape[1]
    Y = np.array([
        rng.choice(n_classes, p=probs[i])
        for i in range(n)
    ], dtype=np.int64)
    
    return X, Y, probs


def sample_dataset(cfg: DataConfig) -> Tuple[Array, Array, Array]:
    """
    Sample a dataset using the provided configuration.
    
    Convenience wrapper around dpg that uses DataConfig parameters.
    
    Args:
        cfg: DataConfig instance
        
    Returns:
        X, Y, probs tuple from dpg
    """
    rng = np.random.default_rng(cfg.seed)
    sampler = cfg.make_sampler()
    p_fn = cfg.get_p_fn()
    return dpg(cfg.n, sampler, p_fn, d=cfg.d, rng=rng)


 
# MNIST Data Utilities
 

class MNISTProbabilityFunction:
    """
    Maps normalized indices to MNIST images and class probabilities.
    
    Creates a bridge between:
    - 1D latent space X ∈ [0, 1] (normalized dataset index)
    - 10-class probability distribution P(Y|X)
    - Actual MNIST images
    
    For counterfactual generation in index space:
    - x* is a factual instance (normalized index)
    - x̄ is a counterfactual (different normalized index)
    """
    
    def __init__(
        self,
        dataset: Dataset,
        model: Optional[nn.Module] = None,
        use_true_labels: bool = True,
        temperature: float = 1.0,
    ):
        """
        Args:
            dataset: MNIST dataset (torchvision)
            model: Optional trained classifier for predictions
            use_true_labels: If True and no model, use ground truth labels
            temperature: Softmax temperature for probability smoothing
        """
        self.dataset = dataset
        self.n_samples = len(dataset)
        self.n_classes = 10
        self.model = model
        self.use_true_labels = use_true_labels
        self.temperature = temperature
        self.device = next(model.parameters()).device if model else "cpu"
        
        # Pre-compute labels for fast lookup
        self._labels = np.array([dataset[i][1] for i in range(self.n_samples)])
        
        # Lazy image cache
        self._image_cache = {}
    
    def index_to_normalized(self, idx: int) -> float:
        """Convert dataset index to normalized value in [0, 1]."""
        return idx / (self.n_samples - 1)
    
    def normalized_to_index(self, x: float) -> int:
        """Convert normalized value to dataset index."""
        x = np.clip(x, 0.0, 1.0)
        return int(np.round(x * (self.n_samples - 1)))
    
    def get_image(self, x: float) -> torch.Tensor:
        """Get MNIST image for normalized index x."""
        idx = self.normalized_to_index(x)
        if idx not in self._image_cache:
            self._image_cache[idx] = self.dataset[idx][0]
        return self._image_cache[idx]
    
    def get_label(self, x: float) -> int:
        """Get true label for normalized index x."""
        idx = self.normalized_to_index(x)
        return self._labels[idx]
    
    def __call__(self, X: Array) -> Array:
        """
        Compute P(Y|X) for normalized indices.
        
        Args:
            X: Normalized indices, shape (n,) or (n, 1)
            
        Returns:
            probs: Probability distribution, shape (n, 10)
        """
        X = np.asarray(X).flatten()
        n = len(X)
        probs = np.zeros((n, self.n_classes))
        
        if self.model is not None:
            self.model.eval()
            for i, x in enumerate(X):
                img = self.get_image(x).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(img)
                    p = nn.Softmax(dim=1)(logits / self.temperature).cpu().numpy()
                probs[i] = p[0]
        else:
            # One-hot encoding with label smoothing
            epsilon = 0.01
            for i, x in enumerate(X):
                label = self.get_label(x)
                probs[i] = epsilon / self.n_classes
                probs[i, label] = 1.0 - epsilon + epsilon / self.n_classes
        
        return probs
    
    def get_images_batch(self, X: Array) -> torch.Tensor:
        """Get batch of images for normalized indices."""
        X = np.asarray(X).flatten()
        return torch.stack([self.get_image(x) for x in X])


def mnist_dpg(
    n: int,
    mnist_prob_fn: MNISTProbabilityFunction,
    rng: Optional[np.random.Generator] = None,
    sample_mode: str = "uniform",
    return_images: bool = False,
) -> Union[Tuple[Array, Array, Array], Tuple[Array, Array, Array, Array]]:
    """
    Data Generating Process for MNIST using normalized indices.
    
    Args:
        n: Number of samples
        mnist_prob_fn: MNISTProbabilityFunction instance
        rng: Random number generator
        sample_mode: "uniform" (random) or "sequential" (first n)
        return_images: If True, also return actual images
        
    Returns:
        X: Normalized indices, shape (n, 1)
        Y: Class labels, shape (n,)
        probs: Probability distributions, shape (n, 10)
        images (optional): Images, shape (n, 1, img_size, img_size)
    """
    rng = np.random.default_rng() if rng is None else rng
    
    if sample_mode == "uniform":
        indices = rng.choice(mnist_prob_fn.n_samples, size=n, replace=False)
        X = np.array([mnist_prob_fn.index_to_normalized(i) for i in indices])
    else:
        indices = np.linspace(0, mnist_prob_fn.n_samples - 1, n, dtype=int)
        X = np.linspace(0, 1, n)
    
    X = X.reshape(-1, 1)
    Y = np.array([mnist_prob_fn._labels[i] for i in indices], dtype=np.int64)
    probs = mnist_prob_fn(X)
    
    # Normalize probabilities
    probs = np.clip(probs, 0.0, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    
    if return_images:
        images = np.array([mnist_prob_fn.dataset[i][0].numpy() for i in indices])
        return X, Y, probs, images
    
    return X, Y, probs


def get_mnist_images_subset(
    dataset: Dataset,
    n_samples: int = 500,
    seed: int = 42,
) -> Tuple[Array, Array, Array]:
    """
    Extract a subset of MNIST images as numpy arrays.
    
    Args:
        dataset: MNIST dataset
        n_samples: Number of images to extract
        seed: Random seed
        
    Returns:
        images: Array of images, shape (n, img_size, img_size)
        labels: Array of labels, shape (n,)
        indices: Array of dataset indices
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_samples, replace=False)
    images = np.array([dataset[i][0].numpy().squeeze() for i in indices])
    labels = np.array([dataset[i][1] for i in indices])
    return images, labels, indices
