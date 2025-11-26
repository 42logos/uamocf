from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Literal, Sized
import numpy as np
from numpy.typing import ArrayLike
import torch
from torch.utils.data import Dataset

# abstract class for data generator
class DataGenerator(ABC):
    @abstractmethod
    def sample(self, n: int, seed: int) -> Tuple[ArrayLike, ArrayLike]:
        """Sample n data points.

        Returns:
            X: Features, shape (n, d)
            y: Labels in one-hot encoding, shape (n, n_classes)
        """
        pass

#abstract class for Probability Function
class Prob_FN(ABC):
    n_classes: int = 2
    
    @abstractmethod
    def __call__(self, x: ArrayLike) -> ArrayLike:
        """Given input features x, return class probabilities.

        Args:
            x: Input features, shape (2), only supports for one sample at a time.

        Returns:
            probs: Class probabilities, shape  (n_classes)
        """
        pass

class DatasetsDG(DataGenerator):
    """Data generator that samples from given datasets.

    Args:
        Dataset: A PyTorch Dataset object.
    """
    def __init__(self, dataset: Dataset, num_classes: int):
        
        # check if the dataset has __len__ and __getitem__ methods
        if not isinstance(dataset, Sized) or not hasattr(dataset, '__getitem__'):
            raise ValueError("The dataset must be a PyTorch Dataset object with __len__ and __getitem__ methods.")
        
        
        self.dataset = dataset
        self.num_classes = num_classes
        
        self.rng = np.random.default_rng()
    
    def sample(self, n: int, seed: int) -> Tuple[ArrayLike, ArrayLike]:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.dataset), size=n, replace=True) 
        x,y= zip(*[self.dataset[i] for i in indices])
        X = torch.stack(x)
        y = torch.nn.functional.one_hot(torch.tensor(y), num_classes=self.num_classes).float()
        return X, y
    
class PlaneDG(DataGenerator):
    def __init__(self, p_fn: Prob_FN):
        self.p_fn = p_fn
        self.rng = np.random.default_rng()
        self.n_classes = p_fn.n_classes

    def sample(self, n: int, seed: int, type: Literal['NumPy', 'Torch']='NumPy') -> Tuple[ArrayLike, ArrayLike]:
        rng = np.random.default_rng(seed)
        X = rng.uniform(0, 1, size=(n, 2)) 
        probs = np.array([self.p_fn(X[i]) for i in range(n)]) 
        y_indices = np.array([rng.choice(self.n_classes, p=probs[i]) for i in range(n)])
        y = np.zeros((n, self.n_classes))
        y[np.arange(n), y_indices] = 1
        return (X, y) if type=='NumPy' else (torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    
    
class circle_prob(Prob_FN):
    """Probability function for circle dataset."""

    n_classes: int = 2
    sigma: float = 0.1
    max_uncertainty: float = 0.69314718056  # ln(2)
    min_uncertainty: float = 0.0
    r_inner: float = 0.3
    r_outer: float = 0.7

    def __call__(self, x: ArrayLike) -> ArrayLike:
        X = np.asarray(x)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        r = np.linalg.norm(X, axis=-1)

        base = ((r <= self.r_inner) | (r >= self.r_outer)).astype(float)
        dist_to_boundary = np.minimum(np.abs(r - self.r_inner), np.abs(r - self.r_outer))
        alpha = self.max_uncertainty * np.exp(-(dist_to_boundary / self.sigma) ** 2)

        p1 = base * (1 - alpha) + 0.5 * alpha
        p1 = np.clip(p1, 0.0, self.max_uncertainty)
        
        # Convert to multi-class probability distribution
        probs = np.zeros((X.shape[0], self.n_classes))
        probs[:, 0] = 1 - p1  # class 0
        probs[:, 1] = p1       # class 1
        # For more than 2 classes, distribute remaining probability uniformly
        if self.n_classes > 2:
            # Rescale so that all classes sum to 1
            remaining = 1.0 - probs[:, 0] - probs[:, 1]
            probs[:, 2:] = remaining[:, None] / (self.n_classes - 2)
            # Renormalize
            probs = probs / probs.sum(axis=-1, keepdims=True)
        
        return probs
    
    
class moon_prob(Prob_FN):
    """Probability function for moon dataset."""

    n_classes: int = 2
    sigma: float = 0.4
    max_uncertainty: float = 0.75
    min_uncertainty: float = 0.15

    def __call__(self, x: ArrayLike) -> ArrayLike:
        X= np.asarray(x)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        x1, x2 = X[:, 0], X[:, 1]
        decision_boundary = np.sin(np.pi * x1) / 2
        dist_to_boundary = np.abs(x2 - decision_boundary)
        alpha = self.max_uncertainty * np.exp(-(dist_to_boundary / self.sigma) ** 2)
        p1 = 0.5 * alpha + (1 - alpha) * (x2 > decision_boundary).astype(float)
        p1 = np.clip(p1, self.min_uncertainty, self.max_uncertainty)
        
        # Convert to multi-class probability distribution
        probs = np.zeros((X.shape[0], self.n_classes))
        probs[:, 0] = 1 - p1  # class 0
        probs[:, 1] = p1       # class 1
        # For more than 2 classes, we can define regions or blend
        if self.n_classes > 2:
            # Example: distribute probability based on distance from origin
            # This is a simple extension; you may customize for your use case
            r = np.linalg.norm(X, axis=-1)
            for c in range(2, self.n_classes):
                # Create class-specific probability based on angular position
                angle = np.arctan2(X[:, 1], X[:, 0])
                sector_prob = 0.5 * (1 + np.cos(angle - 2 * np.pi * c / self.n_classes))
                probs[:, c] = sector_prob * alpha * 2  # small contribution
            # Renormalize
            probs = probs / probs.sum(axis=-1, keepdims=True)
        
        return probs