from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union
from typing_extensions import Literal

from core import models
from core.models import EnsembleModel

import numpy as np
import torch
from pymoo.problems.functional import FunctionalProblem
from torch import nn

Array = Union[np.ndarray, torch.Tensor]

# =============================================================================
# Tensor Utility Functions
# =============================================================================

def is_tensor(x: Array) -> bool:
    """Check if input is a PyTorch tensor."""
    return isinstance(x, torch.Tensor)


def to_numpy(x: Array) -> np.ndarray:
    """Convert tensor or numpy array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x: Array, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert numpy array or tensor to tensor on specified device."""
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(np.asarray(x).astype(np.float32))
    if device is not None:
        t = t.to(device)
    return t

def _gower_distance_tensor(x: torch.Tensor, y: torch.Tensor, feature_range: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized feature-wise dissimilarity (Gower distance) using tensors.
    
    δ_G(u, v) = |u - v| / range for numerical features
    
    Args:
        x, y: Feature vectors (flattened)
        feature_range: Range of each feature
        
    Returns:
        Per-feature dissimilarity, shape (d,)
    """
    return torch.clamp(torch.abs(x - y) / feature_range, max=1.0)

def _k_nearest_tensor(x: torch.Tensor, X_obs: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k nearest neighbors to x in X_obs using tensors.
    
    Args:
        x: Query point (flattened)
        X_obs: Observed data points (N, d)
        k: Number of neighbors
        
    Returns:
        nearest_points: shape (k, d)
        distances: shape (k,)
    """
    dists = torch.norm(X_obs - x, dim=1)
    nearest_idx = torch.argsort(dists)[:k]
    return X_obs[nearest_idx], dists[nearest_idx]

def _gower_distance(x: Array, y: Array, feature_range: Array) -> Array:
    """
    Compute normalized feature-wise dissimilarity (Gower distance).
    
    δ_G(u, v) = |u - v| / range for numerical features
    
    Args:
        x, y: Feature vectors
        feature_range: Range of each feature
        
    Returns:
        Per-feature dissimilarity, shape (d,)
    """
    return np.minimum(np.abs(x - y) / feature_range, 1.0)

def _k_nearest(x: Array, X_obs: Array, k: int) -> Tuple[Array, Array]:
    """
    Find k nearest neighbors to x in X_obs.
    
    Args:
        x: Query point
        X_obs: Observed data points
        k: Number of neighbors
        
    Returns:
        nearest_points: shape (k, d)
        distances: shape (k,)
    """
    dists = np.linalg.norm(X_obs - x, axis=1)
    nearest_idx = np.argsort(dists)[:k]
    return X_obs[nearest_idx], dists[nearest_idx]


def aleatoric_from_models(
    model_ensemble: EnsembleModel, 
    x: Array,
    return_tensor: bool = False,
) -> Array:
    """
    Average aleatoric uncertainty across an ensemble of models.
    Supports both single point (d,) and batch (N, d).
    
    Args:
        model_ensemble: Ensemble of trained models
        x: Input points, shape (d,) or (N, d) - can be tensor or numpy
        return_tensor: If True, return tensor; if False, return numpy.
                       Default False for backward compatibility.
                       If input is tensor and return_tensor not specified,
                       returns tensor to avoid unnecessary conversion.
    
    Returns:
        Aleatoric uncertainty, shape (N,) or scalar
    """
    input_is_tensor = is_tensor(x)
    
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]  # (1, d)
    
    x_t = to_tensor(x, device=model_ensemble.device)
    
    # Collect probabilities from all models
    all_probs = []
    with torch.no_grad():
        for m in model_ensemble.models:
            logits = m(x_t)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs)
    
    # Stack: (n_models, N, n_classes)
    all_probs = torch.stack(all_probs)
    
    # Aleatoric uncertainty = Mean of Entropies
    # Entropy of each model: -sum(p log p)
    entropies = -torch.sum(all_probs * torch.log(all_probs + 1e-12), dim=-1)  # (n_models, N)
    mean_entropy = torch.mean(entropies, dim=0)  # (N,)
    
    # Decide output format
    should_return_tensor = return_tensor or input_is_tensor
    
    if should_return_tensor:
        return mean_entropy if is_batch else mean_entropy[0]
    else:
        result = mean_entropy.cpu().numpy()
        return result if is_batch else result[0]

def Total_uncertainty(
    model_ensemble: EnsembleModel, 
    x: Array, 
    device: Optional[torch.device] = None,
    return_tensor: bool = False,
) -> Union[float, Array]:
    """
    Total uncertainty = Entropy of the mean predictive distribution.
    
    Correct formulation:
    1. Get softmax probabilities from each model
    2. Average the probabilities (not logits!)
    3. Compute entropy of the averaged distribution
    
    Supports both single point (d,) and batch (N, d).
    
    Args:
        model_ensemble: Ensemble of trained models
        x: Input points, shape (d,) or (N, d) - can be tensor or numpy
        device: PyTorch device (defaults to model_ensemble.device)
        return_tensor: If True, return tensor; if False, return numpy.
                       If input is tensor and return_tensor not specified,
                       returns tensor to avoid unnecessary conversion.
    
    Returns:
        Total uncertainty, shape (N,) or scalar
    """
    input_is_tensor = is_tensor(x)
    
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]  # (1, d)
    
    if device is None:
        device = model_ensemble.device
    x_t = to_tensor(x, device=device)
    
    # Collect probabilities from all models
    all_probs = []
    with torch.no_grad():
        for m in model_ensemble.models:
            logits = m(x_t)
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs)
    
    # Stack: (n_models, N, n_classes)
    all_probs = torch.stack(all_probs)
    
    # Mean probabilities across models: (N, n_classes)
    mean_probs = torch.mean(all_probs, dim=0)
    
    # Total uncertainty = entropy of mean prediction
    ent = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12), dim=-1)  # (N,)
    
    # Decide output format
    should_return_tensor = return_tensor or input_is_tensor
    
    if should_return_tensor:
        return ent if is_batch else ent[0]
    else:
        result = ent.cpu().numpy()
        return float(result[0]) if not is_batch else result

def epistemic_from_models(
    model_ensemble: EnsembleModel, 
    x: Array,
    return_tensor: bool = False,
) -> Union[float, Array]:
    """
    Epistemic uncertainty from an ensemble of models.
    
    EU = TU - AU = H[E[p]] - E[H[p]]
    
    Where:
    - TU = Total Uncertainty = Entropy of mean prediction
    - AU = Aleatoric Uncertainty = Mean of individual entropies
    - EU = Epistemic Uncertainty = Disagreement between models
    
    EU is always >= 0 by Jensen's inequality.
    
    Supports both single point (d,) and batch (N, d).
    Supports both tensor and numpy inputs.
    
    Args:
        model_ensemble: Ensemble of trained models
        x: Input points, shape (d,) or (N, d) - can be tensor or numpy
        return_tensor: If True, return tensor; if False, return numpy.
                       If input is tensor and return_tensor not specified,
                       returns tensor to avoid unnecessary conversion.
    
    Returns:
        Epistemic uncertainty, shape (N,) or scalar
    """
    input_is_tensor = is_tensor(x)
    should_return_tensor = return_tensor or input_is_tensor
    
    # Compute using tensors internally
    AU = aleatoric_from_models(model_ensemble, x, return_tensor=True)
    TU = Total_uncertainty(model_ensemble, x, return_tensor=True)
    EU = TU - AU
    
    # EU should be non-negative by Jensen's inequality, but clamp for numerical stability
    if should_return_tensor:
        return torch.clamp(EU, min=0.0)
    else:
        return np.maximum(to_numpy(EU), 0.0)

def make_cf_problem(
    model: nn.Module,
    x_factual: torch.Tensor,
    y_target: torch.Tensor,
    X_obs: torch.Tensor,
    k_neighbors: int = 10,
    use_soft_validity: bool = True,
    normalize_similarity: bool = True,
    sparsity_eps: float = 0.005,
    ensemble: Optional[EnsembleModel] = None,
    device: Optional[torch.device] = None,
) -> FunctionalProblem:
    """
    Create a counterfactual problem for 2D feature space.
    
    Objectives:
    - o1: Validity (soft or hard)
    - o2: Similarity
    - o3: Sparsity
    - o4: Plausibility
    - o5: Aleatoric Uncertainty (negated if maximizing)
    - o6: Epistemic Uncertainty
    
    Args:
        model: Primary trained classifier
        x_star: Factual instance, shape (d,)
        y_target: Target class(es), shape (1,) or (k,)
        X_obs: Observed data, shape (n, d)
        weights: Neighbor weights, shape (k,)
        config: CFConfig instance
        ensemble: Optional model ensemble for uncertainty
        device: PyTorch device
        
    Returns:
        pymoo FunctionalProblem
    """
    
    model.eval()
    if ensemble is not None:
        for m in ensemble.models:
            m.eval()

    w= torch.ones(k_neighbors, device=device)/k_neighbors
    
    # Get flattened dimensions
    p = x_factual.flatten().shape[0]  # Total number of features (e.g., 256 for 16x16)
    input_shape = x_factual.shape  # Store original shape for reshaping (e.g., (1, 1, 16, 16) for CNN)
    
    # Flatten X_obs for computing bounds and distances
    X_obs_flat = X_obs.reshape(X_obs.shape[0], -1).to(device)  # (n_samples, p)
    
    # Feature bounds and range (computed on flattened data)
    xl = torch.min(X_obs_flat, dim=0).values  # (p,)
    xu = torch.max(X_obs_flat, dim=0).values  # (p,)
    feature_range_flat = xu - xl
    feature_range_flat[feature_range_flat == 0] = 1.0
    
    target_labels = y_target.view(-1).long().tolist() 
    
    # Flatten tensors for objective functions
    x_factual_flat = x_factual.flatten().to(device)
    
    # Convert numpy array to tensor with proper shape for model input
    def _to_tensor(x: Array) -> torch.Tensor:
        """Convert numpy array to flattened tensor."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x.astype(np.float32)).to(device)
        return x.to(device)
    
    def _to_model_input(x: Array) -> torch.Tensor:
        """Convert flat array to tensor with proper shape for model."""
        x_t = _to_tensor(x)
        # Reshape to original input shape (e.g., (1, 1, 16, 16) for CNN)
        reshaped = x_t.view(input_shape)
        # Ensure batch dimension exists
        if reshaped.dim() == len(input_shape) and input_shape[0] != 1:
            # No batch dimension, add one
            reshaped = reshaped.unsqueeze(0)
        return reshaped
    
    # OBJECTIVE FUNCTIONS 
    def o1_validity(x: Array) -> float:
        """Validity: 1 - P(target) or hard constraint."""
        x_t = _to_model_input(x)
        
        with torch.no_grad():
            logits = model(x_t)
            # Handle both 1D and 2D outputs
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            probs = torch.softmax(logits, dim=1)
        
        if use_soft_validity:
            prob_target = probs[0, target_labels].sum().item()
            return 1.0 - prob_target
        else:
            pred = probs.argmax(dim=1).item()
            return 0.0 if pred in target_labels else 1.0
    
    def o2_similarity(x: Array) -> float:
        """Similarity: mean Gower distance to factual."""
        x_t = _to_tensor(x)
        if normalize_similarity:
            return float(_gower_distance_tensor(x_t, x_factual_flat, feature_range_flat).mean().item())
        return float(torch.norm(x_t - x_factual_flat).item())
    
    def o3_sparsity(x: Array) -> float:
        """Sparsity: number of changed features."""
        x_t = _to_tensor(x)
        changed = torch.abs(x_t - x_factual_flat) > sparsity_eps
        return float(changed.sum().item())
    
    def o4_plausibility(x: Array) -> float:
        """Plausibility: weighted distance to k-nearest neighbors."""
        x_t = _to_tensor(x)
        nearest_samples, _ = _k_nearest_tensor(x_t, X_obs_flat, k_neighbors)
        per_sample = torch.stack([
            _gower_distance_tensor(x_t, xi, feature_range_flat).mean()
            for xi in nearest_samples
        ])
        return float((per_sample * w).sum().item())
    
    def o5_aleatoric_uncertainty(x: Array) -> float:
        """Aleatoric Uncertainty (negated to maximize)."""
        if ensemble is None or len(ensemble.models) == 0:
            raise ValueError("Ensemble required for aleatoric uncertainty.")
        au = aleatoric_from_models(ensemble, x)
        if isinstance(au, np.ndarray) and au.ndim > 0:
            au = au[0]
        elif isinstance(au, np.ndarray) and au.ndim == 0:
            au = au.item()
        elif isinstance(au, torch.Tensor) and au.ndim > 0:
            au = au[0].item()
        return float(-au)  # Negative to maximize AU
    
    def o6_epistemic_uncertainty(x: Array) -> float:
        """Epistemic Uncertainty (minimize - prefer confident regions)."""
        if ensemble is None or len(ensemble.models) == 0:
            raise ValueError("Ensemble required for epistemic uncertainty.")
        eu = epistemic_from_models(ensemble, x)
        if isinstance(eu, np.ndarray) and eu.ndim > 0:
            eu = eu[0]
        elif isinstance(eu, np.ndarray) and eu.ndim == 0:
            eu = eu.item()
        elif isinstance(eu, torch.Tensor) and eu.ndim > 0:
            eu = eu[0].item()
        return float(eu)  # Positive to minimize EU (prefer low epistemic uncertainty)
    
    # Build objective list based on available ensemble
    # Order: [Validity, Epistemic, Sparsity, Aleatoric]
    # This order matches what vis_tools/app.py expects
    if ensemble is not None and len(ensemble.models) > 0:
        objectives = [
            o1_validity,               # Index 0: Minimize 1 - P(target)
            o6_epistemic_uncertainty,  # Index 1: Minimize EU (prefer confident regions)
            o3_sparsity,               # Index 2: Minimize changed features
            o5_aleatoric_uncertainty,  # Index 3: Maximize AU (minimize -AU)
        ]
    else:
        objectives = [
            o1_validity,
            o2_similarity,
            o3_sparsity,
            o4_plausibility,
        ]
    
    return FunctionalProblem(
        n_var=p,
        objs=objectives,
        xl=xl.cpu().numpy(),
        xu=xu.cpu().numpy(),
        elementwise=True,
    )
