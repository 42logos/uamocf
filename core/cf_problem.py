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


def aleatoric_from_models(model_ensemble:EnsembleModel, x: Array, device: Optional[torch.device]=None) -> Union[float, Array]:
    """
    Average aleatoric uncertainty across an ensemble of models.
    Supports both single point (d,) and batch (N, d).
    """
    x = np.asarray(x)
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :] # (1, d)
        
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    
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
    entropies = -torch.sum(all_probs * torch.log(all_probs + 1e-12), dim=-1) # (n_models, N)
    mean_entropy = torch.mean(entropies, dim=0).cpu().numpy() # (N,)
    
    if not is_batch:
        return float(mean_entropy[0])
    return mean_entropy

def Total_uncertainty(model_ensemble:EnsembleModel, x: Array, device: Optional[torch.device]=None) -> Union[float, Array]:
    """
    Entropy of the predictive distribution from a single model.
    Supports both single point (d,) and batch (N, d).
    """
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]
    
    if device is None:
        device = model_ensemble.device
    x = torch.from_numpy(x.astype(np.float32)).to(device) if isinstance(x, np.ndarray) else x.to(device)

    with torch.no_grad():
        probs = torch.softmax(model_ensemble(x), dim=-1)
        
    ent = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1).cpu().numpy()  # (N,)
    
    if not is_batch:
        return float(ent[0])
    return ent

def epistemic_from_models(model_ensemble:EnsembleModel, x: Array, device: Optional[torch.device]=None) -> Union[float, Array]:
    """
    Epistemic uncertainty from an ensemble of models.
    Supports both single point (d,) and batch (N, d).
    """
    AU=aleatoric_from_models(model_ensemble, x, device)
    TU=Total_uncertainty(model_ensemble, x, device)
    return TU - AU

def make_cf_problem(
    model: nn.Module,
    x_factual: torch.Tensor,
    y_target: torch.Tensor,
    X_obs: torch.Tensor,
    k_neighbors: int = 10,
    use_soft_validity: bool = True,
    normalize_similarity: bool = True,
    sparsity_eps: float = 0.5,
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
        return x_t.view(input_shape)
    
    # OBJECTIVE FUNCTIONS 
    def o1_validity(x: Array) -> float:
        """Validity: 1 - P(target) or hard constraint."""
        x_t = _to_model_input(x)
        
        with torch.no_grad():
            probs = nn.Softmax(dim=1)(model(x_t))
        
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
        au = aleatoric_from_models(ensemble, x, device)
        return -au  # Negative to maximize AU
    
    def o6_epistemic_uncertainty(x: Array) -> float:
        """Epistemic Uncertainty (minimize - prefer confident regions)."""
        if ensemble is None or len(ensemble.models) == 0:
            raise ValueError("Ensemble required for epistemic uncertainty.")
        eu = epistemic_from_models(ensemble, x, device)
        return eu  # Positive to minimize EU (prefer low epistemic uncertainty)
    
    # Build objective list based on available ensemble
    # Order: [Validity, Sparsity, AU (maximize), EU (minimize)]
    if ensemble is not None and len(ensemble.models) > 0:
        objectives = [
            o1_validity,           # Minimize 1 - P(target)
            o3_sparsity,           # Minimize changed features
            o5_aleatoric_uncertainty,  # Maximize AU (minimize -AU)
            o6_epistemic_uncertainty,  # Minimize EU (prefer confident regions)
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
