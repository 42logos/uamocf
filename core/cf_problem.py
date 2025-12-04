from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union
from typing_extensions import Literal

from core import models
from core.models import EnsembleModel

import numpy as np
import torch
from pymoo.core.problem import Problem
from torch import nn

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

def _gower_distance_batch(X: torch.Tensor, y: torch.Tensor, feature_range: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized feature-wise dissimilarity (Gower distance) for batch.
    
    Args:
        X: Batch of feature vectors, shape (N, d)
        y: Reference point, shape (d,)
        feature_range: Range of each feature, shape (d,)
        
    Returns:
        Per-sample mean Gower distance, shape (N,)
    """
    per_feature = torch.clamp(torch.abs(X - y) / feature_range, max=1.0)  # (N, d)
    return per_feature.mean(dim=1)  # (N,)

def _k_nearest_batch(X: torch.Tensor, X_obs: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k nearest neighbors for each point in batch X.
    
    Args:
        X: Query points, shape (N, d)
        X_obs: Observed data points, shape (M, d)
        k: Number of neighbors
        
    Returns:
        nearest_indices: shape (N, k)
        distances: shape (N, k)
    """
    # Compute pairwise distances: (N, M)
    dists = torch.cdist(X, X_obs)
    # Get k nearest for each query point
    distances, nearest_idx = torch.topk(dists, k, dim=1, largest=False)
    return nearest_idx, distances


def aleatoric_from_models_tensor(model_ensemble: EnsembleModel, x_t: torch.Tensor) -> torch.Tensor:
    """
    Average aleatoric uncertainty across an ensemble of models.
    Pure torch tensor version - no numpy conversions.
    
    Args:
        model_ensemble: Ensemble of models
        x_t: Input tensor on device, shape (N, d) or (N, C, H, W)
        
    Returns:
        Aleatoric uncertainty tensor, shape (N,)
    """
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
    entropies = -torch.sum(all_probs * torch.log(all_probs + 1e-12), dim=-1)  # (n_models, N)
    mean_entropy = torch.mean(entropies, dim=0)  # (N,)
    
    return mean_entropy


def aleatoric_from_models(model_ensemble: EnsembleModel, x) -> np.ndarray:
    """
    Average aleatoric uncertainty across an ensemble of models.
    Legacy wrapper that returns numpy array for backward compatibility.
    """
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]
        
    x_t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x.astype(np.float32))
    x_t = x_t.to(model_ensemble.device)
    
    result = aleatoric_from_models_tensor(model_ensemble, x_t)
    return result.cpu().numpy()

def total_uncertainty_tensor(model_ensemble: EnsembleModel, x_t: torch.Tensor) -> torch.Tensor:
    """
    Total uncertainty = Entropy of the mean predictive distribution.
    Pure torch tensor version - no numpy conversions.
    
    Args:
        model_ensemble: Ensemble of models
        x_t: Input tensor on device, shape (N, d) or (N, C, H, W)
        
    Returns:
        Total uncertainty tensor, shape (N,)
    """
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
    
    return ent


def epistemic_from_models_tensor(model_ensemble: EnsembleModel, x_t: torch.Tensor) -> torch.Tensor:
    """
    Epistemic uncertainty from an ensemble of models.
    Pure torch tensor version - no numpy conversions.
    
    EU = TU - AU = H[E[p]] - E[H[p]]
    
    Args:
        model_ensemble: Ensemble of models
        x_t: Input tensor on device, shape (N, d) or (N, C, H, W)
        
    Returns:
        Epistemic uncertainty tensor, shape (N,)
    """
    AU = aleatoric_from_models_tensor(model_ensemble, x_t)
    TU = total_uncertainty_tensor(model_ensemble, x_t)
    EU = TU - AU
    # EU should be non-negative by Jensen's inequality, clamp for numerical stability
    return torch.clamp(EU, min=0.0)


def Total_uncertainty(model_ensemble: EnsembleModel, x, device: Optional[torch.device] = None) -> Union[float, np.ndarray]:
    """
    Total uncertainty = Entropy of the mean predictive distribution.
    Legacy wrapper that returns numpy array for backward compatibility.
    """
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]
    
    if device is None:
        device = model_ensemble.device
    x_t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x.astype(np.float32))
    x_t = x_t.to(device)
    
    result = total_uncertainty_tensor(model_ensemble, x_t)
    result_np = result.cpu().numpy()
    
    if not is_batch:
        return float(result_np[0])
    return result_np


def epistemic_from_models(model_ensemble: EnsembleModel, x) -> Union[float, np.ndarray]:
    """
    Epistemic uncertainty from an ensemble of models.
    Legacy wrapper that returns numpy array for backward compatibility.
    """
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]
        
    x_t = x if isinstance(x, torch.Tensor) else torch.from_numpy(x.astype(np.float32))
    x_t = x_t.to(model_ensemble.device)
    
    result = epistemic_from_models_tensor(model_ensemble, x_t)
    result_np = result.cpu().numpy()
    
    if not is_batch:
        return float(result_np[0])
    return result_np

class TorchCFProblem(Problem):
    """
    Counterfactual optimization problem with full GPU batch computation.
    
    All computations are done on GPU using torch tensors. Only converts to
    numpy at the pymoo interface boundary for maximum efficiency.
    
    Objectives (with ensemble):
    - o1: Validity (1 - P(target))
    - o2: Epistemic Uncertainty
    - o3: Sparsity
    - o4: -Aleatoric Uncertainty (negated to maximize)
    
    Objectives (without ensemble):
    - o1: Validity
    - o2: Similarity
    - o3: Sparsity
    - o4: Plausibility
    """
    
    def __init__(
        self,
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
    ):
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ensemble = ensemble
        self.use_soft_validity = use_soft_validity
        self.normalize_similarity = normalize_similarity
        self.sparsity_eps = sparsity_eps
        self.k_neighbors = k_neighbors
        
        model.eval()
        if ensemble is not None:
            for m in ensemble.models:
                m.eval()
        
        # Store dimensions
        self.p = x_factual.flatten().shape[0]
        self.input_shape = x_factual.shape
        
        # Store tensors on device
        self.X_obs_flat = X_obs.reshape(X_obs.shape[0], -1).to(self.device)
        self.x_factual_flat = x_factual.flatten().to(self.device)
        
        # Compute bounds
        xl = torch.min(self.X_obs_flat, dim=0).values
        xu = torch.max(self.X_obs_flat, dim=0).values
        self.feature_range = xu - xl
        self.feature_range[self.feature_range == 0] = 1.0
        
        # Store target labels
        self.target_labels = y_target.view(-1).long().to(self.device)
        
        # Neighbor weights
        self.w = torch.ones(k_neighbors, device=self.device) / k_neighbors
        
        # Determine number of objectives
        n_obj = 4
        
        # Initialize parent Problem
        super().__init__(
            n_var=self.p,
            n_obj=n_obj,
            xl=xl.cpu().numpy(),
            xu=xu.cpu().numpy(),
        )
    
    def _to_model_input(self, X_t: torch.Tensor) -> torch.Tensor:
        """Reshape flat batch to model input shape."""
        batch_size = X_t.shape[0]
        # Determine target shape based on input_shape
        if len(self.input_shape) == 4:  # (1, C, H, W)
            target_shape = (batch_size, self.input_shape[1], self.input_shape[2], self.input_shape[3])
        elif len(self.input_shape) == 3:  # (C, H, W)
            target_shape = (batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        else:  # (d,) or (1, d)
            target_shape = (batch_size, self.p)
        return X_t.view(target_shape)
    
    def _evaluate(self, X: np.ndarray, out: dict, *args, **kwargs):
        """
        Batch evaluation of all objectives on GPU.
        
        Args:
            X: Decision variables, shape (pop_size, n_var)
            out: Output dictionary for objectives
        """
        # ONE conversion to GPU tensor per generation
        X_t = torch.from_numpy(X).float().to(self.device)
        batch_size = X_t.shape[0]
        
        # Reshape for model input
        X_model = self._to_model_input(X_t)
        
        # ===== OBJECTIVE 1: Validity =====
        with torch.no_grad():
            logits = self.model(X_model)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            probs = torch.softmax(logits, dim=1)
        
        if self.use_soft_validity:
            # Sum probabilities of target classes
            prob_target = probs[:, self.target_labels].sum(dim=1)  # (N,)
            f1_validity = 1.0 - prob_target
        else:
            preds = probs.argmax(dim=1)
            # Check if prediction is in target labels
            is_valid = torch.isin(preds, self.target_labels)
            f1_validity = torch.where(is_valid, torch.zeros_like(preds, dtype=torch.float32), torch.ones_like(preds, dtype=torch.float32))
        
        # ===== OBJECTIVE 3: Sparsity (computed before 2 for efficiency) =====
        changed = torch.abs(X_t - self.x_factual_flat) > self.sparsity_eps  # (N, p)
        f3_sparsity = changed.sum(dim=1).float()  # (N,)
        
        if self.ensemble is not None and len(self.ensemble.models) > 0:
            # ===== OBJECTIVE 2: Epistemic Uncertainty =====
            f2_epistemic = epistemic_from_models_tensor(self.ensemble, X_model)  # (N,)
            
            # ===== OBJECTIVE 4: -Aleatoric Uncertainty (negated to maximize) =====
            f4_aleatoric_neg = -aleatoric_from_models_tensor(self.ensemble, X_model)  # (N,)
            
            # Stack objectives: [Validity, Epistemic, Sparsity, -AU]
            F = torch.stack([f1_validity, f2_epistemic, f3_sparsity, f4_aleatoric_neg], dim=1)
        else:
            # ===== OBJECTIVE 2: Similarity =====
            if self.normalize_similarity:
                f2_similarity = _gower_distance_batch(X_t, self.x_factual_flat, self.feature_range)
            else:
                f2_similarity = torch.norm(X_t - self.x_factual_flat, dim=1)
            
            # ===== OBJECTIVE 4: Plausibility (k-nearest neighbors) =====
            nearest_idx, _ = _k_nearest_batch(X_t, self.X_obs_flat, self.k_neighbors)  # (N, k)
            
            # Compute weighted distance to neighbors
            f4_plausibility = torch.zeros(batch_size, device=self.device)
            for i in range(batch_size):
                neighbor_points = self.X_obs_flat[nearest_idx[i]]  # (k, p)
                dists = _gower_distance_batch(neighbor_points, X_t[i], self.feature_range)  # (k,)
                f4_plausibility[i] = (dists * self.w).sum()
            
            # Stack objectives: [Validity, Similarity, Sparsity, Plausibility]
            F = torch.stack([f1_validity, f2_similarity, f3_sparsity, f4_plausibility], dim=1)
        
        # ONE conversion to numpy at output
        out['F'] = F.cpu().numpy()


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
) -> TorchCFProblem:
    """
    Create a counterfactual problem for feature space optimization.
    
    This is a factory function that creates a TorchCFProblem with full GPU
    batch computation support. All objectives are computed on GPU and only
    converted to numpy at the pymoo interface.
    
    Objectives (with ensemble):
    - o1: Validity (1 - P(target))
    - o2: Epistemic Uncertainty
    - o3: Sparsity
    - o4: -Aleatoric Uncertainty (negated to maximize)
    
    Objectives (without ensemble):
    - o1: Validity
    - o2: Similarity
    - o3: Sparsity
    - o4: Plausibility
    
    Args:
        model: Primary trained classifier
        x_factual: Factual instance, shape (d,) or (1, C, H, W)
        y_target: Target class(es), shape (1,) or (k,)
        X_obs: Observed data, shape (n, d) or (n, C, H, W)
        k_neighbors: Number of neighbors for plausibility
        use_soft_validity: Use soft validity (probability) or hard (prediction)
        normalize_similarity: Use Gower distance (True) or L2 norm (False)
        sparsity_eps: Threshold for considering a feature changed
        ensemble: Optional model ensemble for uncertainty objectives
        device: PyTorch device
        
    Returns:
        TorchCFProblem instance for pymoo optimization
    """
    return TorchCFProblem(
        model=model,
        x_factual=x_factual,
        y_target=y_target,
        X_obs=X_obs,
        k_neighbors=k_neighbors,
        use_soft_validity=use_soft_validity,
        normalize_similarity=normalize_similarity,
        sparsity_eps=sparsity_eps,
        ensemble=ensemble,
        device=device,
    )
