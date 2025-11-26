"""
Uncertainty quantification utilities.

Provides functions for computing aleatoric, epistemic, and total uncertainty
from model ensembles using the entropy-based decomposition.

Uncertainty Decomposition:
- Total Uncertainty (TU): Entropy of the mean prediction = H(E[p])
- Aleatoric Uncertainty (AU): Mean of individual entropies = E[H(p)]
- Epistemic Uncertainty (EU): TU - AU (model disagreement)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

Array = np.ndarray


# =============================================================================
# Core Entropy Functions
# =============================================================================

def entropy(probs: Array) -> Array:
    """
    Compute Shannon entropy along the last dimension.
    
    H(p) = -Σ p_i * log(p_i)
    
    Args:
        probs: Probability array, shape (..., n_classes)
        
    Returns:
        entropy: Entropy values, shape (...)
    """
    return -np.sum(probs * np.log(probs + 1e-12), axis=-1)


def entropy_torch(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy (PyTorch version).
    
    Args:
        probs: Probability tensor, shape (..., n_classes)
        
    Returns:
        entropy: Entropy values, shape (...)
    """
    return -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)


def softmax_logits(logits: torch.Tensor) -> Array:
    """Convert logits to numpy probability array."""
    return torch.softmax(logits, dim=-1).cpu().numpy()


# =============================================================================
# Uncertainty from Model Ensemble
# =============================================================================

def _get_probs_from_model(
    model: nn.Module,
    x: torch.Tensor,
) -> torch.Tensor:
    """Get probability predictions from a single model."""
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1)
    return probs


def _collect_ensemble_probs(
    models: Sequence[nn.Module],
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Collect probability predictions from all ensemble members.
    
    Args:
        models: Sequence of trained models
        x: Input tensor, shape (N, d)
        
    Returns:
        all_probs: Stacked probabilities, shape (n_models, N, n_classes)
    """
    all_probs = []
    for m in models:
        probs = _get_probs_from_model(m, x)
        all_probs.append(probs)
    return torch.stack(all_probs)


def aleatoric_from_models(
    models: Sequence[nn.Module],
    x: Array,
    device: torch.device,
) -> Union[float, Array]:
    """
    Compute aleatoric uncertainty (AU) from an ensemble.
    
    AU = E[H(p)] = mean entropy of individual model predictions
    
    This represents the irreducible uncertainty in the data.
    
    Args:
        models: Sequence of trained models
        x: Input features, shape (d,) or (N, d)
        device: PyTorch device
        
    Returns:
        AU values, scalar or shape (N,)
    """
    x = np.asarray(x)
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]
    
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    all_probs = _collect_ensemble_probs(models, x_t)
    
    # Entropy of each model: (n_models, N)
    entropies = entropy_torch(all_probs)
    
    # Mean entropy across models: (N,)
    mean_entropy = torch.mean(entropies, dim=0).cpu().numpy()
    
    if not is_batch:
        return float(mean_entropy[0])
    return mean_entropy


def total_uncertainty(
    model: nn.Module,
    x: Array,
    device: torch.device,
) -> Union[float, Array]:
    """
    Compute total uncertainty from a single model.
    
    TU = H(p) = entropy of the prediction
    
    Args:
        model: Trained model
        x: Input features, shape (d,) or (N, d)
        device: PyTorch device
        
    Returns:
        TU values, scalar or shape (N,)
    """
    x = np.asarray(x)
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]
    
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    
    with torch.no_grad():
        probs = softmax_logits(model(x_t))
    
    ent = entropy(probs)
    
    if not is_batch:
        return float(ent[0])
    return ent


def total_uncertainty_ensemble(
    models: Sequence[nn.Module],
    x: Array,
    device: torch.device,
) -> Union[float, Array]:
    """
    Compute total uncertainty from an ensemble.
    
    TU = H(E[p]) = entropy of the mean prediction
    
    Args:
        models: Sequence of trained models
        x: Input features, shape (d,) or (N, d)
        device: PyTorch device
        
    Returns:
        TU values, scalar or shape (N,)
    """
    x = np.asarray(x)
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]
    
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    all_probs = _collect_ensemble_probs(models, x_t)
    
    # Mean probability across models: (N, n_classes)
    mean_probs = torch.mean(all_probs, dim=0).cpu().numpy()
    
    # Entropy of the mean
    ent = entropy(mean_probs)
    
    if not is_batch:
        return float(ent[0])
    return ent


def epistemic_from_models(
    models: Sequence[nn.Module],
    x: Array,
    device: torch.device,
) -> Union[float, Array]:
    """
    Compute epistemic uncertainty from an ensemble.
    
    EU = TU - AU = H(E[p]) - E[H(p)]
    
    This represents the model uncertainty (disagreement between models).
    
    Args:
        models: Sequence of trained models
        x: Input features, shape (d,) or (N, d)
        device: PyTorch device
        
    Returns:
        EU values, scalar or shape (N,)
    """
    tu = total_uncertainty_ensemble(models, x, device)
    au = aleatoric_from_models(models, x, device)
    return tu - au


# =============================================================================
# Comprehensive Uncertainty Decomposition
# =============================================================================

@dataclass
class UncertaintyResult:
    """Container for uncertainty decomposition results."""
    
    total: Union[float, Array]  # TU = H(E[p])
    aleatoric: Union[float, Array]  # AU = E[H(p)]
    epistemic: Union[float, Array]  # EU = TU - AU
    mean_probs: Array  # Mean prediction across ensemble
    all_probs: Optional[Array] = None  # Individual model predictions
    predicted_class: Optional[int] = None


def compute_uncertainty_decomposition(
    models: Sequence[nn.Module],
    x: Array,
    device: torch.device,
    return_probs: bool = False,
) -> UncertaintyResult:
    """
    Compute full uncertainty decomposition from an ensemble.
    
    This function computes:
    - TU (Total Uncertainty): entropy of the mean prediction
    - AU (Aleatoric Uncertainty): mean of individual entropies
    - EU (Epistemic Uncertainty): TU - AU
    
    Args:
        models: Sequence of trained models
        x: Input features, shape (d,) for single point
        device: PyTorch device
        return_probs: If True, include all individual probabilities
        
    Returns:
        UncertaintyResult with all uncertainty components
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    all_probs_t = _collect_ensemble_probs(models, x_t)
    
    # Convert to numpy for consistent interface
    all_probs = all_probs_t.cpu().numpy()  # (n_models, N, n_classes)
    
    # Mean prediction: (N, n_classes)
    mean_probs = np.mean(all_probs, axis=0)
    
    # Total Uncertainty: H(E[p])
    tu = entropy(mean_probs)
    
    # Aleatoric Uncertainty: E[H(p)]
    individual_entropies = np.array([entropy(p) for p in all_probs])
    au = np.mean(individual_entropies, axis=0)
    
    # Epistemic Uncertainty: TU - AU
    eu = tu - au
    
    # Get scalar values for single input
    if mean_probs.shape[0] == 1:
        tu = float(tu[0])
        au = float(au[0])
        eu = float(eu[0])
        pred_class = int(np.argmax(mean_probs[0]))
        mean_probs = mean_probs[0]
    else:
        pred_class = None
    
    return UncertaintyResult(
        total=tu,
        aleatoric=au,
        epistemic=eu,
        mean_probs=mean_probs,
        all_probs=all_probs if return_probs else None,
        predicted_class=pred_class,
    )


# =============================================================================
# Batch Uncertainty Computation (for optimization objectives)
# =============================================================================

def batch_aleatoric(
    models: Sequence[nn.Module],
    X: Array,
    device: torch.device,
    batch_size: int = 1024,
) -> Array:
    """
    Compute aleatoric uncertainty for a batch of points efficiently.
    
    Args:
        models: Sequence of trained models
        X: Input features, shape (N, d)
        device: PyTorch device
        batch_size: Processing batch size
        
    Returns:
        AU values, shape (N,)
    """
    X = np.asarray(X)
    n = len(X)
    au = np.zeros(n)
    
    for i in range(0, n, batch_size):
        batch = X[i:i + batch_size]
        au[i:i + len(batch)] = aleatoric_from_models(models, batch, device)
    
    return au


def batch_epistemic(
    models: Sequence[nn.Module],
    X: Array,
    device: torch.device,
    batch_size: int = 1024,
) -> Array:
    """
    Compute epistemic uncertainty for a batch of points efficiently.
    
    Args:
        models: Sequence of trained models
        X: Input features, shape (N, d)
        device: PyTorch device
        batch_size: Processing batch size
        
    Returns:
        EU values, shape (N,)
    """
    X = np.asarray(X)
    n = len(X)
    eu = np.zeros(n)
    
    for i in range(0, n, batch_size):
        batch = X[i:i + batch_size]
        eu[i:i + len(batch)] = epistemic_from_models(models, batch, device)
    
    return eu


def batch_total(
    models: Sequence[nn.Module],
    X: Array,
    device: torch.device,
    batch_size: int = 1024,
) -> Array:
    """
    Compute total uncertainty for a batch of points efficiently.
    
    Args:
        models: Sequence of trained models (uses ensemble mean)
        X: Input features, shape (N, d)
        device: PyTorch device
        batch_size: Processing batch size
        
    Returns:
        TU values, shape (N,)
    """
    X = np.asarray(X)
    n = len(X)
    tu = np.zeros(n)
    
    for i in range(0, n, batch_size):
        batch = X[i:i + batch_size]
        tu[i:i + len(batch)] = total_uncertainty_ensemble(models, batch, device)
    
    return tu
