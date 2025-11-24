
"""
Uncertainty utilities: entropy, aleatoric, and epistemic calculations.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Union

import numpy as np
import torch
from torch import nn

Array = np.ndarray


def entropy(probs: Array) -> Array:
    """
    Compute Shannon entropy for probabilities along the last dimension.
    """
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=-1)


def softmax_logits(logits: torch.Tensor) -> Array:
    return torch.softmax(logits, dim=-1).cpu().numpy()


def aleatoric_from_models(models: Sequence[nn.Module], x: Array, device: torch.device) -> Union[float, Array]:
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
        for m in models:
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


def total_uncertainty(model: nn.Module, x: Array, device: torch.device) -> Union[float, Array]:
    """
    Entropy of the predictive distribution from a single model.
    Supports both single point (d,) and batch (N, d).
    """
    x = np.asarray(x)
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]
        
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    with torch.no_grad():
        probs = softmax_logits(model(x_t))
        
    ent = entropy(probs) # (N,)
    
    if not is_batch:
        return float(ent[0])
    return ent
