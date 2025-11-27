"""
Uncertainty utilities: entropy, aleatoric, and epistemic calculations.

This module re-exports uncertainty functions from core.cf_problem.
Thin wrappers are provided for backward compatibility with the old API
that accepts a list of models instead of EnsembleModel.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
from torch import nn

# Re-export core uncertainty functions directly
from core.cf_problem import (
    aleatoric_from_models as core_aleatoric_from_models,
    Total_uncertainty as core_total_uncertainty,
    epistemic_from_models as core_epistemic_from_models,
)
from core.models import EnsembleModel

Array = np.ndarray


def entropy(probs: Array) -> Array:
    """Compute Shannon entropy for probabilities along the last dimension."""
    return -np.sum(probs * np.log(probs + 1e-12), axis=-1)


def softmax_logits(logits: torch.Tensor) -> Array:
    """Convert logits to probabilities as numpy array."""
    return torch.softmax(logits, dim=-1).cpu().numpy()


def _ensure_ensemble(models: Sequence[nn.Module], device: torch.device) -> EnsembleModel:
    """Convert model sequence to EnsembleModel if needed."""
    if isinstance(models, EnsembleModel):
        return models
    ensemble = EnsembleModel(list(models))
    ensemble.to(device)
    return ensemble


# Backward-compatible wrappers that accept Sequence[nn.Module] instead of EnsembleModel

def aleatoric_from_models(models: Sequence[nn.Module], x: Array, device: torch.device) -> Union[float, Array]:
    """Wrapper around core.cf_problem.aleatoric_from_models for backward compatibility."""
    return core_aleatoric_from_models(_ensure_ensemble(models, device), x)


def total_uncertainty(model: nn.Module, x: Array, device: torch.device) -> Union[float, Array]:
    """Entropy of the predictive distribution from a single model."""
    x = np.asarray(x)
    is_batch = x.ndim > 1
    if not is_batch:
        x = x[None, :]
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    with torch.no_grad():
        ent = entropy(softmax_logits(model(x_t)))
    return float(ent[0]) if not is_batch else ent


def total_uncertainty_ensemble(models: Sequence[nn.Module], x: Array, device: torch.device) -> Union[float, Array]:
    """Wrapper around core.cf_problem.Total_uncertainty for backward compatibility."""
    return core_total_uncertainty(_ensure_ensemble(models, device), x, device)


def epistemic_from_models(models: Sequence[nn.Module], x: Array, device: torch.device) -> Union[float, Array]:
    """Wrapper around core.cf_problem.epistemic_from_models for backward compatibility."""
    return core_epistemic_from_models(_ensure_ensemble(models, device), x)
