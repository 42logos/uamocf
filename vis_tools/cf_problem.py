
"""
Counterfactual objective builder using pymoo's FunctionalProblem.

This module provides a simplified interface for 2D synthetic experiments.
For more advanced use cases with higher-dimensional data, use core.cf_problem directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from pymoo.problems.functional import FunctionalProblem
from torch import nn

# Re-export core utilities - these are the actual implementations
from core.cf_problem import (
    _gower_distance,
    _gower_distance_tensor,
    _k_nearest,
    _k_nearest_tensor,
    aleatoric_from_models,
    Total_uncertainty,
    epistemic_from_models,
    make_cf_problem as core_make_cf_problem,
)
from core.models import EnsembleModel

Array = np.ndarray


@dataclass
class CFConfig:
    """Configuration for counterfactual problem (vis_tools simplified interface)."""
    k_neighbors: int = 5
    sparsity_eps: float = 0.005
    feature_weights: Optional[Array] = None
    use_soft_validity: bool = True


def make_cf_problem(
    model: nn.Module,
    x_star: torch.Tensor,
    y_target: torch.Tensor,
    X_obs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    config: Optional[CFConfig] = None,
    ensemble: Optional[Sequence[nn.Module]] = None,
    bayesian_model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> FunctionalProblem:
    """
    Create a pymoo FunctionalProblem for 2D synthetic counterfactual experiments.
    
    This is a thin wrapper around core.cf_problem.make_cf_problem with a simplified
    interface for backward compatibility with vis_tools experiments.
    
    For new code, prefer using core.cf_problem.make_cf_problem directly.
    """
    if config is None:
        config = CFConfig()
    
    # Convert ensemble list to EnsembleModel if needed
    ensemble_model = None
    if ensemble is not None and len(ensemble) > 0:
        if isinstance(ensemble, EnsembleModel):
            ensemble_model = ensemble
        else:
            ensemble_model = EnsembleModel(list(ensemble))
            if device is not None:
                ensemble_model.to(device)
    
    # Delegate to core implementation
    return core_make_cf_problem(
        model=model,
        x_factual=x_star,
        y_target=y_target,
        X_obs=X_obs,
        k_neighbors=config.k_neighbors,
        use_soft_validity=config.use_soft_validity,
        normalize_similarity=True,
        sparsity_eps=config.sparsity_eps,
        ensemble=ensemble_model,
        device=device,
    )
