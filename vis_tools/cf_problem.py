
"""
Counterfactual objective builder using pymoo's FunctionalProblem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pymoo.problems.functional import FunctionalProblem
from torch import nn

from .uncertainty import aleatoric_from_models, total_uncertainty, total_uncertainty_ensemble

Array = np.ndarray


@dataclass
class CFConfig:
    k_neighbors: int = 5
    sparsity_eps: float = 0.005
    feature_weights: Optional[Array] = None  # shape (k_neighbors,)
    use_soft_validity: bool = True


def _prepare_arrays(x_star: torch.Tensor, X_obs: torch.Tensor, weights: Optional[torch.Tensor]) -> Tuple[Array, Array, Optional[Array]]:
    x_star_np = x_star.detach().cpu().numpy().astype(np.float32)
    X_obs_np = X_obs.detach().cpu().numpy().astype(np.float32)
    w_np: Optional[Array] = None
    if weights is not None:
        w_np = weights.detach().cpu().numpy().astype(np.float32)
        if w_np.ndim == 0:
            w_np = w_np.reshape(1)
        w_np = w_np / (w_np.sum() + 1e-12)
    return x_star_np, X_obs_np, w_np


def make_cf_problem(
    model: nn.Module,
    x_star: torch.Tensor,
    y_target: torch.Tensor,
    X_obs: torch.Tensor,
    weights: Optional[torch.Tensor],
    config: CFConfig,
    ensemble: Optional[Sequence[nn.Module]] = None,
    bayesian_model: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> FunctionalProblem:
    """
    Create a pymoo FunctionalProblem encapsulating the counterfactual objectives.
    """
    device = device or next(model.parameters()).device
    model.eval()
    if bayesian_model is not None:
        bayesian_model.eval()

    x_star_np, X_obs_np, w_np = _prepare_arrays(x_star, X_obs, weights)
    feature_range = X_obs_np.max(axis=0) - X_obs_np.min(axis=0)
    feature_range[feature_range == 0] = 1.0

    target_labels = y_target.view(-1).long().tolist()
    p = x_star_np.shape[0]

    def delta_G_vec(x: Array, y: Array) -> Array:
        return np.minimum(np.abs(x - y) / feature_range, 1.0)

    def k_nearest(x: Array, k: int) -> Array:
        dists = np.linalg.norm(X_obs_np - x, axis=1)
        nearest_indices = np.argsort(dists)[:k]
        return X_obs_np[nearest_indices]

    def o1_validity(x: Array) -> float:
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = nn.Softmax(dim=1)(model(x_t))
        prob_target = probs[0, target_labels].sum().item()
        return 1.0 - prob_target if config.use_soft_validity else (0.0 if probs.argmax(dim=1)[0].item() in target_labels else 1.0)

    def o2_similarity(x: Array) -> float:
        return float(delta_G_vec(x, x_star_np).mean())

    def o3_sparsity(x: Array) -> float:
        diff = np.abs(x - x_star_np) > config.sparsity_eps
        return float(diff.sum())

    def o4_plausibility(x: Array) -> float:
        nearest_samples = k_nearest(x, k=config.k_neighbors)
        per_sample = np.array([delta_G_vec(x, xi).mean() for xi in nearest_samples])
        if w_np is None:
            return float(per_sample.mean())
        return float((per_sample * w_np).sum())

    def aleatoric_uncertainty(x: Array) -> float:
        if ensemble is None or len(ensemble) == 0:
            raise ValueError("Ensemble models must be provided for aleatoric uncertainty.")
        return aleatoric_from_models(ensemble, x, device)
    
    # since our goal is maximizing the Aleatoric uncertainty, we define the objective as its negative
    def o5_aleatoric_uncertainty(x: Array) -> float:
        return -aleatoric_uncertainty(x)

    def o6_epistemic_uncertainty(x: Array) -> float:
        if ensemble is None:
            raise ValueError("Ensemble must be provided for epistemic uncertainty.")
        total = total_uncertainty_ensemble(ensemble, x, device)
        alea = aleatoric_uncertainty(x)
        return total - alea

    return FunctionalProblem(
        n_var=p,
        objs=[o1_validity, o6_epistemic_uncertainty, o3_sparsity, o5_aleatoric_uncertainty],
        xl=X_obs_np.min(axis=0),
        xu=X_obs_np.max(axis=0),
        elementwise=True,
    )
