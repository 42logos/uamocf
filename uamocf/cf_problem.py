"""
Counterfactual problem formulation for multi-objective optimization.

Provides problem builders for both 2D feature space and MNIST image space,
with configurable objectives including uncertainty-aware objectives.

Objectives:
- o1: Validity - P(target class) or hard class constraint
- o2: Similarity - Distance to factual instance
- o3: Sparsity - Number of changed features
- o4: Plausibility - Distance to k-nearest observed samples
- o5: Aleatoric Uncertainty (maximized via negative)
- o6: Epistemic Uncertainty (minimized or maximized)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from pymoo.problems.functional import FunctionalProblem
from torch import nn

from .uncertainty import (
    aleatoric_from_models,
    epistemic_from_models,
    total_uncertainty_ensemble,
)

Array = np.ndarray


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CFConfig:
    """Configuration for counterfactual problem.
    
    Note: Both AU and EU are maximized by default (we search for regions
    with high uncertainty). The objectives return negative values so that
    pymoo minimization maximizes the actual uncertainty values.
    """
    
    k_neighbors: int = 5
    sparsity_eps: float = 0.005  # Threshold for considering a feature "changed"
    use_soft_validity: bool = True  # Use probability vs hard class
    
    # Normalization options
    normalize_similarity: bool = True
    normalize_plausibility: bool = True


@dataclass
class ImageCFConfig(CFConfig):
    """Configuration for image-space counterfactual problem."""
    
    pixel_range: Tuple[float, float] = (-1.0, 1.0)  # MNIST normalized range
    img_size: int = 16
    sparsity_eps: float = 0.05  # 5% of pixel range
    k_neighbors: int = 5


# =============================================================================
# Utility Functions
# =============================================================================

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


# =============================================================================
# Feature Space Counterfactual Problem (2D Synthetic)
# =============================================================================

def make_cf_problem(
    model: nn.Module,
    x_star: torch.Tensor,
    y_target: torch.Tensor,
    X_obs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    config: CFConfig = None,
    ensemble: Optional[Sequence[nn.Module]] = None,
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
    config = config or CFConfig()
    device = device or next(model.parameters()).device
    
    model.eval()
    if ensemble is not None:
        for m in ensemble:
            m.eval()
    
    # Convert to numpy
    x_star_np = x_star.detach().cpu().numpy().astype(np.float32)
    X_obs_np = X_obs.detach().cpu().numpy().astype(np.float32)
    
    if weights is not None:
        w_np = weights.detach().cpu().numpy().astype(np.float32)
        w_np = w_np / (w_np.sum() + 1e-12)
    else:
        w_np = np.ones(config.k_neighbors) / config.k_neighbors
    
    # Feature bounds and range
    xl = X_obs_np.min(axis=0)
    xu = X_obs_np.max(axis=0)
    feature_range = xu - xl
    feature_range[feature_range == 0] = 1.0
    
    target_labels = y_target.view(-1).long().tolist()
    p = x_star_np.shape[0]
    
    # =========== OBJECTIVE FUNCTIONS ===========
    
    def o1_validity(x: Array) -> float:
        """Validity: 1 - P(target) or hard constraint."""
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = nn.Softmax(dim=1)(model(x_t))
        
        if config.use_soft_validity:
            prob_target = probs[0, target_labels].sum().item()
            return 1.0 - prob_target
        else:
            pred = probs.argmax(dim=1).item()
            return 0.0 if pred in target_labels else 1.0
    
    def o2_similarity(x: Array) -> float:
        """Similarity: mean Gower distance to factual."""
        if config.normalize_similarity:
            return float(_gower_distance(x, x_star_np, feature_range).mean())
        return float(np.linalg.norm(x - x_star_np))
    
    def o3_sparsity(x: Array) -> float:
        """Sparsity: number of changed features."""
        changed = np.abs(x - x_star_np) > config.sparsity_eps
        return float(changed.sum())
    
    def o4_plausibility(x: Array) -> float:
        """Plausibility: weighted distance to k-nearest neighbors."""
        nearest_samples, _ = _k_nearest(x, X_obs_np, config.k_neighbors)
        per_sample = np.array([
            _gower_distance(x, xi, feature_range).mean()
            for xi in nearest_samples
        ])
        return float((per_sample * w_np).sum())
    
    def o5_aleatoric_uncertainty(x: Array) -> float:
        """Aleatoric Uncertainty (negated to maximize)."""
        if ensemble is None or len(ensemble) == 0:
            raise ValueError("Ensemble required for aleatoric uncertainty.")
        au = aleatoric_from_models(ensemble, x, device)
        return -au  # Negative to maximize AU
    
    def o6_epistemic_uncertainty(x: Array) -> float:
        """Epistemic Uncertainty (minimize - prefer confident regions)."""
        if ensemble is None or len(ensemble) == 0:
            raise ValueError("Ensemble required for epistemic uncertainty.")
        eu = epistemic_from_models(ensemble, x, device)
        return eu  # Positive to minimize EU (prefer low epistemic uncertainty)
    
    # Build objective list based on available ensemble
    # Order: [Validity, Sparsity, AU (maximize), EU (minimize)]
    if ensemble is not None and len(ensemble) > 0:
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
        xl=xl,
        xu=xu,
        elementwise=True,
    )


# =============================================================================
# Image Space Counterfactual Problem (MNIST)
# =============================================================================

def make_cf_problem_image_space(
    model: nn.Module,
    x_star_image: Array,
    target_class: int,
    X_obs_images: Optional[Array] = None,
    ensemble: Optional[Sequence[nn.Module]] = None,
    config: ImageCFConfig = None,
    device: Optional[torch.device] = None,
) -> FunctionalProblem:
    """
    Create a counterfactual problem in image pixel space.
    
    The decision variables are img_size × img_size pixel values.
    This allows generating NEW images, not just finding existing ones.
    
    Objectives:
    - o1: Validity - P(target class)
    - o2: Similarity - L2 distance in pixel space (normalized)
    - o3: Sparsity - fraction of changed pixels
    - o4: Plausibility - distance to k-nearest observed images
    - o5: Aleatoric Uncertainty (negated to maximize)
    - o6: Epistemic Uncertainty
    
    Args:
        model: Trained MNIST classifier
        x_star_image: Factual image, shape (img_size, img_size) or (1, img_size, img_size)
        target_class: Desired counterfactual class (0-9)
        X_obs_images: Observed images for plausibility, shape (n, img_size, img_size)
        ensemble: Optional model ensemble for uncertainty
        config: ImageCFConfig instance
        device: PyTorch device
        
    Returns:
        pymoo FunctionalProblem
    """
    config = config or ImageCFConfig()
    device = device or next(model.parameters()).device
    img_size = config.img_size
    
    model.eval()
    if ensemble is not None:
        for m in ensemble:
            m.eval()
    
    # Flatten factual image
    x_star_flat = x_star_image.flatten()
    n_pixels = len(x_star_flat)
    
    # Pixel bounds
    xl = np.full(n_pixels, config.pixel_range[0])
    xu = np.full(n_pixels, config.pixel_range[1])
    
    # Flatten observed images
    if X_obs_images is not None:
        X_obs_flat = X_obs_images.reshape(len(X_obs_images), -1)
    else:
        X_obs_flat = None
    
    # Helper: convert flat array to tensor
    def _to_tensor(x_flat: Array) -> torch.Tensor:
        img = x_flat.reshape(1, 1, img_size, img_size)
        return torch.tensor(img, dtype=torch.float32).to(device)
    
    # Helper: get probabilities from model
    def _get_probs(x_flat: Array, m: nn.Module = None) -> Array:
        m = m if m is not None else model
        m.eval()
        with torch.no_grad():
            logits = m(_to_tensor(x_flat))
            probs = nn.Softmax(dim=1)(logits).cpu().numpy()[0]
        return probs
    
    # =========== OBJECTIVE FUNCTIONS ===========
    
    def o1_validity(x: Array) -> float:
        """Validity: 1 - P(target)."""
        probs = _get_probs(x)
        return 1.0 - probs[target_class]
    
    def o2_similarity(x: Array) -> float:
        """Similarity: normalized L2 distance."""
        diff = x - x_star_flat
        l2_dist = np.linalg.norm(diff)
        max_dist = np.sqrt(n_pixels) * (config.pixel_range[1] - config.pixel_range[0])
        return l2_dist / max_dist
    
    def o3_sparsity(x: Array) -> float:
        """Sparsity: fraction of changed pixels."""
        changed = np.abs(x - x_star_flat) > config.sparsity_eps
        return float(np.sum(changed)) / n_pixels
    
    def o4_plausibility(x: Array) -> float:
        """Plausibility: mean L2 distance to k-nearest images."""
        if X_obs_flat is None:
            return 0.0
        _, dists = _k_nearest(x, X_obs_flat, config.k_neighbors)
        return float(np.mean(dists)) / np.sqrt(n_pixels)
    
    def o5_aleatoric_uncertainty(x: Array) -> float:
        """Aleatoric Uncertainty (negated to maximize)."""
        if ensemble is None or len(ensemble) == 0:
            # Use single model entropy
            probs = _get_probs(x)
            entropy = -np.sum(probs * np.log(probs + 1e-12))
            return -entropy
        
        # Average entropy across ensemble
        au = 0.0
        for m in ensemble:
            probs = _get_probs(x, m)
            au += -np.sum(probs * np.log(probs + 1e-12))
        au /= len(ensemble)
        return -au  # Negative to maximize
    
    def o6_epistemic_uncertainty(x: Array) -> float:
        """Epistemic Uncertainty (negated to maximize)."""
        if ensemble is None or len(ensemble) == 0:
            return 0.0
        
        # Collect predictions from all models
        all_probs = np.array([_get_probs(x, m) for m in ensemble])
        
        # TU: entropy of mean prediction
        mean_probs = np.mean(all_probs, axis=0)
        TU = -np.sum(mean_probs * np.log(mean_probs + 1e-12))
        
        # AU: mean of individual entropies
        AU = np.mean([-np.sum(p * np.log(p + 1e-12)) for p in all_probs])
        
        # EU = TU - AU (always non-negative by Jensen's inequality)
        EU = TU - AU
        return EU  # Positive to minimize (we want low EU / confident regions)
    
    # Build objective list
    if ensemble is not None and len(ensemble) > 0:
        objectives = [
            o1_validity,           # Minimize 1 - P(target)
            o3_sparsity,           # Minimize changed pixels
            o5_aleatoric_uncertainty,  # Maximize AU (minimize -AU)
            o6_epistemic_uncertainty,  # Minimize EU (prefer confident regions)
        ]
        print(f"Objectives: [Validity, Sparsity, AU (max), EU (min)]")
    else:
        objectives = [
            o1_validity,
            o2_similarity,
            o3_sparsity,
            o4_plausibility,
        ]
        print(f"Objectives: [Validity, Similarity, Sparsity, Plausibility]")
    
    return FunctionalProblem(
        n_var=n_pixels,
        objs=objectives,
        xl=xl,
        xu=xu,
        elementwise=True,
    )


# =============================================================================
# Factory Functions
# =============================================================================

def create_2d_cf_problem(
    model: nn.Module,
    x_star: Array,
    y_target: int,
    X_obs: Array,
    ensemble: Optional[Sequence[nn.Module]] = None,
    config: CFConfig = None,
    device: Optional[torch.device] = None,
) -> FunctionalProblem:
    """
    Convenience function to create 2D CF problem from numpy arrays.
    
    Args:
        model: Trained classifier
        x_star: Factual point, shape (d,)
        y_target: Target class
        X_obs: Observed data, shape (n, d)
        ensemble: Optional model ensemble
        config: CFConfig
        device: PyTorch device
        
    Returns:
        FunctionalProblem
    """
    device = device or next(model.parameters()).device
    config = config or CFConfig()
    
    x_star_t = torch.tensor(x_star, dtype=torch.float32, device=device)
    y_target_t = torch.tensor([y_target], dtype=torch.long, device=device)
    X_obs_t = torch.tensor(X_obs, dtype=torch.float32, device=device)
    weights = torch.ones(config.k_neighbors, device=device)
    
    return make_cf_problem(
        model=model,
        x_star=x_star_t,
        y_target=y_target_t,
        X_obs=X_obs_t,
        weights=weights,
        config=config,
        ensemble=ensemble,
        device=device,
    )


def create_mnist_cf_problem(
    model: nn.Module,
    x_star_image: Array,
    target_class: int,
    X_obs_images: Optional[Array] = None,
    ensemble: Optional[Sequence[nn.Module]] = None,
    img_size: int = 16,
    device: Optional[torch.device] = None,
) -> FunctionalProblem:
    """
    Convenience function to create MNIST image-space CF problem.
    
    Args:
        model: Trained MNIST classifier
        x_star_image: Factual image
        target_class: Target digit class
        X_obs_images: Observed images for plausibility
        ensemble: Optional model ensemble
        img_size: Image size (16 or 28)
        device: PyTorch device
        
    Returns:
        FunctionalProblem
    """
    config = ImageCFConfig(img_size=img_size)
    
    return make_cf_problem_image_space(
        model=model,
        x_star_image=x_star_image,
        target_class=target_class,
        X_obs_images=X_obs_images,
        ensemble=ensemble,
        config=config,
        device=device,
    )
