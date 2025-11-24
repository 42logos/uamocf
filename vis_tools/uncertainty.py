
"""
Uncertainty utilities: entropy, aleatoric, and epistemic calculations.
"""

from __future__ import annotations

from typing import Iterable, Sequence

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
    return torch.softmax(logits, dim=1).cpu().numpy()


def aleatoric_from_models(models: Sequence[nn.Module], x: Array, device: torch.device) -> float:
    """
    Average aleatoric uncertainty across an ensemble of models at a single point.
    """
    x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    total = 0.0
    with torch.no_grad():
        for m in models:
            probs = softmax_logits(m(x_t))
            total += float(entropy(probs)[0])
    return total / len(models)


def total_uncertainty(model: nn.Module, x: Array, device: torch.device) -> float:
    """
    Entropy of the predictive distribution from a single model.
    """
    x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = softmax_logits(model(x_t))
    return float(entropy(probs)[0])
