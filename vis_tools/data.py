
"""
Data generation utilities and probability helpers.

This module re-exports core.data classes and provides thin functional wrappers
for backward compatibility with 2-class experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np

# Re-export ALL core data classes and functions
from core.data import (
    DataGenerator,
    Prob_FN,
    PlaneDG,
    DatasetsDG,
    circle_prob,
    moon_prob,
    SamplingConfig,
    dpg,
    sample_from_config,
)

Array = np.ndarray

# -----------------------------
# Backward-compatible aliases
# -----------------------------

# DataConfig is an alias for SamplingConfig for backward compatibility
DataConfig = SamplingConfig


def sample_dataset(cfg: "DataConfig") -> Tuple[Array, Array, Array]:
    """
    Backward-compatible wrapper around core.data.sample_from_config.
    """
    X, Y, p = sample_from_config(cfg)
    return np.asarray(X), np.asarray(Y), np.asarray(p)


# -----------------------------
# Functional wrappers using core classes
# These provide backward-compatible functional interface for 2-class
# -----------------------------

# Create singleton instances of core probability classes
_circle_prob_instance = circle_prob()
_moon_prob_instance = moon_prob()


def boundary_focus_prob(X: Array, **kwargs) -> Array:
    """
    Functional wrapper around core.data.circle_prob.
    Returns P(Y=1|X) - the probability of class 1.
    
    For full control, use core.data.circle_prob class directly.
    """
    probs = np.asarray(_circle_prob_instance(X))
    return probs[:, 1] if probs.ndim > 1 else probs[1]


def moon_focus_prob(X: Array, **kwargs) -> Array:
    """
    Functional wrapper around core.data.moon_prob.
    Returns P(Y=1|X) - the probability of class 1.
    
    For full control, use core.data.moon_prob class directly.
    """
    probs = np.asarray(_moon_prob_instance(X))
    return probs[:, 1] if probs.ndim > 1 else probs[1]


def make_aleatoric_uncertainty(p_fn: Callable[[Array], Array]) -> Callable[[Array], Array]:
    """
    Wrap a probability surface p_fn to return Shannon entropy over Bernoulli.
    """
    def wrapped(X: Array) -> Array:
        p1 = p_fn(X)
        p0 = 1.0 - p1
        probs = np.stack([p0, p1], axis=-1)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
        return entropy
    return wrapped
