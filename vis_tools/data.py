
"""
Data generation utilities and probability helpers used across the notebook.

The goal is to keep all stochastic pieces in one place so they can be reused
from scripts, notebooks, or tests without copy/paste.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np

Array = np.ndarray


# -----------------------------
# Probability surfaces
# -----------------------------
def boundary_focus_prob(
    X: Array,
    r_inner: float = np.sqrt(2.0),
    r_outer: float = 3.0,
    sigma: float = 0.85,
    max_uncertainty: float = 0.89,
) -> Array:
    """
    Synthetic conditional probability P(Y=1|X) that is most uncertain near two
    circular decision boundaries.
    """
    X = np.asarray(X)
    r = np.linalg.norm(X, axis=-1)

    base = ((r <= r_inner) | (r >= r_outer)).astype(float)
    dist_to_boundary = np.minimum(np.abs(r - r_inner), np.abs(r - r_outer))
    alpha = max_uncertainty * np.exp(-(dist_to_boundary / sigma) ** 2)

    p1 = base * (1 - alpha) + 0.5 * alpha
    return np.clip(p1, 0.0, max_uncertainty)


def moon_focus_prob(
    X: Array,
    sigma: float = 0.4,
    max_uncertainty: float = 0.75,
    min_uncertainty: float = 0.15,
) -> Array:
    """
    Synthetic conditional probability P(Y=1|X) for a noisy two-moon decision
    boundary. Uncertainty is highest near the sinusoidal boundary.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    x1, x2 = X[:, 0], X[:, 1]
    decision_boundary = np.sin(np.pi * x1) / 2
    dist_to_boundary = np.abs(x2 - decision_boundary)
    alpha = max_uncertainty * np.exp(-(dist_to_boundary / sigma) ** 2)
    p1 = 0.5 * alpha + (1 - alpha) * (x2 > decision_boundary).astype(float)
    return np.clip(p1, min_uncertainty, max_uncertainty)


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


# -----------------------------
# Sampling
# -----------------------------
@dataclass
class DataConfig:
    """Configuration for synthetic sampling."""

    n: int = 1500
    d: int = 2
    low: float = -1.0
    high: float = 1.0
    seed: Optional[int] = 42
    p_fn: Callable[[Array], Array] = moon_focus_prob

    def make_sampler(self) -> Callable[..., Array]:
        rng = np.random.default_rng(self.seed)
        return partial(rng.uniform, low=self.low, high=self.high)


def dpg(
    n: int,
    x_sampler: Callable[..., Array],
    p_fn: Callable[[Array], Array],
    d: int = 2,
    rng: Optional[np.random.Generator] = None,
    x_sampler_kwargs: Optional[dict] = None,
) -> Tuple[Array, Array, Array]:
    """
    Draws (X, Y) pairs where X ~ P_X (via x_sampler) and Y|X ~ Bernoulli(p_fn(X)).
    Returns X, Y, and the underlying p1 surface values.
    """
    rng = np.random.default_rng() if rng is None else rng
    x_sampler_kwargs = {} if x_sampler_kwargs is None else dict(x_sampler_kwargs)

    if "size" not in x_sampler_kwargs:
        x_sampler_kwargs["size"] = (n, d)

    X = np.asarray(x_sampler(**x_sampler_kwargs))
    p1 = np.clip(p_fn(X), 0.0, 1.0)
    Y = rng.binomial(1, p1).astype(np.int64)
    return X, Y, p1


def sample_dataset(cfg: DataConfig) -> Tuple[Array, Array, Array]:
    """
    Convenience wrapper to sample a full dataset given a DataConfig.
    """
    rng = np.random.default_rng(cfg.seed)
    sampler = cfg.make_sampler()
    return dpg(cfg.n, sampler, cfg.p_fn, d=cfg.d, rng=rng)
