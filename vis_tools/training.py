"""
Training utilities for PyTorch classifiers.

This module re-exports training functions from core.training.
All implementations are in core; this module provides backward compatibility.
"""

from __future__ import annotations

# Re-export everything from core.training
from core.training import (
    TrainConfig,
    TrainResult,
    make_loaders,
    train_model,
    train_ensemble,
)

# Re-export for backward compatibility
__all__ = [
    "TrainConfig",
    "TrainResult", 
    "make_loaders",
    "train_model",
    "train_ensemble",
]
