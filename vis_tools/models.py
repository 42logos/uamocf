"""
Model definitions and lightweight wrappers.

This module re-exports model classes from core.models to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn

# Re-export from core to avoid duplication
from core.models import (
    _mlp_layers,
    SimpleNN,
    EnsembleModel,
    SimpleCNN,
    VAE,
    reload_vae_model,
)

# Keep these vis_tools specific utilities
@dataclass
class DeviceConfig:
    prefer_cuda: bool = True

    def pick(self) -> torch.device:
        if self.prefer_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def to_device(models: Iterable[nn.Module], device: torch.device) -> List[nn.Module]:
    """
    Move a sequence of models to the given device.
    """
    return [m.to(device) for m in models]
