
"""
Model definitions and lightweight wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn


def _mlp_layers(input_dim: int, hidden_dim: int, depth: int, output_dim: int) -> nn.Sequential:
    layers = []
    in_dim = input_dim
    for _ in range(depth):
        layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class SimpleNN(nn.Module):
    """
    Small fully-connected network used in the notebook experiments.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 16,
        depth: int = 4,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.net = _mlp_layers(input_dim, hidden_dim, depth, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


class EnsembleModel(nn.Module):
    """
    Simple mean-ensembling wrapper around a list of models.
    """

    def __init__(self, models: Sequence[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = [m(x) for m in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

    def eval(self) -> "EnsembleModel":  # type: ignore[override]
        super().eval()
        for m in self.models:
            m.eval()
        return self


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
