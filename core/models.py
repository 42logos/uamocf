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
    
class SimpleCNN(nn.Module):
    """
    Simple CNN for image data.
    """

    def __init__(self, h,w,input_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(1, h, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # Conv layer 2
            nn.Conv2d(h, 2*h, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_layers(torch.ones(1,input_channels,h,w)).view(self.conv_layers(torch.ones(1,input_channels,h,w)).size(0), -1).size(1), 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class EnsembleModel(nn.Module):
    """
    Simple mean-ensembling wrapper around a list of models.
    """

    def __init__(self, models: Sequence[nn.Module]):
        if len(models) == 0:
            raise ValueError("EnsembleModel requires at least one model.")
        
        super().__init__()
        self.models = nn.ModuleList(models)
        self.device = next(self.models[0].parameters()).device

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = [m(x) for m in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

    def eval(self) -> "EnsembleModel":  # type: ignore[override]
        super().eval()
        for m in self.models:
            m.eval()
        return self

