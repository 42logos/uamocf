"""
Model definitions and lightweight wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn
from torch.nn import functional as F


import os,sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
save_path = os.path.join(project_root, 'saved_models')


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
        
    def to(self, device: torch.device) -> "EnsembleModel":  # type: ignore[override]
        super().to(device)
        for m in self.models:
            m.to(device)
        self.device = device
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        outputs = [m(x) for m in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

    def eval(self) -> "EnsembleModel":  # type: ignore[override]
        super().eval()
        for m in self.models:
            m.eval()
        return self



class VAE(nn.Module):
    def __init__(self, latent_dim=2, in_channels=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # -------- Encoder --------
        # Input: (B, C, 16, 16)
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),           # 8 -> 4
            nn.ReLU(inplace=True),
        )
        # Now feature map is (B, 64, 4, 4) -> 64*4*4 = 1024
        self.enc_fc = nn.Linear(64 * 4 * 4, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # -------- Decoder --------
        self.dec_fc = nn.Linear(latent_dim, 128)
        self.dec_fc2 = nn.Linear(128, 64 * 4 * 4)
        self.dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 4 -> 8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.Sigmoid()  # output in [0,1]
        )

    # ---- Encoder step: x -> μ, logσ² ----
    def encode(self, x):
        h = self.enc_conv(x)               # (B, 64, 4, 4)
        h = h.view(x.size(0), -1)          # (B, 1024)
        h = F.relu(self.enc_fc(h))         # (B, 128)
        mu = self.fc_mu(h)                 # (B, latent_dim)
        logvar = self.fc_logvar(h)         # (B, latent_dim)
        return mu, logvar

    # ---- Reparameterization: μ, logσ² -> z ----
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---- Decoder step: z -> x̂ ----
    def decode(self, z):
        h = F.relu(self.dec_fc(z))         # (B, 128)
        h = F.relu(self.dec_fc2(h))        # (B, 64*4*4)
        h = h.view(z.size(0), 64, 4, 4)    # (B, 64, 4, 4)
        x_hat = self.dec_deconv(h)         # (B, C, 16, 16)
        return x_hat

    # ---- Full forward: x -> x̂, μ, logσ² ----
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar



def reload_vae_model(path: str=save_path, device: str='cpu') -> List[VAE]:
    weights_files = [f for f in os.listdir(path) if f.startswith('vae_mnist_digit_') and f.endswith('.pth')]
    vae_models = []
    for wf in weights_files:
        digit = int(wf.split('_')[-1].split('.')[0])
        model_digit = VAE(latent_dim=2, in_channels=1)
        model_digit.load_state_dict(torch.load(os.path.join(path, wf), map_location=device, weights_only=True))
        model_digit.to(device)
        model_digit.eval()
        vae_models.append((digit, model_digit))

    vae_models.sort(key=lambda x: x[0])
    return [model for _, model in vae_models]