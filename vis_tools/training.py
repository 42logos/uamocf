
"""
Training utilities for PyTorch classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

from .models import SimpleNN

Array = np.ndarray


@dataclass
class TrainConfig:
    batch_size: int = 32
    val_batch_size: int = 256
    epochs: int = 250
    lr: float = 1e-3
    val_split: float = 0.2
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    shuffle: bool = True
    seed: Optional[int] = None
    progress: bool = True
    num_workers: int = 0

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


@dataclass
class TrainResult:
    model: nn.Module
    train_loss: float
    val_loss: float
    val_accuracy: float
    history: List[float] = field(default_factory=list)
    X_train: Optional[Array] = None
    y_train: Optional[Array] = None


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(
    X: Array,
    y: Array,
    cfg: TrainConfig,
) -> Tuple[DataLoader, DataLoader]:
    device = cfg.resolve_device()
    _seed_everything(cfg.seed)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)

    val_size = int(len(dataset) * cfg.val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, val_loader


def train_model(
    model: nn.Module,
    X: Array,
    y: Array,
    cfg: TrainConfig,
) -> TrainResult:
    device = cfg.resolve_device()
    model = model.to(device)
    train_loader, val_loader = make_loaders(X, y, cfg)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    history: List[float] = []
    progress_iter = tqdm(range(cfg.epochs), desc="Training", ncols=80, unit="epoch", disable=not cfg.progress)

    for _ in progress_iter:
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        history.append(epoch_loss)
        progress_iter.set_postfix({"Train Loss": epoch_loss})

    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_accuracy = correct / len(val_loader.dataset)

    return TrainResult(
        model=model,
        train_loss=history[-1] if history else 0.0,
        val_loss=val_loss,
        val_accuracy=val_accuracy,
        history=history,
        X_train=X,
        y_train=y,
    )


def train_ensemble(
    num_models: int,
    X: Array,
    y: Array,
    cfg: TrainConfig,
    model_factory: Callable[[], nn.Module] = lambda: SimpleNN(),
    resample_fn: Optional[Callable[[int], Tuple[Array, Array]]] = None,
) -> List[TrainResult]:
    """
    Train multiple models on (possibly re-sampled) data.
    """
    results: List[TrainResult] = []
    for idx in range(num_models):
        model = model_factory()
        run_cfg = TrainConfig(**vars(cfg))
        run_cfg.seed = None if cfg.seed is None else cfg.seed + idx

        Xi, yi = (X, y) if resample_fn is None else resample_fn(idx)
        res = train_model(model, Xi, yi, run_cfg)
        results.append(res)
    return results
