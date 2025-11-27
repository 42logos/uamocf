
"""
Orchestrates the synthetic experiment:
1) sample data
2) train base model
3) train ensemble
4) build and solve the multi-objective CF problem

Uses core.optimization for NSGA-II configuration and execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

# Import from core
from core.optimization import (
    NSGAConfig,
    run_nsga,
    FactualBasedSampling,
    MixedSampling,
    ValidCFCallback,
)

from .cf_problem import CFConfig, make_cf_problem
from .data import DataConfig, sample_dataset
from .models import EnsembleModel, SimpleNN
from .training import TrainConfig, TrainResult, train_model, train_ensemble

Array = np.ndarray

# Re-export NSGAConfig as NSGACfg for backward compatibility
NSGACfg = NSGAConfig


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ensemble_size: int = 10
    ensemble_resample: bool = True
    cf: CFConfig = field(default_factory=CFConfig)
    nsga: NSGACfg = field(default_factory=NSGACfg)
    x_star: Optional[Array] = None
    y_target: Optional[int] = None


@dataclass
class ExperimentArtifacts:
    X: Array
    y: Array
    p_true: Array
    base: TrainResult
    ensemble: List[TrainResult]
    problem: object
    nsga_result: object


def _train_base_model(X: Array, y: Array, cfg: TrainConfig) -> TrainResult:
    model = SimpleNN()
    return train_model(model, X, y, cfg)


def _train_ensemble(data_cfg: DataConfig, train_cfg: TrainConfig, ensemble_size: int, resample: bool) -> List[TrainResult]:
    X, y, _ = sample_dataset(data_cfg)

    def _resample(idx: int) -> Tuple[Array, Array]:
        cfg = DataConfig(**vars(data_cfg))
        if cfg.seed is not None:
            cfg.seed += idx
        return sample_dataset(cfg)[:2]

    resample_fn = _resample if resample else None
    return train_ensemble(
        num_models=ensemble_size,
        X=X,
        y=y,
        cfg=train_cfg,
        model_factory=lambda: SimpleNN(),
        resample_fn=resample_fn,
    )


def run_experiment(cfg: ExperimentConfig) -> ExperimentArtifacts:
    # 1) sample data
    X, y, p_true = sample_dataset(cfg.data)

    # 2) train base model
    base_result = _train_base_model(X, y, cfg.train)
    device = cfg.train.resolve_device()

    # 3) train ensemble (optionally resampling per model)
    ensemble_results = _train_ensemble(cfg.data, cfg.train, cfg.ensemble_size, cfg.ensemble_resample)
    ensemble_models: Sequence[nn.Module] = [r.model for r in ensemble_results]

    # 4) CF problem setup
    x_star = cfg.x_star if cfg.x_star is not None else X[0]
    y_target = cfg.y_target if cfg.y_target is not None else int(1 - y[0])
    x_star_t = torch.tensor(x_star, dtype=torch.float32, device=device)
    y_target_t = torch.tensor([y_target], dtype=torch.long, device=device)
    X_obs_t = torch.tensor(X, dtype=torch.float32, device=device)
    weights = torch.ones(cfg.cf.k_neighbors, device=device)

    ensemble_wrapper = EnsembleModel(list(ensemble_models)).to(device)
    problem = make_cf_problem(
        model=base_result.model,
        x_star=x_star_t,
        y_target=y_target_t,
        X_obs=X_obs_t,
        weights=weights,
        config=cfg.cf,
        ensemble=ensemble_models,
        bayesian_model=ensemble_wrapper,
        device=device,
    )

    # 5) solve
    nsga_res = run_nsga(problem, cfg.nsga)

    return ExperimentArtifacts(
        X=X,
        y=y,
        p_true=p_true,
        base=base_result,
        ensemble=ensemble_results,
        problem=problem,
        nsga_result=nsga_res,
    )


def quick_run() -> ExperimentArtifacts:
    """
    Convenience entrypoint with lighter defaults for quick iteration.
    """
    cfg = ExperimentConfig(
        train=TrainConfig(epochs=60, progress=False),
        nsga=NSGACfg(n_gen=60, verbose=False),
        ensemble_size=5,
    )
    return run_experiment(cfg)


if __name__ == "__main__":
    artifacts = quick_run()
    F = artifacts.nsga_result.F
    print(f"Finished. Pareto front shape: {F.shape}")
