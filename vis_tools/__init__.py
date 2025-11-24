"""
Utility package for the synthetic MMO counterfactual experiments.
"""

from .data import (
    DataConfig,
    boundary_focus_prob,
    make_aleatoric_uncertainty,
    moon_focus_prob,
    sample_dataset,
)
from .models import EnsembleModel, SimpleNN
from .plotting import plot_proba
from .training import TrainConfig, train_model, train_ensemble
from .state import export_state, import_state

# Optional: pipeline components require pymoo; guard import to keep base modules usable.
try:
    from .pipeline import ExperimentConfig, quick_run, run_experiment
except Exception:  # pragma: no cover - dependency guard
    ExperimentConfig = None  # type: ignore
    quick_run = None  # type: ignore
    run_experiment = None  # type: ignore
