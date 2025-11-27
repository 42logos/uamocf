"""
Utility package for the synthetic MMO counterfactual experiments.

This package provides high-level visualization and experiment utilities.
Low-level model and algorithm implementations are imported from the core package.
"""

from .data import (
    # vis_tools specific (backward-compat wrappers)
    DataConfig,
    boundary_focus_prob,
    make_aleatoric_uncertainty,
    moon_focus_prob,
    sample_dataset,
    # Re-exported from core.data
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
# Models are re-exported from core via vis_tools.models
from .models import EnsembleModel, SimpleNN, DeviceConfig, to_device
from .plotting import plot_proba
# Training re-exports from core.training
from .training import TrainConfig, TrainResult, train_model, train_ensemble, make_loaders
from .state import export_state, import_state

# Re-export core uncertainty functions for convenience
from .uncertainty import (
    entropy,
    aleatoric_from_models,
    epistemic_from_models,
    total_uncertainty,
    total_uncertainty_ensemble,
)

# Re-export core cf_problem utilities
from .cf_problem import (
    CFConfig,
    make_cf_problem,
    # Core utilities
    _gower_distance,
    _gower_distance_tensor,
    _k_nearest,
    _k_nearest_tensor,
)

# Optional: pipeline and optimization components require pymoo
try:
    from .pipeline import (
        ExperimentConfig,
        ExperimentArtifacts,
        quick_run,
        run_experiment,
        # Re-exported from core.optimization
        NSGACfg,  # Alias for NSGAConfig
    )
    from core.optimization import (
        NSGAConfig,
        run_nsga,
        FactualBasedSampling,
        GaussianFactualSampling,
        MixedSampling,
        ValidCFCallback,
        ValidCounterfactualTermination,
        ConvergenceTermination,
    )
except Exception:  # pragma: no cover - dependency guard
    ExperimentConfig = None  # type: ignore
    ExperimentArtifacts = None  # type: ignore
    quick_run = None  # type: ignore
    run_experiment = None  # type: ignore
    NSGACfg = None  # type: ignore
    NSGAConfig = None  # type: ignore
    run_nsga = None  # type: ignore
