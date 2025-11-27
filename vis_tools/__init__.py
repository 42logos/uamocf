"""
vis_tools package for visualization utilities.

This package provides:
- state: State management for the visualization app (export/import)
- plotting: Visualization helpers for design space and objective space

For model training, data generation, and optimization, use the core package directly:
- core.models: SimpleNN, EnsembleModel, etc.
- core.data: SamplingConfig, sample_from_config, moon_prob, etc.
- core.training: TrainConfig, train_model, train_ensemble
- core.cf_problem: make_cf_problem, uncertainty functions
- core.optimization: NSGAConfig, run_nsga
"""

from .plotting import plot_proba
from .state import export_state, import_state, AppState, CFResult, save_state, load_state, create_state_from_pymoo_result, create_and_save_state
