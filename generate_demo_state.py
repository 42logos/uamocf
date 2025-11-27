"""
Generate a demo state file for the visualization app.

This script runs the full experiment pipeline and saves the results
to a state file that can be loaded in the Streamlit app.

Usage:
    python generate_demo_state.py [output_path]
    
    output_path: Optional path for the output file (default: demo_state.pkl)
"""

import sys
import os
from typing import Optional
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from vis_tools import pipeline, state
from vis_tools.pipeline import ExperimentConfig, TrainConfig, NSGACfg, DataConfig


def generate_demo_state(
    output_path: str = "demo_state.pkl",
    n_samples: int = 500,
    n_epochs: int = 100,
    ensemble_size: int = 5,
    pop_size: int = 100,
    n_gen: int = 200,
    x_star: Optional[np.ndarray] = None,
    seed: int = 42,
    verbose: bool = True
) -> state.AppState:
    """
    Generate a demo state by running the experiment pipeline.
    
    Args:
        output_path: Path to save the state file
        n_samples: Number of data samples to generate
        n_epochs: Number of training epochs per model
        ensemble_size: Number of models in the ensemble
        pop_size: NSGA-II population size
        n_gen: Number of NSGA-II generations
        x_star: Factual point coordinates (default: [-0.8, -0.7])
        seed: Random seed for reproducibility
        verbose: Print progress messages
        
    Returns:
        AppState object with all experiment results
    """
    if x_star is None:
        x_star = np.array([-0.8, -0.7])
    
    if verbose:
        print("Starting demo state generation...")
        print(f"  Samples: {n_samples}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Ensemble size: {ensemble_size}")
        print(f"  NSGA-II: pop_size={pop_size}, n_gen={n_gen}")
        print(f"  x*: {x_star}")
    
    # Configure experiment
    cfg = ExperimentConfig(
        data=DataConfig(n=n_samples, seed=seed),
        train=TrainConfig(epochs=n_epochs, progress=verbose),
        ensemble_size=ensemble_size,
        nsga=NSGACfg(pop_size=pop_size, n_gen=n_gen, verbose=verbose, seed=seed),
        x_star=x_star
    )
    
    if verbose:
        print("\nRunning experiment pipeline...")
    
    # Run experiment
    artifacts = pipeline.run_experiment(cfg)
    
    if verbose:
        print(f"\nExperiment finished.")
        print(f"  Pareto front size: {len(artifacts.nsga_result.F)}")
    
    # Convert to AppState using new API
    if verbose:
        print("\nConverting to AppState...")
    
    app_state = state.from_experiment_artifacts(artifacts, x_star=x_star)
    
    if verbose:
        print(f"  Data shape: {app_state.data[0].shape}")
        print(f"  Models: {len(app_state.models)}")
        print(f"  CF results: X={app_state.cf_results.X.shape}, F={app_state.cf_results.F.shape}")
        print(f"  F_obs shape: {app_state.F_obs.shape}")
        print(f"  F_star: {app_state.F_star}")
        print(f"  x_star: {app_state.x_star}")
    
    # Save state
    if verbose:
        print(f"\nSaving state to {output_path}...")
    
    state.save_state(app_state, output_path)
    
    if verbose:
        print(f"State saved successfully!")
        print(f"\nYou can now load this file in the app using 'State Management' > 'Import'")
    
    return app_state


def main():
    """Main entry point."""
    # Get output path from command line or use default
    output_path = sys.argv[1] if len(sys.argv) > 1 else "demo_state.pkl"
    
    generate_demo_state(output_path=output_path)


if __name__ == "__main__":
    main()
