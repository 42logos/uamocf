"""
Generate a demo state file for the visualization app.

This script demonstrates manual training and optimization workflow
(without using the pipeline) to create state files for visualization.

Usage:
    python generate_demo_state.py [output_path]
    
    output_path: Optional path for the output file (default: demo_state.pkl)
"""

import sys
import os
from typing import Optional, List
import numpy as np
import torch
from torch import nn

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from vis_tools import state
from vis_tools.models import SimpleNN, EnsembleModel
from vis_tools.data import sample_dataset, DataConfig
from vis_tools.training import train_model, train_ensemble, TrainConfig
from vis_tools.cf_problem import make_cf_problem, CFConfig
from core.optimization import NSGAConfig, run_nsga


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
    Generate a demo state by manually training models and running optimization.
    
    This demonstrates the full manual workflow without using the pipeline.
    
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
        x_star = np.array([-0.8, -0.7], dtype=np.float32)
    
    if verbose:
        print("=== Manual Demo State Generation ===")
        print(f"  Samples: {n_samples}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Ensemble size: {ensemble_size}")
        print(f"  NSGA-II: pop_size={pop_size}, n_gen={n_gen}")
        print(f"  x*: {x_star}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"\nUsing device: {device}")
    
    # =====================
    # Step 1: Generate Data
    # =====================
    if verbose:
        print("\n[1/4] Generating data...")
    
    data_cfg = DataConfig(n=n_samples, seed=seed)
    X, y, p_true = sample_dataset(data_cfg)
    
    if verbose:
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Class distribution: {np.bincount(y)}")
    
    # =====================
    # Step 2: Train Ensemble
    # =====================
    if verbose:
        print("\n[2/4] Training ensemble models...")
    
    train_cfg = TrainConfig(epochs=n_epochs, progress=verbose)
    
    # Train ensemble with resampling for diversity
    def resample_data(idx: int):
        cfg = DataConfig(n=n_samples, seed=seed + idx)
        X_i, y_i, _ = sample_dataset(cfg)
        return X_i, y_i
    
    ensemble_results = train_ensemble(
        num_models=ensemble_size,
        X=X, y=y,
        cfg=train_cfg,
        model_factory=lambda: SimpleNN(),
        resample_fn=resample_data
    )
    
    # Extract models from results
    models: List[nn.Module] = [r.model for r in ensemble_results]
    
    if verbose:
        print(f"  Trained {len(models)} models")
    
    # =====================
    # Step 3: Setup CF Problem
    # =====================
    if verbose:
        print("\n[3/4] Setting up counterfactual problem...")
    
    # Convert to tensors
    x_star_t = torch.tensor(x_star, dtype=torch.float32, device=device)
    X_obs_t = torch.tensor(X, dtype=torch.float32, device=device)
    
    # Determine target class (opposite of x_star's predicted class)
    with torch.no_grad():
        pred = models[0](x_star_t.unsqueeze(0))
        current_class = pred.argmax(dim=1).item()
    y_target = 1 - current_class
    y_target_t = torch.tensor([y_target], dtype=torch.long, device=device)
    
    if verbose:
        print(f"  x* predicted class: {current_class}")
        print(f"  Target class: {y_target}")
    
    # Create weights for sparsity objective
    cf_cfg = CFConfig(k_neighbors=5)
    weights = torch.ones(cf_cfg.k_neighbors, device=device)
    
    # Create ensemble model wrapper
    ensemble_model = EnsembleModel(models).to(device)
    
    # Create the CF problem
    problem = make_cf_problem(
        model=models[0],
        x_star=x_star_t,
        y_target=y_target_t,
        X_obs=X_obs_t,
        weights=weights,
        config=cf_cfg,
        ensemble=models,
        bayesian_model=ensemble_model,
        device=device
    )
    
    if verbose:
        print(f"  Problem created with {problem.n_var} variables, {problem.n_obj} objectives")
    
    # =====================
    # Step 4: Run NSGA-II
    # =====================
    if verbose:
        print("\n[4/4] Running NSGA-II optimization...")
    
    nsga_cfg = NSGAConfig(
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        verbose=verbose
    )
    
    result = run_nsga(problem, nsga_cfg)
    
    if verbose:
        print(f"  Pareto front size: {len(result.F) if result.F is not None else 0}")
    
    # =====================
    # Create and Save State
    # =====================
    if verbose:
        print("\nCreating AppState and saving...")
    
    app_state = state.create_and_save_state(
        filepath=output_path,
        X=X, y=y, models=models,
        pymoo_result=result,
        problem=problem,
        x_star=x_star,
        p_true=p_true
    )
    
    if verbose:
        print(f"\n=== State saved to {output_path} ===")
        print(f"  Data: X={app_state.data[0].shape}")
        print(f"  Models: {len(app_state.models)}")
        if app_state.cf_results:
            print(f"  CF results: X={app_state.cf_results.X.shape}, F={app_state.cf_results.F.shape}")
        if app_state.F_obs is not None:
            print(f"  F_obs: {app_state.F_obs.shape}")
        print(f"  F_star: {app_state.F_star}")
        print(f"  x_star: {app_state.x_star}")
        print(f"\nLoad this file in the app using 'State Management' > 'Import'")
    
    return app_state


def main():
    """Main entry point."""
    output_path = sys.argv[1] if len(sys.argv) > 1 else "demo_state.pkl"
    generate_demo_state(output_path=output_path)


if __name__ == "__main__":
    main()
