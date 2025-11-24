import sys
import os
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from vis_tools import pipeline, state
from vis_tools.pipeline import ExperimentConfig, TrainConfig, NSGACfg, DataConfig

def main():
    print("Starting demo state generation...")
    
    # Configure for a high-quality demo
    # Note: noise_sigma=0.4 is default in moon_focus_prob, so we don't need to pass it unless we change p_fn
    cfg = ExperimentConfig(
        data=DataConfig(n=500, seed=42),
        train=TrainConfig(epochs=100, progress=True),
        ensemble_size=5,
        nsga=NSGACfg(pop_size=100, n_gen=200, verbose=True, seed=42),
        x_star=np.array([-0.8, -0.7])
    )
    
    print("Running experiment pipeline...")
    artifacts = pipeline.run_experiment(cfg)
    
    print("Experiment finished.")
    print(f"Pareto front size: {len(artifacts.nsga_result.F)}")
    
    # Prepare data for export
    # data tuple: (X, y, p_true)
    data_tuple = (artifacts.X, artifacts.y, artifacts.p_true)
    
    # Models: List of trained models
    # artifacts.ensemble is a list of TrainResult, each has .model
    models = [r.model for r in artifacts.ensemble]
    
    # CF Results: Object with X and F
    class CFResult:
        def __init__(self, X, F):
            self.X = X
            self.F = F
            
    cf_results = CFResult(artifacts.nsga_result.X, artifacts.nsga_result.F)
    
    # Calculate F_obs (Objectives for observed data)
    print("Calculating objectives for observed data...")
    # The problem object is a pymoo FunctionalProblem. 
    # We can evaluate it on X (observed data).
    # Note: The problem was defined with elementwise=True, so we might need to loop or check if it supports vectorization.
    # pymoo FunctionalProblem usually supports vectorization if the functions do.
    # Let's check if we can just call evaluate(X).
    
    # However, the problem definition in make_cf_problem uses torch models and expects input.
    # Let's try evaluating.
    try:
        F_obs = artifacts.problem.evaluate(artifacts.X)
    except Exception as e:
        print(f"Vectorized evaluation failed: {e}. Falling back to loop.")
        # Fallback
        F_list = []
        for i in range(len(artifacts.X)):
            f = artifacts.problem.evaluate(artifacts.X[i])
            F_list.append(f)
        F_obs = np.array(F_list)
        
    print(f"F_obs shape: {F_obs.shape}")
    
    # Export
    print("Exporting state...")
    state_bytes = state.export_state(
        data_tuple,
        models,
        cf_results,
        F_obs
    )
    
    output_path = "demo_state.pkl"
    with open(output_path, "wb") as f:
        f.write(state_bytes)
        
    print(f"State saved to {output_path}")
    print("You can now load this file in the OptiView Pro app using the 'State Management' sidebar.")

if __name__ == "__main__":
    main()
