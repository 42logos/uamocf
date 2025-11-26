"""
UAMOCF Demo: 2D Feature Space Counterfactual Generation

This script demonstrates how to use the uamocf package for generating
counterfactual explanations with uncertainty quantification.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

# Import from our package
from uamocf import (
    # Data generation
    DataConfig,
    sample_dataset,
    dpg,
    
    # Models
    SimpleNN,
    EnsembleModel,
    TorchProbaEstimator,
    
    # Training
    TrainConfig,
    train_ensemble,
    
    # Counterfactual problem
    CFConfig,
    make_cf_problem,
    
    # Optimization
    NSGAConfig,
    run_nsga2,
    
    # Uncertainty
    compute_uncertainty_decomposition,
    
    # Visualization
    plot_proba,
    plot_uncertainty_heatmap,
    plot_pareto_front_2d,
)


def main():
    # ==========================================================================
    # 1. Data Generation
    # ==========================================================================
    print("=" * 60)
    print("1. Data Generation")
    print("=" * 60)
    
    # Configure data generation
    data_config = DataConfig(
        n=1000,
        d=2,
        n_classes=2,
        p_fn_name="moon",
    )

    # Generate data
    X, y, probs = sample_dataset(data_config)
    print(f"Generated {len(X)} samples with {data_config.n_classes} classes")

    # Visualize the dataset
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data')
    plt.savefig('data_visualization.png', dpi=150)
    plt.close()
    print("Saved: data_visualization.png")

    # ==========================================================================
    # 2. Train Ensemble Model
    # ==========================================================================
    print("\n" + "=" * 60)
    print("2. Training Ensemble Model")
    print("=" * 60)
    
    # Configure training
    train_config = TrainConfig(
        epochs=100,
        lr=0.01,
        batch_size=32,
    )

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Train ensemble
    train_results = train_ensemble(
        num_models=5,
        X=X,
        y=y,
        cfg=train_config,
        model_factory=lambda: SimpleNN(input_dim=2, hidden_dim=64, output_dim=2),
    )

    # Create ensemble from trained models
    ensemble = EnsembleModel([r.model for r in train_results])
    
    print(f"Ensemble trained with {len(ensemble.models)} models")
    print(f"Average final accuracy: {np.mean([r.val_accuracy for r in train_results]):.2%}")

    # ==========================================================================
    # 3. Visualize Decision Boundary and Uncertainty
    # ==========================================================================
    print("\n" + "=" * 60)
    print("3. Visualization")
    print("=" * 60)

    # Plot decision boundary
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_proba(ensemble, X, y, grid_resolution=100, ax=ax, device=device)
    ax.set_title('Decision Boundary')
    plt.savefig('decision_boundary.png', dpi=150)
    plt.close()
    print("Saved: decision_boundary.png")

    # Uncertainty heatmap
    fig, ax = plot_uncertainty_heatmap(
        ensemble.models,
        X,
        device=device,
        uncertainty_type='epistemic',
    )
    ax.set_title('Epistemic Uncertainty')
    plt.savefig('uncertainty_heatmap.png', dpi=150)
    plt.close()
    print("Saved: uncertainty_heatmap.png")

    # ==========================================================================
    # 4. Generate Counterfactuals
    # ==========================================================================
    print("\n" + "=" * 60)
    print("4. Counterfactual Generation")
    print("=" * 60)
    
    # Select a factual point from class 0
    class_0_indices = np.where(y == 0)[0]
    factual_idx = class_0_indices[0]
    factual = X[factual_idx]

    print(f"Factual point: {factual}")
    print(f"Factual class: {y[factual_idx]}")

    # Configure counterfactual generation
    cf_config = CFConfig()

    nsga_config = NSGAConfig(
        pop_size=100,
        n_gen=50,
    )

    # Create CF problem
    # Convert to tensors
    factual_t = torch.from_numpy(factual.astype(np.float32)).to(device)
    target_t = torch.tensor([1], device=device)  # target class 1
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    
    problem = make_cf_problem(
        model=ensemble.models[0],
        x_star=factual_t,
        y_target=target_t,
        X_obs=X_t,
        config=cf_config,
        ensemble=ensemble.models,
        device=device,
    )

    # Run optimization
    print("Running NSGA-II optimization...")
    result = run_nsga2(
        problem=problem,
        config=nsga_config,
    )

    if result.X is not None:
        print(f"Found {len(result.X)} counterfactual candidates")
    else:
        print("No counterfactuals found")
        return

    # ==========================================================================
    # 5. Analyze Results
    # ==========================================================================
    print("\n" + "=" * 60)
    print("5. Results Analysis")
    print("=" * 60)
    
    # Plot Pareto front
    if result.F is not None and len(result.F) > 0:
        fig, ax = plot_pareto_front_2d(
            result.F,
            objective_names=['Validity', 'Epistemic', 'Sparsity', 'Aleatoric'],
        )
        plt.savefig('pareto_front.png', dpi=150)
        plt.close()
        print("Saved: pareto_front.png")

    # Visualize counterfactuals in feature space
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.3, label='Training data')
    plt.scatter(factual[0], factual[1], c='green', s=200, marker='*', label='Factual', zorder=5)

    if result.X is not None and len(result.X) > 0:
        plt.scatter(result.X[:, 0], result.X[:, 1], c='yellow', s=100, marker='o', 
                    edgecolors='black', label='Counterfactuals', zorder=4)
        
        # Draw lines from factual to counterfactuals
        for cf in result.X[:10]:  # Show lines to first 10 CFs
            plt.plot([factual[0], cf[0]], [factual[1], cf[1]], 'k--', alpha=0.3)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Counterfactuals in Feature Space')
    plt.legend()
    plt.savefig('counterfactuals.png', dpi=150)
    plt.close()
    print("Saved: counterfactuals.png")

    # Compute uncertainties for best counterfactuals
    print("\nTop 5 counterfactuals by distance:")
    sorted_idx = np.argsort(result.F[:, 0])[:5]
    for i, idx in enumerate(sorted_idx):
        cf = result.X[idx]
        unc = compute_uncertainty_decomposition(
            ensemble.models, cf, device
        )
        print(f"  {i+1}. Distance: {result.F[idx, 0]:.4f}, "
              f"Invalidity: {result.F[idx, 1]:.4f}, "
              f"EU: {unc.epistemic:.4f}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
