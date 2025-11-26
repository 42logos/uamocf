"""
Unified experiment pipelines for UAMOCF.

This module provides high-level pipeline functions that orchestrate the complete
workflow for both 2D feature space and MNIST counterfactual experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union, Callable
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset

from .data import (
    DataConfig,
    MNISTConfig,
    sample_dataset,
    mnist_dpg,
)
from .models import (
    SimpleNN,
    MNISTClassifier,
    EnsembleModel,
)
from .training import (
    TrainConfig,
    TrainResult,
    train_ensemble,
)
from .uncertainty import (
    UncertaintyResult,
    compute_uncertainty_decomposition,
)
from .cf_problem import (
    CFConfig,
    ImageCFConfig,
    make_cf_problem,
    make_cf_problem_image_space,
)
from .optimization import (
    NSGAConfig,
    run_nsga2,
)


# =============================================================================
# Experiment Configurations
# =============================================================================

@dataclass
class Experiment2DConfig:
    """Complete configuration for 2D feature space experiments."""
    
    # Data configuration
    data_config: DataConfig = field(default_factory=DataConfig)
    
    # Training configuration
    train_config: TrainConfig = field(default_factory=TrainConfig)
    n_ensemble: int = 5
    
    # Counterfactual configuration
    cf_config: CFConfig = field(default_factory=CFConfig)
    
    # Optimization configuration
    nsga_config: NSGAConfig = field(default_factory=NSGAConfig)
    
    # Experiment metadata
    experiment_name: str = "2d_experiment"
    random_seed: Optional[int] = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "data_config": {
                "n": self.data_config.n,
                "d": self.data_config.d,
                "n_classes": self.data_config.n_classes,
            },
            "train_config": {
                "n_epochs": self.train_config.n_epochs,
                "lr": self.train_config.lr,
                "batch_size": self.train_config.batch_size,
            },
            "n_ensemble": self.n_ensemble,
            "cf_config": {
                "n_classes": self.cf_config.n_classes,
            },
            "nsga_config": {
                "pop_size": self.nsga_config.pop_size,
                "n_gen": self.nsga_config.n_gen,
            },
            "experiment_name": self.experiment_name,
            "random_seed": self.random_seed,
        }


@dataclass
class ExperimentMNISTConfig:
    """Complete configuration for MNIST experiments."""
    
    # Data configuration
    mnist_config: MNISTConfig = field(default_factory=MNISTConfig)
    
    # Counterfactual configuration
    cf_config: ImageCFConfig = field(default_factory=ImageCFConfig)
    
    # Optimization configuration
    nsga_config: NSGAConfig = field(default_factory=NSGAConfig)
    
    # Experiment metadata
    experiment_name: str = "mnist_experiment"
    random_seed: Optional[int] = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "mnist_config": {
                "img_size": self.mnist_config.img_size,
                "n_ensemble": self.mnist_config.n_ensemble,
                "num_epochs": self.mnist_config.num_epochs,
            },
            "cf_config": {
                "original_label": self.cf_config.original_label,
                "target_label": self.cf_config.target_label,
            },
            "nsga_config": {
                "pop_size": self.nsga_config.pop_size,
                "n_gen": self.nsga_config.n_gen,
            },
            "experiment_name": self.experiment_name,
            "random_seed": self.random_seed,
        }


# =============================================================================
# Experiment Results
# =============================================================================

@dataclass
class ExperimentResult:
    """Container for experiment results."""
    
    # Core results
    counterfactuals: np.ndarray
    objectives: np.ndarray
    
    # Model artifacts
    ensemble: EnsembleModel
    train_result: Optional[TrainResult] = None
    
    # Factual information
    factual: Optional[np.ndarray] = None
    factual_label: Optional[int] = None
    target_label: Optional[int] = None
    
    # Uncertainty information
    uncertainties: Optional[Dict[str, np.ndarray]] = None
    
    # Optimization history
    optimization_history: Optional[List[Dict[str, Any]]] = None
    
    # Configuration used
    config: Optional[Union[Experiment2DConfig, ExperimentMNISTConfig]] = None
    
    def get_best_counterfactuals(
        self,
        n: int = 5,
        objective_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the top n counterfactuals sorted by a specific objective.
        
        Args:
            n: Number of counterfactuals to return.
            objective_idx: Index of objective to sort by (ascending).
            
        Returns:
            Tuple of (counterfactuals, objectives) sorted by the specified objective.
        """
        sorted_idx = np.argsort(self.objectives[:, objective_idx])[:n]
        return self.counterfactuals[sorted_idx], self.objectives[sorted_idx]
    
    def get_pareto_efficient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get Pareto-efficient counterfactuals.
        
        Returns:
            Tuple of (counterfactuals, objectives) for Pareto-efficient solutions.
        """
        is_efficient = np.ones(len(self.objectives), dtype=bool)
        
        for i, obj in enumerate(self.objectives):
            if is_efficient[i]:
                # Check if any other point dominates this one
                is_efficient[is_efficient] = ~np.all(
                    self.objectives[is_efficient] <= obj, axis=1
                ) | np.all(self.objectives[is_efficient] == obj, axis=1)
                is_efficient[i] = True
        
        return self.counterfactuals[is_efficient], self.objectives[is_efficient]


# =============================================================================
# Pipeline Functions
# =============================================================================

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_2d_experiment(
    config: Experiment2DConfig,
    factual: np.ndarray,
    target_class: int,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run a complete 2D feature space counterfactual experiment.
    
    Args:
        config: Experiment configuration.
        factual: The factual point to generate counterfactuals for.
        target_class: The desired counterfactual class.
        device: Torch device (auto-detected if None).
        verbose: Whether to print progress information.
        
    Returns:
        ExperimentResult containing counterfactuals and related information.
    """
    # Set random seed
    if config.random_seed is not None:
        set_random_seed(config.random_seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print(f"Using device: {device}")
    
    # Generate data
    if verbose:
        print("Generating training data...")
    X, y, probs = sample_dataset(config.data_config)
    
    # Train ensemble
    if verbose:
        print(f"Training ensemble of {config.n_ensemble} models...")
    
    train_result = train_ensemble(
        X=X,
        y=y,
        n_models=config.n_ensemble,
        cfg=config.train_config,
        model_factory=lambda: SimpleNN(
            input_dim=config.data_config.d,
            hidden_dim=config.train_config.hidden_dim,
            output_dim=config.data_config.n_classes,
        ),
        device=device,
    )
    
    # Create CF problem
    if verbose:
        print("Creating counterfactual problem...")
    problem = make_cf_problem(
        model=train_result.ensemble.models[0],
        ensemble_models=train_result.ensemble.models,
        factual=factual,
        target_class=target_class,
        config=config.cf_config,
        device=device,
    )
    
    # Run optimization
    if verbose:
        print("Running NSGA-II optimization...")
    result = run_nsga2(
        problem=problem,
        cfg=config.nsga_config,
        factual=factual,
    )
    
    if result.X is None:
        if verbose:
            print("Warning: No valid counterfactuals found.")
        counterfactuals = np.array([])
        objectives = np.array([])
    else:
        counterfactuals = result.X
        objectives = result.F
    
    # Compute uncertainties for counterfactuals
    uncertainties = None
    if len(counterfactuals) > 0:
        unc_results = []
        for cf in counterfactuals:
            unc = compute_uncertainty_decomposition(
                train_result.ensemble.models, cf, device
            )
            unc_results.append(unc)
        
        uncertainties = {
            "total": np.array([u.total for u in unc_results]),
            "aleatoric": np.array([u.aleatoric for u in unc_results]),
            "epistemic": np.array([u.epistemic for u in unc_results]),
        }
    
    # Get factual label
    factual_tensor = torch.tensor(factual.reshape(1, -1), dtype=torch.float32, device=device)
    with torch.no_grad():
        factual_probs = train_result.ensemble.predict_proba(factual_tensor)
        factual_label = int(factual_probs.argmax(dim=1).item())
    
    return ExperimentResult(
        counterfactuals=counterfactuals,
        objectives=objectives,
        ensemble=train_result.ensemble,
        train_result=train_result,
        factual=factual,
        factual_label=factual_label,
        target_label=target_class,
        uncertainties=uncertainties,
        config=config,
    )


def run_mnist_experiment(
    config: ExperimentMNISTConfig,
    factual_image: np.ndarray,
    target_class: int,
    train_dataset: Optional[TensorDataset] = None,
    test_dataset: Optional[TensorDataset] = None,
    ensemble: Optional[EnsembleModel] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run a complete MNIST counterfactual experiment.
    
    Args:
        config: Experiment configuration.
        factual_image: The factual image (flattened or 2D).
        target_class: The desired counterfactual class.
        train_dataset: Pre-loaded training dataset (optional).
        test_dataset: Pre-loaded test dataset (optional).
        ensemble: Pre-trained ensemble (optional).
        device: Torch device (auto-detected if None).
        verbose: Whether to print progress information.
        
    Returns:
        ExperimentResult containing counterfactual images and related information.
    """
    # Set random seed
    if config.random_seed is not None:
        set_random_seed(config.random_seed)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print(f"Using device: {device}")
    
    # Load data if not provided
    if train_dataset is None or test_dataset is None:
        if verbose:
            print("Loading MNIST data...")
        train_dataset, test_dataset = mnist_dpg(
            root="./data",
            img_size=config.mnist_config.img_size,
        )
    
    # Train ensemble if not provided
    train_result = None
    if ensemble is None:
        if verbose:
            print(f"Training ensemble of {config.mnist_config.n_ensemble} models...")
        
        from .training import train_mnist_ensemble
        train_result = train_mnist_ensemble(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            cfg=config.mnist_config,
            device=device,
        )
        ensemble = train_result.ensemble
    
    # Flatten factual image if needed
    if factual_image.ndim > 1:
        factual_flat = factual_image.flatten()
    else:
        factual_flat = factual_image
    
    # Get factual label
    factual_tensor = torch.tensor(
        factual_flat.reshape(1, -1), 
        dtype=torch.float32, 
        device=device
    )
    with torch.no_grad():
        factual_probs = ensemble.predict_proba(factual_tensor)
        factual_label = int(factual_probs.argmax(dim=1).item())
    
    if verbose:
        print(f"Factual predicted as class {factual_label}, target class is {target_class}")
    
    # Update CF config with labels
    cf_config = ImageCFConfig(
        img_size=config.mnist_config.img_size,
        original_label=factual_label,
        target_label=target_class,
    )
    
    # Create CF problem
    if verbose:
        print("Creating counterfactual problem...")
    problem = make_cf_problem_image_space(
        model=ensemble.models[0],
        ensemble_models=ensemble.models,
        original_image=factual_flat,
        config=cf_config,
        device=device,
    )
    
    # Run optimization
    if verbose:
        print("Running NSGA-II optimization...")
    result = run_nsga2(
        problem=problem,
        cfg=config.nsga_config,
        factual=factual_flat,
    )
    
    if result.X is None:
        if verbose:
            print("Warning: No valid counterfactuals found.")
        counterfactuals = np.array([])
        objectives = np.array([])
    else:
        counterfactuals = result.X
        objectives = result.F
    
    # Compute uncertainties for counterfactuals
    uncertainties = None
    if len(counterfactuals) > 0:
        unc_results = []
        for cf in counterfactuals:
            unc = compute_uncertainty_decomposition(
                ensemble.models, cf, device
            )
            unc_results.append(unc)
        
        uncertainties = {
            "total": np.array([u.total for u in unc_results]),
            "aleatoric": np.array([u.aleatoric for u in unc_results]),
            "epistemic": np.array([u.epistemic for u in unc_results]),
        }
    
    return ExperimentResult(
        counterfactuals=counterfactuals,
        objectives=objectives,
        ensemble=ensemble,
        train_result=train_result,
        factual=factual_flat,
        factual_label=factual_label,
        target_label=target_class,
        uncertainties=uncertainties,
        config=config,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def save_experiment_result(
    result: ExperimentResult,
    path: Union[str, Path],
    save_models: bool = True,
) -> None:
    """
    Save experiment results to disk.
    
    Args:
        result: ExperimentResult to save.
        path: Directory path to save results.
        save_models: Whether to save model weights.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # Save counterfactuals and objectives
    np.save(path / "counterfactuals.npy", result.counterfactuals)
    np.save(path / "objectives.npy", result.objectives)
    
    if result.factual is not None:
        np.save(path / "factual.npy", result.factual)
    
    # Save metadata
    metadata = {
        "factual_label": result.factual_label,
        "target_label": result.target_label,
        "n_counterfactuals": len(result.counterfactuals),
    }
    
    if result.config is not None:
        metadata["config"] = result.config.to_dict()
    
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save uncertainties
    if result.uncertainties is not None:
        np.savez(
            path / "uncertainties.npz",
            **result.uncertainties
        )
    
    # Save model weights
    if save_models and result.ensemble is not None:
        models_path = path / "models"
        models_path.mkdir(exist_ok=True)
        
        for i, model in enumerate(result.ensemble.models):
            torch.save(
                model.state_dict(),
                models_path / f"model_{i}.pt"
            )


def load_experiment_result(
    path: Union[str, Path],
    model_factory: Optional[Callable[[], torch.nn.Module]] = None,
    device: Optional[torch.device] = None,
) -> ExperimentResult:
    """
    Load experiment results from disk.
    
    Args:
        path: Directory path containing saved results.
        model_factory: Factory function to create model instances (required to load models).
        device: Device to load models to.
        
    Returns:
        Loaded ExperimentResult.
    """
    path = Path(path)
    
    # Load counterfactuals and objectives
    counterfactuals = np.load(path / "counterfactuals.npy")
    objectives = np.load(path / "objectives.npy")
    
    factual = None
    if (path / "factual.npy").exists():
        factual = np.load(path / "factual.npy")
    
    # Load metadata
    with open(path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    # Load uncertainties
    uncertainties = None
    if (path / "uncertainties.npz").exists():
        with np.load(path / "uncertainties.npz") as data:
            uncertainties = {key: data[key] for key in data.files}
    
    # Load models if factory provided
    ensemble = None
    models_path = path / "models"
    if model_factory is not None and models_path.exists():
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        models = []
        model_files = sorted(models_path.glob("model_*.pt"))
        
        for model_file in model_files:
            model = model_factory()
            model.load_state_dict(torch.load(model_file, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
        
        if models:
            ensemble = EnsembleModel(models)
    
    return ExperimentResult(
        counterfactuals=counterfactuals,
        objectives=objectives,
        ensemble=ensemble,
        factual=factual,
        factual_label=metadata.get("factual_label"),
        target_label=metadata.get("target_label"),
        uncertainties=uncertainties,
    )
