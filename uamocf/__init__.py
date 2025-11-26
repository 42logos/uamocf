"""
UAMOCF - Uncertainty-Aware Multi-Objective Counterfactual Generation.

A Python package for generating counterfactual explanations using multi-objective
optimization with uncertainty quantification through deep ensembles.

Key Features:
- Multi-objective optimization using NSGA-II
- Uncertainty quantification via deep ensembles (aleatoric/epistemic decomposition)
- Support for 2D feature space and MNIST image space experiments
- Configurable objectives: proximity, validity, uncertainty
- Comprehensive visualization tools

Example Usage:
    >>> from uamocf import run_2d_experiment, Experiment2DConfig
    >>> config = Experiment2DConfig(n_ensemble=5)
    >>> result = run_2d_experiment(config, factual=np.array([0.5, 0.5]), target_class=1)
    >>> print(f"Found {len(result.counterfactuals)} counterfactuals")
"""

__version__ = "0.1.0"
__author__ = "UAMOCF Team"

from .data import (
    DataConfig,
    MNISTConfig,
    boundary_focus_prob_multiclass,
    moon_focus_prob_multiclass,
    dpg,
    sample_dataset,
    MNISTProbabilityFunction,
    mnist_dpg,
    get_mnist_images_subset,
)

from .models import (
    SimpleNN,
    MNISTClassifier,
    EnsembleModel,
    TorchProbaEstimator,
    DeviceConfig,
)

from .uncertainty import (
    UncertaintyResult,
    entropy,
    softmax_logits,
    aleatoric_from_models,
    total_uncertainty_ensemble,
    epistemic_from_models,
    compute_uncertainty_decomposition,
)

from .training import (
    TrainConfig,
    MNISTTrainConfig,
    TrainResult,
    train_model,
    train_image_model,
    train_ensemble,
    train_mnist_ensemble,
    extract_models,
)

from .cf_problem import (
    CFConfig,
    ImageCFConfig,
    make_cf_problem,
    make_cf_problem_image_space,
)

from .optimization import (
    NSGAConfig,
    FactualBasedSampling,
    ValidCFCallback,
    ValidCounterfactualTermination,
    run_nsga2,
)

from .visualization import (
    get_class_colors,
    get_decision_boundary_grid,
    plot_proba,
    plot_uncertainty_heatmap,
    visualize_image_counterfactuals,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
    plot_ensemble_decision_boundaries,
)

from .pipeline import (
    Experiment2DConfig,
    ExperimentMNISTConfig,
    ExperimentResult,
    run_2d_experiment,
    run_mnist_experiment,
    save_experiment_result,
    load_experiment_result,
    set_random_seed,
)

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Data
    "DataConfig",
    "MNISTConfig",
    "boundary_focus_prob_multiclass",
    "moon_focus_prob_multiclass",
    "dpg",
    "sample_dataset",
    "MNISTProbabilityFunction",
    "mnist_dpg",
    "get_mnist_images_subset",
    # Models
    "SimpleNN",
    "MNISTClassifier",
    "EnsembleModel",
    "TorchProbaEstimator",
    "DeviceConfig",
    # Uncertainty
    "UncertaintyResult",
    "entropy",
    "softmax_logits",
    "aleatoric_from_models",
    "total_uncertainty_ensemble",
    "epistemic_from_models",
    "compute_uncertainty_decomposition",
    # Training
    "TrainConfig",
    "MNISTTrainConfig",
    "TrainResult",
    "train_model",
    "train_image_model",
    "train_ensemble",
    "train_mnist_ensemble",
    "extract_models",
    # CF Problem
    "CFConfig",
    "ImageCFConfig",
    "make_cf_problem",
    "make_cf_problem_image_space",
    # Optimization
    "NSGAConfig",
    "FactualBasedSampling",
    "ValidCFCallback",
    "ValidCounterfactualTermination",
    "run_nsga2",
    # Visualization
    "get_class_colors",
    "get_decision_boundary_grid",
    "plot_proba",
    "plot_uncertainty_heatmap",
    "visualize_image_counterfactuals",
    "plot_pareto_front_2d",
    "plot_pareto_front_3d",
    "plot_ensemble_decision_boundaries",
    # Pipeline
    "Experiment2DConfig",
    "ExperimentMNISTConfig",
    "ExperimentResult",
    "run_2d_experiment",
    "run_mnist_experiment",
    "save_experiment_result",
    "load_experiment_result",
    "set_random_seed",
]
