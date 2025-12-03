"""
NSGA-II optimization utilities for counterfactual generation.

Provides custom sampling, callbacks, and termination criteria for
multi-objective counterfactual optimization.

Implements MOC (Multi-Objective Counterfactuals) framework adjustments:
1. ICE Curve Variance Initialization - biases initial population towards influential features
2. Conditional Mutator with Transformation Trees - ensures plausible mutations
3. Modified Crowding Distance - diversity in both objective and feature space
4. Penalization - constraint handling for validity

All components support direct tensor computation to avoid unnecessary numpy conversions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowdingSurvival
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.core.termination import Termination
from pymoo.core.survival import Survival
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.tree import DecisionTreeRegressor
import torch


Array = Union[np.ndarray, torch.Tensor]


# =============================================================================
# Tensor/Numpy Utility Functions
# =============================================================================

def to_tensor(x: Array, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert array to tensor, avoiding copy if already a tensor on correct device."""
    if isinstance(x, torch.Tensor):
        if device is not None and x.device != device:
            return x.to(device)
        return x
    # numpy array
    t = torch.from_numpy(np.asarray(x).astype(np.float32))
    if device is not None:
        t = t.to(device)
    return t


def to_numpy(x: Array) -> np.ndarray:
    """Convert array to numpy, avoiding copy if already numpy."""
    if isinstance(x, np.ndarray):
        return x
    # torch tensor
    return x.detach().cpu().numpy()


def is_tensor(x: Array) -> bool:
    """Check if x is a torch tensor."""
    return isinstance(x, torch.Tensor)


def get_device(x: Array) -> Optional[torch.device]:
    """Get device of tensor, or None for numpy."""
    if isinstance(x, torch.Tensor):
        return x.device
    return None


@dataclass
class FeatureTypeConfig:
    """
    Configuration for feature types in mixed-integer optimization (MIES).
    
    Supports:
    - continuous: Numerical features (SBX crossover, Gaussian/polynomial mutation)
    - categorical: Discrete features with multiple levels (uniform crossover, uniform sampling mutation)
    - binary: Binary features (uniform crossover, flip mutation)
    """
    # Feature type for each variable: 'continuous', 'categorical', or 'binary'
    # If None, all features are treated as continuous
    feature_types: Optional[List[str]] = None
    
    # For categorical features: list of admissible levels for each categorical feature
    # Dict mapping feature index -> list of admissible values
    categorical_levels: Optional[Dict[int, List[float]]] = None
    
    # Indices of non-actionable features (permanently fixed to x*)
    fixed_features: Optional[List[int]] = None
    
    # Feature bounds for actionability (capping extreme values)
    # If None, derived from problem bounds or X_obs
    actionable_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None


@dataclass
class NSGAConfig:
    """Configuration for NSGA-II optimization with MOC framework."""
    
    pop_size: int = 100
    n_gen: int = 200
    seed: Optional[int] = 42
    verbose: bool = True
    
    # Crossover parameters
    crossover_prob: float = 0.9
    crossover_eta: float = 15
    
    # Mutation parameters (per-variable probability will be 1/n_var if None)
    mutation_prob: Optional[float] = None
    mutation_eta: float = 20
    
    # Custom termination
    use_valid_cf_termination: bool = True
    min_valid_cf: int = 25
    validity_threshold: float = 0.5  # F[0] < threshold means valid
    min_gen: int = 50
    max_gen: int = 500
    
    # MOC Framework Adjustments
    use_ice_initialization: bool = True      # ICE curve variance initialization
    use_conditional_mutator: bool = True     # Conditional mutator with transformation trees
    use_modified_crowding: bool = True       # Modified crowding distance (feature + objective)
    use_penalization: bool = True            # Constraint handling via penalization
    use_feature_reset: bool = True           # Reset features to x* after mutation (sparsity)
    use_mies: bool = True                    # Mixed Integer Evolutionary Strategies
    
    # ICE initialization parameters
    ice_p_min: float = 0.01                  # Min probability of changing a feature
    ice_p_max: float = 0.99                  # Max probability of changing a feature
    ice_n_samples: int = 50                  # Number of samples for ICE curve estimation
    
    # Penalization parameters
    validity_epsilon: float = 0.5            # Threshold for constraint violation
    
    # Crowding distance weighting
    cd_objective_weight: float = 0.5         # Weight for objective space crowding
    cd_feature_weight: float = 0.5           # Weight for feature space crowding
    
    # Feature reset parameters (sparsity enforcement)
    feature_reset_prob: float = 0.1          # Probability of resetting each feature to x*
    
    # Feature type configuration for MIES
    feature_config: Optional[FeatureTypeConfig] = None

class FactualBasedSampling(FloatRandomSampling):
    """
    Initialize population around the factual instance.
    
    Instead of uniform random sampling, generates samples near x*
    with small perturbations. This helps the optimization start
    from relevant regions of the search space.
    """
    
    def __init__(self, x_star: Array, noise_scale: float = 0.1):
        """
        Args:
            x_star: Factual instance to sample around
            noise_scale: Standard deviation of Gaussian noise as fraction of range
        """
        super().__init__()
        self.x_star = np.asarray(x_star).flatten()
        self.noise_scale = noise_scale
    
    def _do(self, problem, n_samples, **kwargs) -> Array:
        """Generate samples around the factual."""
        # Tile factual to create n_samples copies
        X = np.tile(self.x_star, (n_samples, 1))
        
        # Add uniform noise
        noise = np.random.uniform(
            -self.noise_scale,
            self.noise_scale,
            X.shape
        )
        X = X + noise
        
        # Clip to problem bounds
        X = np.clip(X, problem.xl, problem.xu)
        
        return X


# =============================================================================
# MOC Adjustment 1: ICE Curve Variance Initialization
# =============================================================================

def compute_ice_curves(
    model: Callable,
    x_star: Array,
    X_obs: Array,
    n_samples: int = 50,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute Individual Conditional Expectation (ICE) curve variance for each feature.
    
    The ICE curve shows how the model's prediction changes when varying a single
    feature while keeping all others fixed at x_star's values.
    
    Supports both tensor and numpy inputs. Computation is done on GPU if device is CUDA.
    
    Args:
        model: Callable that takes (N, d) input and returns predictions
        x_star: Factual instance, shape (d,) - tensor or numpy
        X_obs: Observed data for sampling feature values, shape (N, d) - tensor or numpy
        n_samples: Number of samples per feature for ICE curve
        device: PyTorch device
        
    Returns:
        sigma_ice: Standard deviation of ICE curve for each feature, shape (d,) as tensor
    """
    # Determine device
    if device is None:
        if is_tensor(x_star):
            device = x_star.device
        elif is_tensor(X_obs):
            device = X_obs.device
        elif hasattr(model, 'parameters'):
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
    
    # Convert to tensors on device
    x_star_t = to_tensor(x_star, device).flatten()
    X_obs_t = to_tensor(X_obs, device)
    if X_obs_t.dim() > 2:
        X_obs_t = X_obs_t.reshape(X_obs_t.shape[0], -1)
    
    n_features = x_star_t.shape[0]
    sigma_ice = torch.zeros(n_features, device=device)
    
    for j in range(n_features):
        # Sample values for feature j from observed data
        feature_min = X_obs_t[:, j].min()
        feature_max = X_obs_t[:, j].max()
        feature_values = torch.linspace(feature_min, feature_max, n_samples, device=device)
        
        # Create modified instances: vary feature j, keep others at x_star
        X_ice = x_star_t.unsqueeze(0).repeat(n_samples, 1)  # (n_samples, d)
        X_ice[:, j] = feature_values
        
        # Get model predictions
        with torch.no_grad():
            if hasattr(model, 'forward'):
                preds = model(X_ice)
                if preds.dim() > 1:
                    # Get probability of positive class or last class
                    preds = torch.softmax(preds, dim=-1)[:, -1]
            else:
                # Callable that may not be a PyTorch model
                preds = model(X_ice)
                if not isinstance(preds, torch.Tensor):
                    preds = torch.tensor(preds, device=device, dtype=torch.float32)
        
        # Compute standard deviation of ICE curve
        sigma_ice[j] = preds.std()
    
    return sigma_ice


def ice_to_probability(
    sigma_ice: Array,
    p_min: float = 0.01,
    p_max: float = 0.99,
) -> torch.Tensor:
    """
    Transform ICE curve standard deviations to probabilities of changing features.
    
    Higher variance → higher probability of initialization different from x_star.
    
    Formula: P(value differs) = p_min + (σ_j - min(σ)) * (p_max - p_min) / (max(σ) - min(σ))
    
    Supports both tensor and numpy inputs.
    
    Args:
        sigma_ice: Standard deviation of ICE curves, shape (d,) - tensor or numpy
        p_min: Minimum probability of changing a feature
        p_max: Maximum probability of changing a feature
        
    Returns:
        p_change: Probability of changing each feature, shape (d,) as tensor
    """
    # Convert to tensor if needed
    if not is_tensor(sigma_ice):
        sigma_ice = torch.tensor(sigma_ice, dtype=torch.float32)
    
    sigma_min = sigma_ice.min()
    sigma_max = sigma_ice.max()
    
    if sigma_max - sigma_min < 1e-10:
        # All features have same importance, use uniform probability
        return torch.full_like(sigma_ice, (p_min + p_max) / 2)
    
    p_change = p_min + (sigma_ice - sigma_min) * (p_max - p_min) / (sigma_max - sigma_min)
    return torch.clamp(p_change, p_min, p_max)


class ICEVarianceSampling(FloatRandomSampling):
    """
    MOC ICE Curve Variance Initialization.
    
    Biases initial population towards changing features that have high
    influence on the model's prediction (measured by ICE curve variance).
    
    Features with higher ICE variance are more likely to be initialized
    with values different from x_star.
    
    Supports tensor inputs for GPU-accelerated computation.
    """
    
    def __init__(
        self,
        x_star: Array,
        X_obs: Array,
        model: Callable,
        p_min: float = 0.01,
        p_max: float = 0.99,
        n_samples: int = 50,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            x_star: Factual instance (tensor or numpy)
            X_obs: Observed data for sampling (tensor or numpy)
            model: Model for computing ICE curves
            p_min: Minimum probability of changing a feature
            p_max: Maximum probability of changing a feature
            n_samples: Number of samples for ICE curve estimation
            device: PyTorch device
        """
        super().__init__()
        
        # Determine device
        if device is None:
            if is_tensor(x_star):
                device = x_star.device
            elif is_tensor(X_obs):
                device = X_obs.device
            else:
                device = torch.device('cpu')
        
        self.device = device
        
        # Store as tensors
        self.x_star = to_tensor(x_star, device).flatten()
        self.X_obs = to_tensor(X_obs, device)
        if self.X_obs.dim() > 2:
            self.X_obs = self.X_obs.reshape(self.X_obs.shape[0], -1)
        
        self.model = model
        self.p_min = p_min
        self.p_max = p_max
        self.n_samples = n_samples
        
        # Precompute ICE curve variances and change probabilities (on GPU if available)
        self.sigma_ice = compute_ice_curves(
            model, self.x_star, self.X_obs, n_samples, device
        )
        self.p_change = ice_to_probability(self.sigma_ice, p_min, p_max)
    
    def _do(self, problem, n_samples, **kwargs) -> np.ndarray:
        """Generate samples using ICE variance-based initialization."""
        n_features = self.x_star.shape[0]
        
        # Generate on device for speed
        X = torch.zeros((n_samples, n_features), device=self.device)
        
        # Random values for comparison
        rand_vals = torch.rand((n_samples, n_features), device=self.device)
        
        # For each feature, decide whether to change or keep original
        for j in range(n_features):
            change_mask = rand_vals[:, j] < self.p_change[j]
            # Sample from observed data for changed features
            n_change = change_mask.sum().item()
            if n_change > 0:
                indices = torch.randint(0, self.X_obs.shape[0], (n_change,), device=self.device)
                X[change_mask, j] = self.X_obs[indices, j]
            # Keep original for unchanged features
            X[~change_mask, j] = self.x_star[j]
        
        # Clip to problem bounds and convert to numpy for pymoo
        xl = torch.tensor(problem.xl, device=self.device, dtype=torch.float32)
        xu = torch.tensor(problem.xu, device=self.device, dtype=torch.float32)
        X = torch.clamp(X, xl, xu)
        
        return X.cpu().numpy()


# =============================================================================
# MOC Adjustment 2: Conditional Mutator with Transformation Trees
# =============================================================================

class TransformationTreeMutator:
    """
    Transformation Trees for conditional value generation.
    
    Trains a decision tree for each feature to learn the conditional
    distribution P(feature_j | other_features) from observed data.
    
    Supports both tensor and numpy inputs (converts to numpy internally
    since sklearn decision trees require numpy).
    """
    
    def __init__(
        self,
        X_obs: Array,
        max_depth: int = 10,
        min_samples_leaf: int = 5,
    ):
        """
        Train transformation trees on observed data.
        
        Args:
            X_obs: Observed data, shape (N, d) - can be tensor or numpy
            max_depth: Maximum tree depth
            min_samples_leaf: Minimum samples per leaf
        """
        # Convert to numpy (sklearn requires numpy)
        X_obs_np = to_numpy(X_obs)
        self.X_obs = X_obs_np.reshape(X_obs_np.shape[0], -1)
        self.n_features = self.X_obs.shape[1]
        self.trees: List[DecisionTreeRegressor] = []
        
        # Train a tree for each feature
        for j in range(self.n_features):
            # Features: all other features
            X_train = np.delete(self.X_obs, j, axis=1)
            # Target: feature j
            y_train = self.X_obs[:, j]
            
            tree = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
            )
            tree.fit(X_train, y_train)
            self.trees.append(tree)
        
        # Store leaf assignments for efficient sampling
        self._leaf_samples = self._compute_leaf_samples()
    
    def _compute_leaf_samples(self) -> List[Dict[int, np.ndarray]]:
        """Precompute which training samples fall into each leaf."""
        leaf_samples = []
        
        for j, tree in enumerate(self.trees):
            X_train = np.delete(self.X_obs, j, axis=1)
            leaf_ids = tree.apply(X_train)
            
            # Group sample indices by leaf
            leaf_dict = {}
            for idx, leaf_id in enumerate(leaf_ids):
                if leaf_id not in leaf_dict:
                    leaf_dict[leaf_id] = []
                leaf_dict[leaf_id].append(self.X_obs[idx, j])
            
            # Convert to arrays
            leaf_dict = {k: np.array(v) for k, v in leaf_dict.items()}
            leaf_samples.append(leaf_dict)
        
        return leaf_samples
    
    def sample_conditional(
        self,
        x: np.ndarray,
        feature_idx: int,
    ) -> float:
        """
        Sample a plausible value for feature_idx conditioned on other features.
        
        Args:
            x: Current individual, shape (d,) - can be tensor or numpy
            feature_idx: Index of feature to sample
            
        Returns:
            Sampled value for the feature
        """
        # Convert to numpy (sklearn requires numpy)
        x_np = to_numpy(x)
        
        # Get tree for this feature
        tree = self.trees[feature_idx]
        
        # Prepare input (all features except feature_idx)
        x_other = np.delete(x_np, feature_idx).reshape(1, -1)
        
        # Find which leaf this instance falls into
        leaf_id = tree.apply(x_other)[0]
        
        # Sample from training points in same leaf
        leaf_values = self._leaf_samples[feature_idx].get(leaf_id)
        
        if leaf_values is not None and len(leaf_values) > 0:
            return float(np.random.choice(leaf_values))
        else:
            # Fallback: use tree prediction (mean of leaf)
            return float(tree.predict(x_other)[0])


class ConditionalMutation(Mutation):
    """
    MOC Conditional Mutator.
    
    Uses transformation trees to generate plausible feature values
    conditioned on other features, ensuring mutations stay on-manifold.
    
    Key implementation details:
    1. Mutations are applied in randomized order (important for dependencies)
    2. Each mutation conditions on the current state of other features
    3. Falls back to standard polynomial mutation if tree sampling fails
    """
    
    def __init__(
        self,
        transformation_trees: TransformationTreeMutator,
        prob: float = None,
        eta: float = 20,
        use_conditional_prob: float = 0.7,
    ):
        """
        Args:
            transformation_trees: Trained TransformationTreeMutator
            prob: Mutation probability per variable
            eta: Distribution index for fallback polynomial mutation
            use_conditional_prob: Probability of using conditional vs standard mutation
        """
        super().__init__()
        self.trees = transformation_trees
        self.prob = prob
        self.eta = eta
        self.use_conditional_prob = use_conditional_prob
        
        # Standard polynomial mutator as fallback
        self._pm = PM(prob=prob, eta=eta)
    
    def _do(self, problem, X, **kwargs) -> np.ndarray:
        """
        Apply conditional mutation to population.
        
        Args:
            problem: pymoo Problem
            X: Population decision variables, shape (n_pop, n_var)
            
        Returns:
            Mutated population, shape (n_pop, n_var)
        """
        n_pop, n_var = X.shape
        X_mutated = X.copy()
        
        # Per-variable mutation probability
        prob = self.prob if self.prob is not None else (1.0 / n_var)
        
        for i in range(n_pop):
            # Randomize feature order for mutation (critical for conditional logic)
            feature_order = np.random.permutation(n_var)
            
            for j in feature_order:
                if np.random.random() < prob:
                    if np.random.random() < self.use_conditional_prob:
                        # Use conditional mutation via transformation tree
                        new_value = self.trees.sample_conditional(X_mutated[i], j)
                        X_mutated[i, j] = new_value
                    else:
                        # Use standard polynomial mutation for this gene
                        delta = (problem.xu[j] - problem.xl[j]) * 0.1
                        X_mutated[i, j] += np.random.uniform(-delta, delta)
        
        # Clip to bounds
        X_mutated = np.clip(X_mutated, problem.xl, problem.xu)
        
        return X_mutated


# =============================================================================
# MOC Adjustment: MIES (Mixed Integer Evolutionary Strategies)
# =============================================================================

class MIESMutation(Mutation):
    """
    Mixed Integer Evolutionary Strategies (MIES) Mutator.
    
    Handles mixed discrete and continuous search spaces:
    - Continuous features: Scaled Gaussian mutation (similar to polynomial mutation)
    - Categorical features: Uniform sampling from admissible levels
    - Binary features: Flip mutation
    
    Based on Li et al. [19] from the MOC paper.
    """
    
    def __init__(
        self,
        feature_types: List[str],
        categorical_levels: Optional[Dict[int, List[float]]] = None,
        prob: float = None,
        eta: float = 20,
        sigma_scale: float = 0.1,
    ):
        """
        Args:
            feature_types: List of types for each feature: 'continuous', 'categorical', 'binary'
            categorical_levels: Dict mapping categorical feature index -> admissible values
            prob: Mutation probability per variable
            eta: Distribution index for continuous mutation
            sigma_scale: Scale for Gaussian mutation (fraction of feature range)
        """
        super().__init__()
        self.feature_types = feature_types
        self.categorical_levels = categorical_levels or {}
        self.prob = prob
        self.eta = eta
        self.sigma_scale = sigma_scale
    
    def _do(self, problem, X, **kwargs) -> np.ndarray:
        """
        Apply MIES mutation to population.
        
        Args:
            problem: pymoo Problem
            X: Population decision variables, shape (n_pop, n_var)
            
        Returns:
            Mutated population, shape (n_pop, n_var)
        """
        n_pop, n_var = X.shape
        X_mutated = X.copy()
        
        # Per-variable mutation probability
        prob = self.prob if self.prob is not None else (1.0 / n_var)
        
        for i in range(n_pop):
            for j in range(n_var):
                if np.random.random() < prob:
                    ftype = self.feature_types[j] if j < len(self.feature_types) else 'continuous'
                    
                    if ftype == 'continuous':
                        # Scaled Gaussian mutation
                        sigma = (problem.xu[j] - problem.xl[j]) * self.sigma_scale
                        X_mutated[i, j] += np.random.normal(0, sigma)
                        
                    elif ftype == 'categorical':
                        # Uniform sampling from admissible levels
                        if j in self.categorical_levels:
                            X_mutated[i, j] = np.random.choice(self.categorical_levels[j])
                        else:
                            # Fallback: sample uniformly from bounds
                            X_mutated[i, j] = np.random.uniform(problem.xl[j], problem.xu[j])
                            
                    elif ftype == 'binary':
                        # Flip mutation
                        X_mutated[i, j] = 1.0 - X_mutated[i, j]
        
        # Clip continuous features to bounds
        for j in range(n_var):
            ftype = self.feature_types[j] if j < len(self.feature_types) else 'continuous'
            if ftype == 'continuous':
                X_mutated[:, j] = np.clip(X_mutated[:, j], problem.xl[j], problem.xu[j])
            elif ftype == 'binary':
                X_mutated[:, j] = np.clip(X_mutated[:, j], 0, 1)
        
        return X_mutated


class MIESConditionalMutation(Mutation):
    """
    Combined MIES + Conditional Mutation.
    
    Uses transformation trees for conditional sampling when available,
    with MIES-style mutations as fallback based on feature types.
    """
    
    def __init__(
        self,
        feature_types: List[str],
        transformation_trees: Optional[TransformationTreeMutator] = None,
        categorical_levels: Optional[Dict[int, List[float]]] = None,
        prob: float = None,
        use_conditional_prob: float = 0.7,
        sigma_scale: float = 0.1,
    ):
        super().__init__()
        self.feature_types = feature_types
        self.trees = transformation_trees
        self.categorical_levels = categorical_levels or {}
        self.prob = prob
        self.use_conditional_prob = use_conditional_prob
        self.sigma_scale = sigma_scale
    
    def _do(self, problem, X, **kwargs) -> np.ndarray:
        n_pop, n_var = X.shape
        X_mutated = X.copy()
        prob = self.prob if self.prob is not None else (1.0 / n_var)
        
        for i in range(n_pop):
            # Randomize feature order for conditional mutation
            feature_order = np.random.permutation(n_var)
            
            for j in feature_order:
                if np.random.random() < prob:
                    ftype = self.feature_types[j] if j < len(self.feature_types) else 'continuous'
                    
                    # Try conditional mutation first
                    if self.trees is not None and np.random.random() < self.use_conditional_prob:
                        new_value = self.trees.sample_conditional(X_mutated[i], j)
                        X_mutated[i, j] = new_value
                    else:
                        # MIES mutation based on feature type
                        if ftype == 'continuous':
                            sigma = (problem.xu[j] - problem.xl[j]) * self.sigma_scale
                            X_mutated[i, j] += np.random.normal(0, sigma)
                        elif ftype == 'categorical':
                            if j in self.categorical_levels:
                                X_mutated[i, j] = np.random.choice(self.categorical_levels[j])
                            else:
                                X_mutated[i, j] = np.random.uniform(problem.xl[j], problem.xu[j])
                        elif ftype == 'binary':
                            X_mutated[i, j] = 1.0 - X_mutated[i, j]
        
        # Clip to bounds
        for j in range(n_var):
            ftype = self.feature_types[j] if j < len(self.feature_types) else 'continuous'
            if ftype == 'continuous':
                X_mutated[:, j] = np.clip(X_mutated[:, j], problem.xl[j], problem.xu[j])
            elif ftype == 'binary':
                X_mutated[:, j] = np.clip(X_mutated[:, j], 0, 1)
        
        return X_mutated


# =============================================================================
# MOC Adjustment: Feature Reset to x* (Sparsity Enforcement)
# =============================================================================

class FeatureResetOperator:
    """
    Post-mutation operator that resets features to x* with low probability.
    
    From MOC paper: "After recombination and mutation, some feature values are 
    randomly set to the values of x* with a given (low) probability—another 
    control parameter—to prevent all features from deviating from x*."
    
    This encourages sparsity in counterfactual explanations.
    """
    
    def __init__(
        self,
        x_star: np.ndarray,
        reset_prob: float = 0.1,
        fixed_features: Optional[List[int]] = None,
    ):
        """
        Args:
            x_star: Factual instance values, shape (d,)
            reset_prob: Probability of resetting each feature to x*
            fixed_features: Indices of features always fixed to x* (non-actionable)
        """
        self.x_star = np.asarray(x_star).flatten()
        self.reset_prob = reset_prob
        self.fixed_features = set(fixed_features) if fixed_features else set()
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Apply feature reset to population.
        
        Args:
            X: Population, shape (n_pop, n_var)
            
        Returns:
            Population with some features reset to x*
        """
        n_pop, n_var = X.shape
        X_reset = X.copy()
        
        for i in range(n_pop):
            for j in range(n_var):
                # Always reset non-actionable features
                if j in self.fixed_features:
                    X_reset[i, j] = self.x_star[j]
                # Randomly reset other features with low probability
                elif np.random.random() < self.reset_prob:
                    X_reset[i, j] = self.x_star[j]
        
        return X_reset


class ActionabilityEnforcer:
    """
    Enforces actionability constraints on candidates.
    
    From MOC paper:
    1. Caps extreme values to predefined bounds
    2. Permanently fixes non-actionable features to x*
    """
    
    def __init__(
        self,
        x_star: np.ndarray,
        fixed_features: Optional[List[int]] = None,
        actionable_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        """
        Args:
            x_star: Factual instance values, shape (d,)
            fixed_features: Indices of non-actionable features
            actionable_bounds: (lower, upper) bounds for capping, each shape (d,)
        """
        self.x_star = np.asarray(x_star).flatten()
        self.fixed_features = set(fixed_features) if fixed_features else set()
        self.actionable_bounds = actionable_bounds
    
    def __call__(self, X: np.ndarray) -> np.ndarray:
        """
        Enforce actionability on population.
        
        Args:
            X: Population, shape (n_pop, n_var)
            
        Returns:
            Population with actionability enforced
        """
        n_pop, n_var = X.shape
        X_enforced = X.copy()
        
        # Fix non-actionable features
        for j in self.fixed_features:
            if j < n_var:
                X_enforced[:, j] = self.x_star[j]
        
        # Cap to actionable bounds
        if self.actionable_bounds is not None:
            lower, upper = self.actionable_bounds
            X_enforced = np.clip(X_enforced, lower, upper)
        
        return X_enforced


# =============================================================================
# MOC Adjustment: MIES Crossover (Mixed Types)
# =============================================================================

from pymoo.core.crossover import Crossover


class MIESCrossover(Crossover):
    """
    Mixed Integer Evolutionary Strategies Crossover.
    
    From MOC paper:
    - Numerical features: Simulated Binary Crossover (SBX)
    - Categorical/Binary features: Uniform Crossover
    """
    
    def __init__(
        self,
        feature_types: List[str],
        prob: float = 0.9,
        eta: float = 15,
    ):
        """
        Args:
            feature_types: List of types for each feature
            prob: Crossover probability
            eta: Distribution index for SBX
        """
        # n_parents=2, n_offsprings=2
        super().__init__(n_parents=2, n_offsprings=2)
        self.feature_types = feature_types
        self.prob = prob
        self.eta = eta
    
    def _do(self, problem, X, **kwargs):
        """
        Apply MIES crossover.
        
        Args:
            problem: pymoo Problem
            X: Parents, shape (n_parents=2, n_matings, n_var)
            
        Returns:
            Offspring, shape (n_offsprings=2, n_matings, n_var)
        """
        n_parents, n_matings, n_var = X.shape
        # Output shape must be (n_offsprings, n_matings, n_var)
        Y = np.full((2, n_matings, n_var), np.nan)
        
        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]  # Parents indexed as (parent_idx, mating_idx, var)
            c1, c2 = p1.copy(), p2.copy()
            
            if np.random.random() < self.prob:
                for j in range(n_var):
                    ftype = self.feature_types[j] if j < len(self.feature_types) else 'continuous'
                    
                    if ftype == 'continuous':
                        # SBX crossover for continuous features
                        if np.random.random() < 0.5:
                            if abs(p1[j] - p2[j]) > 1e-10:
                                c1[j], c2[j] = self._sbx_single(
                                    p1[j], p2[j], 
                                    problem.xl[j], problem.xu[j], 
                                    self.eta
                                )
                    else:
                        # Uniform crossover for categorical/binary
                        if np.random.random() < 0.5:
                            c1[j], c2[j] = p2[j], p1[j]
            
            Y[0, k] = c1
            Y[1, k] = c2
        
        return Y
    
    def _sbx_single(self, p1: float, p2: float, xl: float, xu: float, eta: float) -> Tuple[float, float]:
        """SBX for a single continuous variable."""
        u = np.random.random()
        
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (eta + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
        
        c1 = 0.5 * ((p1 + p2) - beta * abs(p2 - p1))
        c2 = 0.5 * ((p1 + p2) + beta * abs(p2 - p1))
        
        c1 = np.clip(c1, xl, xu)
        c2 = np.clip(c2, xl, xu)
        
        return c1, c2


# =============================================================================
# MOC Adjustment 3: Modified Crowding Distance (Feature + Objective Space)
# =============================================================================

def gower_distance_tensor(
    x: torch.Tensor, 
    y: torch.Tensor, 
    feature_range: torch.Tensor
) -> torch.Tensor:
    """
    Compute Gower distance between two individuals using tensors.
    
    Gower distance normalizes each feature by its range.
    """
    diff = torch.abs(x - y) / torch.clamp(feature_range, min=1e-10)
    return torch.mean(torch.clamp(diff, max=1.0))


def gower_distance(x: Array, y: Array, feature_range: Array) -> float:
    """
    Compute Gower distance between two individuals.
    
    Gower distance normalizes each feature by its range.
    Supports both tensor and numpy inputs.
    """
    if is_tensor(x) and is_tensor(y) and is_tensor(feature_range):
        return gower_distance_tensor(x, y, feature_range).item()
    
    # Convert to numpy for computation
    x_np = to_numpy(x)
    y_np = to_numpy(y)
    fr_np = to_numpy(feature_range)
    
    diff = np.abs(x_np - y_np) / np.maximum(fr_np, 1e-10)
    return float(np.mean(np.minimum(diff, 1.0)))


def compute_crowding_distance_L1_tensor(F: torch.Tensor) -> torch.Tensor:
    """
    Compute crowding distance in objective space using L1 norm (tensor version).
    
    For each objective, sort by that objective and assign distance
    as the difference between neighbors, normalized by range.
    
    Args:
        F: Objective values, shape (n, m) as tensor
        
    Returns:
        Crowding distances, shape (n,) as tensor
    """
    n, m = F.shape
    device = F.device
    
    if n <= 2:
        return torch.full((n,), float('inf'), device=device)
    
    cd = torch.zeros(n, device=device)
    
    for obj in range(m):
        # Sort by this objective
        sorted_idx = torch.argsort(F[:, obj])
        
        # Objective range
        obj_range = F[:, obj].max() - F[:, obj].min()
        if obj_range < 1e-10:
            continue
        
        # Boundary points get infinite distance
        cd[sorted_idx[0]] = float('inf')
        cd[sorted_idx[-1]] = float('inf')
        
        # Interior points: distance to neighbors
        for k in range(1, n - 1):
            cd[sorted_idx[k]] += (
                F[sorted_idx[k + 1], obj] - F[sorted_idx[k - 1], obj]
            ) / obj_range
    
    return cd


def compute_crowding_distance_L1(F: Array) -> Array:
    """
    Compute crowding distance in objective space using L1 norm.
    
    Supports both tensor and numpy inputs.
    
    Args:
        F: Objective values, shape (n, m)
        
    Returns:
        Crowding distances, shape (n,)
    """
    if is_tensor(F):
        return compute_crowding_distance_L1_tensor(F)
    
    # Numpy version
    n, m = F.shape
    
    if n <= 2:
        return np.full(n, np.inf)
    
    cd = np.zeros(n)
    
    for obj in range(m):
        # Sort by this objective
        sorted_idx = np.argsort(F[:, obj])
        
        # Objective range
        obj_range = F[:, obj].max() - F[:, obj].min()
        if obj_range < 1e-10:
            continue
        
        # Boundary points get infinite distance
        cd[sorted_idx[0]] = np.inf
        cd[sorted_idx[-1]] = np.inf
        
        # Interior points: distance to neighbors
        for k in range(1, n - 1):
            cd[sorted_idx[k]] += (
                F[sorted_idx[k + 1], obj] - F[sorted_idx[k - 1], obj]
            ) / obj_range
    
    return cd


def compute_crowding_distance_feature_space_tensor(
    X: torch.Tensor,
    feature_range: torch.Tensor,
) -> torch.Tensor:
    """
    Compute crowding distance in feature space using Gower distance (tensor version).
    
    For each individual, sum distances to 2 nearest neighbors.
    
    Args:
        X: Decision variables, shape (n, d) as tensor
        feature_range: Range of each feature, shape (d,) as tensor
        
    Returns:
        Crowding distances in feature space, shape (n,) as tensor
    """
    n = X.shape[0]
    device = X.device
    
    if n <= 2:
        return torch.full((n,), float('inf'), device=device)
    
    cd_feature = torch.zeros(n, device=device)
    
    # Compute pairwise Gower distances efficiently
    # Normalize X by feature range
    X_norm = X / torch.clamp(feature_range, min=1e-10)
    
    for i in range(n):
        # Compute distance to all others
        diffs = torch.abs(X_norm[i] - X_norm)  # (n, d)
        diffs = torch.clamp(diffs, max=1.0)
        distances = diffs.mean(dim=1)  # (n,)
        distances[i] = float('inf')  # Exclude self
        
        # Get 2 smallest distances
        sorted_dists, _ = torch.sort(distances)
        cd_feature[i] = sorted_dists[0] + sorted_dists[1]
    
    return cd_feature


def compute_crowding_distance_feature_space(
    X: Array,
    feature_range: Array,
) -> Array:
    """
    Compute crowding distance in feature space using Gower distance.
    
    For each individual, sum distances to 2 nearest neighbors.
    Supports both tensor and numpy inputs.
    
    Args:
        X: Decision variables, shape (n, d)
        feature_range: Range of each feature, shape (d,)
        
    Returns:
        Crowding distances in feature space, shape (n,)
    """
    if is_tensor(X) and is_tensor(feature_range):
        return compute_crowding_distance_feature_space_tensor(X, feature_range)
    
    # Numpy version
    X_np = to_numpy(X)
    fr_np = to_numpy(feature_range)
    
    n = X_np.shape[0]
    
    if n <= 2:
        return np.full(n, np.inf)
    
    cd_feature = np.zeros(n)
    
    for i in range(n):
        # Compute Gower distance to all other individuals
        distances = []
        for j in range(n):
            if i != j:
                distances.append(gower_distance(X_np[i], X_np[j], fr_np))
        
        distances = np.array(distances)
        distances.sort()
        
        # Sum of distances to 2 nearest neighbors
        cd_feature[i] = distances[0] + (distances[1] if len(distances) > 1 else 0)
    
    return cd_feature


def modified_crowding_distance_tensor(
    F: torch.Tensor,
    X: torch.Tensor,
    feature_range: torch.Tensor,
    weight_obj: float = 0.5,
    weight_feature: float = 0.5,
) -> torch.Tensor:
    """
    MOC Modified Crowding Distance (tensor version).
    
    Combines crowding distance in objective space (L1 norm) and
    feature space (Gower distance) with equal weighting.
    """
    device = F.device
    
    cd_obj = compute_crowding_distance_L1_tensor(F)
    cd_feature = compute_crowding_distance_feature_space_tensor(X, feature_range)
    
    # Handle infinite values
    inf_mask_obj = torch.isinf(cd_obj)
    inf_mask_feature = torch.isinf(cd_feature)
    
    cd_obj_finite = cd_obj.clone()
    cd_feature_finite = cd_feature.clone()
    
    if not inf_mask_obj.all():
        max_obj = cd_obj[~inf_mask_obj].max() * 2
        cd_obj_finite[inf_mask_obj] = max_obj
    
    if not inf_mask_feature.all():
        max_feature = cd_feature[~inf_mask_feature].max() * 2
        cd_feature_finite[inf_mask_feature] = max_feature
    
    # Normalize to [0, 1]
    def normalize_tensor(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min < 1e-10:
            return torch.ones_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    cd_obj_norm = normalize_tensor(cd_obj_finite)
    cd_feature_norm = normalize_tensor(cd_feature_finite)
    
    # Combined crowding distance
    cd_combined = weight_obj * cd_obj_norm + weight_feature * cd_feature_norm
    
    # Restore infinite values for boundary points
    cd_combined[inf_mask_obj | inf_mask_feature] = float('inf')
    
    return cd_combined


def modified_crowding_distance(
    F: Array,
    X: Array,
    feature_range: Array,
    weight_obj: float = 0.5,
    weight_feature: float = 0.5,
) -> Array:
    """
    MOC Modified Crowding Distance.
    
    Combines crowding distance in objective space (L1 norm) and
    feature space (Gower distance) with equal weighting.
    
    This ensures diversity in both the objective values AND the
    actual feature changes, providing more diverse actionable advice.
    
    Supports both tensor and numpy inputs.
    
    Args:
        F: Objective values, shape (n, m)
        X: Decision variables, shape (n, d)
        feature_range: Range of each feature, shape (d,)
        weight_obj: Weight for objective space crowding
        weight_feature: Weight for feature space crowding
        
    Returns:
        Combined crowding distances, shape (n,)
    """
    if is_tensor(F) and is_tensor(X) and is_tensor(feature_range):
        return modified_crowding_distance_tensor(F, X, feature_range, weight_obj, weight_feature)
    
    # Numpy version
    F_np = to_numpy(F)
    X_np = to_numpy(X)
    fr_np = to_numpy(feature_range)
    
    cd_obj = compute_crowding_distance_L1(F_np)
    cd_feature = compute_crowding_distance_feature_space(X_np, fr_np)
    
    # Handle infinite values
    cd_obj_finite = np.where(np.isinf(cd_obj), np.nanmax(cd_obj[~np.isinf(cd_obj)]) * 2, cd_obj)
    cd_feature_finite = np.where(np.isinf(cd_feature), np.nanmax(cd_feature[~np.isinf(cd_feature)]) * 2, cd_feature)
    
    # Normalize to [0, 1]
    def normalize(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min < 1e-10:
            return np.ones_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    cd_obj_norm = normalize(cd_obj_finite)
    cd_feature_norm = normalize(cd_feature_finite)
    
    # Combined crowding distance
    cd_combined = weight_obj * cd_obj_norm + weight_feature * cd_feature_norm
    
    # Restore infinite values for boundary points
    cd_combined[np.isinf(cd_obj) | np.isinf(cd_feature)] = np.inf
    
    return cd_combined


# =============================================================================
# MOC Adjustment 4: Penalization (Constraint Handling)
# =============================================================================

def penalized_nondominated_sort(
    F: np.ndarray,
    constraint_violations: np.ndarray,
) -> List[np.ndarray]:
    """
    MOC Penalization: Nondominated sorting with constraint handling.
    
    After standard nondominated sorting, constraint violators are
    reassigned to later fronts based on their violation magnitude.
    
    This ensures that infeasible solutions (CFs too far from target)
    are less likely to be selected for the next generation.
    
    Args:
        F: Objective values, shape (n, m)
        constraint_violations: Constraint violation for each individual, shape (n,)
                              0 = feasible, >0 = magnitude of violation
                              
    Returns:
        List of fronts, each front is array of indices
    """
    n = F.shape[0]
    
    # Step 1: Standard nondominated sorting
    nds = NonDominatedSorting()
    fronts = nds.do(F, return_rank=False)
    
    # Convert to lists for modification
    fronts = [list(f) for f in fronts]
    n_original_fronts = len(fronts)
    
    # Step 2: Identify and remove violators from their fronts
    violators = []
    for front_idx, front in enumerate(fronts):
        to_remove = []
        for ind_idx in front:
            if constraint_violations[ind_idx] > 0:
                violators.append((ind_idx, constraint_violations[ind_idx]))
                to_remove.append(ind_idx)
        for idx in to_remove:
            front.remove(idx)
    
    # Step 3: Sort violators by violation magnitude (least violating first)
    violators.sort(key=lambda x: x[1])
    
    # Step 4: Assign violators to penalty fronts
    for i, (ind_idx, _) in enumerate(violators):
        penalty_front_idx = n_original_fronts + i
        while penalty_front_idx >= len(fronts):
            fronts.append([])
        fronts[penalty_front_idx].append(ind_idx)
    
    # Remove empty fronts
    fronts = [np.array(f) for f in fronts if len(f) > 0]
    
    return fronts


class PenalizedRankAndCrowdingSurvival(Survival):
    """
    MOC Survival operator with penalization and modified crowding distance.
    
    Combines:
    1. Penalized nondominated sorting (constraint handling)
    2. Modified crowding distance (feature + objective space diversity)
    """
    
    def __init__(
        self,
        validity_epsilon: float = 0.5,
        weight_obj: float = 0.5,
        weight_feature: float = 0.5,
        nds: NonDominatedSorting = None,
    ):
        """
        Args:
            validity_epsilon: Threshold for validity constraint
            weight_obj: Weight for objective space crowding distance
            weight_feature: Weight for feature space crowding distance
            nds: NonDominatedSorting instance
        """
        super().__init__(filter_infeasible=False)
        self.validity_epsilon = validity_epsilon
        self.weight_obj = weight_obj
        self.weight_feature = weight_feature
        self.nds = nds if nds is not None else NonDominatedSorting()
        self._feature_range = None
    
    def set_feature_range(self, feature_range: np.ndarray):
        """Set feature range for Gower distance computation."""
        self._feature_range = feature_range
    
    def _do(
        self,
        problem: Problem,
        pop: Population,
        *args,
        n_survive: int = None,
        **kwargs,
    ) -> Population:
        """
        Select survivors using penalized sorting and modified crowding.
        
        Args:
            problem: pymoo Problem
            pop: Current population
            n_survive: Number of survivors to select
            
        Returns:
            Surviving population
        """
        F = pop.get("F")
        X = pop.get("X")
        
        if n_survive is None:
            n_survive = len(pop)
        
        # Compute constraint violations (validity objective is F[:, 0])
        # Violation = max(0, F[:, 0] - epsilon)
        constraint_violations = np.maximum(0, F[:, 0] - self.validity_epsilon)
        
        # Penalized nondominated sorting
        fronts = penalized_nondominated_sort(F, constraint_violations)
        
        # Feature range for Gower distance
        if self._feature_range is None:
            self._feature_range = problem.xu - problem.xl
            self._feature_range[self._feature_range == 0] = 1.0
        
        # Assign ranks to all individuals based on their front
        ranks = np.zeros(len(pop), dtype=int)
        for rank, front in enumerate(fronts):
            ranks[front] = rank
        
        # Compute modified crowding distance for ALL individuals
        all_cd = modified_crowding_distance(
            F, X, self._feature_range,
            self.weight_obj, self.weight_feature
        )
        
        # Set rank and crowding distance on population (required by pymoo's tournament selection)
        pop.set("rank", ranks)
        pop.set("crowding", all_cd)
        
        # Select survivors front by front
        survivors = []
        for front in fronts:
            if len(survivors) + len(front) <= n_survive:
                # Entire front fits
                survivors.extend(front)
            else:
                # Need to select subset based on modified crowding distance
                n_remaining = n_survive - len(survivors)
                
                # Get crowding distances for this front
                cd_front = all_cd[front]
                
                # Select individuals with highest crowding distance
                sorted_by_cd = np.argsort(-cd_front)  # Descending order
                selected = front[sorted_by_cd[:n_remaining]]
                survivors.extend(selected)
                break
        
        return pop[survivors]


class GaussianFactualSampling(FloatRandomSampling):
    """
    Sample from Gaussian distribution centered at factual.
    
    Alternative to uniform noise - may work better for smooth problems.
    """
    
    def __init__(self, x_star: Array, sigma: float = 0.2):
        super().__init__()
        self.x_star = np.asarray(x_star).flatten()
        self.sigma = sigma
    
    def _do(self, problem, n_samples, **kwargs) -> Array:
        X = np.random.normal(
            loc=self.x_star,
            scale=self.sigma,
            size=(n_samples, len(self.x_star))
        )
        return np.clip(X, problem.xl, problem.xu)


class MixedSampling(FloatRandomSampling):
    """
    Mix factual-based and random sampling.
    
    A fraction of the population starts near factual, the rest is random.
    """
    
    def __init__(
        self,
        x_star: Array,
        factual_fraction: float = 0.5,
        noise_scale: float = 0.2,
    ):
        super().__init__()
        self.x_star = np.asarray(x_star).flatten()
        self.factual_fraction = factual_fraction
        self.noise_scale = noise_scale
    
    def _do(self, problem, n_samples, **kwargs) -> Array:
        n_factual = int(n_samples * self.factual_fraction)
        n_random = n_samples - n_factual
        
        # Factual-based samples
        X_factual = np.tile(self.x_star, (n_factual, 1))
        X_factual += np.random.uniform(
            -self.noise_scale, self.noise_scale, X_factual.shape
        )
        
        # Random samples
        X_random = np.random.uniform(
            problem.xl, problem.xu,
            size=(n_random, len(self.x_star))
        )
        
        X = np.vstack([X_factual, X_random])
        return np.clip(X, problem.xl, problem.xu)

@dataclass
class ProgressEntry:
    """Single entry in optimization progress history."""
    gen: int
    n_valid_pop: int
    n_valid_archive: int
    best_validity: float
    best_p_target: float
    mean_sparsity: float = 0.0


class ValidCFCallback(Callback):
    """
    Callback to track valid counterfactuals during optimization.
    
    Prints progress and maintains history of valid CF counts and best validity.
    """
    
    def __init__(
        self,
        validity_threshold: float = 0.5,
        print_every: int = 10,
        verbose: bool = True,
    ):
        """
        Args:
            validity_threshold: F[0] < threshold means valid CF
            print_every: Print progress every N generations
            verbose: Whether to print progress
        """
        super().__init__()
        self.validity_threshold = validity_threshold
        self.print_every = print_every
        self.verbose = verbose
        self.history: List[ProgressEntry] = []
    
    def notify(self, algorithm):
        """Called each generation."""
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")
        
        if F is None:
            return
        
        # Ensure F is 2D
        if F.ndim == 1:
            F = F.reshape(1, -1)
        
        # Handle empty F
        if F.size == 0 or F.shape[0] == 0:
            return
        
        # Count valid CFs in population
        n_valid_pop = int(np.sum(F[:, 0] < self.validity_threshold))
        
        # Count valid CFs in archive (Pareto optimal)
        n_valid_archive = 0
        mean_sparsity = 0.0
        if algorithm.opt is not None:
            F_opt = algorithm.opt.get("F")
            if F_opt is not None and F_opt.size > 0:
                # Ensure F_opt is 2D
                if F_opt.ndim == 1:
                    F_opt = F_opt.reshape(1, -1)
                # Only process if we have at least one column
                if F_opt.shape[1] > 0:
                    n_valid_archive = int(np.sum(F_opt[:, 0] < self.validity_threshold))
                    # Mean sparsity of Pareto front (objective index 1 is sparsity)
                    mean_sparsity = float(F_opt[:, 1].mean()) if F_opt.shape[1] > 1 else 0.0
        
        # Best validity
        best_validity = float(F[:, 0].min())
        best_p_target = 1.0 - best_validity
        
        # Store history
        entry = ProgressEntry(
            gen=gen,
            n_valid_pop=n_valid_pop,
            n_valid_archive=n_valid_archive,
            best_validity=best_validity,
            best_p_target=best_p_target,
            mean_sparsity=mean_sparsity,
        )
        self.history.append(entry)
        
        # Print progress
        if self.verbose and (gen % self.print_every == 0 or gen == 1):
            print(
                f"Gen {gen:4d} | "
                f"Valid CFs (pop): {n_valid_pop:3d} | "
                f"Valid CFs (archive): {n_valid_archive:3d} | "
                f"Best P(target): {best_p_target:.3f} | "
                f"Mean Sparsity: {mean_sparsity:.3f}"
            )
    
    def get_summary(self) -> Dict:
        """Get summary of optimization progress."""
        if not self.history:
            return {}
        
        return {
            "total_generations": len(self.history),
            "final_valid_pop": self.history[-1].n_valid_pop,
            "final_valid_archive": self.history[-1].n_valid_archive,
            "best_p_target": self.history[-1].best_p_target,
            "final_mean_sparsity": self.history[-1].mean_sparsity,
            "history": self.history,
        }


class ValidCounterfactualTermination(Termination):
    """
    Terminate when enough valid counterfactuals are found.
    
    Conditions:
    - At least min_valid_cf solutions with P(target) > (1 - validity_threshold)
    - After minimum generations have passed
    - Or when maximum generations reached
    """
    
    def __init__(
        self,
        min_valid_cf: int = 10,
        validity_threshold: float = 0.5,
        min_gen: int = 20,
        max_gen: int = 200,
    ):
        """
        Args:
            min_valid_cf: Minimum number of valid CFs required
            validity_threshold: F[0] < threshold means valid
            min_gen: Minimum generations before allowing termination
            max_gen: Maximum generations (fallback)
        """
        super().__init__()
        self.min_valid_cf = min_valid_cf
        self.validity_threshold = validity_threshold
        self.min_gen = min_gen
        self.max_gen = max_gen
    
    def _update(self, algorithm) -> float:
        """
        Return progress as a float between 0 and 1.
        Returns >= 1.0 when termination criteria are met.
        """
        n_gen = algorithm.n_gen
        F = algorithm.pop.get("F")
        
        # Calculate progress based on generation count
        gen_progress = n_gen / self.max_gen
        
        if F is not None:
            n_valid = np.sum(F[:, 0] < self.validity_threshold)
            
            # Check if conditions met: enough valid CFs after min generations
            if n_gen >= self.min_gen and n_valid >= self.min_valid_cf:
                return 1.0  # Terminate
        
        # Fallback: max generations reached
        if n_gen >= self.max_gen:
            return 1.0  # Terminate
        
        return gen_progress  # Continue, report progress


class ConvergenceTermination(Termination):
    """
    Terminate when improvement in hypervolume stalls.
    
    Alternative termination based on convergence rather than
    counting valid solutions.
    """
    
    def __init__(
        self,
        patience: int = 50,
        min_improvement: float = 0.001,
        min_gen: int = 100,
        max_gen: int = 500,
    ):
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_gen = min_gen
        self.max_gen = max_gen
        self._best_validity = float('inf')
        self._no_improve_count = 0
    
    def _update(self, algorithm) -> float:
        """
        Return progress as a float between 0 and 1.
        Returns >= 1.0 when termination criteria are met.
        """
        n_gen = algorithm.n_gen
        F = algorithm.pop.get("F")
        
        # Calculate progress based on generation count
        gen_progress = n_gen / self.max_gen
        
        if F is not None:
            current_best = F[:, 0].min()
            
            if self._best_validity - current_best > self.min_improvement:
                self._best_validity = current_best
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            
            if (n_gen >= self.min_gen and 
                self._no_improve_count >= self.patience):
                return 1.0  # Terminate
        
        if n_gen >= self.max_gen:
            return 1.0  # Terminate
        
        return gen_progress  # Continue
    
    
def run_nsga(
    problem: Problem,
    config: NSGAConfig,
    sampling: Optional[FloatRandomSampling] = None,
) -> Result:
    """Run NSGA-II optimization on the given problem with specified configuration."""
    # Use default mutation probability if not specified (1/n_var is default for PM)
    mutation_prob = config.mutation_prob if config.mutation_prob is not None else (1.0 / problem.n_var)
    
    algorithm = NSGA2(
        pop_size=config.pop_size,
        sampling=sampling or FloatRandomSampling(),
        crossover=SBX(prob=config.crossover_prob, eta=int(config.crossover_eta)),
        mutation=PM(prob=mutation_prob, eta=int(config.mutation_eta)),
        eliminate_duplicates=True,
    )
    
    # Setup termination criteria
    if config.use_valid_cf_termination:
        termination = ValidCounterfactualTermination(
            min_valid_cf=config.min_valid_cf,
            validity_threshold=config.validity_threshold,
            min_gen=config.min_gen,
            max_gen=config.max_gen,
        )
    else:
        termination = get_termination("n_gen", config.n_gen)
    
    # Setup callback
    callback = ValidCFCallback(
        validity_threshold=config.validity_threshold,
        print_every=10,
        verbose=config.verbose,
    )
    
    # Run optimization
    # Note: verbose=False to avoid duplicate output (callback handles printing)
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=config.seed,
        callback=callback,
        verbose=False,
    )
    
    # Store callback history in the algorithm's callback for later retrieval
    # The callback is accessible via result.algorithm.callback
    
    return result


class MOCCallback(Callback):
    """
    Extended callback for MOC that applies feature reset and actionability
    enforcement after each generation.
    """
    
    def __init__(
        self,
        validity_threshold: float = 0.5,
        print_every: int = 10,
        verbose: bool = True,
        feature_reset_op: Optional[FeatureResetOperator] = None,
        actionability_enforcer: Optional[ActionabilityEnforcer] = None,
    ):
        super().__init__()
        self.validity_threshold = validity_threshold
        self.print_every = print_every
        self.verbose = verbose
        self.feature_reset_op = feature_reset_op
        self.actionability_enforcer = actionability_enforcer
        self.history: List[ProgressEntry] = []
    
    def notify(self, algorithm):
        """Called each generation - apply post-mutation operators and track progress."""
        gen = algorithm.n_gen
        
        # Apply feature reset and actionability enforcement to population
        if self.feature_reset_op is not None or self.actionability_enforcer is not None:
            X = algorithm.pop.get("X")
            if X is not None:
                if self.feature_reset_op is not None:
                    X = self.feature_reset_op(X)
                if self.actionability_enforcer is not None:
                    X = self.actionability_enforcer(X)
                algorithm.pop.set("X", X)
        
        # Track progress (same as ValidCFCallback)
        F = algorithm.pop.get("F")
        if F is None:
            return
        
        if F.ndim == 1:
            F = F.reshape(1, -1)
        if F.size == 0 or F.shape[0] == 0:
            return
        
        n_valid_pop = int(np.sum(F[:, 0] < self.validity_threshold))
        
        n_valid_archive = 0
        mean_sparsity = 0.0
        if algorithm.opt is not None:
            F_opt = algorithm.opt.get("F")
            if F_opt is not None and F_opt.size > 0:
                if F_opt.ndim == 1:
                    F_opt = F_opt.reshape(1, -1)
                if F_opt.shape[1] > 0:
                    n_valid_archive = int(np.sum(F_opt[:, 0] < self.validity_threshold))
                    mean_sparsity = float(F_opt[:, 1].mean()) if F_opt.shape[1] > 1 else 0.0
        
        best_validity = float(F[:, 0].min())
        best_p_target = 1.0 - best_validity
        
        entry = ProgressEntry(
            gen=gen,
            n_valid_pop=n_valid_pop,
            n_valid_archive=n_valid_archive,
            best_validity=best_validity,
            best_p_target=best_p_target,
            mean_sparsity=mean_sparsity,
        )
        self.history.append(entry)
        
        if self.verbose and (gen % self.print_every == 0 or gen == 1):
            print(
                f"Gen {gen:4d} | "
                f"Valid CFs (pop): {n_valid_pop:3d} | "
                f"Valid CFs (archive): {n_valid_archive:3d} | "
                f"Best P(target): {best_p_target:.3f} | "
                f"Mean Sparsity: {mean_sparsity:.3f}"
            )
    
    def get_summary(self) -> Dict:
        """Get summary of optimization progress."""
        if not self.history:
            return {}
        return {
            "total_generations": len(self.history),
            "final_valid_pop": self.history[-1].n_valid_pop,
            "final_valid_archive": self.history[-1].n_valid_archive,
            "best_p_target": self.history[-1].best_p_target,
            "final_mean_sparsity": self.history[-1].mean_sparsity,
            "history": self.history,
        }


def run_moc_nsga(
    problem: Problem,
    config: NSGAConfig,
    x_star: Array,
    X_obs: Array,
    model: Callable = None,
    device: Optional[torch.device] = None,
) -> Result:
    """
    Run NSGA-II with full MOC (Multi-Objective Counterfactuals) framework.
    
    Implements all MOC modifications from Dandl et al.:
    1. ICE Curve Variance Initialization - biases initial population towards influential features
    2. MIES (Mixed Integer Evolutionary Strategies) - handles mixed discrete/continuous features
    3. Conditional Mutator with Transformation Trees - plausible on-manifold mutations
    4. Feature Reset to x* - enforces sparsity by resetting features with low probability
    5. Modified Crowding Distance - diversity in feature + objective space (Gower + L1)
    6. Penalization - constraint handling for validity via front reassignment
    7. Actionability - caps extreme values and fixes non-actionable features
    
    Args:
        problem: pymoo Problem for counterfactual optimization
        config: NSGAConfig with MOC parameters (including feature_config for MIES)
        x_star: Factual instance, shape (d,) - can be tensor or numpy
        X_obs: Observed data, shape (N, d) - can be tensor or numpy
        model: Model for ICE curve computation (required if use_ice_initialization=True)
        device: PyTorch device
        
    Returns:
        pymoo Result object
    """
    # Convert to numpy for pymoo (optimization operates on numpy)
    x_star_np = to_numpy(x_star).flatten()
    X_obs_np = to_numpy(X_obs).reshape(-1, x_star_np.shape[0])
    n_var = problem.n_var
    
    # Get feature configuration
    feature_config = config.feature_config or FeatureTypeConfig()
    feature_types = feature_config.feature_types or ['continuous'] * n_var
    fixed_features = feature_config.fixed_features or []
    
    if config.verbose:
        print("=" * 60)
        print("[MOC] Multi-Objective Counterfactual Framework")
        print("=" * 60)
    
    # ==========================================================================
    # MOC Adjustment 1: ICE Curve Variance Initialization
    # ==========================================================================
    if config.use_ice_initialization and model is not None:
        sampling = ICEVarianceSampling(
            x_star=x_star,
            X_obs=X_obs,
            model=model,
            p_min=config.ice_p_min,
            p_max=config.ice_p_max,
            n_samples=config.ice_n_samples,
            device=device,
        )
        if config.verbose:
            print(f"[MOC] ICE Variance Initialization enabled")
            print(f"      Feature change probabilities: min={sampling.p_change.min():.3f}, "
                  f"max={sampling.p_change.max():.3f}, mean={sampling.p_change.mean():.3f}")
    else:
        sampling = FactualBasedSampling(x_star=x_star_np, noise_scale=0.2)
        if config.verbose:
            print(f"[MOC] Using Factual-Based Sampling (ICE disabled or no model)")
    
    # ==========================================================================
    # MOC Adjustment 2: MIES Crossover (Mixed Types)
    # ==========================================================================
    if config.use_mies and any(ft != 'continuous' for ft in feature_types):
        crossover = MIESCrossover(
            feature_types=feature_types,
            prob=config.crossover_prob,
            eta=config.crossover_eta,
        )
        if config.verbose:
            n_cont = sum(1 for ft in feature_types if ft == 'continuous')
            n_cat = sum(1 for ft in feature_types if ft == 'categorical')
            n_bin = sum(1 for ft in feature_types if ft == 'binary')
            print(f"[MOC] MIES Crossover enabled")
            print(f"      Features: {n_cont} continuous, {n_cat} categorical, {n_bin} binary")
    else:
        crossover = SBX(prob=config.crossover_prob, eta=int(config.crossover_eta))
        if config.verbose:
            print(f"[MOC] Using SBX Crossover (all continuous features)")
    
    # ==========================================================================
    # MOC Adjustment 3: Conditional Mutator with MIES support
    # ==========================================================================
    mutation_prob = config.mutation_prob if config.mutation_prob is not None else (1.0 / n_var)
    
    if config.use_conditional_mutator:
        transformation_trees = TransformationTreeMutator(X_obs_np)
        
        if config.use_mies and any(ft != 'continuous' for ft in feature_types):
            # Combined MIES + Conditional mutation
            mutation = MIESConditionalMutation(
                feature_types=feature_types,
                transformation_trees=transformation_trees,
                categorical_levels=feature_config.categorical_levels,
                prob=mutation_prob,
                use_conditional_prob=0.7,
            )
            if config.verbose:
                print(f"[MOC] MIES + Conditional Mutator enabled (70% conditional)")
        else:
            mutation = ConditionalMutation(
                transformation_trees=transformation_trees,
                prob=mutation_prob,
                eta=config.mutation_eta,
                use_conditional_prob=0.7,
            )
            if config.verbose:
                print(f"[MOC] Conditional Mutator enabled (70% conditional, 30% polynomial)")
    elif config.use_mies and any(ft != 'continuous' for ft in feature_types):
        mutation = MIESMutation(
            feature_types=feature_types,
            categorical_levels=feature_config.categorical_levels,
            prob=mutation_prob,
        )
        if config.verbose:
            print(f"[MOC] MIES Mutator enabled (no conditional)")
    else:
        mutation = PM(prob=mutation_prob, eta=int(config.mutation_eta))
        if config.verbose:
            print(f"[MOC] Using standard Polynomial Mutation")
    
    # ==========================================================================
    # MOC Adjustment 4: Feature Reset to x* (Sparsity Enforcement)
    # ==========================================================================
    feature_reset_op = None
    if config.use_feature_reset:
        feature_reset_op = FeatureResetOperator(
            x_star=x_star_np,
            reset_prob=config.feature_reset_prob,
            fixed_features=fixed_features,
        )
        if config.verbose:
            print(f"[MOC] Feature Reset enabled (prob={config.feature_reset_prob:.2f})")
            if fixed_features:
                print(f"      Non-actionable features: {fixed_features}")
    
    # ==========================================================================
    # MOC Adjustment 5: Actionability Enforcement
    # ==========================================================================
    actionability_enforcer = None
    if fixed_features or feature_config.actionable_bounds is not None:
        # Derive actionable bounds from X_obs if not specified
        actionable_bounds = feature_config.actionable_bounds
        if actionable_bounds is None:
            actionable_bounds = (X_obs_np.min(axis=0), X_obs_np.max(axis=0))
        
        actionability_enforcer = ActionabilityEnforcer(
            x_star=x_star_np,
            fixed_features=fixed_features,
            actionable_bounds=actionable_bounds,
        )
        if config.verbose:
            print(f"[MOC] Actionability Enforcer enabled")
            print(f"      Bounds derived from observed data range")
    
    # ==========================================================================
    # MOC Adjustment 6 & 7: Modified Crowding Distance + Penalization
    # ==========================================================================
    if config.use_modified_crowding or config.use_penalization:
        survival = PenalizedRankAndCrowdingSurvival(
            validity_epsilon=config.validity_epsilon,
            weight_obj=config.cd_objective_weight,
            weight_feature=config.cd_feature_weight,
        )
        feature_range = problem.xu - problem.xl
        feature_range[feature_range == 0] = 1.0
        survival.set_feature_range(feature_range)
        
        if config.verbose:
            if config.use_modified_crowding:
                print(f"[MOC] Modified Crowding Distance enabled "
                      f"(obj={config.cd_objective_weight:.1f}, feature={config.cd_feature_weight:.1f})")
            if config.use_penalization:
                print(f"[MOC] Penalization enabled (epsilon={config.validity_epsilon:.2f})")
    else:
        survival = RankAndCrowdingSurvival()
        if config.verbose:
            print(f"[MOC] Using standard Rank and Crowding Survival")
    
    # ==========================================================================
    # Create NSGA-II Algorithm with Full MOC Components
    # ==========================================================================
    algorithm = NSGA2(
        pop_size=config.pop_size,
        sampling=sampling,
        crossover=crossover,
        mutation=mutation,
        survival=survival,
        eliminate_duplicates=True,
    )
    
    # Setup termination criteria
    if config.use_valid_cf_termination:
        termination = ValidCounterfactualTermination(
            min_valid_cf=config.min_valid_cf,
            validity_threshold=config.validity_threshold,
            min_gen=config.min_gen,
            max_gen=config.max_gen,
        )
    else:
        termination = get_termination("n_gen", config.n_gen)
    
    # Setup MOC callback with feature reset and actionability
    callback = MOCCallback(
        validity_threshold=config.validity_threshold,
        print_every=10,
        verbose=config.verbose,
        feature_reset_op=feature_reset_op,
        actionability_enforcer=actionability_enforcer,
    )
    
    if config.verbose:
        print(f"\n[MOC] Starting NSGA-II optimization...")
        print(f"      Population: {config.pop_size}, Max generations: {config.max_gen}")
        print(f"      Features: {n_var}, Objectives: {problem.n_obj}")
        print("-" * 60)
    
    # Run optimization
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=config.seed,
        callback=callback,
        verbose=False,
    )
    
    return result