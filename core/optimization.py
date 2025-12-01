"""
NSGA-II optimization utilities for counterfactual generation.

Provides custom sampling, callbacks, and termination criteria for
multi-objective counterfactual optimization.

Includes MOC-style modifications:
- Modified crowding distance (objective space L1 + feature space Gower)
- Conditional Mutator (samples from conditional distributions using transformation trees)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.core.termination import Termination
from pymoo.core.survival import Survival
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import torch


Array = Union[np.ndarray, torch.Tensor]


# =============================================================================
# Conditional Mutator using Transformation Trees
# =============================================================================

class TransformationTree:
    """
    A transformation tree for conditional sampling of feature values.
    
    Uses a Decision Tree Regressor to model the conditional distribution
    P(x_j | x_{-j}) for numerical features, enabling plausible mutations.
    """
    
    def __init__(self, feature_idx: int, max_depth: int = 5, min_samples_leaf: int = 10):
        """
        Args:
            feature_idx: Index of the feature this tree predicts
            max_depth: Maximum depth of the decision tree
            min_samples_leaf: Minimum samples per leaf for variance estimation
        """
        self.feature_idx = feature_idx
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.leaf_stats: Dict[int, Tuple[float, float]] = {}  # leaf_id -> (mean, std)
        self._is_fitted = False
    
    def fit(self, X_obs: np.ndarray) -> 'TransformationTree':
        """
        Train the transformation tree on observed data.
        
        Args:
            X_obs: Observed training data, shape (n_samples, n_features)
            
        Returns:
            self
        """
        from sklearn.tree import DecisionTreeRegressor
        
        n_samples, n_features = X_obs.shape
        
        # Target: the feature we want to predict
        y = X_obs[:, self.feature_idx]
        
        # Features: all other features (exclude the target feature)
        feature_mask = np.ones(n_features, dtype=bool)
        feature_mask[self.feature_idx] = False
        X = X_obs[:, feature_mask]
        
        # Train decision tree
        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        self.tree.fit(X, y)
        
        # Compute statistics for each leaf node
        leaf_ids = self.tree.apply(X)
        unique_leaves = np.unique(leaf_ids)
        
        for leaf_id in unique_leaves:
            mask = leaf_ids == leaf_id
            leaf_values = y[mask]
            self.leaf_stats[leaf_id] = (
                float(np.mean(leaf_values)),
                float(np.std(leaf_values)) + 1e-6  # Add small epsilon for stability
            )
        
        self._is_fitted = True
        return self
    
    def sample(self, x: np.ndarray, rng: np.random.Generator) -> float:
        """
        Sample a plausible value for this feature given other features.
        
        Args:
            x: Current feature vector, shape (n_features,)
            rng: Random number generator
            
        Returns:
            Sampled feature value from conditional distribution
        """
        if not self._is_fitted:
            raise RuntimeError("TransformationTree must be fitted before sampling")
        
        # Prepare input (exclude the target feature)
        n_features = len(x)
        feature_mask = np.ones(n_features, dtype=bool)
        feature_mask[self.feature_idx] = False
        x_input = x[feature_mask].reshape(1, -1)
        
        # Find the leaf node
        leaf_id = self.tree.apply(x_input)[0]
        
        # Get statistics for this leaf
        if leaf_id in self.leaf_stats:
            mean, std = self.leaf_stats[leaf_id]
        else:
            # Fallback: use tree prediction with small noise
            mean = self.tree.predict(x_input)[0]
            std = 0.1
        
        # Sample from Gaussian approximation of conditional distribution
        sampled_value = rng.normal(mean, std)
        
        return sampled_value


class ConditionalMutator(Mutation):
    """
    MOC-style Conditional Mutator for plausible counterfactual generation.
    
    Instead of random mutations, this operator samples feature values from
    their conditional distributions P(x_j | x_{-j}), learned via transformation
    trees trained on observed data. This produces more plausible mutations
    that respect the data distribution.
    
    Key features:
    - Trains a transformation tree for each feature on X_obs
    - Mutates features in randomized order (since mutations are dependent)
    - Samples new values from conditional distributions
    
    Reference: Dandl et al. "Multi-Objective Counterfactual Explanations"
    """
    
    def __init__(self,
                 X_obs: np.ndarray,
                 prob: float = 0.2,
                 max_depth: int = 5,
                 min_samples_leaf: int = 10,
                 fallback_std: float = 0.1):
        """
        Args:
            X_obs: Observed training data for learning conditionals, shape (n_samples, n_features)
            prob: Probability of mutating each feature
            max_depth: Maximum depth of transformation trees
            min_samples_leaf: Minimum samples per leaf in trees
            fallback_std: Standard deviation for fallback random mutation
        """
        super().__init__()
        self.X_obs = np.asarray(X_obs)
        self.prob = prob
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.fallback_std = fallback_std
        
        # Build transformation trees for each feature
        self.n_features = self.X_obs.shape[1]
        self.trees: List[TransformationTree] = []
        self._build_trees()
    
    def _build_trees(self):
        """Train a transformation tree for each feature."""
        for j in range(self.n_features):
            tree = TransformationTree(
                feature_idx=j,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf
            )
            tree.fit(self.X_obs)
            self.trees.append(tree)
    
    def _do(self, problem, X, **kwargs) -> np.ndarray:
        """
        Perform conditional mutation on the population.
        
        Args:
            problem: The optimization problem
            X: Decision variables of offspring, shape (n_offspring, n_var)
            
        Returns:
            Mutated decision variables
        """
        n_offspring, n_var = X.shape
        X_mutated = X.copy()
        
        # Get bounds
        xl, xu = problem.xl, problem.xu
        
        # Random generator
        rng = np.random.default_rng()
        
        for i in range(n_offspring):
            # Randomize feature order for this individual
            # (important since mutations are conditionally dependent)
            feature_order = rng.permutation(n_var)
            
            for j in feature_order:
                # Decide whether to mutate this feature
                if rng.random() < self.prob:
                    # Sample from conditional distribution
                    if j < len(self.trees):
                        new_value = self.trees[j].sample(X_mutated[i], rng)
                    else:
                        # Fallback for features beyond trained trees
                        current_value = X_mutated[i, j]
                        feature_range = xu[j] - xl[j]
                        new_value = current_value + rng.normal(0, self.fallback_std * feature_range)
                    
                    # Clip to bounds
                    X_mutated[i, j] = np.clip(new_value, xl[j], xu[j])
        
        return X_mutated


class HybridMutator(Mutation):
    """
    Hybrid mutation combining Conditional Mutator with Polynomial Mutation.
    
    With probability `conditional_prob`, uses the ConditionalMutator for
    plausible mutations. Otherwise, falls back to standard PM for exploration.
    """
    
    def __init__(self,
                 X_obs: np.ndarray,
                 conditional_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 pm_eta: float = 20,
                 max_depth: int = 5,
                 min_samples_leaf: int = 10):
        """
        Args:
            X_obs: Observed training data
            conditional_prob: Probability of using conditional mutation vs PM
            mutation_prob: Per-feature mutation probability
            pm_eta: Distribution index for polynomial mutation
            max_depth: Max depth for transformation trees
            min_samples_leaf: Min samples per leaf in trees
        """
        super().__init__()
        self.conditional_prob = conditional_prob
        
        # Initialize both mutators
        self.conditional_mutator = ConditionalMutator(
            X_obs=X_obs,
            prob=mutation_prob,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        self.pm_mutator = PM(prob=mutation_prob, eta=pm_eta)
    
    def _do(self, problem, X, **kwargs) -> np.ndarray:
        """Apply hybrid mutation."""
        n_offspring = X.shape[0]
        X_mutated = X.copy()
        
        rng = np.random.default_rng()
        
        for i in range(n_offspring):
            if rng.random() < self.conditional_prob:
                # Use conditional mutation for this individual
                X_mutated[i:i+1] = self.conditional_mutator._do(
                    problem, X_mutated[i:i+1], **kwargs
                )
            else:
                # Use polynomial mutation for exploration
                X_mutated[i:i+1] = self.pm_mutator._do(
                    problem, X_mutated[i:i+1], **kwargs
                )
        
        return X_mutated


# =============================================================================
# Gower Distance for Feature Space Diversity
# =============================================================================
# Note: Reuses _gower_distance from cf_problem.py for numerical features.
# This module extends it with categorical feature support for crowding distance.

from core.cf_problem import _gower_distance


def gower_distance_mixed(x1: np.ndarray, x2: np.ndarray, 
                         feature_ranges: np.ndarray,
                         feature_types: Optional[np.ndarray] = None) -> float:
    """
    Compute Gower distance between two feature vectors with mixed types.
    
    Extends the numerical-only Gower distance with categorical support:
    - Numerical: normalized by feature range |x_j - y_j| / R_j
    - Categorical/Binary: indicator function I(x_j != y_j)
    
    Args:
        x1, x2: Feature vectors of shape (d,)
        feature_ranges: Range of each numerical feature (max - min), shape (d,)
        feature_types: Array indicating feature type per dimension.
                      'numerical' for continuous, 'categorical' for discrete.
                      If None, all features are treated as numerical (uses existing impl).
    
    Returns:
        Gower distance in [0, 1]
    """
    if feature_types is None:
        # All numerical - use existing implementation
        per_feature = _gower_distance(x1, x2, feature_ranges)
        return float(np.mean(per_feature))
    
    d = len(x1)
    distances = np.zeros(d)
    
    for j in range(d):
        if feature_types[j] == 'numerical':
            # Numerical: normalize by range
            if feature_ranges[j] > 0:
                distances[j] = np.abs(x1[j] - x2[j]) / feature_ranges[j]
            else:
                distances[j] = 0.0
        else:
            # Categorical/Binary: indicator function
            distances[j] = 1.0 if x1[j] != x2[j] else 0.0
    
    return float(np.mean(distances))


def compute_gower_crowding_distance(X: np.ndarray, 
                                    feature_ranges: np.ndarray,
                                    feature_types: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute crowding distance in feature space using Gower distance.
    
    For each individual, compute the sum of Gower distances to its
    nearest neighbors (similar to standard crowding distance but in feature space).
    
    Args:
        X: Population decision variables, shape (n_pop, n_var)
        feature_ranges: Range of each feature, shape (n_var,)
        feature_types: Feature type indicators
        
    Returns:
        Feature space crowding distances, shape (n_pop,)
    """
    n_pop = X.shape[0]
    
    if n_pop <= 2:
        return np.full(n_pop, np.inf)
    
    # Compute pairwise Gower distances
    gower_matrix = np.zeros((n_pop, n_pop))
    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            dist = gower_distance_mixed(X[i], X[j], feature_ranges, feature_types)
            gower_matrix[i, j] = dist
            gower_matrix[j, i] = dist
    
    # For crowding: sum of distances to two nearest neighbors
    crowding = np.zeros(n_pop)
    for i in range(n_pop):
        # Get distances to all others (excluding self)
        dists = gower_matrix[i].copy()
        dists[i] = np.inf  # Exclude self
        
        # Find two nearest neighbors
        sorted_idx = np.argsort(dists)
        if n_pop > 2:
            crowding[i] = dists[sorted_idx[0]] + dists[sorted_idx[1]]
        else:
            crowding[i] = dists[sorted_idx[0]]
    
    return crowding


def compute_objective_crowding_distance_l1(F: np.ndarray) -> np.ndarray:
    """
    Compute crowding distance in objective space using L1 norm.
    
    Standard crowding distance but using L1 (Manhattan) distance
    instead of the standard per-objective boundary distance.
    
    Args:
        F: Objective values, shape (n_pop, n_obj)
        
    Returns:
        Objective space crowding distances, shape (n_pop,)
    """
    n_pop, n_obj = F.shape
    
    if n_pop <= 2:
        return np.full(n_pop, np.inf)
    
    # Normalize objectives to [0, 1] for fair comparison
    f_min = F.min(axis=0)
    f_max = F.max(axis=0)
    f_range = f_max - f_min
    f_range[f_range == 0] = 1.0  # Avoid division by zero
    
    F_norm = (F - f_min) / f_range
    
    # Compute pairwise L1 distances
    l1_matrix = np.zeros((n_pop, n_pop))
    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            dist = np.sum(np.abs(F_norm[i] - F_norm[j]))  # L1 norm
            l1_matrix[i, j] = dist
            l1_matrix[j, i] = dist
    
    # For crowding: sum of distances to two nearest neighbors
    crowding = np.zeros(n_pop)
    for i in range(n_pop):
        dists = l1_matrix[i].copy()
        dists[i] = np.inf  # Exclude self
        
        sorted_idx = np.argsort(dists)
        if n_pop > 2:
            crowding[i] = dists[sorted_idx[0]] + dists[sorted_idx[1]]
        else:
            crowding[i] = dists[sorted_idx[0]]
    
    return crowding


def compute_moc_crowding_distance(X: np.ndarray, 
                                   F: np.ndarray,
                                   feature_ranges: np.ndarray,
                                   feature_types: Optional[np.ndarray] = None,
                                   alpha: float = 0.5) -> np.ndarray:
    """
    Compute MOC-style modified crowding distance.
    
    Combines objective space and feature space distances:
    CD_MOC = alpha * CD_objective + (1 - alpha) * CD_feature
    
    This encourages diversity in both objective space (different trade-offs)
    and feature space (different actionable changes).
    
    Args:
        X: Decision variables (features), shape (n_pop, n_var)
        F: Objective values, shape (n_pop, n_obj)
        feature_ranges: Range of each feature for Gower normalization
        feature_types: Feature type indicators ('numerical' or 'categorical')
        alpha: Weight for objective space distance (default 0.5 = equal weighting)
        
    Returns:
        Combined crowding distances, shape (n_pop,)
    """
    # Compute crowding in objective space (L1 norm)
    cd_objective = compute_objective_crowding_distance_l1(F)
    
    # Compute crowding in feature space (Gower distance)
    cd_feature = compute_gower_crowding_distance(X, feature_ranges, feature_types)
    
    # Normalize both to [0, 1] for fair combination
    # Handle infinity values
    cd_obj_finite = cd_objective[np.isfinite(cd_objective)]
    cd_feat_finite = cd_feature[np.isfinite(cd_feature)]
    
    if len(cd_obj_finite) > 0:
        cd_obj_max = cd_obj_finite.max() if cd_obj_finite.max() > 0 else 1.0
        cd_objective_norm = np.where(np.isfinite(cd_objective), 
                                      cd_objective / cd_obj_max, 
                                      np.inf)
    else:
        cd_objective_norm = cd_objective
    
    if len(cd_feat_finite) > 0:
        cd_feat_max = cd_feat_finite.max() if cd_feat_finite.max() > 0 else 1.0
        cd_feature_norm = np.where(np.isfinite(cd_feature), 
                                    cd_feature / cd_feat_max, 
                                    np.inf)
    else:
        cd_feature_norm = cd_feature
    
    # Combine with equal weighting (alpha = 0.5)
    combined = alpha * cd_objective_norm + (1 - alpha) * cd_feature_norm
    
    return combined


# =============================================================================
# Custom Survival for MOC Crowding Distance
# =============================================================================

class MOCRankAndCrowdingSurvival(Survival):
    """
    Survival operator using MOC-style modified crowding distance.
    
    Implements the Avila et al. approach where crowding distance
    combines both objective space (L1) and feature space (Gower) distances.
    This promotes diversity in both objectives and actionable feature changes.
    """
    
    def __init__(self, 
                 feature_ranges: np.ndarray,
                 feature_types: Optional[np.ndarray] = None,
                 alpha: float = 0.5,
                 nds: Optional[NonDominatedSorting] = None):
        """
        Args:
            feature_ranges: Range of each feature (max - min) for Gower distance
            feature_types: Array of 'numerical' or 'categorical' for each feature
            alpha: Weight for objective space distance (0.5 = equal weighting)
            nds: Non-dominated sorting instance

        Returns:
            
        """
        super().__init__(filter_infeasible=True)
        self.feature_ranges = feature_ranges
        self.feature_types = feature_types
        self.alpha = alpha
        self.nds = nds if nds is not None else NonDominatedSorting()
    
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        """Perform survival selection with MOC crowding distance."""
        
        if n_survive is None:
            n_survive = len(pop)
        
        # Get objective values and decision variables
        F = pop.get("F").astype(float, copy=False)
        X = pop.get("X").astype(float, copy=False)
        
        # Non-dominated sorting
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        
        # Initialize crowding distance for all individuals
        crowding = np.full(len(pop), np.nan)
        
        # Assign ranks
        ranks = np.full(len(pop), -1)
        
        survivors = []
        
        for k, front in enumerate(fronts):
            # Get individuals in this front
            I = np.array(front)
            
            # Assign rank
            ranks[I] = k
            
            # Compute MOC crowding distance for this front
            F_front = F[I]
            X_front = X[I]
            
            cd = compute_moc_crowding_distance(
                X_front, F_front, 
                self.feature_ranges,
                self.feature_types,
                self.alpha
            )
            
            # Store crowding distance for individuals in this front
            crowding[I] = cd
            
            # Check if adding this front exceeds capacity
            if len(survivors) + len(I) <= n_survive:
                # Add entire front
                survivors.extend(I.tolist())
            else:
                # Need to select subset based on crowding distance
                n_remaining = n_survive - len(survivors)
                
                if n_remaining > 0:
                    # Select individuals with largest crowding distance
                    sorted_by_cd = np.argsort(-cd)  # Descending order
                    selected = I[sorted_by_cd[:n_remaining]]
                    survivors.extend(selected.tolist())
                
                break
        
        # Set attributes on population before subsetting
        pop.set("crowding", crowding)
        pop.set("rank", ranks)
        
        # Return survivors - crowding values are preserved
        return pop[survivors]


class MOCNSGA2(NSGA2):
    """
    NSGA-II with MOC-style modified crowding distance.
    
    This variant uses a combined crowding distance that considers both:
    1. Objective space diversity (L1 norm between objectives)
    2. Feature space diversity (Gower distance between decision variables)
    
    This approach, based on Avila et al., is particularly suited for
    counterfactual generation where we want diverse feature changes,
    not just diverse objective trade-offs.
    """
    
    def __init__(self,
                 feature_ranges: np.ndarray,
                 feature_types: Optional[np.ndarray] = None,
                 crowding_alpha: float = 0.5,
                 **kwargs):
        """
        Args:
            feature_ranges: Range of each feature (xu - xl) for Gower normalization
            feature_types: Feature type indicators ('numerical'/'categorical')
            crowding_alpha: Weight for objective space (0.5 = equal weighting)
            **kwargs: Additional NSGA2 arguments
        """
        # Create custom survival operator
        survival = MOCRankAndCrowdingSurvival(
            feature_ranges=feature_ranges,
            feature_types=feature_types,
            alpha=crowding_alpha
        )
        
        # Initialize parent NSGA2 with custom survival
        super().__init__(survival=survival, **kwargs)
        
        self.feature_ranges = feature_ranges
        self.feature_types = feature_types
        self.crowding_alpha = crowding_alpha


@dataclass
class NSGAConfig:
    """Configuration for NSGA-II optimization."""
    
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
    
    # MOC-style crowding distance parameters
    use_moc_crowding: bool = True  # Use MOC modified crowding distance
    crowding_alpha: float = 0.5   # Weight for objective space (0.5 = equal weighting)
    feature_types: Optional[np.ndarray] = None  # 'numerical' or 'categorical' per feature
    
    # Conditional Mutator parameters (MOC-style)
    use_conditional_mutator: bool = True  # Use conditional mutator for plausible mutations
    conditional_mutator_prob: float = 0.7  # Probability of using conditional vs PM mutation
    tree_max_depth: int = 5  # Max depth of transformation trees
    tree_min_samples_leaf: int = 10  # Min samples per leaf in transformation trees

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
        self._last_printed_gen = -1  # Track last printed generation to avoid duplicates
    
    def notify(self, algorithm):
        """Called each generation."""
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")
        
        if F is None or F.size == 0:
            return
        
        # Ensure F is 2D
        if F.ndim == 1:
            F = F.reshape(1, -1)
        
        # Check if F has valid shape
        if F.shape[0] == 0 or F.shape[1] == 0:
            return
        
        # Count valid CFs in population
        n_valid_pop = int(np.sum(F[:, 0] < self.validity_threshold))
        
        # Count valid CFs in archive (Pareto optimal solutions)
        # Use non-dominated sorting on current population to find Pareto front
        n_valid_archive = 0
        mean_sparsity = 0.0
        
        try:
            from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
            nds = NonDominatedSorting()
            fronts = nds.do(F)
            if len(fronts) > 0 and len(fronts[0]) > 0:
                # Get first Pareto front
                pareto_indices = fronts[0]
                F_pareto = F[pareto_indices]
                n_valid_archive = int(np.sum(F_pareto[:, 0] < self.validity_threshold))
                # Mean sparsity of Pareto front (objective index 2 is sparsity in MOC)
                if F_pareto.shape[1] > 2:
                    mean_sparsity = float(F_pareto[:, 2].mean())
                elif F_pareto.shape[1] > 1:
                    mean_sparsity = float(F_pareto[:, 1].mean())
        except Exception:
            pass
        
        # Best validity
        best_validity = float(F[:, 0].min())
        best_p_target = 1.0 - best_validity
        
        # Store history (only once per generation)
        if len(self.history) == 0 or self.history[-1].gen != gen:
            entry = ProgressEntry(
                gen=gen,
                n_valid_pop=n_valid_pop,
                n_valid_archive=n_valid_archive,
                best_validity=best_validity,
                best_p_target=best_p_target,
                mean_sparsity=mean_sparsity,
            )
            self.history.append(entry)
        
        # Print progress (avoid duplicates)
        should_print = (gen % self.print_every == 0 or gen == 1) and gen != self._last_printed_gen
        if self.verbose and should_print:
            self._last_printed_gen = gen
            pareto_size = len(fronts[0]) if (fronts and len(fronts) > 0) else 0
            print(
                f"Gen {gen:4d} | "
                f"Valid CFs (pop): {n_valid_pop:3d} | "
                f"Pareto front: {pareto_size:3d} | "
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
            # Ensure F is 2D
            if F.ndim == 1:
                F = F.reshape(1, -1)
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
            # Ensure F is 2D
            if F.ndim == 1:
                F = F.reshape(1, -1)
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
    X_obs: Optional[np.ndarray] = None,
) -> Result:
    """
    Run NSGA-II optimization on the given problem with specified configuration.
    
    Supports MOC-style modifications:
    - Modified crowding distance (objective space + feature space diversity)
    - Conditional Mutator (plausible mutations via transformation trees)
    
    Args:
        problem: The optimization problem
        config: NSGA-II configuration
        sampling: Optional custom sampling strategy
        X_obs: Observed training data for conditional mutator, shape (n_samples, n_features).
               Required if config.use_conditional_mutator is True.
               
    Returns:
        Optimization result
    """
    # Use default mutation probability if not specified (1/n_var is default for PM)
    mutation_prob = config.mutation_prob if config.mutation_prob is not None else (1.0 / problem.n_var)
    
    # Compute feature ranges from problem bounds
    feature_ranges = problem.xu - problem.xl
    feature_ranges[feature_ranges == 0] = 1.0  # Avoid division by zero
    
    # Setup mutation operator
    if config.use_conditional_mutator and X_obs is not None:
        # Use MOC-style Conditional Mutator (hybrid with PM)
        mutation = HybridMutator(
            X_obs=X_obs,
            conditional_prob=config.conditional_mutator_prob,
            mutation_prob=mutation_prob,
            pm_eta=int(config.mutation_eta),
            max_depth=config.tree_max_depth,
            min_samples_leaf=config.tree_min_samples_leaf
        )
    else:
        # Use standard Polynomial Mutation
        mutation = PM(prob=mutation_prob, eta=int(config.mutation_eta))
    
    if config.use_moc_crowding:
        # Use MOC-style NSGA-II with modified crowding distance
        algorithm = MOCNSGA2(
            pop_size=config.pop_size,
            feature_ranges=feature_ranges,
            feature_types=config.feature_types,
            crowding_alpha=config.crowding_alpha,
            sampling=sampling or FloatRandomSampling(),
            crossover=SBX(prob=config.crossover_prob, eta=int(config.crossover_eta)),
            mutation=mutation,
            eliminate_duplicates=True,
        )
    else:
        # Use standard NSGA-II
        algorithm = NSGA2(
            pop_size=config.pop_size,
            sampling=sampling or FloatRandomSampling(),
            crossover=SBX(prob=config.crossover_prob, eta=int(config.crossover_eta)),
            mutation=mutation,
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