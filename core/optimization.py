"""
NSGA-II optimization utilities for counterfactual generation.

Provides custom sampling, callbacks, and termination criteria for
multi-objective counterfactual optimization.

Includes MOC-style modifications:
- Modified crowding distance (objective space L1 + feature space Gower)
- Conditional Mutator (samples from conditional distributions using transformation trees)

All runtime computations use torch tensors on GPU for efficiency.
Numpy is only used at pymoo interface boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
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
        
        Note: sklearn DecisionTree requires numpy, so we keep numpy for tree operations.
        The transformation tree training is done once at init, not per-generation.
        
        Args:
            problem: The optimization problem
            X: Decision variables of offspring, shape (n_offspring, n_var)
            
        Returns:
            Mutated decision variables (numpy array for pymoo)
        """
        n_offspring, n_var = X.shape
        X_mutated = X.copy()
        
        # Get bounds as numpy (problem.xl/xu are numpy arrays)
        xl, xu = problem.xl, problem.xu
        
        # Random generator
        rng = np.random.default_rng()
        
        for i in range(n_offspring):
            # Randomize feature order for this individual
            feature_order = rng.permutation(n_var)
            
            for j in feature_order:
                if rng.random() < self.prob:
                    if j < len(self.trees):
                        new_value = self.trees[j].sample(X_mutated[i], rng)
                    else:
                        current_value = X_mutated[i, j]
                        feature_range = xu[j] - xl[j]
                        new_value = current_value + rng.normal(0, self.fallback_std * feature_range)
                    
                    X_mutated[i, j] = np.clip(new_value, xl[j], xu[j])
        
        return X_mutated


class ResetOperator(Mutation):
    """
    Reset Operator for counterfactual generation base on paper MOC.
    
    After recombination and mutation, this operator randomly resets some feature
    values back to the factual instance x* with a given probability p_reset.
    This prevents all features from drifting away from x* simultaneously,
    encouraging sparser counterfactuals.
    
    Mathematical formulation:
    For each feature j independently, with probability p_reset:
        x_j^new = x_j^*  (reset to factual)
    Otherwise:
        x_j^new = x̃_j    (keep mutated value)
    
    In vector form with Bernoulli mask B ~ Bernoulli(p_reset)^p:
        x_new = B ⊙ x* + (1 - B) ⊙ x̃
    
    Expected effect on sparsity objective:
        E[||x_new - x*||_0 | x̃] = (1 - p_reset) * ||x̃ - x*||_0
    
    Reference: Dandl et al. "Multi-Objective Counterfactual Explanations"
    """
    
    def __init__(self, x_factual, p_reset: float = 0.05, device: Optional[torch.device] = None):
        """
        Args:
            x_factual: The factual instance x*, can be numpy array or torch tensor
            p_reset: Probability of resetting each feature to its factual value.
                     Should be small (e.g., 0.01-0.1) to avoid too much attraction.
            device: Torch device for GPU computation
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Store factual as torch tensor on device
        if isinstance(x_factual, torch.Tensor):
            self.x_factual = x_factual.flatten().to(self.device)
        else:
            self.x_factual = torch.from_numpy(np.asarray(x_factual).flatten().astype(np.float32)).to(self.device)
        
        self.p_reset = p_reset
    
    def _do(self, problem, X, **kwargs) -> np.ndarray:
        """
        Apply reset operator using GPU batch computation.
        
        Args:
            problem: The optimization problem
            X: Decision variables of offspring, shape (n_offspring, n_var)
            
        Returns:
            Decision variables with some features reset to factual values (numpy for pymoo)
        """
        # Convert to GPU tensor for batch operation
        X_t = torch.from_numpy(X.astype(np.float32)).to(self.device)
        n_offspring, n_var = X_t.shape
        
        # Generate Bernoulli mask on GPU
        B = torch.rand(n_offspring, n_var, device=self.device) < self.p_reset
        
        # Apply reset: x_new = B ⊙ x* + (1 - B) ⊙ x̃
        X_reset = torch.where(B, self.x_factual.unsqueeze(0).expand(n_offspring, -1), X_t)
        
        # Return as numpy for pymoo
        return X_reset.cpu().numpy()


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
        
        # TODO use batch processing on GPU for efficiency, implement ideal: using mask
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
# Gower Distance for Feature Space Diversity (Torch-based)
# =============================================================================

def _get_device() -> torch.device:
    """Get default device for torch operations."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def gower_distance_batch_torch(X: torch.Tensor, 
                               feature_ranges: torch.Tensor,
                               feature_types: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Compute pairwise Gower distances for entire population using torch.
    
    Args:
        X: Population decision variables, shape (n_pop, n_var) on device
        feature_ranges: Range of each feature, shape (n_var,) on device
        feature_types: Array indicating feature type per dimension (numpy array)
    
    Returns:
        Pairwise Gower distance matrix, shape (n_pop, n_pop)
    """
    n_pop, n_var = X.shape
    device = X.device
    
    if feature_types is None:
        # All numerical - vectorized computation
        # Compute pairwise absolute differences: |X[i] - X[j]| for all i, j
        # Using broadcasting: X[i] - X[j] = X[:, None, :] - X[None, :, :]
        diff = torch.abs(X.unsqueeze(1) - X.unsqueeze(0))  # (n_pop, n_pop, n_var)
        
        # Normalize by feature ranges
        per_feature = torch.clamp(diff / feature_ranges, max=1.0)  # (n_pop, n_pop, n_var)
        
        # Mean over features
        return per_feature.mean(dim=2)  # (n_pop, n_pop)
    else:
        # Mixed types - need per-feature handling
        distances = torch.zeros(n_pop, n_pop, n_var, device=device)
        
        for j in range(n_var):
            if feature_types[j] == 'numerical':
                if feature_ranges[j] > 0:
                    distances[:, :, j] = torch.abs(X[:, j].unsqueeze(1) - X[:, j].unsqueeze(0)) / feature_ranges[j]
            else:
                # Categorical: indicator function
                distances[:, :, j] = (X[:, j].unsqueeze(1) != X[:, j].unsqueeze(0)).float()
        
        return distances.mean(dim=2)


def compute_gower_crowding_distance_torch(X: torch.Tensor, 
                                          feature_ranges: torch.Tensor,
                                          feature_types: Optional[np.ndarray] = None) -> torch.Tensor:
    """
    Compute crowding distance in feature space using Gower distance (torch version).
    
    Args:
        X: Population decision variables, shape (n_pop, n_var) on device
        feature_ranges: Range of each feature, shape (n_var,) on device
        feature_types: Feature type indicators
        
    Returns:
        Feature space crowding distances, shape (n_pop,) on device
    """
    n_pop = X.shape[0]
    device = X.device
    
    if n_pop <= 2:
        return torch.full((n_pop,), float('inf'), device=device)
    
    # Compute pairwise Gower distances
    gower_matrix = gower_distance_batch_torch(X, feature_ranges, feature_types)
    
    # Set diagonal to inf (exclude self)
    gower_matrix.fill_diagonal_(float('inf'))
    
    # For crowding: sum of distances to two nearest neighbors
    sorted_dists, _ = torch.sort(gower_matrix, dim=1)
    
    if n_pop > 2:
        crowding = sorted_dists[:, 0] + sorted_dists[:, 1]
    else:
        crowding = sorted_dists[:, 0]
    
    return crowding


def compute_objective_crowding_distance_l1_torch(F: torch.Tensor) -> torch.Tensor:
    """
    Compute crowding distance in objective space using L1 norm (torch version).
    
    Args:
        F: Objective values, shape (n_pop, n_obj) on device
        
    Returns:
        Objective space crowding distances, shape (n_pop,) on device
    """
    n_pop, n_obj = F.shape
    device = F.device
    
    if n_pop <= 2:
        return torch.full((n_pop,), float('inf'), device=device)
    
    # Normalize objectives to [0, 1]
    f_min = F.min(dim=0).values
    f_max = F.max(dim=0).values
    f_range = f_max - f_min
    f_range[f_range == 0] = 1.0
    
    F_norm = (F - f_min) / f_range
    
    # Compute pairwise L1 distances using broadcasting
    l1_matrix = torch.abs(F_norm.unsqueeze(1) - F_norm.unsqueeze(0)).sum(dim=2)  # (n_pop, n_pop)
    
    # Set diagonal to inf
    l1_matrix.fill_diagonal_(float('inf'))
    
    # Sum of distances to two nearest neighbors
    sorted_dists, _ = torch.sort(l1_matrix, dim=1)
    
    if n_pop > 2:
        crowding = sorted_dists[:, 0] + sorted_dists[:, 1]
    else:
        crowding = sorted_dists[:, 0]
    
    return crowding


def compute_moc_crowding_distance_torch(X: torch.Tensor, 
                                         F: torch.Tensor,
                                         feature_ranges: torch.Tensor,
                                         feature_types: Optional[np.ndarray] = None,
                                         alpha: float = 0.5) -> torch.Tensor:
    """
    Compute MOC-style modified crowding distance using torch.
    
    Combines objective space and feature space distances:
    CD_MOC = alpha * CD_objective + (1 - alpha) * CD_feature
    
    Args:
        X: Decision variables (features), shape (n_pop, n_var) on device
        F: Objective values, shape (n_pop, n_obj) on device
        feature_ranges: Range of each feature, shape (n_var,) on device
        feature_types: Feature type indicators
        alpha: Weight for objective space distance
        
    Returns:
        Combined crowding distances, shape (n_pop,) on device
    """
    device = X.device
    
    # Compute crowding in both spaces
    cd_objective = compute_objective_crowding_distance_l1_torch(F)
    cd_feature = compute_gower_crowding_distance_torch(X, feature_ranges, feature_types)
    
    # Normalize both to [0, 1]
    # Handle infinity values
    cd_obj_finite_mask = torch.isfinite(cd_objective)
    cd_feat_finite_mask = torch.isfinite(cd_feature)
    
    # Normalize objective crowding
    if cd_obj_finite_mask.any():
        cd_obj_max = cd_objective[cd_obj_finite_mask].max()
        cd_obj_max = cd_obj_max if cd_obj_max > 0 else 1.0
        cd_objective_norm = torch.where(cd_obj_finite_mask, cd_objective / cd_obj_max, 
                                        torch.tensor(float('inf'), device=device))
    else:
        cd_objective_norm = cd_objective
    
    # Normalize feature crowding
    if cd_feat_finite_mask.any():
        cd_feat_max = cd_feature[cd_feat_finite_mask].max()
        cd_feat_max = cd_feat_max if cd_feat_max > 0 else 1.0
        cd_feature_norm = torch.where(cd_feat_finite_mask, cd_feature / cd_feat_max,
                                      torch.tensor(float('inf'), device=device))
    else:
        cd_feature_norm = cd_feature
    
    # Combine
    combined = alpha * cd_objective_norm + (1 - alpha) * cd_feature_norm
    
    return combined


# Legacy numpy wrappers for backward compatibility
def gower_distance_mixed(x1: np.ndarray, x2: np.ndarray, 
                         feature_ranges: np.ndarray,
                         feature_types: Optional[np.ndarray] = None) -> float:
    """Compute Gower distance between two feature vectors (legacy numpy wrapper)."""
    device = _get_device()
    X = torch.from_numpy(np.stack([x1, x2]).astype(np.float32)).to(device)
    fr = torch.from_numpy(feature_ranges.astype(np.float32)).to(device)
    dist_matrix = gower_distance_batch_torch(X, fr, feature_types)
    return float(dist_matrix[0, 1].cpu().numpy())


def compute_gower_crowding_distance(X: np.ndarray, 
                                    feature_ranges: np.ndarray,
                                    feature_types: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute crowding distance in feature space (legacy numpy wrapper)."""
    device = _get_device()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    fr_t = torch.from_numpy(feature_ranges.astype(np.float32)).to(device)
    result = compute_gower_crowding_distance_torch(X_t, fr_t, feature_types)
    return result.cpu().numpy()


def compute_objective_crowding_distance_l1(F: np.ndarray) -> np.ndarray:
    """Compute crowding distance in objective space (legacy numpy wrapper)."""
    device = _get_device()
    F_t = torch.from_numpy(F.astype(np.float32)).to(device)
    result = compute_objective_crowding_distance_l1_torch(F_t)
    return result.cpu().numpy()


def compute_moc_crowding_distance(X: np.ndarray, 
                                   F: np.ndarray,
                                   feature_ranges: np.ndarray,
                                   feature_types: Optional[np.ndarray] = None,
                                   alpha: float = 0.5) -> np.ndarray:
    """Compute MOC-style modified crowding distance (legacy numpy wrapper)."""
    device = _get_device()
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    F_t = torch.from_numpy(F.astype(np.float32)).to(device)
    fr_t = torch.from_numpy(feature_ranges.astype(np.float32)).to(device)
    result = compute_moc_crowding_distance_torch(X_t, F_t, fr_t, feature_types, alpha)
    return result.cpu().numpy()


# =============================================================================
# Custom Survival for MOC Crowding Distance
# =============================================================================

class MOCRankAndCrowdingSurvival(Survival):
    """
    Survival operator using MOC-style modified crowding distance with constraint handling.
    
    Implements the Avila et al. approach where:
    1. Crowding distance combines both objective space (L1) and feature space (Gower) distances
    2. Constraint handling demotes invalid candidates to worst fronts
    
    Constraint Handling (based on Dandl et al. / Deb et al.):
    - Validity constraint: v(x) = max(0, o1(x) - epsilon)
    - If v(x) = 0: feasible (prediction close enough to target)
    - If v(x) > 0: constraint violation (demoted to worst fronts)
    - Violators are sorted by violation amount and placed in individual fronts
      after all feasible fronts, ensuring feasible solutions always survive first.
    """
    
    def __init__(self, 
                 feature_ranges: np.ndarray,
                 feature_types: Optional[np.ndarray] = None,
                 alpha: float = 0.5,
                 nds: Optional[NonDominatedSorting] = None,
                 use_constraint_handling: bool = False,
                 validity_eps: float = 0.0,
                 validity_obj_idx: int = 0):
        """
        Args:
            feature_ranges: Range of each feature (max - min) for Gower distance
            feature_types: Array of 'numerical' or 'categorical' for each feature
            alpha: Weight for objective space distance (0.5 = equal weighting)
            nds: Non-dominated sorting instance
            use_constraint_handling: Whether to apply validity constraint handling
            validity_eps: Tolerance epsilon for validity constraint.
                          v(x) = max(0, o1(x) - epsilon)
                          If epsilon=0, any o1 > 0 is a violation.
            validity_obj_idx: Index of validity objective in F (default: 0)
        """
        super().__init__(filter_infeasible=True)
        self.feature_ranges = feature_ranges
        self.feature_types = feature_types
        self.alpha = alpha
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.use_constraint_handling = use_constraint_handling
        self.validity_eps = validity_eps
        self.validity_obj_idx = validity_obj_idx
    
    def _do(self, problem, pop, *args, n_survive=None, **kwargs):
        """Perform survival selection with MOC crowding distance and constraint handling."""
        
        if n_survive is None:
            n_survive = len(pop)
        
        # Get objective values and decision variables
        F = pop.get("F").astype(float, copy=False)
        X = pop.get("X").astype(float, copy=False)
        
        # =====================================================================
        # Step 1: Standard non-dominated sorting (ignoring constraints)
        # =====================================================================
        fronts = self.nds.do(F, n_stop_if_ranked=n_survive)
        
        # Initialize arrays for all individuals
        crowding = np.full(len(pop), np.nan)
        ranks = np.full(len(pop), -1)
        violations = np.zeros(len(pop))  # Store violation amounts
        
        # =====================================================================
        # Step 2: Identify constraint violators (if constraint handling enabled)
        # v(x) = max(0, o1(x) - epsilon)
        # =====================================================================
        if self.use_constraint_handling:
            validity_obj = F[:, self.validity_obj_idx]
            violations = np.maximum(0.0, validity_obj - self.validity_eps)
            violator_mask = violations > 0
            violator_indices = np.where(violator_mask)[0]
            feasible_mask = ~violator_mask
        else:
            violator_indices = np.array([], dtype=int)
            feasible_mask = np.ones(len(pop), dtype=bool)
        
        # =====================================================================
        # Step 3: Process feasible fronts (remove violators from fronts)
        # =====================================================================
        feasible_fronts = []
        for front in fronts:
            front_arr = np.array(front)
            # Remove violators from this front
            feasible_in_front = front_arr[feasible_mask[front_arr]]
            if len(feasible_in_front) > 0:
                feasible_fronts.append(feasible_in_front)
        
        # =====================================================================
        # Step 4: Compute crowding distance and select survivors from feasible fronts
        # =====================================================================
        survivors = []
        current_rank = 0
        
        for front in feasible_fronts:
            I = front
            
            # Assign rank to feasible individuals
            ranks[I] = current_rank
            
            # Compute MOC crowding distance for this front
            if len(I) > 0:
                F_front = F[I]
                X_front = X[I]
                
                cd = compute_moc_crowding_distance(
                    X_front, F_front, 
                    self.feature_ranges,
                    self.feature_types,
                    self.alpha
                )
                
                # Store crowding distance
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
            
            current_rank += 1
        
        # =====================================================================
        # Step 5: If still need more survivors, add violators sorted by violation
        # Each violator gets its own front (F_{K+1}, F_{K+2}, ...)
        # Sorted ascending by violation: smallest violation = best among violators
        # =====================================================================
        if len(survivors) < n_survive and len(violator_indices) > 0:
            # Sort violators by violation amount (ascending)
            violator_violations = violations[violator_indices]
            sorted_violator_order = np.argsort(violator_violations)
            sorted_violators = violator_indices[sorted_violator_order]
            
            # Assign each violator to its own "virtual" front after feasible fronts
            # The violator with smallest violation gets rank K+1, next gets K+2, etc.
            for i, violator_idx in enumerate(sorted_violators):
                ranks[violator_idx] = current_rank + i + 1
                # Crowding distance is irrelevant for singleton fronts, set to 0
                crowding[violator_idx] = 0.0
                
                if len(survivors) < n_survive:
                    survivors.append(violator_idx)
        
        # Set attributes on population before subsetting
        pop.set("crowding", crowding)
        pop.set("rank", ranks)
        pop.set("violation", violations)  # Store violations for debugging/analysis
        
        # Return survivors - crowding values are preserved
        return pop[survivors]


class MOCNSGA2(NSGA2):
    """
    NSGA-II with MOC-style modified crowding distance and constraint handling.
    
    This variant uses a combined crowding distance that considers both:
    1. Objective space diversity (L1 norm between objectives)
    2. Feature space diversity (Gower distance between decision variables)
    
    Constraint Handling (Dandl et al. / Deb et al.):
    - Validity constraint based on objective o1 and tolerance epsilon
    - v(x) = max(0, o1(x) - epsilon)
    - Feasible solutions (v=0) are always preferred over violators
    - Violators are demoted to worst fronts, sorted by violation amount
    
    This approach, based on Avila et al. and Dandl et al., is particularly suited for
    counterfactual generation where we want diverse feature changes,
    not just diverse objective trade-offs, while ensuring validity.
    """
    
    def __init__(self,
                 feature_ranges: np.ndarray,
                 feature_types: Optional[np.ndarray] = None,
                 crowding_alpha: float = 0.5,
                 use_constraint_handling: bool = False,
                 validity_eps: float = 0.0,
                 validity_obj_idx: int = 0,
                 **kwargs):
        """
        Args:
            feature_ranges: Range of each feature (xu - xl) for Gower normalization
            feature_types: Feature type indicators ('numerical'/'categorical')
            crowding_alpha: Weight for objective space (0.5 = equal weighting)
            use_constraint_handling: Enable validity constraint handling
            validity_eps: Tolerance epsilon for validity constraint.
                          v(x) = max(0, o1(x) - epsilon)
                          Default 0.0 means any prediction outside Y0 is a violation.
            validity_obj_idx: Index of validity objective in F (default: 0)
            **kwargs: Additional NSGA2 arguments
        """
        # Create custom survival operator with constraint handling
        survival = MOCRankAndCrowdingSurvival(
            feature_ranges=feature_ranges,
            feature_types=feature_types,
            alpha=crowding_alpha,
            use_constraint_handling=use_constraint_handling,
            validity_eps=validity_eps,
            validity_obj_idx=validity_obj_idx
        )
        
        # Initialize parent NSGA2 with custom survival
        super().__init__(survival=survival, **kwargs)
        
        self.feature_ranges = feature_ranges
        self.feature_types = feature_types
        self.crowding_alpha = crowding_alpha
        self.use_constraint_handling = use_constraint_handling
        self.validity_eps = validity_eps
        self.validity_obj_idx = validity_obj_idx


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
    
    # Constraint handling parameters (Dandl et al. / Deb et al.)
    # Penalizes candidates whose validity objective o1 exceeds tolerance epsilon
    # v(x) = max(0, o1(x) - epsilon); violators are demoted to worst fronts
    use_constraint_handling: bool = False  # Enable validity constraint handling
    validity_eps: float = 0.0  # Tolerance epsilon for validity constraint
                               # If epsilon=0, any o1 > 0 (prediction not in Y0) is a violation
                               # If epsilon=0.1, predictions within 0.1 of Y0 are still feasible
    validity_obj_idx: int = 0  # Index of validity objective in F (default: 0 = first objective)
    
    # Conditional Mutator parameters (MOC-style)
    use_conditional_mutator: bool = True  # Use conditional mutator for plausible mutations
    conditional_mutator_prob: float = 0.7  # Probability of using conditional vs PM mutation
    tree_max_depth: int = 5  # Max depth of transformation trees
    tree_min_samples_leaf: int = 10  # Min samples per leaf in transformation trees
    
    # Reset Operator parameters (MOC-style)
    # After mutation, randomly reset features to factual values to encourage sparsity
    use_reset_operator: bool = True  # Enable MOC-style reset operator
    reset_prob: float = 0.05  # Probability of resetting each feature to factual value
                              # Should be small (0.01-0.1) to prevent over-attraction to x*

class FactualBasedSampling(FloatRandomSampling):
    """
    Initialize population around the factual instance using GPU computation.
    
    Instead of uniform random sampling, generates samples near x*
    with small perturbations. This helps the optimization start
    from relevant regions of the search space.
    """
    
    def __init__(self, x_star, noise_scale: float = 0.1, device: Optional[torch.device] = None):
        """
        Args:
            x_star: Factual instance to sample around (numpy or torch tensor)
            noise_scale: Standard deviation of Gaussian noise as fraction of range
            device: Torch device for GPU computation
        """
        super().__init__()
        self.device = device if device is not None else _get_device()
        
        # Store factual as torch tensor
        if isinstance(x_star, torch.Tensor):
            self.x_star = x_star.flatten().to(self.device)
        else:
            self.x_star = torch.from_numpy(np.asarray(x_star).flatten().astype(np.float32)).to(self.device)
        
        self.noise_scale = noise_scale
    
    def _do(self, problem, n_samples, **kwargs) -> np.ndarray:
        """Generate samples around the factual using GPU."""
        # Get bounds as tensors
        xl = torch.from_numpy(problem.xl.astype(np.float32)).to(self.device)
        xu = torch.from_numpy(problem.xu.astype(np.float32)).to(self.device)
        
        # Tile factual
        X = self.x_star.unsqueeze(0).expand(n_samples, -1).clone()
        
        # Add uniform noise on GPU
        noise = torch.empty_like(X).uniform_(-self.noise_scale, self.noise_scale)
        X = X + noise
        
        # Clip to bounds
        X = torch.clamp(X, xl, xu)
        
        return X.cpu().numpy()


class GaussianFactualSampling(FloatRandomSampling):
    """
    Sample from Gaussian distribution centered at factual using GPU.
    
    Alternative to uniform noise - may work better for smooth problems.
    """
    
    def __init__(self, x_star, sigma: float = 0.2, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else _get_device()
        
        if isinstance(x_star, torch.Tensor):
            self.x_star = x_star.flatten().to(self.device)
        else:
            self.x_star = torch.from_numpy(np.asarray(x_star).flatten().astype(np.float32)).to(self.device)
        
        self.sigma = sigma
    
    def _do(self, problem, n_samples, **kwargs) -> np.ndarray:
        xl = torch.from_numpy(problem.xl.astype(np.float32)).to(self.device)
        xu = torch.from_numpy(problem.xu.astype(np.float32)).to(self.device)
        
        # Sample from Gaussian on GPU
        X = torch.randn(n_samples, len(self.x_star), device=self.device) * self.sigma + self.x_star
        X = torch.clamp(X, xl, xu)
        
        return X.cpu().numpy()


class MixedSampling(FloatRandomSampling):
    """
    Mix factual-based and random sampling using GPU.
    
    A fraction of the population starts near factual, the rest is random.
    """
    
    def __init__(
        self,
        x_star,
        factual_fraction: float = 0.5,
        noise_scale: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device if device is not None else _get_device()
        
        if isinstance(x_star, torch.Tensor):
            self.x_star = x_star.flatten().to(self.device)
        else:
            self.x_star = torch.from_numpy(np.asarray(x_star).flatten().astype(np.float32)).to(self.device)
        
        self.factual_fraction = factual_fraction
        self.noise_scale = noise_scale
    
    def _do(self, problem, n_samples, **kwargs) -> np.ndarray:
        xl = torch.from_numpy(problem.xl.astype(np.float32)).to(self.device)
        xu = torch.from_numpy(problem.xu.astype(np.float32)).to(self.device)
        
        n_factual = int(n_samples * self.factual_fraction)
        n_random = n_samples - n_factual
        n_var = len(self.x_star)
        
        # Factual-based samples on GPU
        X_factual = self.x_star.unsqueeze(0).expand(n_factual, -1).clone()
        X_factual += torch.empty_like(X_factual).uniform_(-self.noise_scale, self.noise_scale)
        
        # Random samples on GPU
        X_random = torch.empty(n_random, n_var, device=self.device).uniform_(0, 1)
        X_random = xl + X_random * (xu - xl)
        
        # Combine and clip
        X = torch.cat([X_factual, X_random], dim=0)
        X = torch.clamp(X, xl, xu)
        
        return X.cpu().numpy()


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
    
    
class MutationWithReset(Mutation):
    """
    Wrapper that applies reset operator after any mutation.
    
    This combines any mutation operator with the MOC-style reset operator,
    applying mutation first and then the reset step.
    """
    
    def __init__(self, mutation: Mutation, reset: ResetOperator):
        """
        Args:
            mutation: The base mutation operator (PM, ConditionalMutator, or HybridMutator)
            reset: The reset operator to apply after mutation
        """
        super().__init__()
        self.mutation = mutation
        self.reset = reset
    
    def _do(self, problem, X, **kwargs) -> np.ndarray:
        """
        Apply mutation followed by reset.
        
        1. First apply the base mutation operator
        2. Then apply the reset operator to pull some features back to x*
        
        Args:
            problem: The optimization problem
            X: Decision variables, shape (n_offspring, n_var)
            
        Returns:
            Mutated and reset decision variables
        """
        # Step 1: Apply base mutation
        X_mutated = self.mutation._do(problem, X, **kwargs)
        
        # Step 2: Apply reset operator
        X_reset = self.reset._do(problem, X_mutated, **kwargs)
        
        return X_reset


def run_nsga(
    problem: Problem,
    config: NSGAConfig,
    sampling: Optional[FloatRandomSampling] = None,
    X_obs = None,
    x_factual = None,
    device: Optional[torch.device] = None,
) -> Result:
    """
    Run NSGA-II optimization on the given problem with specified configuration.
    
    Supports MOC-style modifications:
    - Modified crowding distance (objective space + feature space diversity)
    - Conditional Mutator (plausible mutations via transformation trees)
    - Reset Operator (randomly reset features to factual values for sparsity)
    
    All computations are done on GPU where possible for efficiency.
    
    Args:
        problem: The optimization problem (TorchCFProblem recommended)
        config: NSGA-II configuration
        sampling: Optional custom sampling strategy
        X_obs: Observed training data for conditional mutator.
               Can be numpy array or torch tensor, shape (n_samples, n_features).
               Required if config.use_conditional_mutator is True.
        x_factual: The factual instance x*.
                   Can be numpy array or torch tensor, shape (n_features,) or (1, n_features).
                   Required if config.use_reset_operator is True.
        device: Torch device for GPU computation
               
    Returns:
        Optimization result
    """
    if device is None:
        device = _get_device()
    
    # Use default mutation probability if not specified (1/n_var is default for PM)
    mutation_prob = config.mutation_prob if config.mutation_prob is not None else (1.0 / problem.n_var)
    
    # Compute feature ranges from problem bounds
    feature_ranges = problem.xu - problem.xl
    feature_ranges[feature_ranges == 0] = 1.0  # Avoid division by zero
    
    # Convert X_obs to numpy if needed for sklearn-based ConditionalMutator
    X_obs_np = None
    if X_obs is not None:
        if isinstance(X_obs, torch.Tensor):
            X_obs_np = X_obs.cpu().numpy() if X_obs.is_cuda else X_obs.numpy()
        else:
            X_obs_np = np.asarray(X_obs)
        # Flatten if needed
        if X_obs_np.ndim > 2:
            X_obs_np = X_obs_np.reshape(X_obs_np.shape[0], -1)
    
    # Setup base mutation operator
    if config.use_conditional_mutator and X_obs_np is not None:
        # Use MOC-style Conditional Mutator (hybrid with PM)
        # Note: ConditionalMutator uses sklearn which needs numpy for training
        base_mutation = HybridMutator(
            X_obs=X_obs_np,
            conditional_prob=config.conditional_mutator_prob,
            mutation_prob=mutation_prob,
            pm_eta=int(config.mutation_eta),
            max_depth=config.tree_max_depth,
            min_samples_leaf=config.tree_min_samples_leaf
        )
    else:
        # Use standard Polynomial Mutation
        base_mutation = PM(prob=mutation_prob, eta=int(config.mutation_eta))
    
    # Optionally wrap with reset operator
    if config.use_reset_operator and x_factual is not None:
        # Create reset operator with GPU support
        reset_op = ResetOperator(
            x_factual=x_factual,
            p_reset=config.reset_prob,
            device=device
        )
        # Wrap mutation with reset
        mutation = MutationWithReset(mutation=base_mutation, reset=reset_op)
    else:
        mutation = base_mutation
    
    # Disable duplicate elimination for high-dimensional problems to avoid memory issues
    # scipy.spatial.distance.cdist in pymoo's duplicate elimination can cause MemoryError
    eliminate_duplicates = problem.n_var < 500  # Only use for low-dimensional problems
    
    if config.use_moc_crowding:
        # Use MOC-style NSGA-II with modified crowding distance and optional constraint handling
        algorithm = MOCNSGA2(
            pop_size=config.pop_size,
            feature_ranges=feature_ranges,
            feature_types=config.feature_types,
            crowding_alpha=config.crowding_alpha,
            use_constraint_handling=config.use_constraint_handling,
            validity_eps=config.validity_eps,
            validity_obj_idx=config.validity_obj_idx,
            sampling=sampling or FloatRandomSampling(),
            crossover=SBX(prob=config.crossover_prob, eta=int(config.crossover_eta)),
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates,
        )
    else:
        # Use standard NSGA-II
        algorithm = NSGA2(
            pop_size=config.pop_size,
            sampling=sampling or FloatRandomSampling(),
            crossover=SBX(prob=config.crossover_prob, eta=int(config.crossover_eta)),
            mutation=mutation,
            eliminate_duplicates=eliminate_duplicates,
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
    result = minimize(
        problem,
        algorithm,
        termination,
        seed=config.seed,
        callback=callback,
        verbose=False,
    )
    
    return result