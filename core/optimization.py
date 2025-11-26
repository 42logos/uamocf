"""
NSGA-II optimization utilities for counterfactual generation.

Provides custom sampling, callbacks, and termination criteria for
multi-objective counterfactual optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.core.termination import Termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import torch


Array = Union[np.ndarray, torch.Tensor]


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
    
    def notify(self, algorithm):
        """Called each generation."""
        gen = algorithm.n_gen
        F = algorithm.pop.get("F")
        
        if F is None:
            return
        
        # Count valid CFs in population
        n_valid_pop = int(np.sum(F[:, 0] < self.validity_threshold))
        
        # Count valid CFs in archive (Pareto optimal)
        n_valid_archive = 0
        mean_sparsity = 0.0
        if algorithm.opt is not None:
            F_opt = algorithm.opt.get("F")
            if F_opt is not None:
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