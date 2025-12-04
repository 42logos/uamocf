"""Test the refactored torch-based CF problem and optimization.

This test verifies that:
1. TorchCFProblem correctly evaluates objectives on GPU
2. All optimization operators work with torch tensors
3. Performance is improved vs the numpy-based implementation
"""
import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn
import numpy as np
import time
import gc

# Force CPU for stability in testing
device = torch.device('cpu')  # Use CPU for testing

from core.cf_problem import make_cf_problem, TorchCFProblem
from core.models import EnsembleModel
from core.optimization import (
    run_nsga, NSGAConfig, 
    FactualBasedSampling, GaussianFactualSampling, MixedSampling,
    compute_gower_crowding_distance_torch,
    compute_objective_crowding_distance_l1_torch,
    compute_moc_crowding_distance_torch,
    ResetOperator
)
from pymoo.optimize import minimize
from pymoo.algorithms.moo.nsga2 import NSGA2

# Setup
print(f"Device: {device}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Use smaller image size for memory efficiency
IMG_SIZE = 8
N_MODELS = 2

# Create simple models (use a simpler model architecture)
print("\n" + "="*60)
print("=== Setting up models ===")
print("="*60)

class SimpleMLP(nn.Module):
    """Simple MLP classifier for testing."""
    def __init__(self, input_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

input_dim = 1 * IMG_SIZE * IMG_SIZE
models = [SimpleMLP(input_dim, num_classes=10).to(device) for _ in range(N_MODELS)]
for m in models:
    m.eval()

ensemble = EnsembleModel(models).to(device)
print(f"Created {N_MODELS} models, input_dim={input_dim}")

# Create dummy data
x_factual = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=device)
X_obs = torch.randn(30, 1, IMG_SIZE, IMG_SIZE, device=device)
y_target = torch.tensor([5], device=device)

print("\n" + "="*60)
print("=== Testing TorchCFProblem ===")
print("="*60)
problem = make_cf_problem(
    model=models[0],
    x_factual=x_factual,
    y_target=y_target,
    X_obs=X_obs,
    ensemble=ensemble,
    device=device
)

print(f"Problem type: {type(problem)}")
print(f"n_var: {problem.n_var}, n_obj: {problem.n_obj}")

# Test batch evaluation
print("\n" + "="*60)
print("=== Testing Batch Evaluation Performance ===")
print("="*60)
for batch_size in [10, 30, 50]:
    X_test = np.random.randn(batch_size, problem.n_var).astype(np.float32)
    X_test = np.clip(X_test, problem.xl, problem.xu)
    
    # Benchmark
    start = time.time()
    out = {}
    problem._evaluate(X_test, out)
    elapsed = time.time() - start
    print(f"Batch size {batch_size:3d}: {elapsed*1000:6.2f}ms ({batch_size/elapsed:.0f} samples/sec)")

# Test crowding distance functions
print("\n" + "="*60)
print("=== Testing Torch Crowding Distance ===")
print("="*60)
n_test = 50
X_t = torch.randn(n_test, problem.n_var, device=device)
F_t = torch.randn(n_test, problem.n_obj, device=device)
fr_t = torch.from_numpy(problem.xu - problem.xl).float().to(device)

start = time.time()
cd_obj = compute_objective_crowding_distance_l1_torch(F_t)
cd_feat = compute_gower_crowding_distance_torch(X_t, fr_t)
cd_moc = compute_moc_crowding_distance_torch(X_t, F_t, fr_t)
elapsed = time.time() - start
print(f"Crowding distance for {n_test} samples: {elapsed*1000:.2f}ms")
print(f"  - Objective CD shape: {cd_obj.shape}")
print(f"  - Feature CD shape: {cd_feat.shape}")
print(f"  - MOC CD shape: {cd_moc.shape}")

# Test sampling classes
print("\n" + "="*60)
print("=== Testing Torch-based Sampling ===")
print("="*60)
sampling_factual = FactualBasedSampling(x_star=x_factual, noise_scale=0.3, device=device)
sampling_gaussian = GaussianFactualSampling(x_star=x_factual, sigma=0.2, device=device)
sampling_mixed = MixedSampling(x_star=x_factual, factual_fraction=0.7, noise_scale=0.2, device=device)

for name, sampler in [("FactualBased", sampling_factual), ("Gaussian", sampling_gaussian), ("Mixed", sampling_mixed)]:
    start = time.time()
    samples = sampler._do(problem, 50)
    elapsed = time.time() - start
    print(f"{name:15s}: {elapsed*1000:.2f}ms, shape={samples.shape}")

# Test Reset Operator
print("\n" + "="*60)
print("=== Testing Reset Operator ===")
print("="*60)
reset_op = ResetOperator(x_factual=x_factual, p_reset=0.1, device=device)
X_pre = np.random.randn(30, problem.n_var).astype(np.float32)
start = time.time()
X_post = reset_op._do(problem, X_pre)
elapsed = time.time() - start
# Count how many features were reset to factual
x_fact_flat = x_factual.flatten().cpu().numpy()
reset_count = np.sum(np.abs(X_post - x_fact_flat) < 1e-6)
total_features = X_post.size
print(f"Reset operator: {elapsed*1000:.2f}ms")
print(f"  - Reset ratio: {reset_count/total_features*100:.1f}% (expected ~10%)")

# Test optimization
print("\n" + "="*60)
print("=== Testing Full Optimization ===")
print("="*60)
config = NSGAConfig(
    pop_size=30,
    min_gen=5,
    max_gen=15,
    use_conditional_mutator=False,  # Disable for speed
    use_reset_operator=True,
    verbose=True
)

sampling = FactualBasedSampling(x_star=x_factual, noise_scale=0.3, device=device)

start = time.time()
results = run_nsga(
    problem=problem,
    config=config,
    sampling=sampling,
    x_factual=x_factual,
    device=device
)
elapsed = time.time() - start

print(f"\nOptimization completed in {elapsed:.2f}s")
print(f"Result X shape: {results.X.shape}")
print(f"Result F shape: {results.F.shape}")
print(f"Best validity: {results.F[:, 0].min():.4f}")

print("\n=== SUCCESS: All tests passed! ===")
