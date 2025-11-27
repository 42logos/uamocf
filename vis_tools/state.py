import pickle
import io
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any
import numpy as np
import torch
from torch import nn
from core.models import SimpleNN


@dataclass
class CFResult:
    """Simple container for counterfactual results (X and F arrays)."""
    X: np.ndarray
    F: np.ndarray


@dataclass 
class AppState:
    """Complete application state for visualization."""
    data: Tuple[np.ndarray, np.ndarray, np.ndarray]  # (X, y, p_true)
    models: List[nn.Module]
    cf_results: Optional[CFResult]
    F_obs: Optional[np.ndarray]
    F_star: Optional[np.ndarray]
    x_star: Optional[np.ndarray]


def export_state(data, models, cf_results, F_obs, F_star=None, x_star=None):
    """
    Serializes the application state into a bytes object.
    
    Args:
        data: Tuple of (X, Y, p1) arrays
        models: List of trained PyTorch models
        cf_results: Counterfactual optimization results with X and F attributes
        F_obs: Objective values for observed data points
        F_star: Objective values for the factual point x*
        x_star: The factual point coordinates (x1, x2)
    """
    # Extract X and F from cf_results if it exists
    cf_data = None
    if cf_results is not None:
        cf_data = {
            "X": cf_results.X,
            "F": cf_results.F
        }

    state = {
        "data": data,
        "model_state_dicts": [m.state_dict() for m in models] if models else [],
        "cf_results": cf_data,
        "F_obs": F_obs,
        "F_star": F_star,
        "x_star": x_star,
    }
    
    buffer = io.BytesIO()
    pickle.dump(state, buffer)
    return buffer.getvalue()


def import_state(file_obj, device):
    """
    Deserializes the application state from a file-like object.
    Returns (data, models, cf_results, F_obs, F_star, x_star).
    """
    state = pickle.load(file_obj)
    
    data = state.get("data")
    
    models = []
    model_dicts = state.get("model_state_dicts", [])
    if model_dicts:
        for sd in model_dicts:
            # Assuming default architecture for SimpleNN as per models.py
            model = SimpleNN() 
            model.load_state_dict(sd)
            model.to(device)
            model.eval()
            models.append(model)
            
    cf_results = None
    cf_data = state.get("cf_results")
    if cf_data:
        cf_results = CFResult(cf_data["X"], cf_data["F"])
        
    F_obs = state.get("F_obs")
    F_star = state.get("F_star")
    x_star = state.get("x_star")
    
    return data, models, cf_results, F_obs, F_star, x_star


def save_state(app_state: AppState, filepath: str) -> None:
    """
    Save AppState to a file.
    
    Args:
        app_state: AppState object
        filepath: Path to save the state file
    """
    state_bytes = export_state(
        data=app_state.data,
        models=app_state.models,
        cf_results=app_state.cf_results,
        F_obs=app_state.F_obs,
        F_star=app_state.F_star,
        x_star=app_state.x_star
    )
    with open(filepath, 'wb') as f:
        f.write(state_bytes)


def load_state(filepath: str, device: Optional[torch.device] = None) -> AppState:
    """
    Load AppState from a file.
    
    Args:
        filepath: Path to the state file
        device: Torch device for models (defaults to CPU)
        
    Returns:
        AppState object
    """
    if device is None:
        device = torch.device('cpu')
        
    with open(filepath, 'rb') as f:
        data, models, cf_results, F_obs, F_star, x_star = import_state(f, device)
    
    return AppState(
        data=data,
        models=models,
        cf_results=cf_results,
        F_obs=F_obs,
        F_star=F_star,
        x_star=x_star
    )


def create_state_from_pymoo_result(
    X: np.ndarray,
    y: np.ndarray,
    models: List[nn.Module],
    pymoo_result: Any,
    problem: Any,
    x_star: np.ndarray,
    p_true: Optional[np.ndarray] = None,
) -> AppState:
    """
    Create an AppState from pymoo optimization result and problem.
    
    This is the recommended interface - it automatically calculates F_obs and F_star
    from the problem object.
    
    Args:
        X: Training data features, shape (n_samples, n_features)
        y: Training data labels, shape (n_samples,)
        models: List of trained PyTorch models (ensemble)
        pymoo_result: Result object from pymoo minimize() with .X and .F attributes
        problem: The pymoo Problem used for optimization (has .evaluate() method)
        x_star: Factual point coordinates, shape (n_features,)
        p_true: Optional true probability values, shape (n_samples,)
    
    Returns:
        AppState object ready to be saved with save_state()
    """
    # Create p_true if not provided
    if p_true is None:
        p_true = np.zeros(len(y), dtype=np.float32)
    
    # Create data tuple
    data = (X, y, p_true)
    
    # Extract CF results from pymoo result
    cf_results = CFResult(X=pymoo_result.X, F=pymoo_result.F)
    
    # Calculate F_obs (objectives for observed data)
    try:
        F_obs = np.array(problem.evaluate(X))
    except Exception:
        F_list = []
        for i in range(len(X)):
            f = problem.evaluate(X[i:i+1])
            F_list.append(f[0] if len(f.shape) > 1 else f)
        F_obs = np.array(F_list)
    
    # Calculate F_star (objectives for factual point)
    x_star = np.asarray(x_star)
    try:
        F_star = np.array(problem.evaluate(x_star.reshape(1, -1)))
    except Exception:
        F_star = None
    
    return AppState(
        data=data,
        models=models,
        cf_results=cf_results,
        F_obs=F_obs,
        F_star=F_star,
        x_star=x_star
    )


def create_and_save_state(
    filepath: str,
    X: np.ndarray,
    y: np.ndarray,
    models: List[nn.Module],
    pymoo_result: Any,
    problem: Any,
    x_star: np.ndarray,
    p_true: Optional[np.ndarray] = None,
) -> AppState:
    """
    Create and save state from pymoo result in one step.
    
    This is the simplest interface - just pass your data, models, and pymoo result.
    
    Args:
        filepath: Path to save the state file
        X: Training data features, shape (n_samples, n_features)
        y: Training data labels, shape (n_samples,)
        models: List of trained PyTorch models (ensemble)
        pymoo_result: Result object from pymoo minimize() with .X and .F attributes
        problem: The pymoo Problem used for optimization
        x_star: Factual point coordinates, shape (n_features,)
        p_true: Optional true probability values
    
    Returns:
        AppState object that was saved
    """
    app_state = create_state_from_pymoo_result(
        X=X, y=y, models=models,
        pymoo_result=pymoo_result, problem=problem, x_star=x_star,
        p_true=p_true
    )
    save_state(app_state, filepath)
    return app_state
