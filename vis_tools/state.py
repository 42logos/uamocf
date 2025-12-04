import pickle
import io
from dataclasses import dataclass
from typing import Optional, Tuple, List, Any, Union
import numpy as np
import torch
from torch import nn
from core.models import SimpleNN


def _to_numpy(arr) -> Optional[np.ndarray]:
    """Safely convert tensor or array to numpy for serialization.
    
    Args:
        arr: Input array (numpy, torch tensor, or None)
        
    Returns:
        Numpy array, or None if input was None
    """
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


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
    
    All tensor data is converted to numpy for Streamlit compatibility.
    
    Args:
        data: Tuple of (X, Y, p1) arrays (can be tensors)
        models: List of trained PyTorch models
        cf_results: Counterfactual optimization results with X and F attributes
        F_obs: Objective values for observed data points
        F_star: Objective values for the factual point x*
        x_star: The factual point coordinates (x1, x2)
    """
    # Convert data tuple to numpy
    if data is not None:
        data = tuple(_to_numpy(d) for d in data)
    
    # Extract X and F from cf_results if it exists, convert to numpy
    cf_data = None
    if cf_results is not None:
        cf_data = {
            "X": _to_numpy(cf_results.X),
            "F": _to_numpy(cf_results.F)
        }

    state = {
        "data": data,
        "model_state_dicts": [m.state_dict() for m in models] if models else [],
        "cf_results": cf_data,
        "F_obs": _to_numpy(F_obs),
        "F_star": _to_numpy(F_star),
        "x_star": _to_numpy(x_star),
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
    X,
    y,
    models: List[nn.Module],
    pymoo_result: Any,
    problem: Any,
    x_star,
    p_true = None,
) -> AppState:
    """
    Create an AppState from pymoo optimization result and problem.
    
    This is the recommended interface - it automatically calculates F_obs and F_star
    from the problem object. Accepts both numpy arrays and torch tensors.
    
    Args:
        X: Training data features, shape (n_samples, n_features) - numpy or tensor
        y: Training data labels, shape (n_samples,) - numpy or tensor
        models: List of trained PyTorch models (ensemble)
        pymoo_result: Result object from pymoo minimize() with .X and .F attributes
        problem: The pymoo Problem used for optimization (has .evaluate() method)
        x_star: Factual point coordinates, shape (n_features,) - numpy or tensor
        p_true: Optional true probability values, shape (n_samples,) - numpy or tensor
    
    Returns:
        AppState object ready to be saved with save_state()
    """
    # Convert inputs to numpy for Streamlit compatibility
    X = _to_numpy(X)
    y = _to_numpy(y)
    x_star = _to_numpy(x_star)
    
    # Create p_true if not provided
    if p_true is None:
        p_true = np.zeros(len(y), dtype=np.float32)
    else:
        p_true = _to_numpy(p_true)
    
    # Create data tuple (all numpy)
    data = (X, y, p_true)
    
    # Extract CF results from pymoo result (convert to numpy)
    cf_results = CFResult(X=_to_numpy(pymoo_result.X), F=_to_numpy(pymoo_result.F))
    
    # Calculate F_obs (objectives for observed data) - X is already numpy
    try:
        out = {}
        problem._evaluate(X, out)
        F_obs = _to_numpy(out.get('F'))
    except Exception:
        # Fallback: evaluate one at a time
        F_list = []
        for i in range(len(X)):
            out = {}
            problem._evaluate(X[i:i+1], out)
            f = _to_numpy(out.get('F'))
            F_list.append(f[0] if f is not None and len(f.shape) > 1 else f)
        F_obs = np.array(F_list) if F_list else None
    
    # Calculate F_star (objectives for factual point)
    try:
        out = {}
        problem._evaluate(x_star.reshape(1, -1), out)
        F_star = _to_numpy(out.get('F'))
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
    X,
    y,
    models: List[nn.Module],
    pymoo_result: Any,
    problem: Any,
    x_star,
    p_true = None,
) -> AppState:
    """
    Create and save state from pymoo result in one step.
    
    This is the simplest interface - just pass your data, models, and pymoo result.
    Accepts both numpy arrays and torch tensors.
    
    Args:
        filepath: Path to save the state file
        X: Training data features, shape (n_samples, n_features) - numpy or tensor
        y: Training data labels, shape (n_samples,) - numpy or tensor
        models: List of trained PyTorch models (ensemble)
        pymoo_result: Result object from pymoo minimize() with .X and .F attributes
        problem: The pymoo Problem used for optimization
        x_star: Factual point coordinates, shape (n_features,) - numpy or tensor
        p_true: Optional true probability values - numpy or tensor
    
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
