import pickle
import io
import torch
from .models import SimpleNN

def export_state(data, models, cf_results, F_obs):
    """
    Serializes the application state into a bytes object.
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
        "F_obs": F_obs
    }
    
    buffer = io.BytesIO()
    pickle.dump(state, buffer)
    return buffer.getvalue()

def import_state(file_obj, device):
    """
    Deserializes the application state from a file-like object.
    Returns (data, models, cf_results, F_obs).
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
        # Create a simple object to mimic pymoo result
        class Result:
            def __init__(self, X, F):
                self.X = X
                self.F = F
        cf_results = Result(cf_data["X"], cf_data["F"])
        
    F_obs = state.get("F_obs")
    
    return data, models, cf_results, F_obs
