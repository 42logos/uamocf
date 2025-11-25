import sys
import os
# Add src to path so we can import vis_tools as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from vis_tools import data, models, training, uncertainty, cf_problem, plotting, state

PANEL_HEIGHT = 500

st.set_page_config(page_title="uamocf: Uncertainty-Aware Multi-Objective Counterfactuals", layout="wide")

# --- Session State ---
if "data" not in st.session_state:
    st.session_state.data = None
if "models" not in st.session_state:
    st.session_state.models = None
if "cf_results" not in st.session_state:
    st.session_state.cf_results = None
if "F_obs" not in st.session_state:
    st.session_state.F_obs = None

# --- Sidebar Controls ---

# 0. Compute Configuration
st.sidebar.header("Compute")
available_devices = ["cpu"]
if torch.cuda.is_available():
    available_devices.append("cuda")
selected_device = st.sidebar.selectbox("Device", available_devices, index=0)

# 1. Interaction (Top Priority)
st.sidebar.header("Interaction")
interaction_mode = st.sidebar.radio("Mode", ["Explore (Pan/Zoom/Orbit)", "Select (Lasso/Box)"], index=0)

if st.sidebar.button("Clear All Selections", type="primary"):
    st.session_state.global_indices = set()
    st.session_state.last_design_select = None
    st.session_state.last_obj_select = None
    st.session_state.table_selection = set()
    st.rerun()

# Manual Index Selection (workaround for 3D - hover to see index, then type it here)
st.sidebar.caption("💡 **3D Selection**: Hover over a 3D point to see its index, then enter it below")
col_idx, col_btn = st.sidebar.columns([2, 1])
with col_idx:
    manual_index = st.number_input("Point Index", min_value=0, value=0, step=1, key="manual_idx_input", label_visibility="collapsed")
with col_btn:
    if st.button("Add", key="add_manual_idx"):
        st.session_state.global_indices.add(int(manual_index))
        st.session_state.table_selection = st.session_state.global_indices.copy()
        st.rerun()

# 2. State Management
with st.sidebar.expander("State Management", expanded=False):
    st.subheader("Export")
    if st.session_state.data is not None:
        state_bytes = state.export_state(
            st.session_state.data,
            st.session_state.models,
            st.session_state.cf_results,
            st.session_state.F_obs
        )
        st.download_button(
            label="Download State",
            data=state_bytes,
            file_name="optiview_state.pkl",
            mime="application/octet-stream"
        )
    else:
        st.info("No data to export.")

    st.subheader("Import")
    uploaded_file = st.file_uploader("Load State", type=["pkl"])
    if uploaded_file is not None:
        if st.button("Load State"):
            try:
                # Determine device
                device = torch.device(selected_device)
                d, m, cf, f_obs = state.import_state(uploaded_file, device=device)
                st.session_state.data = d
                st.session_state.models = m
                st.session_state.cf_results = cf
                st.session_state.F_obs = f_obs
                st.success("State loaded successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading state: {e}")

# 3. System Configuration
with st.sidebar.expander("System Configuration", expanded=False):
    st.subheader("Data Generation")
    n_samples = st.slider("Number of Samples", 100, 2000, 500)
    noise_sigma = st.slider("Noise Sigma", 0.1, 1.0, 0.4)
    seed = st.number_input("Random Seed", value=42)
    
    st.subheader("Model Training")
    n_epochs = st.slider("Epochs", 10, 200, 50)
    ensemble_size = st.slider("Ensemble Size", 1, 100, 3)
    
    st.subheader("Search Parameters")
    c1, c2 = st.columns(2)
    x_star_x = c1.number_input("x* (x1)", value=-0.8)
    x_star_y = c2.number_input("x* (x2)", value=-0.7)
    pop_size = st.slider("Population Size", 20, 200, 50)
    n_gen = st.slider("Generations", 50, 500, 200)

# 3. Visualization Settings
with st.sidebar.expander("Visualization Settings", expanded=False):
    st.subheader("Visibility")
    show_sampled = st.checkbox("Show Sampled Points", value=True)
    show_context = st.checkbox("Show Context", value=True)
    show_linking = st.checkbox("Show Linking Line", value=False)
    
    st.subheader("Background")
    background_type = st.selectbox("Design Space Background", ["Probability", "Aleatoric Uncertainty", "Epistemic Uncertainty", "None"], index=0)
    n_contours = st.slider("Background Contour Levels", 5, 100, 20)
    
    st.subheader("Opacity")
    context_opacity = st.slider("Context Opacity", 0.01, 1.0, 0.2)
    pareto_opacity = st.slider("Pareto Opacity (3D)", 0.1, 1.0, 1.0)

    st.subheader("Point Sizes")
    size_pareto = st.slider("Pareto Points", 1, 20, 5)
    size_context = st.slider("Context Points", 1, 20, 3)    
    size_xstar = st.slider("Reference (x*)", 5, 30, 6)
    size_sampled = st.slider("Sampled Points", 1, 20, 7)    

# 4. Range & Filters
with st.sidebar.expander("Range & Filters", expanded=True):
    st.subheader("Design Space Range")
    use_custom_range = st.checkbox("Custom Range", value=False)
    
    custom_x_range = None
    custom_y_range = None

    if use_custom_range:
        # Default bounds
        x_min_def, x_max_def = -3.0, 3.0
        y_min_def, y_max_def = -3.0, 3.0
        
        # Default values (current view)
        x_val_min, x_val_max = -2.5, 2.5
        y_val_min, y_val_max = -2.5, 2.5

        if st.session_state.data is not None:
            X_data = st.session_state.data[0]
            x_d_min, x_d_max = X_data[:, 0].min(), X_data[:, 0].max()
            y_d_min, y_d_max = X_data[:, 1].min(), X_data[:, 1].max()
            
            # Margin 10% of the span for default view
            x_span = x_d_max - x_d_min
            y_span = y_d_max - y_d_min
            margin_x = x_span * 0.1
            margin_y = y_span * 0.1
            
            x_val_min = x_d_min - margin_x
            x_val_max = x_d_max + margin_x
            y_val_min = y_d_min - margin_y
            y_val_max = y_d_max + margin_y
            
            # Slider bounds (allow more zooming out, say 50% margin)
            outer_margin_x = x_span * 0.5
            outer_margin_y = y_span * 0.5
            x_min_def = x_d_min - outer_margin_x
            x_max_def = x_d_max + outer_margin_x
            y_min_def = y_d_min - outer_margin_y
            y_max_def = y_d_max + outer_margin_y

        custom_x_range = st.slider("X Range", float(x_min_def), float(x_max_def), (float(x_val_min), float(x_val_max)))
        custom_y_range = st.slider("Y Range", float(y_min_def), float(y_max_def), (float(y_val_min), float(y_val_max)))

    st.subheader("Objective Filters")
    obj_filters = {}
    if st.session_state.cf_results is not None or st.session_state.F_obs is not None:
        # Combine F and F_obs to find ranges
        F_all = []
        if st.session_state.cf_results is not None and st.session_state.cf_results.F is not None:
            F_all.append(st.session_state.cf_results.F)
        if st.session_state.F_obs is not None:
            F_all.append(st.session_state.F_obs)
        
        if F_all:
            F_combined = np.vstack(F_all)
            obj_names = ["Validity", "Epistemic", "Sparsity", "Aleatoric"]
            
            for i, name in enumerate(obj_names):
                col_values = F_combined[:, i]
                if name == "Sparsity":
                    # Discrete handling for Sparsity
                    unique_vals = np.unique(col_values)
                    unique_vals = np.sort(unique_vals)
                    options = unique_vals.tolist()
                    # Use multiselect for discrete values
                    selected = st.multiselect(f"{name} Values", options, default=options)
                    obj_filters[name] = selected
                else:
                    v_min = float(col_values.min())
                    v_max = float(col_values.max())
                    # Add a small buffer to min/max to ensure points on boundary are included
                    # or just use the exact values.
                    if v_min < v_max:
                        obj_filters[name] = st.slider(f"{name} Range", v_min, v_max, (v_min, v_max))
                    else:
                        st.write(f"{name}: {v_min:.4f}")
                        obj_filters[name] = (v_min, v_max)

# --- Helper Functions ---
def get_indices_from_selection(selection):
    if not selection: return []
    points = selection.get("selection", {}).get("points", [])
    # Flatten customdata if it's a list (sometimes it is)
    indices = []
    for p in points:
        if "customdata" in p:
            cd = p["customdata"]
            if isinstance(cd, list):
                indices.extend(cd)
            else:
                indices.append(cd)
    return list(set(indices))

def filter_mask(F_data, filters):
    if F_data is None: return None
    mask = np.ones(len(F_data), dtype=bool)
    obj_names = ["Validity", "Epistemic", "Sparsity", "Aleatoric"]
    for i, name in enumerate(obj_names):
        if name in filters:
            val = filters[name]
            if isinstance(val, list):
                # Discrete filtering (e.g. for Sparsity)
                mask &= np.isin(F_data[:, i], val)
            else:
                # Range filtering (tuple)
                min_v, max_v = val
                mask &= (F_data[:, i] >= min_v) & (F_data[:, i] <= max_v)
    return mask

# --- Global Selection Logic ---
if "global_indices" not in st.session_state:
    st.session_state.global_indices = set()
if "last_design_select" not in st.session_state:
    st.session_state.last_design_select = None
if "table_selection" not in st.session_state:
    st.session_state.table_selection = set()



# Update from Design Space
curr_design_sel = st.session_state.get("design_select")
if curr_design_sel != st.session_state.last_design_select:
    new_indices = get_indices_from_selection(curr_design_sel)
    if interaction_mode.startswith("Explore"):
        # Additive
        if new_indices:
            st.session_state.global_indices.update(new_indices)
    else:
        # Replace (Select Mode)
        # Only replace if we have a valid selection. 
        # This prevents clearing selection when the plot re-renders (which might reset selection state).
        if new_indices:
            st.session_state.global_indices = set(new_indices)
    st.session_state.last_design_select = curr_design_sel

# Note: 3D objective space selection is not supported by Streamlit.
# Users can hover over 3D points to see their index, then use the sidebar input to select them.

# --- Main Layout ---
st.title("uamocf: Uncertainty-Aware Multi-Objective Counterfactuals")

# Top Row: Design Space & Objective Space
col_design, col_obj = st.columns([1, 1])

# --- Logic: Data & Training ---
# We run this logic first to populate session state, but display it in the columns
if st.sidebar.button("Initialize System (Gen Data + Train)"):
    # 1. Generate Data
    cfg = data.DataConfig(
        n=n_samples,
        seed=seed,
        p_fn=lambda x: data.moon_focus_prob(x, sigma=noise_sigma)
    )
    X, Y, p1 = data.sample_dataset(cfg)
    st.session_state.data = (X, Y, p1)
    # Clear previous results
    st.session_state.cf_results = None
    st.session_state.F_obs = None
    st.session_state.F_star = None
    
    # 2. Train Model
    train_cfg = training.TrainConfig(
        epochs=n_epochs,
        device=selected_device,
        progress=True
    )
    
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    total_steps = ensemble_size * n_epochs
    
    def update_progress(model_idx, epoch, loss):
        current_step = model_idx * n_epochs + epoch + 1
        progress = min(current_step / total_steps, 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Training Model {model_idx+1}/{ensemble_size} - Epoch {epoch+1}/{n_epochs} - Loss: {loss:.4f}")
    
    def resample_fn(idx):
        # Generate fresh data for each model to approximate posterior
        # Use a different seed for each model
        model_seed = seed + idx + 1 # Offset from main seed
        new_cfg = data.DataConfig(
            n=n_samples,
            seed=model_seed,
            p_fn=lambda x: data.moon_focus_prob(x, sigma=noise_sigma)
        )
        X_new, Y_new, _ = data.sample_dataset(new_cfg)
        return X_new, Y_new

    with st.spinner("Training Ensemble..."):
        results = training.train_ensemble(
            num_models=ensemble_size,
            X=X,
            y=Y,
            cfg=train_cfg,
            resample_fn=resample_fn,
            callback=update_progress
        )
    
    progress_bar.empty()
    status_text.empty()
    st.session_state.models = [res.model for res in results]
    st.success("System Initialized!")

# --- Logic: Search ---
if st.sidebar.button("Run Counterfactual Search"):
    if st.session_state.models is None or st.session_state.data is None:
        st.error("Please Initialize System first.")
    else:
        # Ensure models are on the selected device
        device_obj = torch.device(selected_device)
        for m in st.session_state.models:
            m.to(device_obj)

        X, Y, _ = st.session_state.data
        x_star = torch.tensor([[x_star_x, x_star_y]], dtype=torch.float32).to(device_obj)
        
        ensemble_model = models.EnsembleModel(st.session_state.models)
        with torch.no_grad():
            y_pred = ensemble_model(x_star).argmax(dim=1)
        y_target = 1 - y_pred 
        
        X_obs = torch.tensor(X, dtype=torch.float32)
        weights = torch.ones(5)
        
        cf_cfg = cf_problem.CFConfig(k_neighbors=5, use_soft_validity=True)
        
        problem = cf_problem.make_cf_problem(
            model=st.session_state.models[0],
            x_star=x_star.squeeze(0),
            y_target=y_target,
            X_obs=X_obs,
            weights=weights,
            config=cf_cfg,
            ensemble=st.session_state.models,
            bayesian_model=ensemble_model,
            device=torch.device(selected_device)
        )
        
        algorithm = NSGA2(pop_size=pop_size)
        termination = get_termination("n_gen", n_gen)
        
        # Progress Bar for Search
        search_progress = st.progress(0.0)
        search_status = st.empty()
        
        from pymoo.core.callback import Callback

        class ProgressCallback(Callback):
            def __init__(self):
                super().__init__()
                self.n_gen = n_gen
                
            def notify(self, algorithm):
                gen = algorithm.n_gen
                progress = min(gen / self.n_gen, 1.0)
                search_progress.progress(progress)
                search_status.text(f"Optimization Generation {gen}/{self.n_gen}")

        with st.spinner("Running Optimization..."):
            res = minimize(problem, algorithm, termination, callback=ProgressCallback(), verbose=False)
        
        search_progress.empty()
        search_status.empty()
        
        # Evaluate objectives for all observed points for context
        with st.spinner("Evaluating Context..."):
            F_obs = np.array(problem.evaluate(X))
            st.session_state.F_obs = F_obs
            
            # Evaluate objectives for x*
            x_star_np = x_star.cpu().numpy()
            F_star = np.array(problem.evaluate(x_star_np))
            st.session_state.F_star = F_star

        st.session_state.cf_results = res
        st.success("Search Complete!")

# --- Visualization: Design Space (Top Left) ---
with col_design:
    st.subheader("Design Space (Input & ML Model)")
    if st.session_state.models is not None and st.session_state.data is not None:
        X, Y, _ = st.session_state.data
        ensemble_model = models.EnsembleModel(st.session_state.models)
        
        # Use Global Selection
        highlight_indices = list(st.session_state.global_indices)
        
        # Determine Modes based on Interaction Mode
        is_select_mode = interaction_mode.startswith("Select")
        dragmode_2d = "lasso" if is_select_mode else "pan"
        sel_mode_2d = ["points", "box", "lasso"] if is_select_mode else ["points"]
        
        # Apply Filters to Design Space
        # We need to filter X (Observed) and Pareto_X based on obj_filters
        # But X (Observed) might not have F_obs computed yet if search hasn't run.
        # If F_obs is None, we can't filter Observed points by objectives.
        
        mask_obs = None
        if st.session_state.F_obs is not None:
            mask_obs = filter_mask(st.session_state.F_obs, obj_filters)
        
        mask_par = None
        if st.session_state.cf_results is not None and st.session_state.cf_results.F is not None:
            mask_par = filter_mask(st.session_state.cf_results.F, obj_filters)

        # Prepare Data for Plotting
        X_plot = X
        Y_plot = Y if show_sampled else None
        X_indices = np.arange(len(X))
        
        if mask_obs is not None:
            X_plot = X[mask_obs]
            if Y_plot is not None:
                Y_plot = Y[mask_obs]
            X_indices = X_indices[mask_obs]
        
        pareto_X_plot = None
        pareto_indices = None
        
        if st.session_state.cf_results is not None and st.session_state.cf_results.X is not None:
            pareto_X_plot = st.session_state.cf_results.X
            # Global indices for Pareto start after X
            pareto_indices = np.arange(len(X), len(X) + len(pareto_X_plot))
            
            if mask_par is not None:
                pareto_X_plot = pareto_X_plot[mask_par]
                pareto_indices = pareto_indices[mask_par]

        # Generate Plotly Figure
        fig1 = plotting.get_design_space_fig(
            ensemble_model, 
            X_plot, 
            Y_plot, 
            x_star=np.array([x_star_x, x_star_y]),
            pareto_X=pareto_X_plot,
            device="cpu",
            sampled_size=size_sampled,
            pareto_size=size_pareto,
            x_star_size=size_xstar,
            dragmode=dragmode_2d,
            x_range=custom_x_range,
            y_range=custom_y_range,
            background_type=background_type,
            models=st.session_state.models,
            n_contours=n_contours,
            X_indices=X_indices,
            pareto_indices=pareto_indices,
            height=PANEL_HEIGHT
        )
        
        # Add Highlights (Global Selection)
        if highlight_indices:
            # Gather points to highlight
            # Indices < len(X) are Observed
            # Indices >= len(X) are Pareto
            N = len(X)
            high_X = []
            high_Y = []
            
            # Check Observed
            # Only highlight if they are visible (in mask)
            obs_indices = [i for i in highlight_indices if i < N]
            if obs_indices:
                # Filter out indices that are hidden by mask
                if mask_obs is not None:
                    # mask_obs is boolean array corresponding to 0..N-1
                    visible_obs_indices = [i for i in obs_indices if i < len(mask_obs) and mask_obs[i]]
                    obs_indices = visible_obs_indices
                
                if obs_indices:
                    high_X.extend(X[obs_indices, 0])
                    high_Y.extend(X[obs_indices, 1])
            
            # Check Pareto
            if st.session_state.cf_results is not None:
                X_cf = st.session_state.cf_results.X
                if X_cf is not None:
                    par_indices = [i - N for i in highlight_indices if i >= N]
                    if par_indices:
                        # Filter out indices that are hidden by mask
                        if mask_par is not None:
                            visible_par_indices = [i for i in par_indices if i < len(mask_par) and mask_par[i]]
                            par_indices = visible_par_indices
                        
                        # Also filter out indices that are out of bounds of X_cf (if mask_par was None)
                        par_indices = [i for i in par_indices if i < len(X_cf)]

                        if par_indices:
                            high_X.extend(X_cf[par_indices, 0])
                            high_Y.extend(X_cf[par_indices, 1])
            
            if high_X:
                fig1.add_trace(go.Scatter(
                    x=high_X, y=high_Y,
                    mode='markers',
                    marker=dict(size=12, color='yellow', line=dict(width=2, color='black'), symbol='circle-open'),
                    name='Linked Selection',
                    hoverinfo='skip'
                ))

        # Add Table Highlights (Secondary Selection)
        if "table_selection" in st.session_state and st.session_state.table_selection:
            table_indices = list(st.session_state.table_selection)
            N = len(X)
            tbl_X = []
            tbl_Y = []
            
            # Check Observed
            obs_indices = [i for i in table_indices if i < N]
            if obs_indices:
                if mask_obs is not None:
                    obs_indices = [i for i in obs_indices if i < len(mask_obs) and mask_obs[i]]
                
                # Bounds check for X
                obs_indices = [i for i in obs_indices if i < len(X)]
                
                if obs_indices:
                    tbl_X.extend(X[obs_indices, 0])
                    tbl_Y.extend(X[obs_indices, 1])
            
            # Check Pareto
            if st.session_state.cf_results is not None:
                X_cf = st.session_state.cf_results.X
                if X_cf is not None:
                    par_indices = [i - N for i in table_indices if i >= N]
                    if par_indices:
                        if mask_par is not None:
                            par_indices = [i for i in par_indices if i < len(mask_par) and mask_par[i]]
                        
                        # Bounds check for X_cf
                        par_indices = [i for i in par_indices if i < len(X_cf)]
                        
                        if par_indices:
                            tbl_X.extend(X_cf[par_indices, 0])
                            tbl_Y.extend(X_cf[par_indices, 1])
            
            if tbl_X:
                fig1.add_trace(go.Scatter(
                    x=tbl_X, y=tbl_Y,
                    mode='markers',
                    marker=dict(size=14, color='cyan', line=dict(width=3, color='blue'), symbol='circle'),
                    name='Table Selection',
                    hoverinfo='skip'
                ))

        st.plotly_chart(fig1, on_select="rerun", selection_mode=sel_mode_2d, key="design_select", config={'displayModeBar': True}, use_container_width=True)
    else:
        st.info("Initialize System to view Design Space.")

# --- Visualization: Objective Space (Top Right) ---
with col_obj:
    st.subheader("Objective Space (Output & Pareto)")
    if st.session_state.cf_results is not None:
        res = st.session_state.cf_results
        F = res.F # Objectives: 0:Validity, 1:Epistemic, 2:Sparsity, 3:Aleatoric
        X_cf = res.X
        
        # Use Global Selection
        highlight_indices = list(st.session_state.global_indices)
        
        if F is not None:
            # Apply Filters
            mask_par = filter_mask(F, obj_filters)
            if mask_par is None:
                mask_par = np.ones(len(F), dtype=bool)
            
            F_filtered = F[mask_par]
            
            fig_3d = go.Figure()
            
            # Global Index Offset for Pareto
            N = len(st.session_state.data[0]) if st.session_state.data else 0
            
            # Original indices for ALL Pareto points
            all_pareto_indices = np.arange(N, N + len(F))
            
            # Filtered indices
            filtered_pareto_indices = all_pareto_indices[mask_par]

            # Split into Valid and Invalid based on Validity (Index 0) <= 0.5
            valid_mask = F_filtered[:, 0] <= 0.5
            invalid_mask = ~valid_mask

            F_valid = F_filtered[valid_mask]
            idx_valid = filtered_pareto_indices[valid_mask]

            F_invalid = F_filtered[invalid_mask]
            idx_invalid = filtered_pareto_indices[invalid_mask]
            
            # 1. Original observations (Context)
            if show_context and st.session_state.F_obs is not None:
                F_obs = st.session_state.F_obs
                obs_indices = np.arange(len(F_obs))
                
                # Apply Filters to Context
                mask_obs = filter_mask(F_obs, obj_filters)
                if mask_obs is not None:
                    F_obs = F_obs[mask_obs]
                    obs_indices = obs_indices[mask_obs]
                
                if len(F_obs) > 0:
                    fig_3d.add_trace(go.Scatter3d(
                        x=F_obs[:, 1], y=F_obs[:, 0], z=-F_obs[:, 3],
                        mode='markers',
                        marker=dict(
                            size=3, 
                            color='green', 
                            opacity=context_opacity
                        ),
                        customdata=obs_indices,
                        hovertemplate='Obs Idx: %{customdata}<br>Val: %{y:.2f}<br>Epi: %{x:.2f}<br>Ale: %{z:.2f}',
                        name='Original observations in Obj Space'
                    ))

            # 2. Pareto Front which not valid
            if len(F_invalid) > 0:
                fig_3d.add_trace(go.Scatter3d(
                    x=F_invalid[:, 1], y=F_invalid[:, 0], z=-F_invalid[:, 3],
                    mode='markers',
                    marker=dict(
                        size=size_pareto, 
                        color='blue', 
                        symbol='cross',
                        opacity=pareto_opacity
                    ),
                    customdata=idx_invalid,
                    hovertemplate='Pareto Idx: %{customdata}<br>Val: %{y:.2f}<br>Epi: %{x:.2f}<br>Ale: %{z:.2f}',
                    name='Pareto Front which not valid'
                ))

            # 3. Valid Counterfactuals in Obj Space
            if len(F_valid) > 0:
                fig_3d.add_trace(go.Scatter3d(
                    x=F_valid[:, 1], y=F_valid[:, 0], z=-F_valid[:, 3],
                    mode='markers',
                    marker=dict(
                        size=size_pareto, 
                        color='red', 
                        symbol='cross',
                        opacity=pareto_opacity
                    ),
                    customdata=idx_valid,
                    hovertemplate='Pareto Idx: %{customdata}<br>Val: %{y:.2f}<br>Epi: %{x:.2f}<br>Ale: %{z:.2f}',
                    name='Valid Counterfactuals in Obj Space'
                ))
            
            # 4. Factual Instance x* in Obj Space
            if "F_star" in st.session_state and st.session_state.F_star is not None:
                F_star = st.session_state.F_star
                x_star_obj_x = F_star[:, 1]
                x_star_obj_y = F_star[:, 0]
                x_star_obj_z = -F_star[:, 3]
            else:
                # Fallback
                x_star_obj_x = [0]
                x_star_obj_y = [1]
                x_star_obj_z = [0]

            fig_3d.add_trace(go.Scatter3d(
                x=x_star_obj_x, y=x_star_obj_y, z=x_star_obj_z, 
                mode='markers',
                marker=dict(size=size_xstar, color='purple'),
                name='Factual Instance x* in Obj Space'
            ))
            
            # 4. Highlights (Global Selection)
            if highlight_indices:
                high_x, high_y, high_z = [], [], []
                
                # Check Observed
                if st.session_state.F_obs is not None:
                    obs_idxs = [i for i in highlight_indices if i < N]
                    if obs_idxs:
                        # Bounds check
                        obs_idxs = [i for i in obs_idxs if i < len(st.session_state.F_obs)]
                        if obs_idxs:
                            high_x.extend(st.session_state.F_obs[obs_idxs, 1])
                            high_y.extend(st.session_state.F_obs[obs_idxs, 0])
                            high_z.extend(-st.session_state.F_obs[obs_idxs, 3])
                
                # Check Pareto
                par_idxs = [i - N for i in highlight_indices if i >= N]
                if par_idxs:
                    # Bounds check
                    par_idxs = [i for i in par_idxs if i < len(F)]
                    if par_idxs:
                        high_x.extend(F[par_idxs, 1])
                        high_y.extend(F[par_idxs, 0])
                        high_z.extend(-F[par_idxs, 3])
                
                if high_x:
                    fig_3d.add_trace(go.Scatter3d(
                        x=high_x, y=high_y, z=high_z,
                        mode='markers',
                        marker=dict(size=10, color='yellow', line=dict(width=5, color='black'), symbol='circle-open'),
                        name='Linked Selection',
                        hoverinfo='skip'
                    ))

            # 5. Table Highlights (Secondary Selection)
            if "table_selection" in st.session_state and st.session_state.table_selection:
                table_indices = list(st.session_state.table_selection)
                tbl_x, tbl_y, tbl_z = [], [], []
                
                # Check Observed
                if st.session_state.F_obs is not None:
                    obs_idxs = [i for i in table_indices if i < N]
                    if obs_idxs:
                        # Bounds check
                        obs_idxs = [i for i in obs_idxs if i < len(st.session_state.F_obs)]
                        if obs_idxs:
                            tbl_x.extend(st.session_state.F_obs[obs_idxs, 1])
                            tbl_y.extend(st.session_state.F_obs[obs_idxs, 0])
                            tbl_z.extend(-st.session_state.F_obs[obs_idxs, 3])
                
                # Check Pareto
                par_idxs = [i - N for i in table_indices if i >= N]
                if par_idxs:
                    # Bounds check
                    par_idxs = [i for i in par_idxs if i < len(F)]
                    if par_idxs:
                        tbl_x.extend(F[par_idxs, 1])
                        tbl_y.extend(F[par_idxs, 0])
                        tbl_z.extend(-F[par_idxs, 3])
                
                if tbl_x:
                    fig_3d.add_trace(go.Scatter3d(
                        x=tbl_x, y=tbl_y, z=tbl_z,
                        mode='markers',
                        marker=dict(size=12, color='cyan', line=dict(width=5, color='blue'), symbol='circle'),
                        name='Table Selection',
                        hoverinfo='skip'
                    ))

            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='similarity/epistemic uncertainty',
                    yaxis_title='prob-based validity',
                    zaxis_title='plausibility/aleotoric uncertainty',
                    zaxis=dict(autorange='reversed')
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=PANEL_HEIGHT,
                autosize=True,
                dragmode='turntable', # Ensure rotation is default
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            # Note: 3D plots don't support selection callbacks in Streamlit.
            # Use the "Point Index" input in the sidebar to select points by hovering to see their index.
            st.plotly_chart(fig_3d, config={'displayModeBar': True}, use_container_width=True)
            st.caption("💡 Hover over points to see their index, then use sidebar 'Point Index' input to select them")
        else:
            st.warning("No solutions found.")
    else:
        st.info("Run Search to view Objective Space.")

# --- Bottom Row: Parallel Coordinates & Details ---
col_par, col_det = st.columns([1, 1])

with col_par:
    st.subheader("Parallel Coordinates Plot")
    
    # Placeholder for chart to ensure top alignment with Details panel
    chart_placeholder = st.empty()
    
    # Switcher for Data Source
    par_coords_source = st.radio("Data Source", ["Pareto Front", "Observed Data", "All"], index=0, horizontal=True)
    
    if st.session_state.cf_results is not None and st.session_state.F_obs is not None:
        F_cf = st.session_state.cf_results.F
        X_cf = st.session_state.cf_results.X
        F_obs = st.session_state.F_obs
        X_obs = st.session_state.data[0]
        N = len(X_obs)
        
        df_list = []
        
        # 1. Observed Data
        if par_coords_source in ["Observed Data", "All"]:
            df_obs = pd.DataFrame(F_obs, columns=['Validity', 'Epistemic', 'Sparsity', 'Aleatoric'])
            df_obs['x1'] = X_obs[:, 0]
            df_obs['x2'] = X_obs[:, 1]
            df_obs['Type'] = 0.0 # 0 for Observed
            df_obs['Global_Index'] = np.arange(N)
            df_list.append(df_obs)
            
        # 2. Pareto Front
        if par_coords_source in ["Pareto Front", "All"]:
            if F_cf is not None and X_cf is not None:
                df_cf = pd.DataFrame(F_cf, columns=['Validity', 'Epistemic', 'Sparsity', 'Aleatoric'])
                df_cf['x1'] = X_cf[:, 0]
                df_cf['x2'] = X_cf[:, 1]
                df_cf['Type'] = 1.0 # 1 for Pareto
                df_cf['Global_Index'] = np.arange(N, N + len(F_cf))
                df_list.append(df_cf)
        
        if df_list:
            df_res = pd.concat(df_list, ignore_index=True)
            
            # Apply Objective Filters
            mask_filters = filter_mask(df_res[['Validity', 'Epistemic', 'Sparsity', 'Aleatoric']].values, obj_filters)
            if mask_filters is not None:
                df_res = df_res[mask_filters]
            
            # Filter based on selection
            highlight_indices = list(st.session_state.global_indices)
            if highlight_indices:
                # Filter by Global_Index
                df_res = df_res[df_res['Global_Index'].isin(highlight_indices)]

            if not df_res.empty:
                # Dimensions to show
                dims = ['Validity', 'Epistemic', 'Sparsity', 'Aleatoric']
                if par_coords_source == "All":
                    dims.append('Type')
                
                fig_par = px.parallel_coordinates(
                    df_res, 
                    color="Validity", 
                    dimensions=dims,
                    labels={"Validity": "Validity", "Epistemic": "Epistemic", "Sparsity": "Sparsity", "Aleatoric": "Aleatoric", "Type": "Type (0=Obs, 1=CF)"},
                    color_continuous_scale=px.colors.diverging.Tealrose,
                    color_continuous_midpoint=0.5
                )
                fig_par.update_layout(margin=dict(l=40, r=40, b=20, t=40), height=PANEL_HEIGHT)
                chart_placeholder.plotly_chart(fig_par, use_container_width=True)
            else:
                chart_placeholder.info("No points selected for the chosen source.")
        else:
             chart_placeholder.warning("No data available for the chosen source.")

    else:
        chart_placeholder.info("Run Search to view Parallel Coordinates.")

with col_det:
    st.subheader("Details & Inspector")
    
    # Use Global Selection
    highlight_indices = list(st.session_state.global_indices)
    
    if not highlight_indices:
        st.info("Select points in Design or Objective space to view details.")
    else:
        rows = []
        N = len(st.session_state.data[0]) if st.session_state.data else 0
        
        # 1. Observed Points
        obs_indices = [i for i in highlight_indices if i < N]
        if obs_indices and st.session_state.data is not None:
             X_obs = st.session_state.data[0]
             F_obs = st.session_state.F_obs
             for idx in obs_indices:
                 if idx < len(X_obs):
                     row = {"Type": "Observed", "Index": idx, "x1": X_obs[idx, 0], "x2": X_obs[idx, 1]}
                     if F_obs is not None and idx < len(F_obs):
                         row.update({"Validity": F_obs[idx, 0], "Epistemic": F_obs[idx, 1], "Sparsity": F_obs[idx, 2], "Aleatoric": F_obs[idx, 3]})
                     rows.append(row)

        # 2. Pareto Points
        if st.session_state.cf_results is not None:
            X_cf = st.session_state.cf_results.X
            F_cf = st.session_state.cf_results.F
            if X_cf is not None and F_cf is not None:
                par_indices = [i - N for i in highlight_indices if i >= N]
                for idx in par_indices:
                    if 0 <= idx < len(X_cf):
                        row = {"Type": "Pareto", "Index": idx + N, "x1": X_cf[idx, 0], "x2": X_cf[idx, 1]}
                        row.update({"Validity": F_cf[idx, 0], "Epistemic": F_cf[idx, 1], "Sparsity": F_cf[idx, 2], "Aleatoric": F_cf[idx, 3]})
                        rows.append(row)
        
        if rows:
            df_details = pd.DataFrame(rows)
            
            # Apply Objective Filters to Details
            # We need to check if the row values satisfy the filters
            # Columns in df_details are full names: Validity, Epistemic, Sparsity, Aleatoric
            
            mask_details = np.ones(len(df_details), dtype=bool)
            for name, val in obj_filters.items():
                if name in df_details.columns:
                    if isinstance(val, list):
                        # Discrete filtering
                        mask_details &= df_details[name].isin(val)
                    else:
                        # Range filtering
                        min_v, max_v = val
                        mask_details &= (df_details[name] >= min_v) & (df_details[name] <= max_v)
            
            df_details = df_details[mask_details]

            # Rename columns for brevity
            rename_map = {
                "Validity": "Val", 
                "Epistemic": "Epi", 
                "Sparsity": "Spar", 
                "Aleatoric": "Ale"
            }
            df_details.rename(columns=rename_map, inplace=True)
            
            # Columns to show (Reduced set)
            cols = ["Index", "Type", "x1", "x2", "Val", "Epi", "Spar", "Ale"]
            # Filter if they exist
            cols = [c for c in cols if c in df_details.columns]
            
            # Display with selection
            selection = st.dataframe(
                df_details[cols].style.format("{:.4f}", subset=[c for c in cols if c not in ["Type", "Index"]]), 
                use_container_width=True,
                on_select="rerun",
                selection_mode="multi-row",
                key="details_table_view",
                hide_index=True,
                height=PANEL_HEIGHT
            )

            # Handle table selection
            if len(selection.selection.rows) > 0:
                # Map selected rows back to global indices
                # Note: selection.selection.rows are indices into the *displayed* dataframe (df_details[cols])
                # We need to get the "Index" column from the corresponding rows
                selected_indices = df_details.iloc[selection.selection.rows]["Index"].values.tolist()
                
                # Update TABLE selection state (Secondary Selection)
                new_set = set(selected_indices)
                if "table_selection" not in st.session_state or new_set != st.session_state.table_selection:
                    st.session_state.table_selection = new_set
                    st.rerun()
            else:
                # Clear table selection if nothing selected
                if "table_selection" in st.session_state and st.session_state.table_selection:
                    st.session_state.table_selection = set()
                    st.rerun()
        else:
            st.warning("Selected indices out of range.")

