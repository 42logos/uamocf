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

from vis_tools import data, models, training, uncertainty, cf_problem, plotting

st.set_page_config(page_title="OptiView Pro", layout="wide")

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
st.sidebar.title("OptiView Pro")

# 1. Interaction (Top Priority)
st.sidebar.header("Interaction")
interaction_mode = st.sidebar.radio("Mode", ["Explore (Pan/Zoom/Orbit)", "Select (Lasso/Box)"], index=0)

if st.sidebar.button("Clear All Selections", type="primary"):
    st.session_state.global_indices = set()
    st.session_state.last_design_select = None
    st.session_state.last_obj_select = None

# 2. System Configuration
with st.sidebar.expander("System Configuration", expanded=False):
    st.subheader("Data Generation")
    n_samples = st.slider("Number of Samples", 100, 2000, 500)
    noise_sigma = st.slider("Noise Sigma", 0.1, 1.0, 0.4)
    seed = st.number_input("Random Seed", value=42)
    
    st.subheader("Model Training")
    n_epochs = st.slider("Epochs", 10, 200, 50)
    ensemble_size = st.slider("Ensemble Size", 1, 10, 3)
    
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
    context_opacity = st.slider("Context Opacity", 0.01, 0.5, 0.1)
    pareto_opacity = st.slider("Pareto Opacity (3D)", 0.1, 1.0, 0.3)

    st.subheader("Point Sizes")
    size_pareto = st.slider("Pareto Points", 1, 20, 10)
    size_context = st.slider("Context Points", 1, 20,10)    
    size_xstar = st.slider("Reference (x*)", 5, 30, 14)
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
                v_min = float(F_combined[:, i].min())
                v_max = float(F_combined[:, i].max())
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
            min_v, max_v = filters[name]
            mask &= (F_data[:, i] >= min_v) & (F_data[:, i] <= max_v)
    return mask

# --- Global Selection Logic ---
if "global_indices" not in st.session_state:
    st.session_state.global_indices = set()
if "last_design_select" not in st.session_state:
    st.session_state.last_design_select = None
if "last_obj_select" not in st.session_state:
    st.session_state.last_obj_select = None



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
        st.session_state.global_indices = set(new_indices)
    st.session_state.last_design_select = curr_design_sel

# Update from Objective Space
curr_obj_sel = st.session_state.get("obj_select")
if curr_obj_sel != st.session_state.last_obj_select:
    new_indices = get_indices_from_selection(curr_obj_sel)
    # Objective space is always point-based (Explore-like) in this setup
    # But if we are in Select mode, maybe we should replace?
    # Let's follow the mode:
    if interaction_mode.startswith("Explore"):
        if new_indices:
            st.session_state.global_indices.update(new_indices)
    else:
        st.session_state.global_indices = set(new_indices)
    st.session_state.last_obj_select = curr_obj_sel

# --- Main Layout ---
st.title("OptiView Pro: Linked Dual-Space Exploration")

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
    
    # 2. Train Model
    train_cfg = training.TrainConfig(
        epochs=n_epochs,
        device="cpu",
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
        X, Y, _ = st.session_state.data
        x_star = torch.tensor([[x_star_x, x_star_y]], dtype=torch.float32)
        
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
            device=torch.device("cpu")
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
            pareto_indices=pareto_indices
        )
        
        # Add Highlights
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
                    visible_obs_indices = [i for i in obs_indices if mask_obs[i]]
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
                            visible_par_indices = [i for i in par_indices if mask_par[i]]
                            par_indices = visible_par_indices
                        
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
                if show_linking:
                    # Simulate linking line by drawing a line to a fixed point or just text
                    pass

        st.plotly_chart(fig1, on_select="rerun", selection_mode=sel_mode_2d, key="design_select", config={'displayModeBar': True})
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
            
            # 1. Pareto Front (Filtered)
            # Scale sizes for 3D (approx 0.3x of 2D size)
            SIZE_3D_SCALE = 0.3
            
            fig_3d.add_trace(go.Scatter3d(
                x=F_filtered[:, 1], y=F_filtered[:, 0], z=F_filtered[:, 3],
                mode='markers',
                marker=dict(
                    size=size_pareto * SIZE_3D_SCALE, 
                    color=F_filtered[:, 2], # Color by Sparsity
                    colorscale='Viridis', 
                    showscale=True,
                    opacity=pareto_opacity,
                    colorbar=dict(
                        title="Sparsity",
                        thickness=15,
                        len=0.5,
                        x=1.1,
                        y=0.5
                    )
                ),
                text=[f"Sparsity: {s:.2f}" for s in F_filtered[:, 2]],
                customdata=filtered_pareto_indices,
                hovertemplate='Pareto Idx: %{customdata}<br>Val: %{y:.2f}<br>Epi: %{x:.2f}<br>Ale: %{z:.2f}',
                name='Pareto Front'
            ))
            
            # 2. Context Cloud (All Possible Points)
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
                        x=F_obs[:, 1], y=F_obs[:, 0], z=F_obs[:, 3],
                        mode='markers',
                        marker=dict(
                            size=size_context * SIZE_3D_SCALE, 
                            color='gray', 
                            opacity=context_opacity
                        ),
                        customdata=obs_indices,
                        hovertemplate='Obs Idx: %{customdata}',
                        name='Context (All Points)'
                    ))
                
            # 3. x* Marker
            fig_3d.add_trace(go.Scatter3d(
                x=[0], y=[1], z=[0], 
                mode='markers',
                marker=dict(size=size_xstar * SIZE_3D_SCALE, color='red', symbol='cross'),
                name='x* (Reference)'
            ))
            
            # 4. Highlights
            if highlight_indices:
                high_x, high_y, high_z = [], [], []
                
                # Check Observed
                if st.session_state.F_obs is not None:
                    obs_idxs = [i for i in highlight_indices if i < N]
                    if obs_idxs:
                        high_x.extend(st.session_state.F_obs[obs_idxs, 1])
                        high_y.extend(st.session_state.F_obs[obs_idxs, 0])
                        high_z.extend(st.session_state.F_obs[obs_idxs, 3])
                
                # Check Pareto
                par_idxs = [i - N for i in highlight_indices if i >= N]
                if par_idxs:
                    high_x.extend(F[par_idxs, 1])
                    high_y.extend(F[par_idxs, 0])
                    high_z.extend(F[par_idxs, 3])
                
                if high_x:
                    fig_3d.add_trace(go.Scatter3d(
                        x=high_x, y=high_y, z=high_z,
                        mode='markers',
                        marker=dict(size=10, color='yellow', line=dict(width=5, color='black'), symbol='circle-open'),
                        name='Linked Selection',
                        hoverinfo='skip'
                    ))

            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='Epistemic Unc.',
                    yaxis_title='Validity (Prob)',
                    zaxis_title='Aleatoric Unc.'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                height=500,
                autosize=True,
                dragmode='orbit', # Ensure rotation is default
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            # 3D plots don't support box/lasso selection well, so we restrict to points to ensure rotation works
            # We only enable box/lasso if explicitly in Select mode AND if we wanted to support it (which is hard in 3D)
            # For now, we keep 3D as points-only to preserve rotation capabilities.
            st.plotly_chart(fig_3d, on_select="rerun", selection_mode=["points"], key="obj_select", config={'displayModeBar': True})
        else:
            st.warning("No solutions found.")
    else:
        st.info("Run Search to view Objective Space.")

# --- Bottom Row: Parallel Coordinates & Details ---
col_par, col_det = st.columns([1, 1])

with col_par:
    st.subheader("Parallel Coordinates Plot")
    
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
                fig_par.update_layout(margin=dict(l=40, r=40, b=20, t=40), height=500)
                st.plotly_chart(fig_par, use_container_width=True)
            else:
                st.info("No points selected for the chosen source.")
        else:
             st.warning("No data available for the chosen source.")

    else:
        st.info("Run Search to view Parallel Coordinates.")

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
                 row = {"Type": "Observed", "Index": idx, "x1": X_obs[idx, 0], "x2": X_obs[idx, 1]}
                 if F_obs is not None:
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
            for name, (min_v, max_v) in obj_filters.items():
                if name in df_details.columns:
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
            cols = ["Type", "x1", "x2", "Val", "Epi", "Spar", "Ale"]
            # Filter if they exist
            cols = [c for c in cols if c in df_details.columns]
            
            st.dataframe(
                df_details[cols].style.format("{:.4f}", subset=[c for c in cols if c not in ["Type", "Index"]]), 
                height=500
            )
        else:
            st.warning("Selected indices out of range.")

