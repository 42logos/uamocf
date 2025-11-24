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
st.sidebar.title("OptiView Pro Controls")

st.sidebar.header("1. Data & Model")
n_samples = st.sidebar.slider("Number of Samples", 100, 2000, 500)
noise_sigma = st.sidebar.slider("Noise Sigma", 0.1, 1.0, 0.4)
seed = st.sidebar.number_input("Random Seed", value=42)
n_epochs = st.sidebar.slider("Epochs", 10, 200, 50)
ensemble_size = st.sidebar.slider("Ensemble Size", 1, 10, 3)

st.sidebar.header("2. Search Parameters")
x_star_x = st.sidebar.number_input("x* (x1)", value=-0.8)
x_star_y = st.sidebar.number_input("x* (x2)", value=-0.7)
pop_size = st.sidebar.slider("Population Size", 20, 200, 50)
n_gen = st.sidebar.slider("Generations", 50, 500, 200)

st.sidebar.header("3. Visualization")
show_sampled = st.sidebar.checkbox("Show Sampled Points", value=True)
show_context = st.sidebar.checkbox("Show All Possible Points (Context)", value=True)
context_opacity = st.sidebar.slider("Context Opacity", 0.01, 0.5, 0.1)
show_linking = st.sidebar.checkbox("Show Linking Line (Simulated)", value=False)

st.sidebar.header("4. Filters")
sparsity_range = None
if st.session_state.cf_results is not None:
    F = st.session_state.cf_results.F
    if F is not None:
        min_sp, max_sp = int(F[:, 2].min()), int(F[:, 2].max())
        if min_sp < max_sp:
            sparsity_range = st.sidebar.slider("Sparsity Range", min_sp, max_sp, (min_sp, max_sp))
        else:
            st.sidebar.write(f"Sparsity: {min_sp}")

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
    with st.spinner("Training Ensemble..."):
        results = training.train_ensemble(
            num_models=ensemble_size,
            X=X,
            y=Y,
            cfg=train_cfg
        )
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
            model=ensemble_model,
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
        
        with st.spinner("Running Optimization..."):
            res = minimize(problem, algorithm, termination, verbose=False)
        
        # Evaluate objectives for all observed points for context
        with st.spinner("Evaluating Context..."):
            F_obs = np.array(problem.evaluate(X))
            st.session_state.F_obs = F_obs

        st.session_state.cf_results = res
        st.success("Search Complete!")

# --- Visualization: Design Space (Top Left) ---
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

with col_design:
    st.subheader("Design Space (Input & ML Model)")
    if st.session_state.models is not None and st.session_state.data is not None:
        X, Y, _ = st.session_state.data
        ensemble_model = models.EnsembleModel(st.session_state.models)
        
        # Get selection from Objective Space
        obj_sel = st.session_state.get("obj_select")
        highlight_indices = get_indices_from_selection(obj_sel)
        
        # Generate Plotly Figure
        fig1 = plotting.get_design_space_fig(
            ensemble_model, 
            X, 
            Y if show_sampled else None, 
            x_star=np.array([x_star_x, x_star_y]),
            pareto_X=st.session_state.cf_results.X if st.session_state.cf_results else None,
            device="cpu"
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
            obs_indices = [i for i in highlight_indices if i < N]
            if obs_indices:
                high_X.extend(X[obs_indices, 0])
                high_Y.extend(X[obs_indices, 1])
            
            # Check Pareto
            if st.session_state.cf_results is not None:
                X_cf = st.session_state.cf_results.X
                if X_cf is not None:
                    par_indices = [i - N for i in highlight_indices if i >= N]
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

        st.plotly_chart(fig1, on_select="rerun", selection_mode=["box", "lasso"], key="design_select")
    else:
        st.info("Initialize System to view Design Space.")

# --- Visualization: Objective Space (Top Right) ---
with col_obj:
    st.subheader("Objective Space (Output & Pareto)")
    if st.session_state.cf_results is not None:
        res = st.session_state.cf_results
        F = res.F # Objectives: 0:Validity, 1:Epistemic, 2:Sparsity, 3:Aleatoric
        X_cf = res.X
        
        # Get selection from Design Space
        design_sel = st.session_state.get("design_select")
        highlight_indices = get_indices_from_selection(design_sel)
        
        if F is not None:
            # Apply Filters
            mask = np.ones(len(F), dtype=bool)
            if sparsity_range:
                mask = (F[:, 2] >= sparsity_range[0]) & (F[:, 2] <= sparsity_range[1])
            
            F_filtered = F[mask]
            
            fig_3d = go.Figure()
            
            # Global Index Offset for Pareto
            N = len(st.session_state.data[0]) if st.session_state.data else 0
            
            # Original indices for ALL Pareto points
            all_pareto_indices = np.arange(N, N + len(F))
            
            # Filtered indices
            filtered_pareto_indices = all_pareto_indices[mask]
            
            # 1. Pareto Front (Filtered)
            fig_3d.add_trace(go.Scatter3d(
                x=F_filtered[:, 1], y=F_filtered[:, 0], z=F_filtered[:, 3],
                mode='markers',
                marker=dict(
                    size=6, 
                    color=F_filtered[:, 2], # Color by Sparsity
                    colorscale='Viridis', 
                    showscale=True,
                    colorbar=dict(title="Sparsity")
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
                
                fig_3d.add_trace(go.Scatter3d(
                    x=F_obs[:, 1], y=F_obs[:, 0], z=F_obs[:, 3],
                    mode='markers',
                    marker=dict(
                        size=3, 
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
                marker=dict(size=10, color='red', symbol='cross'),
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
                autosize=True
            )
            st.plotly_chart(fig_3d, on_select="rerun", selection_mode=["box", "lasso"], key="obj_select")
        else:
            st.warning("No solutions found.")
    else:
        st.info("Run Search to view Objective Space.")

# --- Bottom Row: Parallel Coordinates & Details ---
col_par, col_det = st.columns([2, 1])

with col_par:
    st.subheader("Parallel Coordinates Plot")
    if st.session_state.cf_results is not None:
        F = st.session_state.cf_results.F
        X_cf = st.session_state.cf_results.X
        
        if F is not None and X_cf is not None:
            # Create DataFrame
            df_res = pd.DataFrame(F, columns=['Validity', 'Epistemic', 'Sparsity', 'Aleatoric'])
            df_res['x1'] = X_cf[:, 0]
            df_res['x2'] = X_cf[:, 1]
            
            fig_par = px.parallel_coordinates(
                df_res, 
                color="Validity", 
                labels={"Validity": "Validity", "Epistemic": "Epistemic", "Sparsity": "Sparsity", "Aleatoric": "Aleatoric"},
                color_continuous_scale=px.colors.diverging.Tealrose,
                color_continuous_midpoint=0.5
            )
            st.plotly_chart(fig_par)
        else:
            st.warning("No solutions found.")
    else:
        st.info("Run Search to view Parallel Coordinates.")

with col_det:
    st.subheader("Details & Inspector")
    if st.session_state.cf_results is not None:
        F = st.session_state.cf_results.F
        X_cf = st.session_state.cf_results.X
        if F is not None and X_cf is not None:
            df_res = pd.DataFrame(F, columns=['Validity', 'Epistemic', 'Sparsity', 'Aleatoric'])
            df_res['x1'] = X_cf[:, 0]
            df_res['x2'] = X_cf[:, 1]
            st.dataframe(df_res.style.highlight_min(axis=0), height=400)
    else:
        st.info("Run Search to view Details.")

