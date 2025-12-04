"""
Visualization helpers.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.inspection import DecisionBoundaryDisplay
from torch import nn

import plotly.graph_objects as go
import plotly.express as px

from core.models import EnsembleModel
from core.cf_problem import aleatoric_from_models, epistemic_from_models


Array = np.ndarray


class TorchProbaEstimator(ClassifierMixin, BaseEstimator):
    """
    sklearn-compatible adapter for a trained PyTorch classifier.
    """

    _estimator_type = "classifier"

    def __init__(self, torch_model: nn.Module, device=None, already_prob: bool = False, batch_size: int = 8192):
        self.torch_model = torch_model
        self.device = device
        self.already_prob = already_prob
        self.batch_size = batch_size

        self.classes_ = None
        self.is_fitted_ = False

        self.torch_model.eval()
        if self.device is None:
            self.device = next(torch_model.parameters()).device

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]

        if y is not None:
            self.classes_ = np.unique(y)
        else:
            p = self.predict_proba(X[:4])
            self.classes_ = np.arange(p.shape[1])

        self.is_fitted_ = True
        return self

    @torch.no_grad()
    def predict_proba(self, X):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        X_tensor = torch.from_numpy(X).float().to(self.device)

        probs_list = []
        for i in range(0, len(X_tensor), self.batch_size):
            xb = X_tensor[i : i + self.batch_size]
            out = self.torch_model(xb)

            if out.ndim == 1:
                out = out.unsqueeze(1)

            if self.already_prob:
                prob = out
            else:
                if out.shape[1] == 1:
                    p1 = torch.sigmoid(out)
                    prob = torch.cat([1 - p1, p1], dim=1)
                else:
                    prob = torch.softmax(out, dim=1)

            probs_list.append(prob.cpu())

        return torch.cat(probs_list, dim=0).numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def plot_proba(
    torch_model: nn.Module,
    X: Array,
    y: Optional[Array] = None,
    class_of_interest: Union[str, int, None] = "auto",
    grid_resolution: int = 300,
    padding: float = 0.8,
    levels: int = 100,
    cmap_single: str = "viridis",
    scatter_kwargs: Optional[Dict] = None,
    ax=None,
    device=None,
    already_prob: bool = False,
    multiclass_cmap: str = "tab10",
    show_grid: bool = True,
    grid_kwargs: Optional[Dict] = None,
    xlabel: str = "x0",
    ylabel: str = "x1",
    show_ticks: bool = True,
):
    """
    Plot probability surfaces or class regions for 2D data.
    """
    X = np.asarray(X)
    assert X.shape[1] == 2, "X must be shape (N,2)"

    est = TorchProbaEstimator(
        torch_model,
        device=device,
        already_prob=already_prob,
    )
    est.fit(X, y)
    n_classes = len(est.classes_)

    base_cmap = plt.cm.get_cmap(multiclass_cmap, n_classes)
    multiclass_colors = base_cmap(np.arange(n_classes))

    if scatter_kwargs is None:
        scatter_kwargs = dict(s=28, marker="o", linewidths=0.9, edgecolor="k", alpha=0.9)

    if grid_kwargs is None:
        grid_kwargs = dict(color="0.85", linestyle="--", linewidth=0.8)

    if class_of_interest == "auto":
        class_of_interest = 1 if n_classes == 2 else None

    y_arr = None if y is None else np.asarray(y)

    def _decorate_axis(a):
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        if show_grid:
            a.grid(True, **grid_kwargs)
        if not show_ticks:
            a.set_xticks([])
            a.set_yticks([])

    if class_of_interest == "all" and n_classes > 2:
        if ax is None:
            fig, axes = plt.subplots(1, n_classes, figsize=(4.2 * n_classes, 3.6), constrained_layout=True)
        else:
            axes = ax
            fig = axes[0].figure

        for k in range(n_classes):
            disp = DecisionBoundaryDisplay.from_estimator(
                est,
                X,
                response_method="predict_proba",
                class_of_interest=k,
                plot_method="contourf",
                levels=levels,
                vmin=0,
                vmax=1,
                cmap=cmap_single,
                ax=axes[k],
                grid_resolution=grid_resolution,
                eps=padding,
            )
            axes[k].set_title(f"Class {k} probability")

            if y_arr is not None:
                for kk in range(n_classes):
                    mask = y_arr == est.classes_[kk]
                    axes[k].scatter(X[mask, 0], X[mask, 1], color=multiclass_colors[kk], **scatter_kwargs)

            _decorate_axis(axes[k])

        fig.colorbar(disp.surface_, ax=axes, orientation="horizontal", fraction=0.05, pad=0.08, label="Probability")
        return fig, axes

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6))
    else:
        fig = ax.figure

    if class_of_interest is None or class_of_interest == "max":
        disp = DecisionBoundaryDisplay.from_estimator(
            est,
            X,
            response_method="predict_proba",
            class_of_interest=None,
            plot_method="contourf",
            levels=levels,
            vmin=0,
            vmax=1,
            ax=ax,
            grid_resolution=grid_resolution,
            eps=padding,
            multiclass_colors=multiclass_colors,
        )
        ax.set_title("Max-probability class regions")

        if y_arr is not None:
            for k in range(n_classes):
                mask = y_arr == est.classes_[k]
                ax.scatter(X[mask, 0], X[mask, 1], color=multiclass_colors[k], **scatter_kwargs)

        max_cmaps = [s.cmap for s in disp.surface_]
        for k in range(n_classes):
            cax = fig.add_axes([0.78, 0.12 + k * 0.06, 0.18, 0.04])
            fig.colorbar(cm.ScalarMappable(norm=None, cmap=max_cmaps[k]), cax=cax, orientation="horizontal")
            cax.set_title(f"P(class {k})", fontsize=9)

        _decorate_axis(ax)
        return fig, ax

    k = int(class_of_interest)
    disp = DecisionBoundaryDisplay.from_estimator(
        est,
        X,
        response_method="predict_proba",
        class_of_interest=k,
        plot_method="contourf",
        levels=levels,
        vmin=0,
        vmax=1,
        cmap=cmap_single,
        ax=ax,
        grid_resolution=grid_resolution,
        eps=padding,
    )
    ax.set_title(f"Class {k} probability surface")

    if y_arr is not None:
        for kk in range(n_classes):
            mask = y_arr == est.classes_[kk]
            ax.scatter(X[mask, 0], X[mask, 1], color=multiclass_colors[kk], **scatter_kwargs)

    fig.colorbar(disp.surface_, ax=ax, label="Probability")
    _decorate_axis(ax)
    return fig, ax


def _to_numpy(arr) -> Optional[np.ndarray]:
    """Safely convert tensor or array to numpy for visualization.
    
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


def get_design_space_fig(
    torch_model: nn.Module,
    X: Array,
    Y: Optional[Array] = None,
    x_star: Optional[Array] = None,
    pareto_X: Optional[Array] = None,
    grid_resolution: int = 100,
    device=None,
    sampled_size: int = 8,
    pareto_size: int = 10,
    x_star_size: int = 15,
    dragmode: str = 'zoom',
    x_range: Optional[tuple] = None,
    y_range: Optional[tuple] = None,
    background_type: str = "Probability",
    models: Optional[Sequence[nn.Module]] = None,
    n_contours: int = 20,
    X_indices: Optional[Array] = None,
    pareto_indices: Optional[Array] = None,
    height: int = 500
) -> go.Figure:
    """
    Generates a Plotly figure for the Design Space (Input Space).
    Includes:
    - Probability Heatmap/Contour
    - Decision Boundary
    - Sampled Points (X, Y)
    - x* (Factual)
    - Pareto Set (Counterfactuals)
    """
    # Ensure all inputs are numpy arrays for Streamlit/Plotly compatibility
    X = _to_numpy(X)
    Y = _to_numpy(Y)
    x_star = _to_numpy(x_star)
    pareto_X = _to_numpy(pareto_X)
    
    # Default indices if not provided
    if X_indices is None:
        X_indices = np.arange(len(X))
    else:
        X_indices = _to_numpy(X_indices)
    
    # 1. Create Grid & Predict
    if x_range is not None:
        x_min, x_max = x_range
    else:
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        
    if y_range is not None:
        y_min, y_max = y_range
    else:
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    Z = None
    colorscale = 'RdBu_r'
    zmin, zmax = 0.0, 1.0
    colorbar_title = "Prob"

    if background_type == "Probability":
        est = TorchProbaEstimator(torch_model, device=device)
        est.fit(X, Y) # Fit to get classes
        probas = est.predict_proba(grid_points)
        if probas.shape[1] == 2:
            Z = probas[:, 1].reshape(xx.shape)
        else:
            Z = probas[:, 1].reshape(xx.shape)
        colorscale = 'RdBu_r'
        colorbar_title = "Prob"
        
    elif background_type == "Aleatoric Uncertainty":
        if models is None:
            # Fallback if no ensemble
            Z = np.zeros_like(xx)
        else:
            # Convert to EnsembleModel if needed, ensure on correct device
            if isinstance(models, EnsembleModel):
                ensemble = models
            else:
                ensemble = EnsembleModel(list(models))
            # Move ensemble to device if specified
            if device is not None:
                ensemble = ensemble.to(device)
            # aleatoric_from_models accepts numpy and returns numpy
            au = aleatoric_from_models(ensemble, grid_points.astype(np.float32))
            Z = np.asarray(au).reshape(xx.shape)
        colorscale = 'Viridis'
        zmin, zmax = None, None
        colorbar_title = "AU"
        
    elif background_type == "Epistemic Uncertainty":
        if models is None:
            Z = np.zeros_like(xx)
        else:
            # Convert to EnsembleModel if needed, ensure on correct device
            if isinstance(models, EnsembleModel):
                ensemble = models
            else:
                ensemble = EnsembleModel(list(models))
            # Move ensemble to device if specified
            if device is not None:
                ensemble = ensemble.to(device)
            # epistemic_from_models accepts numpy and returns numpy
            eu = epistemic_from_models(ensemble, grid_points.astype(np.float32))
            Z = np.asarray(eu).reshape(xx.shape)
        colorscale = 'Plasma'
        zmin, zmax = None, None
        colorbar_title = "EU"

    fig = go.Figure()

    # 2. Contour / Heatmap
    if Z is not None and background_type != "None":
        # Calculate contour size based on range and n_contours
        if zmin is not None and zmax is not None:
            c_start, c_end = zmin, zmax
        else:
            c_start, c_end = Z.min(), Z.max()
            
        if c_end > c_start:
            c_size = (c_end - c_start) / n_contours
        else:
            c_size = 0.1

        contour_kwargs = dict(
            z=Z,
            x=np.linspace(x_min, x_max, grid_resolution),
            y=np.linspace(y_min, y_max, grid_resolution),
            colorscale=colorscale,
            opacity=0.6,
            contours=dict(
                start=c_start,
                end=c_end,
                size=c_size,
                showlines=False
            ),
            colorbar=dict(
                title=colorbar_title,
                thickness=15,
                len=0.5,
                x=1.05,
                y=0.5
            ),
            name=background_type,
            hoverinfo='skip'
        )
        
        if zmin is not None:
            contour_kwargs['zmin'] = zmin
            contour_kwargs['zmax'] = zmax
            
        fig.add_trace(go.Contour(**contour_kwargs))
    
    # 3. Decision Boundary (Line at 0.5) - Only for Probability
    if background_type == "Probability" and Z is not None:
        fig.add_trace(go.Contour(
            z=Z,
            x=np.linspace(x_min, x_max, grid_resolution),
            y=np.linspace(y_min, y_max, grid_resolution),
            contours=dict(
                type='constraint',
                operation='=',
                value=0.5,
            ),
            line=dict(color='white', width=3, dash='dash'),
            showscale=False,
            name='Decision Boundary',
            hoverinfo='skip'
        ))

    # 4. Sampled Points
    if Y is not None:
        # Split by class
        for c in np.unique(Y):
            mask = Y == c
            # Use provided indices if available, else calculate relative
            if X_indices is not None:
                indices = X_indices[mask]
            else:
                indices = np.where(mask)[0]
            
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                marker=dict(
                    size=sampled_size,
                    color='orange' if c==1 else 'blue',
                    line=dict(width=1, color='black')
                ),
                name=f'Class {c}',
                customdata=indices, # Store index for linking
                hovertemplate='Index: %{customdata}<br>x1: %{x:.2f}<br>x2: %{y:.2f}'
            ))

    # 5. x* (Factual)
    if x_star is not None:
        fig.add_trace(go.Scatter(
            x=[x_star[0]],
            y=[x_star[1]],
            mode='markers',
            marker=dict(size=x_star_size, color='red', symbol='star'),
            name='x* (Factual)',
            hoverinfo='name+x+y'
        ))

    # 6. Pareto Set
    if pareto_X is not None:
        # Use provided indices if available
        if pareto_indices is not None:
            p_indices = pareto_indices
        else:
            # Fallback: assume appended
            p_indices = np.arange(len(X), len(X) + len(pareto_X))
        
        fig.add_trace(go.Scatter(
            x=pareto_X[:, 0],
            y=pareto_X[:, 1],
            mode='markers',
            marker=dict(size=pareto_size, color='cyan', symbol='x'),
            name='Pareto Set',
            customdata=p_indices,
            hovertemplate='Pareto Idx: %{customdata}<br>x1: %{x:.2f}<br>x2: %{y:.2f}'
        ))

    fig.update_layout(
        xaxis_title="x1",
        yaxis_title="x2",
        margin=dict(l=0, r=0, b=0, t=30),
        height=height,
        dragmode=dragmode,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig
