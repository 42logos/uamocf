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


def get_design_space_fig(
    torch_model: nn.Module,
    X: Array,
    Y: Optional[Array] = None,
    x_star: Optional[Array] = None,
    pareto_X: Optional[Array] = None,
    grid_resolution: int = 100,
    device=None
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
    X = np.asarray(X)
    
    # 1. Create Grid & Predict
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    est = TorchProbaEstimator(torch_model, device=device)
    est.fit(X, Y) # Fit to get classes
    
    # Predict probabilities for class 1 (assuming binary)
    probas = est.predict_proba(grid_points)
    # Assuming class 1 is the target or interesting one. 
    if probas.shape[1] == 2:
        Z = probas[:, 1].reshape(xx.shape)
    else:
        Z = probas[:, 1].reshape(xx.shape)

    fig = go.Figure()

    # 2. Contour / Heatmap
    fig.add_trace(go.Contour(
        z=Z,
        x=np.linspace(x_min, x_max, grid_resolution),
        y=np.linspace(y_min, y_max, grid_resolution),
        colorscale='RdBu_r', # Diverging Red-Blue (Red=High Prob)
        opacity=0.6,
        contours=dict(
            start=0,
            end=1,
            size=0.1,
            showlines=False
        ),
        name='Probability',
        hoverinfo='skip' # Reduce clutter
    ))
    
    # 3. Decision Boundary (Line at 0.5)
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
            # We need original indices for linking
            indices = np.where(mask)[0]
            
            fig.add_trace(go.Scatter(
                x=X[mask, 0],
                y=X[mask, 1],
                mode='markers',
                marker=dict(
                    size=8,
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
            marker=dict(size=15, color='red', symbol='star'),
            name='x* (Factual)',
            hoverinfo='name+x+y'
        ))

    # 6. Pareto Set
    if pareto_X is not None:
        # Indices for Pareto points? They are new points, so maybe index -1 or separate range
        # We can assign them indices starting from len(X)
        pareto_indices = np.arange(len(X), len(X) + len(pareto_X))
        
        fig.add_trace(go.Scatter(
            x=pareto_X[:, 0],
            y=pareto_X[:, 1],
            mode='markers',
            marker=dict(size=10, color='cyan', symbol='x'),
            name='Pareto Set',
            customdata=pareto_indices,
            hovertemplate='Pareto Idx: %{customdata}<br>x1: %{x:.2f}<br>x2: %{y:.2f}'
        ))

    fig.update_layout(
        xaxis_title="x1",
        yaxis_title="x2",
        margin=dict(l=0, r=0, b=0, t=30),
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig
