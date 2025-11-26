"""
Visualization utilities for counterfactual experiments.

Provides plotting functions for:
- Decision boundaries and probability surfaces
- Uncertainty heatmaps
- Pareto fronts (2D and 3D)
- Image counterfactuals (MNIST)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import cm
from matplotlib.lines import Line2D
from sklearn.inspection import DecisionBoundaryDisplay
from torch import nn

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .models import TorchProbaEstimator
from .uncertainty import (
    aleatoric_from_models,
    epistemic_from_models,
    total_uncertainty_ensemble,
)

Array = np.ndarray


# =============================================================================
# Decision Boundary Grid Helper
# =============================================================================

def get_decision_boundary_grid(
    model: nn.Module,
    X: Array,
    grid_resolution: int = 100,
    range_scale: float = 1.1,
    device=None,
    class_of_interest: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get decision boundary grid data for plotting with external libraries (e.g., Plotly).
    
    Args:
        model: Trained PyTorch classifier
        X: Data points, shape (n, 2)
        grid_resolution: Resolution of probability grid
        range_scale: Scale factor for the plot range (default 1.1 = 110% of data range)
        device: PyTorch device
        class_of_interest: Which class probability to compute (default 1)
        
    Returns:
        Tuple of (xx, yy, ZZ) where:
        - xx: 1D array of x coordinates
        - yy: 1D array of y coordinates  
        - ZZ: 2D array of probabilities, shape (len(yy), len(xx))
    """
    X = np.asarray(X)
    assert X.shape[1] == 2, "X must have 2 features"
    
    # Calculate scaled range based on data extent
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_half_range = (x_max - x_min) / 2
    y_half_range = (y_max - y_min) / 2
    
    # Scale by range_scale
    plot_x_min = x_center - x_half_range * range_scale
    plot_x_max = x_center + x_half_range * range_scale
    plot_y_min = y_center - y_half_range * range_scale
    plot_y_max = y_center + y_half_range * range_scale
    
    # Create grid
    xx = np.linspace(plot_x_min, plot_x_max, grid_resolution)
    yy = np.linspace(plot_y_min, plot_y_max, grid_resolution)
    XX, YY = np.meshgrid(xx, yy)
    grid_points = np.c_[XX.ravel(), YY.ravel()].astype(np.float32)
    
    # Get model predictions
    grid_tensor = torch.from_numpy(grid_points)
    if device is not None:
        grid_tensor = grid_tensor.to(device)
    
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(grid_tensor)
        else:
            logits = model(grid_tensor)
            probs = torch.softmax(logits, dim=-1)
        ZZ = probs[:, class_of_interest].cpu().numpy().reshape(XX.shape)
    
    return xx, yy, ZZ


# =============================================================================
# Color Schemes
# =============================================================================

DEFAULT_CLASS_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#d62728',  # red
    '#2ca02c',  # green
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]


def get_class_colors(n_classes: int) -> List[str]:
    """Get a list of colors for n classes."""
    if n_classes <= len(DEFAULT_CLASS_COLORS):
        return DEFAULT_CLASS_COLORS[:n_classes]
    # Fall back to matplotlib colormap for more classes
    cmap = plt.cm.get_cmap('tab20', n_classes)
    return [cmap(i) for i in range(n_classes)]


# =============================================================================
# Probability Surface Plotting
# =============================================================================

def plot_proba(
    model: nn.Module,
    X: Array,
    y: Optional[Array] = None,
    class_of_interest: Union[str, int, None] = "auto",
    grid_resolution: int = 300,
    padding: Optional[float] = None,
    range_scale: float = 1.1,
    levels: int = 100,
    cmap_single: str = "viridis",
    scatter_kwargs: Optional[Dict] = None,
    ax=None,
    device=None,
    already_prob: bool = False,
    class_colors: Optional[List[str]] = None,
    show_grid: bool = True,
    grid_kwargs: Optional[Dict] = None,
    xlabel: str = "x₁",
    ylabel: str = "x₂",
    show_ticks: bool = True,
    title: Optional[str] = None,
):
    """
    Plot probability surface for a 2D classifier.
    
    Args:
        model: Trained PyTorch classifier
        X: Data points, shape (n, 2)
        y: Labels, shape (n,) - used for scatter coloring
        class_of_interest: Which class probability to show
            - "auto": class 1 for binary, max regions for multi-class
            - None/"max": show max-class regions
            - int: show probability for specific class
            - "all": subplot for each class
        grid_resolution: Resolution of probability grid
        padding: DEPRECATED - use range_scale instead. Absolute padding around data range.
        range_scale: Scale factor for the plot range (default 1.1 = 110% of data range).
            The plot will show from center - scale*half_range to center + scale*half_range.
        levels: Number of contour levels
        cmap_single: Colormap for single-class probability
        scatter_kwargs: Additional kwargs for scatter plot
        ax: Matplotlib axes (created if None)
        device: PyTorch device
        already_prob: If True, model outputs probabilities
        class_colors: Custom colors for each class
        show_grid: Whether to show grid lines
        grid_kwargs: Styling for grid lines
        xlabel, ylabel: Axis labels
        show_ticks: Whether to show axis ticks
        title: Plot title
        
    Returns:
        (fig, ax) or (fig, axes) tuple
    """
    X = np.asarray(X)
    assert X.shape[1] == 2, "X must have 2 features"
    
    # Calculate scaled range based on data extent
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_half_range = (x_max - x_min) / 2
    y_half_range = (y_max - y_min) / 2
    
    # Use the larger range for both axes to maintain aspect ratio, scaled by range_scale
    if padding is not None:
        # Legacy mode: use absolute padding
        plot_x_min = x_min - padding
        plot_x_max = x_max + padding
        plot_y_min = y_min - padding
        plot_y_max = y_max + padding
    else:
        # New mode: scale by range_scale (default 1.1 = 110% of data range)
        plot_x_min = x_center - x_half_range * range_scale
        plot_x_max = x_center + x_half_range * range_scale
        plot_y_min = y_center - y_half_range * range_scale
        plot_y_max = y_center + y_half_range * range_scale
    
    # Create estimator
    est = TorchProbaEstimator(model, device=device, already_prob=already_prob)
    est.fit(X, y)
    n_classes = len(est.classes_)
    
    # Get colors
    if class_colors is None:
        class_colors = get_class_colors(n_classes)
    
    # Default scatter style
    if scatter_kwargs is None:
        scatter_kwargs = dict(s=28, marker="o", linewidths=0.9, edgecolor="k", alpha=0.9)
    
    # Default grid style
    if grid_kwargs is None:
        grid_kwargs = dict(color="0.85", linestyle="--", linewidth=0.8)
    
    # Auto-select class of interest
    if class_of_interest == "auto":
        class_of_interest = 1 if n_classes == 2 else None
    
    y_arr = None if y is None else np.asarray(y)
    
    def _decorate_axis(a, t=None):
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        if show_grid:
            a.grid(True, **grid_kwargs)
        if not show_ticks:
            a.set_xticks([])
            a.set_yticks([])
        if t:
            a.set_title(t)
    
    # === Multi-class subplot mode ===
    if class_of_interest == "all" and n_classes > 2:
        if ax is None:
            fig, axes = plt.subplots(
                1, n_classes, 
                figsize=(4.2 * n_classes, 3.6),
                constrained_layout=True
            )
        else:
            axes = ax
            fig = axes[0].figure
        
        # Create grid with scaled range
        xx, yy = np.meshgrid(
            np.linspace(plot_x_min, plot_x_max, grid_resolution),
            np.linspace(plot_y_min, plot_y_max, grid_resolution)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = est.predict_proba(grid)
        
        for k in range(n_classes):
            Z = probs[:, k].reshape(xx.shape)
            contour = axes[k].contourf(xx, yy, Z, levels=levels, vmin=0, vmax=1, cmap=cmap_single)
            
            if y_arr is not None:
                for kk in range(n_classes):
                    mask = y_arr == est.classes_[kk]
                    axes[k].scatter(
                        X[mask, 0], X[mask, 1],
                        color=class_colors[kk],
                        **scatter_kwargs
                    )
            
            axes[k].set_xlim(plot_x_min, plot_x_max)
            axes[k].set_ylim(plot_y_min, plot_y_max)
            _decorate_axis(axes[k], f"P(class {k})")
        
        fig.colorbar(
            contour, ax=axes,
            orientation="horizontal",
            fraction=0.05, pad=0.08,
            label="Probability"
        )
        return fig, axes
    
    # === Single plot ===
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6))
    else:
        fig = ax.figure
    
    # Max-class regions
    if class_of_interest is None or class_of_interest == "max":
        # Get predictions for coloring using scaled range
        xx, yy = np.meshgrid(
            np.linspace(plot_x_min, plot_x_max, grid_resolution),
            np.linspace(plot_y_min, plot_y_max, grid_resolution)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = est.predict_proba(grid)
        Z = probs.argmax(axis=1).reshape(xx.shape)
        
        # Plot filled contours for each class
        for k in range(n_classes):
            mask = Z == k
            ax.contourf(
                xx, yy, mask.astype(float),
                levels=[0.5, 1.5],
                colors=[class_colors[k]],
                alpha=0.3
            )
        
        # Decision boundaries
        ax.contour(xx, yy, Z, levels=range(n_classes), colors='white', linewidths=2)
        
        if y_arr is not None:
            for k in range(n_classes):
                mask = y_arr == est.classes_[k]
                ax.scatter(X[mask, 0], X[mask, 1], color=class_colors[k], **scatter_kwargs)
        
        ax.set_xlim(plot_x_min, plot_x_max)
        ax.set_ylim(plot_y_min, plot_y_max)
        _decorate_axis(ax, title or "Decision Regions")
        return fig, ax
    
    # Specific class probability - use manual grid instead of DecisionBoundaryDisplay
    k = int(class_of_interest)
    
    # Create grid with scaled range
    xx, yy = np.meshgrid(
        np.linspace(plot_x_min, plot_x_max, grid_resolution),
        np.linspace(plot_y_min, plot_y_max, grid_resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = est.predict_proba(grid)
    Z = probs[:, k].reshape(xx.shape)
    
    contour = ax.contourf(xx, yy, Z, levels=levels, vmin=0, vmax=1, cmap=cmap_single)
    
    if y_arr is not None:
        for kk in range(n_classes):
            mask = y_arr == est.classes_[kk]
            ax.scatter(X[mask, 0], X[mask, 1], color=class_colors[kk], **scatter_kwargs)
    
    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_min, plot_y_max)
    fig.colorbar(contour, ax=ax, label="Probability")
    _decorate_axis(ax, title or f"P(class {k})")
    
    return fig, ax


# =============================================================================
# Uncertainty Heatmap Plotting
# =============================================================================

def plot_uncertainty_heatmap(
    models: Sequence[nn.Module],
    X: Array,
    device,
    uncertainty_type: str = "aleatoric",
    grid_resolution: int = 100,
    padding: Optional[float] = None,
    range_scale: float = 1.1,
    cmap: str = "viridis",
    ax=None,
    title: Optional[str] = None,
    xlabel: str = "x₁",
    ylabel: str = "x₂",
    show_colorbar: bool = True,
    cf_points: Optional[Array] = None,
    x_star: Optional[Array] = None,
    levels: int = 20,
):
    """
    Plot uncertainty heatmap from an ensemble.
    
    Args:
        models: Sequence of trained models
        X: Observed data points, shape (n, 2)
        device: PyTorch device
        uncertainty_type: "aleatoric", "epistemic", or "total"
        grid_resolution: Resolution of uncertainty grid
        padding: DEPRECATED - use range_scale instead. Absolute padding around data range.
        range_scale: Scale factor for the plot range (default 1.1 = 110% of data range).
        cmap: Colormap
        ax: Matplotlib axes
        title: Plot title
        xlabel, ylabel: Axis labels
        show_colorbar: Whether to show colorbar
        cf_points: Optional counterfactual points to overlay
        x_star: Optional factual point to mark
        levels: Number of contour levels
        
    Returns:
        (fig, ax) tuple
    """
    X = np.asarray(X)
    
    # Calculate scaled range based on data extent
    x_data_min, x_data_max = X[:, 0].min(), X[:, 0].max()
    y_data_min, y_data_max = X[:, 1].min(), X[:, 1].max()
    
    x_center = (x_data_min + x_data_max) / 2
    y_center = (y_data_min + y_data_max) / 2
    x_half_range = (x_data_max - x_data_min) / 2
    y_half_range = (y_data_max - y_data_min) / 2
    
    if padding is not None:
        # Legacy mode: use absolute padding
        x_min = x_data_min - padding
        x_max = x_data_max + padding
        y_min = y_data_min - padding
        y_max = y_data_max + padding
    else:
        # New mode: scale by range_scale (default 1.1 = 110% of data range)
        x_min = x_center - x_half_range * range_scale
        x_max = x_center + x_half_range * range_scale
        y_min = y_center - y_half_range * range_scale
        y_max = y_center + y_half_range * range_scale
    
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute uncertainty
    if uncertainty_type == "aleatoric":
        Z = aleatoric_from_models(models, grid, device)
        default_title = "Aleatoric Uncertainty (AU)"
    elif uncertainty_type == "epistemic":
        Z = epistemic_from_models(models, grid, device)
        default_title = "Epistemic Uncertainty (EU)"
    elif uncertainty_type == "total":
        Z = total_uncertainty_ensemble(models, grid, device)
        default_title = "Total Uncertainty (TU)"
    else:
        raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")
    
    Z = Z.reshape(xx.shape)
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
    
    cntr = ax.contourf(xx, yy, Z, levels=levels, cmap=cmap)
    
    if show_colorbar:
        fig.colorbar(cntr, ax=ax, label="Uncertainty")
    
    # Overlay points
    if cf_points is not None:
        ax.scatter(
            cf_points[:, 0], cf_points[:, 1],
            c="red", marker="x", s=50, label="CFs"
        )
    
    if x_star is not None:
        ax.scatter(
            x_star[0], x_star[1],
            c="green", marker="*", s=200, label="x*"
        )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or default_title)
    
    if cf_points is not None or x_star is not None:
        ax.legend()
    
    return fig, ax


def plot_uncertainty_comparison(
    models: Sequence[nn.Module],
    X: Array,
    device,
    x_star: Optional[Array] = None,
    cf_points: Optional[Array] = None,
    figsize: Tuple[int, int] = (15, 4),
):
    """
    Plot side-by-side comparison of AU, EU, and TU.
    
    Args:
        models: Ensemble of models
        X: Observed data
        device: PyTorch device
        x_star: Factual point
        cf_points: Counterfactual points
        figsize: Figure size
        
    Returns:
        (fig, axes) tuple
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    for ax, utype in zip(axes, ["aleatoric", "epistemic", "total"]):
        plot_uncertainty_heatmap(
            models, X, device,
            uncertainty_type=utype,
            ax=ax,
            x_star=x_star,
            cf_points=cf_points,
            show_colorbar=True,
        )
    
    plt.tight_layout()
    return fig, axes


# =============================================================================
# Pareto Front Visualization
# =============================================================================

def plot_pareto_front_2d(
    F: Array,
    F_valid: Optional[Array] = None,
    F_factual: Optional[Array] = None,
    objective_names: Optional[List[str]] = None,
    validity_threshold: float = 0.5,
    figsize: Tuple[int, int] = (15, 10),
):
    """
    Plot 2D projections of the Pareto front.
    
    Args:
        F: All objective values, shape (n, n_obj)
        F_valid: Valid counterfactuals only
        F_factual: Factual instance objectives
        objective_names: Names for each objective
        validity_threshold: Threshold line for validity
        figsize: Figure size
        
    Returns:
        (fig, axes) tuple
    """
    n_obj = F.shape[1]
    
    if objective_names is None:
        objective_names = [f"Objective {i}" for i in range(n_obj)]
    
    # Create all pairs of objectives
    pairs = []
    for i in range(n_obj):
        for j in range(i + 1, n_obj):
            pairs.append((i, j))
    
    n_pairs = len(pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (i, j) in enumerate(pairs):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        # All solutions
        ax.scatter(F[:, i], F[:, j], c='blue', alpha=0.5, label='All')
        
        # Valid solutions
        if F_valid is not None and len(F_valid) > 0:
            ax.scatter(
                F_valid[:, i], F_valid[:, j],
                c='red', marker='x', s=100, label='Valid CFs'
            )
        
        # Factual
        if F_factual is not None:
            ax.scatter(
                F_factual[i], F_factual[j],
                c='green', marker='*', s=200, label='Factual'
            )
        
        # Validity threshold line (if first objective is validity)
        if i == 0:
            ax.axhline(y=validity_threshold, color='green', linestyle='--', alpha=0.5)
        if j == 0:
            ax.axvline(x=validity_threshold, color='green', linestyle='--', alpha=0.5)
        
        ax.set_xlabel(objective_names[i])
        ax.set_ylabel(objective_names[j])
        ax.legend(fontsize=8)
    
    # Hide empty subplots
    for idx in range(len(pairs), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    return fig, axes


def plot_pareto_front_3d(
    F: Array,
    F_valid: Optional[Array] = None,
    F_factual: Optional[Array] = None,
    objective_names: Optional[List[str]] = None,
    obj_indices: Tuple[int, int, int] = (0, 1, 2),
    save_html: Optional[str] = None,
):
    """
    Create interactive 3D Pareto front visualization with Plotly.
    
    Args:
        F: All objective values
        F_valid: Valid counterfactuals only
        F_factual: Factual instance objectives
        objective_names: Names for objectives
        obj_indices: Which 3 objectives to plot
        save_html: Path to save HTML file
        
    Returns:
        Plotly Figure
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for 3D visualization")
    
    i, j, k = obj_indices
    
    if objective_names is None:
        objective_names = [f"Obj {x}" for x in range(F.shape[1])]
    
    fig = go.Figure()
    
    # All solutions
    fig.add_trace(go.Scatter3d(
        x=F[:, i], y=F[:, j], z=F[:, k],
        mode='markers',
        marker=dict(color='blue', size=3, opacity=0.3),
        name='All Solutions'
    ))
    
    # Valid solutions
    if F_valid is not None and len(F_valid) > 0:
        fig.add_trace(go.Scatter3d(
            x=F_valid[:, i], y=F_valid[:, j], z=F_valid[:, k],
            mode='markers',
            marker=dict(color='red', size=5, symbol='cross'),
            name='Valid CFs'
        ))
    
    # Factual
    if F_factual is not None:
        fig.add_trace(go.Scatter3d(
            x=[F_factual[i]], y=[F_factual[j]], z=[F_factual[k]],
            mode='markers',
            marker=dict(color='green', size=8, symbol='diamond'),
            name='Factual'
        ))
    
    fig.update_layout(
        scene=dict(
            xaxis_title=objective_names[i],
            yaxis_title=objective_names[j],
            zaxis_title=objective_names[k],
        ),
        width=900,
        height=700,
    )
    
    if save_html:
        fig.write_html(save_html)
    
    return fig


# =============================================================================
# Image Counterfactual Visualization
# =============================================================================

def visualize_image_counterfactuals(
    model: nn.Module,
    x_star_image: Array,
    cf_images: Array,
    target_class: int,
    factual_label: int,
    device,
    img_size: int = 16,
    n_show: int = 8,
    figsize_per_cf: Tuple[float, float] = (2.5, 7),
):
    """
    Visualize factual and counterfactual images with probabilities.
    
    Creates a 3-row figure:
    - Row 1: Images
    - Row 2: Probability bar charts
    - Row 3: Difference images
    
    Args:
        model: Trained classifier
        x_star_image: Factual image
        cf_images: Counterfactual images, shape (n, n_pixels) or (n, h, w)
        target_class: Target class for counterfactual
        factual_label: True label of factual
        device: PyTorch device
        img_size: Image size
        n_show: Maximum number of CFs to show
        figsize_per_cf: Figure size per counterfactual
        
    Returns:
        Matplotlib Figure
    """
    model.eval()
    n_cfs = min(len(cf_images), n_show)
    
    fig, axes = plt.subplots(
        3, n_cfs + 1,
        figsize=(figsize_per_cf[0] * (n_cfs + 1), figsize_per_cf[1])
    )
    
    # Reshape images if needed
    x_star_flat = x_star_image.flatten()
    x_star_2d = x_star_image.reshape(img_size, img_size)
    
    # Get factual predictions
    with torch.no_grad():
        img_tensor = torch.tensor(
            x_star_flat.reshape(1, 1, img_size, img_size),
            dtype=torch.float32
        ).to(device)
        probs_star = nn.Softmax(dim=1)(model(img_tensor)).cpu().numpy()[0]
    
    # Plot factual
    axes[0, 0].imshow(x_star_2d, cmap='gray', vmin=-1, vmax=1)
    axes[0, 0].set_title(f"Factual x*\nTrue: {factual_label}", fontsize=10)
    axes[0, 0].axis('off')
    
    # Probability bar chart for factual
    colors_star = ['green' if j == factual_label else 'blue' for j in range(10)]
    axes[1, 0].bar(range(10), probs_star, color=colors_star, alpha=0.7)
    axes[1, 0].set_xticks(range(10))
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("P(Y|X)")
    axes[1, 0].set_title("Factual probs")
    
    # No difference for factual
    axes[2, 0].imshow(np.zeros((img_size, img_size)), cmap='RdBu', vmin=-1, vmax=1)
    axes[2, 0].set_title("Diff: N/A")
    axes[2, 0].axis('off')
    
    # Plot counterfactuals
    for i in range(n_cfs):
        cf_flat = cf_images[i].flatten()
        cf_2d = cf_flat.reshape(img_size, img_size)
        
        # Get CF predictions
        with torch.no_grad():
            cf_tensor = torch.tensor(
                cf_flat.reshape(1, 1, img_size, img_size),
                dtype=torch.float32
            ).to(device)
            probs_cf = nn.Softmax(dim=1)(model(cf_tensor)).cpu().numpy()[0]
        
        pred_class = np.argmax(probs_cf)
        
        # CF image
        axes[0, i + 1].imshow(cf_2d, cmap='gray', vmin=-1, vmax=1)
        axes[0, i + 1].set_title(f"CF {i+1}\nPred: {pred_class}", fontsize=9)
        axes[0, i + 1].axis('off')
        
        # Probability bar chart
        colors_cf = ['red' if j == target_class else 'blue' for j in range(10)]
        axes[1, i + 1].bar(range(10), probs_cf, color=colors_cf, alpha=0.7)
        axes[1, i + 1].set_xticks(range(10))
        axes[1, i + 1].set_ylim(0, 1)
        axes[1, i + 1].set_title(f"P(target)={probs_cf[target_class]:.2f}")
        
        # Difference image
        diff = cf_2d - x_star_2d
        axes[2, i + 1].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
        l2 = np.linalg.norm(diff)
        l0 = np.sum(np.abs(diff) > 0.05)
        axes[2, i + 1].set_title(f"L2={l2:.1f}, L0={l0}")
        axes[2, i + 1].axis('off')
    
    plt.suptitle(
        f"MNIST Image-Space Counterfactuals\n"
        f"Factual: {factual_label} → Target: {target_class}",
        fontsize=12
    )
    plt.tight_layout()
    
    return fig


def plot_ensemble_decision_boundaries(
    models: Sequence[nn.Module],
    X: Array,
    y: Array,
    device,
    grid_resolution: int = 300,
    figsize: Tuple[int, int] = (12, 9),
    show_ensemble_mean: bool = True,
):
    """
    Plot decision boundaries for all models in an ensemble.
    
    Args:
        models: Sequence of trained models
        X: Data points
        y: Labels
        device: PyTorch device
        grid_resolution: Resolution of decision boundary
        figsize: Figure size
        show_ensemble_mean: Whether to show mean prediction background
        
    Returns:
        Matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, grid_resolution),
        np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, grid_resolution)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)
    
    # Detect number of classes
    with torch.no_grad():
        sample_out = models[0](grid_tensor[:1])
        n_classes = sample_out.shape[1]
    
    # Model colors
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    class_colors = get_class_colors(n_classes)
    
    # Plot each model's boundary
    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            Z = model(grid_tensor).cpu().numpy()
        Z_class = np.argmax(Z, axis=1).reshape(xx.shape)
        
        for c in range(n_classes - 1):
            ax.contour(
                xx, yy, Z_class,
                levels=[c + 0.5],
                colors=[model_colors[i]],
                alpha=0.6,
                linewidths=1.2,
            )
    
    # Ensemble mean background
    if show_ensemble_mean:
        mean_probs = np.zeros((grid.shape[0], n_classes))
        for model in models:
            model.eval()
            with torch.no_grad():
                probs = nn.Softmax(dim=1)(model(grid_tensor)).cpu().numpy()
            mean_probs += probs
        mean_probs /= len(models)
        
        Z_ensemble = np.argmax(mean_probs, axis=1).reshape(xx.shape)
        ax.contourf(
            xx, yy, Z_ensemble,
            levels=np.arange(-0.5, n_classes, 1),
            colors=class_colors,
            alpha=0.15
        )
    
    # Scatter data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=10, alpha=0.5)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color=model_colors[i], linewidth=2, label=f'Model {i+1}')
        for i in range(len(models))
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"Decision Boundaries of {len(models)} Ensemble Models")
    
    plt.tight_layout()
    return fig
