import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from core.cf_problem import Total_uncertainty, aleatoric_from_models, epistemic_from_models
class TorchProbaEstimator(ClassifierMixin, BaseEstimator):
    """
    sklearn-compatible adapter for a trained PyTorch classifier.
    """

    _estimator_type = "classifier"  # <-- Force sklearn to treat it as classifier

    def __init__(self, torch_model, device=None, already_prob=False, batch_size=8192):
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
        """
        Dummy fit: does NOT train torch_model.
        Only sets fitted attrs for sklearn checks.
        """
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]  # sklearn-friendly

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
            xb = X_tensor[i:i+self.batch_size]
            out = self.torch_model(xb)

            if out.ndim == 1:
                out = out.unsqueeze(1)  # (N,1)

            if self.already_prob:
                prob = out
            else:
                # Binary: single logit -> sigmoid
                if out.shape[1] == 1:
                    p1 = torch.sigmoid(out)
                    prob = torch.cat([1 - p1, p1], dim=1)
                # Binary or multiclass: logits -> softmax
                else:
                    prob = torch.softmax(out, dim=1)

            probs_list.append(prob.cpu())

        return torch.cat(probs_list, dim=0).numpy()

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.inspection import DecisionBoundaryDisplay

def plot_proba(
    torch_model,
    X,
    y=None,
    class_of_interest="auto",   # "auto" | None/"max" | int | "all"
    grid_resolution=300,
    padding=0.8,
    levels=100,
    cmap_single="viridis",
    scatter_kwargs=None,
    ax=None,
    device=None,
    already_prob=False,
    multiclass_cmap="auto",     # "auto" | str (cmap name) | list of colors
    class_colors=None,          # NEW: explicit list of colors for each class

    # ---- NEW: coordinate + grid display ----
    show_grid=True,             # 是否显示网格线
    grid_kwargs=None,           # 网格线样式 dict
    xlabel="x0",                # x 轴名
    ylabel="x1",                # y 轴名
    show_ticks=True,            # 是否显示刻度/坐标值
):
    X = np.asarray(X)
    assert X.shape[1] == 2, "X must be shape (N,2)"

    est = TorchProbaEstimator(
        torch_model,
        device=device,
        already_prob=already_prob
    )
    est.fit(X, y)
    n_classes = len(est.classes_)

    # --- Build exact background class palette ---
    if class_colors is not None:
        # User provided explicit colors
        multiclass_colors = list(class_colors)
    elif multiclass_cmap == "auto":
        # Use a better default palette: blue, orange/yellow, red, ...
        # This ensures class 0=blue, class 1=orange, class 2=red for 3-class
        default_colors = [
            '#1f77b4',  # class 0: blue
            '#ff7f0e',  # class 1: orange
            '#d62728',  # class 2: red
            '#2ca02c',  # class 3: green
            '#9467bd',  # class 4: purple
            '#8c564b',  # class 5: brown
            '#e377c2',  # class 6: pink
            '#7f7f7f',  # class 7: gray
            '#bcbd22',  # class 8: olive
            '#17becf',  # class 9: cyan
        ]
        multiclass_colors = default_colors[:n_classes]
    elif isinstance(multiclass_cmap, list):
        multiclass_colors = list(multiclass_cmap)
    else:
        base_cmap = plt.cm.get_cmap(multiclass_cmap, n_classes)
        multiclass_colors = [base_cmap(i) for i in range(n_classes)]

    if scatter_kwargs is None:
        scatter_kwargs = dict(
            s=28, marker="o",
            linewidths=0.9, edgecolor="k",
            alpha=0.9
        )

    # default grid style
    if grid_kwargs is None:
        grid_kwargs = dict(color="0.85", linestyle="--", linewidth=0.8)

    if class_of_interest == "auto":
        class_of_interest = 1 if n_classes == 2 else None

    y_arr = None if y is None else np.asarray(y)

    # helper to apply axis cosmetics
    def _decorate_axis(a):
        a.set_xlabel(xlabel)
        a.set_ylabel(ylabel)
        if show_grid:
            a.grid(True, **grid_kwargs)
        if not show_ticks:
            a.set_xticks([])
            a.set_yticks([])

    # ---- multi-class: each class prob subplot ----
    if class_of_interest == "all" and n_classes > 2:
        if ax is None:
            fig, axes = plt.subplots(
                1, n_classes, figsize=(4.2*n_classes, 3.6),
                constrained_layout=True
            )
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
                vmin=0, vmax=1,
                cmap=cmap_single,
                ax=axes[k],
                grid_resolution=grid_resolution,
                eps=padding,
            )
            axes[k].set_title(f"Class {k} probability")

            # points colored by TRUE class with SAME palette
            if y_arr is not None:
                for kk in range(n_classes):
                    mask = y_arr == est.classes_[kk]
                    axes[k].scatter(
                        X[mask, 0], X[mask, 1],
                        color=multiclass_colors[kk],
                        **scatter_kwargs
                    )

            _decorate_axis(axes[k])

        fig.colorbar(
            disp.surface_, ax=axes, orientation="horizontal",
            fraction=0.05, pad=0.08, label="Probability"
        )
        return fig, axes

    # ---- single plot ----
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.6))
    else:
        fig = ax.figure

    # ---- max-class regions ----
    if class_of_interest is None or class_of_interest == "max":
        disp = DecisionBoundaryDisplay.from_estimator(
            est,
            X,
            response_method="predict_proba",
            class_of_interest=None,
            plot_method="contourf",
            levels=levels,
            vmin=0, vmax=1,
            ax=ax,
            grid_resolution=grid_resolution,
            eps=padding,
            multiclass_colors=multiclass_colors,  # KEY
        )
        ax.set_title("Max-probability class regions")

        # points colored by TRUE class using SAME background colors
        if y_arr is not None:
            for k in range(n_classes):
                mask = y_arr == est.classes_[k]
                ax.scatter(
                    X[mask, 0], X[mask, 1],
                    color=multiclass_colors[k],
                    **scatter_kwargs
                )

        # sklearn-like per-class mini colorbars
        max_cmaps = [s.cmap for s in disp.surface_]
        for k in range(n_classes):
            cax = fig.add_axes([0.78, 0.12 + k*0.06, 0.18, 0.04])
            fig.colorbar(
                cm.ScalarMappable(norm=None, cmap=max_cmaps[k]),
                cax=cax, orientation="horizontal"
            )
            cax.set_title(f"P(class {k})", fontsize=9)

        _decorate_axis(ax)
        return fig, ax

    # ---- specific class probability surface ----
    k = int(class_of_interest)
    disp = DecisionBoundaryDisplay.from_estimator(
        est,
        X,
        response_method="predict_proba",
        class_of_interest=k,
        plot_method="contourf",
        levels=levels,
        vmin=0, vmax=1,
        cmap=cmap_single,
        ax=ax,
        grid_resolution=grid_resolution,
        eps=padding,
    )
    ax.set_title(f"Class {k} probability surface")

    if y_arr is not None:
        for kk in range(n_classes):
            mask = y_arr == est.classes_[kk]
            ax.scatter(
                X[mask, 0], X[mask, 1],
                color=multiclass_colors[kk],
                **scatter_kwargs
            )

    fig.colorbar(disp.surface_, ax=ax, label="Probability")
    _decorate_axis(ax)
    return fig, ax


# Uncertainty Heatmap Plotting
 
from torch import nn
from typing import Sequence, Optional, Union

Array = Union[np.ndarray, torch.Tensor]

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
        Z = aleatoric_from_models(models, grid)
        default_title = "Aleatoric Uncertainty (AU)"
    elif uncertainty_type == "epistemic":
        Z = epistemic_from_models(models, grid)
        default_title = "Epistemic Uncertainty (EU)"
    elif uncertainty_type == "total":
        Z = Total_uncertainty(models, grid)
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


import plotly.graph_objects as go
import pymoo.core.result as result

def plot_cf_3d(
    results: result.Result,
    x_factual: torch.Tensor,
    context: Union[Array, None] = None
):
    
    if not results.problem == None:
        problem = results.problem
    else:
        raise ValueError("Problem not found in results.")
    
    
    X= results.X
    F_X = problem.evaluate(X)
    
    # Get number of objectives
    n_obj = F_X.shape[1] if F_X.ndim > 1 else 1
    if n_obj < 4:
        raise ValueError(f"plot_cf_3d requires at least 4 objectives, but problem has {n_obj}")

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=problem.evaluate(X)[:,1], y=problem.evaluate(X)[:,0], z=-problem.evaluate(X)[:,3],
        mode='markers',
        marker=dict(color='green', size=3, opacity=0.2),
        name='Original observations in Obj Space'
    ))
    
    # add context points if provided
    if context is not None:
        context = np.asarray(context)
        if context.ndim == 1:
            context = context[None, :]  # Make it 2D if single point
        
        # Evaluate all context points
        context_objs = problem.evaluate(context)
        if context_objs.ndim == 1:
            context_objs = context_objs[None, :]  # Make it 2D if single result
        
        fig.add_trace(go.Scatter3d(
            x=context_objs[:, 1],
            y=context_objs[:, 0],
            z=-context_objs[:, 3],
            mode='markers',
            marker=dict(color='green', size=4, opacity=0.2),
            name='Context Points in Obj Space'
        ))
    
    if results.F is None or len(results.F)==0:
        raise ValueError("F not found in results.")
    if results.X is None or len(results.X)==0:
        raise ValueError("X not found in results.")
        
    F_mmo = results.F 
    X_mmo = results.X
    valided_F_mmo = []
    not_valided_F_mmo = []
    
    for i in range(F_mmo.shape[0]):
        x_candidate = X_mmo[i]
        if F_mmo[i,0] <= 0.5:  # validity threshold
            valided_F_mmo.append(F_mmo[i])
        else:
            not_valided_F_mmo.append(F_mmo[i])
    
    
    if len(valided_F_mmo) == 0:
        raise ValueError("No valid counterfactuals found.")
    
    valided_F_mmo = np.array(valided_F_mmo)
    not_valided_F_mmo = np.array(not_valided_F_mmo)

    fig.add_trace(go.Scatter3d(
        x=not_valided_F_mmo[:,1], y=not_valided_F_mmo[:,0], z=-not_valided_F_mmo[:,3],
        mode='markers',
        marker=dict(color='blue', size=
                    5, opacity=1,symbol='cross'),
        name='Pareto Front which not valid'
    ))

    fig.add_trace(go.Scatter3d(
        x=valided_F_mmo[:,1], y=valided_F_mmo[:,0], z=-valided_F_mmo[:,3],
        mode='markers',
        marker=dict(color='red', size=5, symbol='cross'),
        name='Valid Counterfactuals in Obj Space'
    ))


    # add the fixed x_star point - flatten x_factual properly for evaluation

    factual_obj = problem.evaluate(x_factual)
    # Ensure factual_obj is 1D array
    if factual_obj.ndim == 0:
        factual_obj = np.array([factual_obj])
    elif factual_obj.ndim > 1:
        factual_obj = factual_obj.flatten()
    
    fig.add_trace(go.Scatter3d(
        x=[factual_obj[1]],
        y=[factual_obj[0]],
        z=[-factual_obj[3]],
        mode='markers',
        marker=dict(color='purple', size=6),
        name='Factual Instance x* in Obj Space'
    ))
    


    fig.update_layout(
        scene=dict(
            xaxis_title='similarity/epistemic uncertainty',
            yaxis_title='prob-based validity',
            zaxis_title='plausibility/aleotoric uncertainty'
        ),
        width=900, height=700
    )

    # reverse z axis
    fig.update_layout(scene=dict(zaxis=dict(autorange='reversed')))

    fig.show()

    # save to html
    fig.write_html("pareto_front_3d.html")
