"""
Visualization utilities for the interpretability layer.

Generates SHAP summary plots, feature importance bar charts,
modality contribution charts, dependence plots, calibration
curves, and error analysis visualizations.

All functions return matplotlib Figure objects and optionally
save to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)


def _save(fig: plt.Figure, save_path: str | Path | None) -> None:
    """Persist figure if a path is provided."""
    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        logger.info("Figure saved to %s", p)


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    class_index: int | None = None,
    max_display: int = 20,
    title: str = "SHAP Summary (Beeswarm)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : ndarray
        Shape ``(n_samples, n_features)`` or ``(n_samples, n_features, n_classes)``.
    X : ndarray
        Feature matrix corresponding to SHAP values.
    feature_names : list[str]
        Feature names for labeling.
    class_index : int | None
        If multiclass (3-D), select this class for plotting.
    max_display : int
        Maximum number of features to display.
    title : str
        Plot title.
    save_path : str | Path | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import shap

    vals = np.asarray(shap_values)
    if vals.ndim == 3:
        if class_index is not None:
            vals = vals[:, :, class_index]
        else:
            # Average absolute across classes for summary
            vals = np.mean(np.abs(vals), axis=2) * np.sign(np.sum(vals, axis=2))

    fig = plt.figure(figsize=(10, max(6, max_display * 0.35)))
    shap.summary_plot(
        vals,
        X,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()

    _save(fig, save_path)
    return fig


def plot_shap_bar(
    global_importance: pd.DataFrame,
    top_k: int = 30,
    title: str = "Global Feature Importance (mean |SHAP|)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a horizontal bar chart for global feature importance.

    Parameters
    ----------
    global_importance : pd.DataFrame
        Must contain columns ``feature`` and ``mean_abs_shap``.
    top_k : int
        Number of top features to show.
    title : str
        Plot title.
    save_path : str | Path | None
        Optional path to save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = global_importance.head(top_k).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(5, top_k * 0.35)))
    palette = sns.color_palette("viridis", n_colors=len(df))
    ax.barh(df["feature"], df["mean_abs_shap"], color=palette, edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    _save(fig, save_path)
    return fig


def plot_modality_importance(
    modality_importance: dict[str, dict[str, Any]],
    title: str = "Modality Contribution to Predictions",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a donut chart for per-modality importance.

    Parameters
    ----------
    modality_importance : dict
        Keys = modality names, values = dicts with ``total_importance``
        and ``n_features``.
    title : str
        Plot title.
    save_path : str | Path | None
        Optional path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    modalities = list(modality_importance.keys())
    fractions = [modality_importance[m]["total_importance"] for m in modalities]
    n_feats = [modality_importance[m]["n_features"] for m in modalities]

    labels = [
        f"{m}\n({n}F, {f:.1%})" for m, f, n in zip(modalities, fractions, n_feats)
    ]

    colors = sns.color_palette("Set2", n_colors=len(modalities))

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        fractions,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"width": 0.4, "edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(10)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()

    _save(fig, save_path)
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    feature_index: int,
    X: np.ndarray,
    feature_names: list[str],
    class_index: int | None = None,
    interaction_index: str | int | None = "auto",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a SHAP dependence plot for a single feature.

    Parameters
    ----------
    shap_values : ndarray
        Shape ``(n_samples, n_features)`` or ``(n_samples, n_features, n_classes)``.
    feature_index : int
        Index of the feature to plot.
    X : ndarray
        Feature matrix.
    feature_names : list[str]
        Feature names.
    class_index : int | None
        Class slice for multiclass SHAP values.
    interaction_index : str | int | None
        Feature for color coding interaction. ``"auto"`` uses SHAP's
        automatic detection.
    title : str | None
        Plot title (defaults to feature name).
    save_path : str | Path | None
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import shap

    vals = np.asarray(shap_values)
    if vals.ndim == 3:
        cls_idx = class_index if class_index is not None else 0
        vals = vals[:, :, cls_idx]

    fig = plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feature_index,
        vals,
        X,
        feature_names=feature_names,
        interaction_index=interaction_index,
        show=False,
    )
    plot_title = title or f"SHAP Dependence -- {feature_names[feature_index]}"
    plt.title(plot_title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    _save(fig, save_path)
    return fig


def plot_calibration_curve(
    calibration_data: dict[str, Any],
    class_labels: dict[int, str] | None = None,
    title: str = "Calibration Curves (Reliability Diagram)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate per-class reliability diagrams.

    Parameters
    ----------
    calibration_data : dict
        Output of :func:`compute_calibration` with ``per_class`` key.
    class_labels : dict | None
        Maps class index -> display label.
    title : str
        Plot title.
    save_path : str | Path | None
        Optional path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    per_class = calibration_data["per_class"]
    n_classes = len(per_class)

    default_labels = {
        0: "Same day (0)",
        1: "1-7 days (1)",
        2: "8-30 days (2)",
        3: "31-90 days (3)",
        4: "100+ days (4)",
    }
    if class_labels is None:
        class_labels = default_labels

    colors = sns.color_palette("tab10", n_colors=n_classes)

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4), sharey=True)
    if n_classes == 1:
        axes = [axes]

    for cls in range(n_classes):
        ax = axes[cls]
        data = per_class[cls] if cls in per_class else per_class[str(cls)]
        bins = data["bin_centers"]
        obs = data["observed_freq"]
        pred = data["mean_predicted"]

        # Filter out None bins
        valid = [(b, o, p) for b, o, p in zip(bins, obs, pred) if o is not None]
        if valid:
            vb, vo, vp = zip(*valid)
            ax.plot(vp, vo, "o-", color=colors[cls], markersize=5, linewidth=1.5)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1)
        ax.set_title(
            class_labels.get(cls, f"Class {cls}"), fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("Mean predicted", fontsize=9)
        if cls == 0:
            ax.set_ylabel("Observed frequency", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    _save(fig, save_path)
    return fig


def plot_error_analysis(
    error_profile: dict[str, Any],
    class_labels: dict[int, str] | None = None,
    title: str = "Error Analysis",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate error analysis visualizations.

    Creates a 2x2 figure with:

    1. Top confusion pairs (bar chart)
    2. Error direction distribution (pie)
    3. Per-class error rate (bar chart)
    4. Confidence distribution comparison

    Parameters
    ----------
    error_profile : dict
        Output of :func:`analyze_errors`.
    class_labels : dict | None
        Maps class index -> display label.
    title : str
        Overall figure title.
    save_path : str | Path | None
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    default_labels = {
        0: "Same day (0)",
        1: "1-7 days (1)",
        2: "8-30 days (2)",
        3: "31-90 days (3)",
        4: "100+ days (4)",
    }
    if class_labels is None:
        class_labels = default_labels

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel 1: Top confusion pairs ---
    ax = axes[0, 0]
    pairs = error_profile["confusion_pairs"][:10]
    if pairs:
        labels_list = [f"{p['true']}->{p['pred']}" for p in pairs]
        counts = [p["count"] for p in pairs]
        colors = sns.color_palette("Reds_r", n_colors=len(pairs))
        ax.barh(labels_list[::-1], counts[::-1], color=colors[::-1], edgecolor="white")
        ax.set_xlabel("Count", fontsize=10)
        ax.set_title(
            "Top Confusion Pairs (true -> pred)", fontsize=11, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No errors", ha="center", va="center", fontsize=12)
        ax.set_title("Top Confusion Pairs", fontsize=11, fontweight="bold")

    # --- Panel 2: Error direction ---
    ax = axes[0, 1]
    dir_counts = error_profile["direction_counts"]
    if dir_counts["over_predict"] + dir_counts["under_predict"] > 0:
        dir_labels = ["Over-predict\n(pred > true)", "Under-predict\n(pred < true)"]
        dir_values = [dir_counts["over_predict"], dir_counts["under_predict"]]
        ax.pie(
            dir_values,
            labels=dir_labels,
            colors=["#e74c3c", "#3498db"],
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 10},
        )
        ax.set_title("Error Direction Distribution", fontsize=11, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No errors", ha="center", va="center", fontsize=12)
        ax.set_title("Error Direction", fontsize=11, fontweight="bold")

    # --- Panel 3: Per-class error rate ---
    ax = axes[1, 0]
    per_class = error_profile["per_class_error_rate"]
    classes = sorted(per_class.keys())
    er_values = [per_class[c]["error_rate"] for c in classes]
    bar_labels = [class_labels.get(c, f"Class {c}") for c in classes]
    palette = sns.color_palette("viridis", n_colors=len(classes))
    bars = ax.bar(bar_labels, er_values, color=palette, edgecolor="white")
    for bar, val in zip(bars, er_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2%}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("Error Rate", fontsize=10)
    ax.set_title("Per-Class Error Rate", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(er_values) * 1.2 if er_values else 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # --- Panel 4: Confidence comparison ---
    ax = axes[1, 1]
    conf = error_profile.get("confidence_analysis")
    if conf:
        categories = ["Correct", "Errors"]
        means = [conf["mean_confidence_correct"], conf["mean_confidence_errors"]]
        medians = [conf["median_confidence_correct"], conf["median_confidence_errors"]]
        x = np.arange(len(categories))
        w = 0.35
        ax.bar(x - w / 2, means, w, label="Mean", color="#2196F3", edgecolor="white")
        ax.bar(
            x + w / 2, medians, w, label="Median", color="#FF9800", edgecolor="white"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylabel("Max Predicted Probability", fontsize=10)
        ax.set_title("Prediction Confidence", fontsize=11, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No probability data", ha="center", va="center", fontsize=12)
        ax.set_title("Prediction Confidence", fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    _save(fig, save_path)
    return fig


def plot_per_class_shap(
    per_class_importance: dict[str, list[dict[str, Any]]],
    class_labels: dict[int, str] | None = None,
    top_k: int = 10,
    title: str = "Per-Class Top Features (mean SHAP)",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate per-class bar charts for top SHAP features.

    Parameters
    ----------
    per_class_importance : dict
        Output of :func:`compute_per_class_importance`.
    class_labels : dict | None
        Maps class index -> display label.
    top_k : int
        Features per class.
    title : str
        Figure suptitle.
    save_path : str | Path | None
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    default_labels = {
        0: "Same day (0)",
        1: "1-7 days (1)",
        2: "8-30 days (2)",
        3: "31-90 days (3)",
        4: "100+ days (4)",
    }
    if class_labels is None:
        class_labels = default_labels

    n_classes = len(per_class_importance)
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 6), sharey=False)
    if n_classes == 1:
        axes = [axes]

    for i, (cls_key, features) in enumerate(sorted(per_class_importance.items())):
        ax = axes[i]
        feats = features[:top_k]
        if not feats:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        names = [f["feature"] for f in feats][::-1]
        values = [f["mean_shap"] for f in feats][::-1]

        colors = ["#e74c3c" if v > 0 else "#3498db" for v in values]
        ax.barh(names, values, color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(
            class_labels.get(int(cls_key), f"Class {cls_key}"),
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    _save(fig, save_path)
    return fig
