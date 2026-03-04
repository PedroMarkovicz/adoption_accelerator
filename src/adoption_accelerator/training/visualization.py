"""
Visualization helpers for the modeling phase.

Generates confusion matrix heatmaps, learning curves, and model
comparison charts.  All functions return matplotlib Figure objects
and optionally save to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, learning_curve

from adoption_accelerator.training.evaluation import compute_qwk

logger = logging.getLogger(__name__)

CLASS_LABELS = [
    "Same day (0)",
    "1-7 days (1)",
    "8-30 days (2)",
    "31-90 days (3)",
    "100+ days (4)",
]


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str] | None = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    class_labels : list[str] | None
        Display labels for each class.
    normalize : bool
        Normalize rows to proportions (recall perspective).
    title : str
        Figure title.
    save_path : str | Path | None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if class_labels is None:
        class_labels = CLASS_LABELS

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


def plot_learning_curve(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    splitter: StratifiedKFold,
    train_sizes: np.ndarray | None = None,
    title: str = "Learning Curve",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Compute and plot learning curves (training size vs. QWK).

    Uses sklearn's ``learning_curve`` with a custom QWK scorer.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from sklearn.metrics import make_scorer

    qwk_scorer = make_scorer(compute_qwk)

    if train_sizes is None:
        train_sizes = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

    train_sizes_abs, train_scores, val_scores = learning_curve(
        clone(model),
        X,
        y,
        cv=splitter,
        scoring=qwk_scorer,
        train_sizes=train_sizes,
        n_jobs=-1,
        random_state=42,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(
        train_sizes_abs,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
        color="#2196F3",
    )
    ax.fill_between(
        train_sizes_abs,
        val_mean - val_std,
        val_mean + val_std,
        alpha=0.15,
        color="#FF9800",
    )
    ax.plot(train_sizes_abs, train_mean, "o-", color="#2196F3", label="Training QWK")
    ax.plot(train_sizes_abs, val_mean, "o-", color="#FF9800", label="Validation QWK")
    ax.set_xlabel("Training Set Size", fontsize=11)
    ax.set_ylabel("QWK", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Learning curve saved to %s", save_path)

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metric_col: str = "mean_qwk_threshold",
    title: str = "Baseline Model Comparison",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Generate a horizontal bar chart comparing models by a metric.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Must contain ``model_name`` and *metric_col*.
    metric_col : str
        Column to plot.
    title : str
        Figure title.
    save_path : str | Path | None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    df = comparison_df.sort_values(metric_col, ascending=True).copy()

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.7)))
    palette = sns.color_palette("viridis", n_colors=len(df))
    bars = ax.barh(df["model_name"], df[metric_col], color=palette, edgecolor="white")

    for bar, val in zip(bars, df[metric_col]):
        ax.text(
            val + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel(metric_col.replace("_", " ").title(), fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Model comparison chart saved to %s", save_path)

    return fig


# ── Optuna visualizations ──────────────────────────────────────────


def plot_optuna_history(
    study: "optuna.Study",
    metric_name: str = "QWK",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Optuna optimization history (best value over trials).

    Parameters
    ----------
    study : optuna.Study
        Completed study.
    metric_name : str
        Y-axis label.
    title : str | None
        Figure title.
    save_path : str | Path | None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import optuna

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trial_numbers = [t.number for t in completed]
    trial_values = [t.value for t in completed]

    # Compute running best
    running_best = []
    best_so_far = float("-inf")
    for v in trial_values:
        best_so_far = max(best_so_far, v)
        running_best.append(best_so_far)

    if title is None:
        title = f"Optuna Optimization History -- {study.study_name}"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        trial_numbers,
        trial_values,
        alpha=0.4,
        s=20,
        color="#90CAF9",
        label="Trial value",
    )
    ax.plot(
        trial_numbers, running_best, color="#1565C0", linewidth=2, label="Best value"
    )
    ax.set_xlabel("Trial Number", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Optuna history plot saved to %s", save_path)

    return fig


def plot_optuna_param_importance(
    study: "optuna.Study",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot hyperparameter importance from an Optuna study.

    Parameters
    ----------
    study : optuna.Study
        Completed study.
    title : str | None
        Figure title.
    save_path : str | Path | None
        If provided, save figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from optuna.importance import get_param_importances

    importances = get_param_importances(study)
    params = list(importances.keys())
    values = list(importances.values())

    if title is None:
        title = f"Hyperparameter Importance -- {study.study_name}"

    fig, ax = plt.subplots(figsize=(9, max(4, len(params) * 0.5)))
    palette = sns.color_palette("viridis", n_colors=len(params))
    ax.barh(params[::-1], values[::-1], color=palette, edgecolor="white")
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Param importance plot saved to %s", save_path)

    return fig


def plot_overfitting_diagnostic(
    train_qwks: list[float],
    val_qwks: list[float],
    model_name: str = "Model",
    title: str | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot train-vs-validation QWK per fold.

    Parameters
    ----------
    train_qwks : list[float]
        Per-fold training QWK.
    val_qwks : list[float]
        Per-fold validation QWK.
    model_name : str
        Model name for labeling.
    title : str | None
        Figure title.
    save_path : str | Path | None
        If provided, save figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if title is None:
        title = f"Overfitting Diagnostic -- {model_name}"

    folds = list(range(len(train_qwks)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(folds, train_qwks, "o-", color="#2196F3", label="Train QWK", linewidth=2)
    ax.plot(folds, val_qwks, "o-", color="#FF9800", label="Validation QWK", linewidth=2)
    ax.fill_between(
        folds,
        train_qwks,
        val_qwks,
        alpha=0.15,
        color="#E57373",
        label="Gap",
    )
    ax.set_xlabel("Fold", fontsize=11)
    ax.set_ylabel("QWK", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Overfitting diagnostic saved to %s", save_path)

    return fig
