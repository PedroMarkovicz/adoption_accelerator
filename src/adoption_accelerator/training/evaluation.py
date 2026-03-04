"""
Evaluation utilities for the modeling phase.

Provides QWK computation, threshold optimization on expected values,
comprehensive classification metrics, and helpers for downstream
diagnostic analyses.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ── Core metric ─────────────────────────────────────────────────────


def compute_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Quadratic Weighted Kappa between true and predicted labels."""
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))


# ── Threshold optimization ──────────────────────────────────────────


def _expected_value(y_proba: np.ndarray) -> np.ndarray:
    """Compute E[Y] = sum(i * P(Y=i)) for each sample."""
    classes = np.arange(y_proba.shape[1])
    return y_proba @ classes


def _apply_boundaries(ev: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    """Map expected values to class predictions using sorted boundaries."""
    return np.digitize(ev, sorted(boundaries))


def optimize_thresholds(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_classes: int = 5,
    metric_fn: Callable | None = None,
) -> dict[str, Any]:
    """Optimize 4 threshold boundaries on the expected value to maximize QWK.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels (0..4).
    y_proba : ndarray of shape (n_samples, n_classes)
        Predicted class probabilities.
    n_classes : int
        Number of classes.
    metric_fn : callable | None
        Scoring function ``(y_true, y_pred) -> float``.
        Defaults to :func:`compute_qwk`.

    Returns
    -------
    dict
        ``{"thresholds": list[float], "qwk_optimized": float, "qwk_argmax": float}``
    """
    if metric_fn is None:
        metric_fn = compute_qwk

    y_true = np.asarray(y_true)
    ev = _expected_value(y_proba)

    # Baseline: argmax
    y_pred_argmax = np.argmax(y_proba, axis=1)
    qwk_argmax = metric_fn(y_true, y_pred_argmax)

    # Initial boundaries: evenly spaced
    init_bounds = np.array([0.5, 1.5, 2.5, 3.5])

    def _neg_metric(boundaries):
        preds = _apply_boundaries(ev, boundaries)
        preds = np.clip(preds, 0, n_classes - 1)
        return -metric_fn(y_true, preds)

    result = minimize(
        _neg_metric,
        x0=init_bounds,
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-6, "fatol": 1e-6},
    )
    opt_boundaries = np.sort(result.x)
    y_pred_opt = np.clip(_apply_boundaries(ev, opt_boundaries), 0, n_classes - 1)
    qwk_optimized = metric_fn(y_true, y_pred_opt)

    return {
        "thresholds": opt_boundaries.tolist(),
        "qwk_optimized": float(qwk_optimized),
        "qwk_argmax": float(qwk_argmax),
    }


def apply_thresholds(
    y_proba: np.ndarray,
    thresholds: list[float] | np.ndarray,
    n_classes: int = 5,
) -> np.ndarray:
    """Convert probabilities to class predictions via threshold boundaries.

    Parameters
    ----------
    y_proba : ndarray of shape (n_samples, n_classes)
        Predicted class probabilities.
    thresholds : list[float]
        Sorted boundary values (length = n_classes - 1).
    n_classes : int
        Number of classes.

    Returns
    -------
    np.ndarray
        Integer class predictions.
    """
    ev = _expected_value(y_proba)
    preds = _apply_boundaries(ev, np.array(thresholds))
    return np.clip(preds, 0, n_classes - 1).astype(int)


# ── Comprehensive metrics ───────────────────────────────────────────


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute a full metrics dictionary for multiclass classification.

    Returned keys: ``qwk``, ``accuracy``, ``macro_f1``, ``weighted_f1``,
    ``per_class_precision``, ``per_class_recall``, ``per_class_f1``,
    ``confusion_matrix``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(5))

    metrics: dict[str, Any] = {
        "qwk": compute_qwk(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "per_class_precision": precision_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        ).tolist(),
        "per_class_recall": recall_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        ).tolist(),
        "per_class_f1": f1_score(
            y_true, y_pred, average=None, labels=labels, zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return metrics


# ── Overfitting diagnostic ──────────────────────────────────────────


def compute_overfitting_diagnostic(
    train_qwks: list[float],
    val_qwks: list[float],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """Compute train-vs-validation metric gaps per fold and in aggregate.

    Parameters
    ----------
    train_qwks : list[float]
        Per-fold training QWK scores.
    val_qwks : list[float]
        Per-fold validation QWK scores.
    threshold : float
        Gap above which overfitting is flagged.

    Returns
    -------
    dict
        Diagnostic summary with ``mean_gap``, ``per_fold_gaps``,
        ``overfitting_flag``.
    """
    gaps = [t - v for t, v in zip(train_qwks, val_qwks)]
    mean_gap = float(np.mean(gaps))
    return {
        "mean_train_qwk": float(np.mean(train_qwks)),
        "mean_val_qwk": float(np.mean(val_qwks)),
        "mean_gap": mean_gap,
        "per_fold_gaps": gaps,
        "overfitting_flag": mean_gap > threshold,
        "threshold": threshold,
    }


# ── Statistical significance ───────────────────────────────────────


def test_statistical_significance(
    metrics_a: list[float],
    metrics_b: list[float],
    test_type: str = "t-test",
) -> dict[str, Any]:
    """Perform a paired statistical test on per-fold metric arrays.

    Parameters
    ----------
    metrics_a : list[float]
        Per-fold metrics for model A.
    metrics_b : list[float]
        Per-fold metrics for model B.
    test_type : str
        ``"t-test"`` or ``"wilcoxon"``.

    Returns
    -------
    dict
        ``{"test_type", "statistic", "p_value", "significant_at_005"}``
    """
    from scipy.stats import ttest_rel, wilcoxon

    a = np.array(metrics_a)
    b = np.array(metrics_b)

    if test_type == "t-test":
        stat, p = ttest_rel(a, b)
    elif test_type == "wilcoxon":
        stat, p = wilcoxon(a, b)
    else:
        raise ValueError(
            f"Unknown test_type '{test_type}'. Use 't-test' or 'wilcoxon'."
        )

    return {
        "test_type": test_type,
        "statistic": float(stat),
        "p_value": float(p),
        "significant_at_005": bool(p < 0.05),
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "diff_mean": float(np.mean(a) - np.mean(b)),
    }


# ── Ablation study helper ──────────────────────────────────────────


def run_ablation_study(
    model_family: str,
    model_params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    modality_columns: dict[str, list[int]],
) -> dict[str, dict[str, Any]]:
    """Run CV with subsets of features defined by modality.

    Parameters
    ----------
    model_family : str
        Model family name for :func:`create_model`.
    model_params : dict
        Hyperparameters for the model.
    X : ndarray
        Full feature matrix.
    y : ndarray
        Target vector.
    folds : list[tuple[ndarray, ndarray]]
        Pre-computed fold indices.
    modality_columns : dict[str, list[int]]
        Maps modality names to column index lists.

    Returns
    -------
    dict[str, dict]
        Per-ablation metrics keyed by ablation name.
    """
    from adoption_accelerator.training.model_factory import create_model

    results: dict[str, dict[str, Any]] = {}

    for ablation_name, col_indices in modality_columns.items():
        X_sub = X[:, col_indices]
        fold_qwks: list[float] = []

        for train_idx, val_idx in folds:
            model = create_model(model_family, model_params.copy())
            model.fit(X_sub[train_idx], y[train_idx])
            proba = model.predict_proba(X_sub[val_idx])

            if proba.shape[1] < 5:
                full_proba = np.zeros((proba.shape[0], 5))
                for i, cls in enumerate(model.classes_):
                    full_proba[:, int(cls)] = proba[:, i]
                proba = full_proba

            thresh = optimize_thresholds(y[val_idx], proba)
            fold_qwks.append(thresh["qwk_optimized"])

        results[ablation_name] = {
            "mean_qwk": float(np.mean(fold_qwks)),
            "std_qwk": float(np.std(fold_qwks)),
            "fold_qwks": fold_qwks,
            "n_features": len(col_indices),
        }
        logger.info(
            "Ablation '%s': %d features, QWK=%.4f",
            ablation_name,
            len(col_indices),
            results[ablation_name]["mean_qwk"],
        )

    return results
