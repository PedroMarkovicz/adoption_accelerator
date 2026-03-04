"""
Diagnostic utilities for model interpretability.

Error analysis, calibration assessment, and misclassification profiling
for the final selected model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def analyze_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    X: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Profile misclassified samples and compute error statistics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : ndarray | None
        Predicted probabilities (n_samples, n_classes).
    X : ndarray | None
        Feature matrix (used for feature-value profiling).
    feature_names : list[str] | None
        Feature names for X columns.

    Returns
    -------
    dict
        Error profile with keys:

        * ``n_total`` -- total number of samples
        * ``n_errors`` -- number of misclassified samples
        * ``error_rate`` -- fraction misclassified
        * ``confusion_pairs`` -- sorted list of ``(true, pred, count)``
        * ``direction_counts`` -- counts of over-prediction vs under-prediction
        * ``mean_abs_error`` -- mean |true - pred|
        * ``error_indices`` -- indices of misclassified samples
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    incorrect = y_true != y_pred
    n_total = len(y_true)
    n_errors = int(incorrect.sum())
    error_rate = n_errors / n_total if n_total > 0 else 0.0

    # Confusion pairs (most common misclassifications)
    error_mask = incorrect
    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(y_true[error_mask], y_pred[error_mask]):
        key = (int(t), int(p))
        pairs[key] = pairs.get(key, 0) + 1

    confusion_pairs = sorted(
        [{"true": k[0], "pred": k[1], "count": v} for k, v in pairs.items()],
        key=lambda x: x["count"],
        reverse=True,
    )

    # Direction analysis
    diffs = y_pred[error_mask].astype(int) - y_true[error_mask].astype(int)
    over_predict = int((diffs > 0).sum())
    under_predict = int((diffs < 0).sum())

    direction_counts = {
        "over_predict": over_predict,
        "under_predict": under_predict,
    }

    # Mean absolute error (ordinal distance)
    abs_errors = np.abs(y_true.astype(int) - y_pred.astype(int))
    mean_abs_error = float(abs_errors.mean())

    # Per-class error rate
    per_class_error_rate = {}
    for cls in range(5):
        cls_mask = y_true == cls
        cls_count = int(cls_mask.sum())
        cls_errors = int((y_pred[cls_mask] != cls).sum())
        per_class_error_rate[cls] = {
            "n_samples": cls_count,
            "n_errors": cls_errors,
            "error_rate": round(cls_errors / cls_count, 4) if cls_count > 0 else 0.0,
        }

    # Confidence analysis for errors
    confidence_analysis = None
    if y_proba is not None:
        correct_mask = ~incorrect
        error_conf = y_proba[error_mask].max(axis=1)
        correct_conf = y_proba[correct_mask].max(axis=1)
        confidence_analysis = {
            "mean_confidence_errors": round(float(error_conf.mean()), 4),
            "mean_confidence_correct": round(float(correct_conf.mean()), 4),
            "median_confidence_errors": round(float(np.median(error_conf)), 4),
            "median_confidence_correct": round(float(np.median(correct_conf)), 4),
        }

    return {
        "n_total": n_total,
        "n_errors": n_errors,
        "error_rate": round(error_rate, 4),
        "confusion_pairs": confusion_pairs,
        "direction_counts": direction_counts,
        "mean_abs_error": round(mean_abs_error, 4),
        "per_class_error_rate": per_class_error_rate,
        "confidence_analysis": confidence_analysis,
        "error_indices": np.where(incorrect)[0].tolist(),
    }


def compute_calibration(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute per-class calibration data (reliability diagram inputs).

    For each class, bins the predicted probability and computes the
    observed frequency of that class within each bin.

    Parameters
    ----------
    y_true : array-like
        True labels (0..4).
    y_proba : ndarray of shape (n_samples, n_classes)
        Predicted class probabilities.
    n_bins : int
        Number of probability bins.

    Returns
    -------
    dict
        Per-class calibration data::

            {
              "per_class": {
                0: {"bin_centers": [...], "observed_freq": [...],
                    "mean_predicted": [...], "bin_counts": [...]},
                ...
              },
              "ece": float  (Expected Calibration Error, averaged across classes)
            }
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n_classes = y_proba.shape[1]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    per_class: dict[int, dict[str, Any]] = {}
    ece_components: list[float] = []

    for cls in range(n_classes):
        proba_cls = y_proba[:, cls]
        true_binary = (y_true == cls).astype(float)

        observed_freq = []
        mean_predicted = []
        bin_counts = []

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (proba_cls >= low) & (proba_cls <= high)
            else:
                mask = (proba_cls >= low) & (proba_cls < high)

            n_in_bin = int(mask.sum())
            bin_counts.append(n_in_bin)

            if n_in_bin > 0:
                obs = float(true_binary[mask].mean())
                pred = float(proba_cls[mask].mean())
                observed_freq.append(round(obs, 4))
                mean_predicted.append(round(pred, 4))
            else:
                observed_freq.append(None)
                mean_predicted.append(None)

        # Compute per-class ECE
        cls_ece = 0.0
        total_samples = len(y_true)
        for i in range(n_bins):
            if bin_counts[i] > 0 and observed_freq[i] is not None:
                cls_ece += (bin_counts[i] / total_samples) * abs(
                    observed_freq[i] - mean_predicted[i]
                )
        ece_components.append(cls_ece)

        per_class[cls] = {
            "bin_centers": [round(float(c), 4) for c in bin_centers],
            "observed_freq": observed_freq,
            "mean_predicted": mean_predicted,
            "bin_counts": bin_counts,
        }

    return {
        "per_class": per_class,
        "ece": round(float(np.mean(ece_components)), 6),
        "per_class_ece": {cls: round(v, 6) for cls, v in enumerate(ece_components)},
        "n_bins": n_bins,
    }


def profile_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    shap_values: np.ndarray | None = None,
    X: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    n_cases: int = 10,
) -> list[dict[str, Any]]:
    """Select and annotate the most severe misclassifications.

    Selects samples with the largest ordinal gap (|true - pred|).
    Within ties, selects those with highest confidence on the wrong class.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : ndarray
        Predicted probabilities (n_samples, n_classes).
    shap_values : ndarray | None
        SHAP values for the explained samples.
    X : ndarray | None
        Feature matrix.
    feature_names : list[str] | None
        Feature names.
    n_cases : int
        Number of cases to return.

    Returns
    -------
    list[dict]
        Annotated case dicts with keys: ``index``, ``true_label``,
        ``pred_label``, ``ordinal_gap``, ``confidence``, ``probabilities``,
        and optionally ``top_shap_features``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    incorrect = y_true != y_pred
    if not incorrect.any():
        logger.warning("No misclassified samples found.")
        return []

    error_indices = np.where(incorrect)[0]

    # Score: ordinal gap + confidence on wrong class
    gaps = np.abs(y_true[error_indices].astype(int) - y_pred[error_indices].astype(int))
    confidences = y_proba[error_indices, y_pred[error_indices].astype(int)]

    # Sort by gap (desc), then confidence (desc)
    sort_keys = np.lexsort((confidences, gaps))[::-1]
    selected = error_indices[sort_keys[:n_cases]]

    cases: list[dict[str, Any]] = []
    for idx in selected:
        case: dict[str, Any] = {
            "index": int(idx),
            "true_label": int(y_true[idx]),
            "pred_label": int(y_pred[idx]),
            "ordinal_gap": int(abs(int(y_true[idx]) - int(y_pred[idx]))),
            "confidence": round(float(y_proba[idx].max()), 4),
            "probabilities": [round(float(p), 4) for p in y_proba[idx]],
        }

        # Add top SHAP features for this sample
        if shap_values is not None and feature_names is not None:
            sv = np.asarray(shap_values)
            if sv.ndim == 3:
                # Use SHAP values for the predicted class
                sample_shap = sv[idx, :, int(y_pred[idx])]
            else:
                sample_shap = sv[idx, :]

            top_k = min(5, len(feature_names))
            top_idx = np.argsort(np.abs(sample_shap))[::-1][:top_k]
            case["top_shap_features"] = [
                {
                    "feature": feature_names[i],
                    "shap_value": round(float(sample_shap[i]), 6),
                    "feature_value": (
                        round(float(X[idx, i]), 4) if X is not None else None
                    ),
                }
                for i in top_idx
            ]

        cases.append(case)

    return cases
