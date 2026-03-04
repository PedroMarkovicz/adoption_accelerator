"""
Diagnostic utilities for inference pipeline validation.

These are analysis tools used during pipeline validation (notebook 14)
and CI testing -- they are NOT invoked during production inference.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from adoption_accelerator.inference.predictor import (
    compute_expected_values,
    predict_probabilities,
)

logger = logging.getLogger(__name__)


def compute_prediction_diagnostics(
    predicted_classes: np.ndarray,
    probabilities: np.ndarray,
    training_distribution: dict[int, float] | pd.Series | None = None,
    low_confidence_threshold: float = 0.30,
) -> dict[str, Any]:
    """Compute prediction distribution, confidence statistics, and comparisons.

    Parameters
    ----------
    predicted_classes : ndarray
        Integer class predictions (0-4).
    probabilities : ndarray
        Class probability matrix.
    training_distribution : dict or Series or None
        Training set class distribution (counts or proportions).
    low_confidence_threshold : float
        Confidence floor for low-confidence flagging.

    Returns
    -------
    dict
        Structured diagnostics report.
    """
    n_samples = len(predicted_classes)
    confidence = probabilities.max(axis=1)

    # Prediction distribution
    unique, counts = np.unique(predicted_classes, return_counts=True)
    pred_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    pred_pct = {
        int(k): round(float(v) / n_samples * 100, 2) for k, v in zip(unique, counts)
    }

    # Confidence stats
    low_conf_mask = confidence < low_confidence_threshold
    low_conf_count = int(low_conf_mask.sum())

    diag: dict[str, Any] = {
        "n_samples": n_samples,
        "prediction_distribution": pred_dist,
        "prediction_distribution_pct": pred_pct,
        "confidence": {
            "mean": float(np.mean(confidence)),
            "median": float(np.median(confidence)),
            "min": float(np.min(confidence)),
            "max": float(np.max(confidence)),
            "std": float(np.std(confidence)),
        },
        "low_confidence_count": low_conf_count,
        "low_confidence_pct": round(low_conf_count / n_samples * 100, 2),
    }

    # Sanity checks
    empty_classes = [c for c in range(5) if c not in pred_dist]
    dominant_classes = [c for c, pct in pred_pct.items() if pct > 50.0]
    diag["warnings"] = {
        "empty_classes": empty_classes,
        "dominant_classes": dominant_classes,
    }

    # Compare against training distribution
    if training_distribution is not None:
        if isinstance(training_distribution, pd.Series):
            train_dist = training_distribution.value_counts(normalize=True).sort_index()
            train_pct = {
                int(k): round(float(v) * 100, 2) for k, v in train_dist.items()
            }
        else:
            total = sum(training_distribution.values())
            train_pct = {
                int(k): round(float(v) / total * 100, 2)
                for k, v in training_distribution.items()
            }
        diag["training_distribution_pct"] = train_pct

    return diag


def run_degradation_test(
    model,
    X: pd.DataFrame,
    feature_schema: dict[str, Any],
    modality_columns: dict[str, list[str]],
    thresholds: list[float],
    original_predictions: np.ndarray,
    n_classes: int = 5,
) -> dict[str, Any]:
    """Simulate missing-modality by zeroing out columns and comparing predictions.

    Parameters
    ----------
    model : fitted estimator
    X : pd.DataFrame
        Full feature matrix.
    feature_schema : dict
        Model bundle feature schema.
    modality_columns : dict
        Mapping of modality name to list of column names.
    thresholds : list[float]
        Optimized threshold boundaries.
    original_predictions : ndarray
        Predictions from full-feature input.
    n_classes : int
        Number of classes.

    Returns
    -------
    dict
        Per-modality degradation results.
    """
    from adoption_accelerator.training.evaluation import apply_thresholds as _apply

    results: dict[str, Any] = {}

    for modality_name, cols in modality_columns.items():
        # Filter to columns that exist in X
        valid_cols = [c for c in cols if c in X.columns]
        if not valid_cols:
            results[modality_name] = {"skipped": True, "reason": "no matching columns"}
            continue

        X_degraded = X.copy()
        X_degraded[valid_cols] = 0.0

        try:
            proba_degraded = predict_probabilities(
                model, X_degraded, n_classes=n_classes
            )
            preds_degraded = _apply(proba_degraded, thresholds, n_classes=n_classes)

            changed_mask = preds_degraded != original_predictions
            change_pct = float(changed_mask.mean()) * 100
            mean_shift = float(np.abs(preds_degraded - original_predictions).mean())

            results[modality_name] = {
                "n_columns_zeroed": len(valid_cols),
                "prediction_change_pct": round(change_pct, 2),
                "mean_class_shift": round(mean_shift, 4),
                "completed": True,
            }
            logger.info(
                "Degradation [%s]: zeroed %d cols, %.1f%% preds changed, mean shift=%.4f",
                modality_name,
                len(valid_cols),
                change_pct,
                mean_shift,
            )
        except Exception as exc:
            results[modality_name] = {
                "completed": False,
                "error": str(exc),
            }
            logger.warning("Degradation test [%s] failed: %s", modality_name, exc)

    return results


def verify_reproducibility(
    model,
    X,
    thresholds: list[float],
    first_predictions: np.ndarray,
    n_classes: int = 5,
) -> dict[str, Any]:
    """Re-run prediction and assert bit-identical output.

    Returns
    -------
    dict
        ``{"passed": bool, "detail": str}``
    """
    from adoption_accelerator.training.evaluation import apply_thresholds as _apply

    proba = predict_probabilities(model, X, n_classes=n_classes)
    preds = _apply(proba, thresholds, n_classes=n_classes)

    identical = np.array_equal(preds, first_predictions)
    detail = (
        "Bit-identical"
        if identical
        else f"{int((preds != first_predictions).sum())} mismatches"
    )

    logger.info(
        "Reproducibility test: %s -- %s", "PASS" if identical else "FAIL", detail
    )
    return {"passed": identical, "detail": detail}


def profile_inference_latency(
    model,
    X,
    thresholds: list[float],
    n_repeats: int = 3,
    n_classes: int = 5,
) -> dict[str, Any]:
    """Measure prediction latency over multiple runs.

    Returns
    -------
    dict
        Timing breakdown with total, per-sample, and throughput metrics.
    """
    from adoption_accelerator.training.evaluation import apply_thresholds as _apply

    n_samples = X.shape[0] if hasattr(X, "shape") else len(X)

    predict_times = []
    ev_times = []
    thresh_times = []

    for _ in range(n_repeats):
        t0 = time.perf_counter()
        proba = model.predict_proba(X)
        t1 = time.perf_counter()
        compute_expected_values(proba)
        t2 = time.perf_counter()
        _ = _apply(proba, thresholds, n_classes=n_classes)
        t3 = time.perf_counter()

        predict_times.append(t1 - t0)
        ev_times.append(t2 - t1)
        thresh_times.append(t3 - t2)

    total_times = [p + e + t for p, e, t in zip(predict_times, ev_times, thresh_times)]
    mean_total = float(np.mean(total_times))
    per_sample_ms = (mean_total / n_samples) * 1000

    profile = {
        "n_repeats": n_repeats,
        "n_samples": n_samples,
        "total_batch_seconds": round(mean_total, 4),
        "per_sample_ms": round(per_sample_ms, 4),
        "throughput_samples_per_second": round(n_samples / mean_total, 1)
        if mean_total > 0
        else 0,
        "breakdown": {
            "predict_proba_ms": round(float(np.mean(predict_times)) * 1000, 4),
            "expected_value_ms": round(float(np.mean(ev_times)) * 1000, 4),
            "threshold_application_ms": round(float(np.mean(thresh_times)) * 1000, 4),
        },
    }

    logger.info(
        "Latency profile: total=%.3fs, per-sample=%.4fms, throughput=%.0f samples/s",
        mean_total,
        per_sample_ms,
        profile["throughput_samples_per_second"],
    )
    return profile
