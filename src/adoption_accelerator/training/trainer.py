"""
Cross-validation trainer.

Runs K-fold cross-validation with per-fold metric computation, optional
threshold optimization, and structured result aggregation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from adoption_accelerator.training.evaluation import (
    apply_thresholds,
    compute_classification_metrics,
    optimize_thresholds,
)

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    """Structured cross-validation result container."""

    model_name: str
    n_splits: int
    fold_metrics: list[dict[str, Any]] = field(default_factory=list)
    fold_thresholds: list[dict[str, Any] | None] = field(default_factory=list)
    oof_predictions: np.ndarray | None = None
    oof_probabilities: np.ndarray | None = None
    oof_true: np.ndarray | None = None
    aggregated: dict[str, Any] = field(default_factory=dict)
    training_time_seconds: float = 0.0


def cross_validate_model(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    splitter: StratifiedKFold,
    threshold_optimize: bool = True,
    model_name: str | None = None,
) -> CVResult:
    """Run K-fold cross-validation for a classifier.

    For each fold: fit, predict class probabilities (if available) or
    hard labels, compute metrics, and optionally optimize thresholds.

    Parameters
    ----------
    model : BaseEstimator
        Scikit-learn-compatible classifier (will be cloned per fold).
    X : DataFrame or ndarray
        Feature matrix.
    y : Series or ndarray
        Target vector.
    splitter : StratifiedKFold
        Pre-configured splitter.
    threshold_optimize : bool
        Whether to run threshold optimization on predicted probabilities.
    model_name : str | None
        Human-readable model name.

    Returns
    -------
    CVResult
        Structured result with per-fold and aggregated metrics.
    """
    if model_name is None:
        model_name = type(model).__name__

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n_samples = X_arr.shape[0]
    n_classes = 5

    oof_preds = np.full(n_samples, -1, dtype=int)
    oof_proba = np.zeros((n_samples, n_classes), dtype=float)
    oof_true = y_arr.copy()

    result = CVResult(model_name=model_name, n_splits=splitter.n_splits)
    has_proba = hasattr(model, "predict_proba")

    t0 = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X_arr, y_arr)):
        X_train, X_val = X_arr[train_idx], X_arr[val_idx]
        y_train, y_val = y_arr[train_idx], y_arr[val_idx]

        fold_model = clone(model)

        # Suppress LightGBM verbosity
        fit_params: dict[str, Any] = {}
        model_cls = type(fold_model).__name__
        if model_cls == "LGBMClassifier":
            fit_params["callbacks"] = [_lgbm_silent_callback()]

        fold_model.fit(X_train, y_train, **fit_params)

        # Training set predictions (for overfitting diagnostic)
        if has_proba:
            train_proba = fold_model.predict_proba(X_train)
            if train_proba.shape[1] < n_classes:
                full_tp = np.zeros((train_proba.shape[0], n_classes))
                for i, cls in enumerate(fold_model.classes_):
                    full_tp[:, int(cls)] = train_proba[:, i]
                train_proba = full_tp
            y_pred_train = np.argmax(train_proba, axis=1)
        else:
            y_pred_train = fold_model.predict(X_train).astype(int)
        train_qwk = float(compute_classification_metrics(y_train, y_pred_train)["qwk"])

        # Predictions
        if has_proba:
            proba = fold_model.predict_proba(X_val)
            # Guard against models returning fewer columns
            if proba.shape[1] < n_classes:
                full_proba = np.zeros((proba.shape[0], n_classes))
                for i, cls in enumerate(fold_model.classes_):
                    full_proba[:, int(cls)] = proba[:, i]
                proba = full_proba
            y_pred_argmax = np.argmax(proba, axis=1)
            oof_proba[val_idx] = proba
        else:
            proba = None
            y_pred_argmax = fold_model.predict(X_val).astype(int)

        oof_preds[val_idx] = y_pred_argmax

        # Base metrics (argmax)
        fold_met = compute_classification_metrics(y_val, y_pred_argmax, proba)
        fold_met["qwk_argmax"] = fold_met["qwk"]
        fold_met["train_qwk"] = train_qwk

        # Threshold optimization
        thresh_result = None
        if threshold_optimize and proba is not None:
            thresh_result = optimize_thresholds(y_val, proba)
            fold_met["qwk_threshold"] = thresh_result["qwk_optimized"]
            y_pred_thresh = apply_thresholds(proba, thresh_result["thresholds"])
            fold_met["accuracy_threshold"] = float(np.mean(y_val == y_pred_thresh))
        else:
            fold_met["qwk_threshold"] = fold_met["qwk"]

        result.fold_metrics.append(fold_met)
        result.fold_thresholds.append(thresh_result)

        logger.info(
            "[%s] Fold %d/%d  QWK(argmax)=%.4f  QWK(thresh)=%.4f",
            model_name,
            fold_idx + 1,
            splitter.n_splits,
            fold_met["qwk_argmax"],
            fold_met["qwk_threshold"],
        )

    result.training_time_seconds = time.time() - t0
    result.oof_predictions = oof_preds
    result.oof_probabilities = oof_proba if has_proba else None
    result.oof_true = oof_true

    # Aggregate across folds
    result.aggregated = _aggregate_fold_metrics(
        result.fold_metrics, result.training_time_seconds
    )

    logger.info(
        "[%s] CV complete  mean_QWK(argmax)=%.4f  mean_QWK(thresh)=%.4f  time=%.1fs",
        model_name,
        result.aggregated["mean_qwk_argmax"],
        result.aggregated["mean_qwk_threshold"],
        result.training_time_seconds,
    )
    return result


def _aggregate_fold_metrics(
    fold_metrics: list[dict[str, Any]],
    training_time: float,
) -> dict[str, Any]:
    """Compute mean/std of fold-level metrics."""
    qwk_argmax = [m["qwk_argmax"] for m in fold_metrics]
    qwk_thresh = [m["qwk_threshold"] for m in fold_metrics]
    accuracy = [m["accuracy"] for m in fold_metrics]
    macro_f1 = [m["macro_f1"] for m in fold_metrics]
    weighted_f1 = [m["weighted_f1"] for m in fold_metrics]
    train_qwk = [m.get("train_qwk", 0.0) for m in fold_metrics]

    agg = {
        "mean_qwk_argmax": float(np.mean(qwk_argmax)),
        "std_qwk_argmax": float(np.std(qwk_argmax)),
        "qwk_argmax_folds": qwk_argmax,
        "mean_qwk_threshold": float(np.mean(qwk_thresh)),
        "std_qwk_threshold": float(np.std(qwk_thresh)),
        "qwk_threshold_folds": qwk_thresh,
        "mean_accuracy": float(np.mean(accuracy)),
        "std_accuracy": float(np.std(accuracy)),
        "mean_macro_f1": float(np.mean(macro_f1)),
        "std_macro_f1": float(np.std(macro_f1)),
        "mean_weighted_f1": float(np.mean(weighted_f1)),
        "std_weighted_f1": float(np.std(weighted_f1)),
        "mean_train_qwk": float(np.mean(train_qwk)),
        "train_qwk_folds": train_qwk,
        "overfitting_gap": float(np.mean(train_qwk)) - float(np.mean(qwk_thresh)),
        "training_time_seconds": training_time,
    }
    return agg


def _lgbm_silent_callback():
    """Return a LightGBM callback that suppresses iteration logging."""
    import lightgbm as lgb

    return lgb.log_evaluation(period=-1)
