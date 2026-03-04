"""
Core prediction logic for inference.

Isolated from formatting, validation, and orchestration so that it can
be called independently by the frontend service layer and the agent
system tools.  Model-agnostic -- accepts any sklearn-compatible
estimator with ``predict_proba``.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def predict_probabilities(
    model,
    X,
    n_classes: int = 5,
) -> np.ndarray:
    """Generate class probability predictions.

    Parameters
    ----------
    model : fitted estimator
        Must expose ``predict_proba(X) -> ndarray``.
    X : array-like of shape (n_samples, n_features)
        Input feature matrix.
    n_classes : int
        Expected number of classes.

    Returns
    -------
    np.ndarray of shape (n_samples, n_classes)
        Class probability matrix.

    Raises
    ------
    ValueError
        If output shape or probability constraints are violated.
    """
    proba = model.predict_proba(X)

    # Shape check
    if proba.shape[1] != n_classes:
        raise ValueError(
            f"predict_proba returned {proba.shape[1]} classes, expected {n_classes}"
        )

    # Range check
    if np.any(proba < 0) or np.any(proba > 1):
        raise ValueError("Predicted probabilities contain values outside [0, 1]")

    # Sum-to-one check (per row)
    row_sums = proba.sum(axis=1)
    max_deviation = float(np.max(np.abs(row_sums - 1.0)))
    if max_deviation > 1e-6:
        raise ValueError(
            f"Probability rows do not sum to 1.0 (max deviation: {max_deviation:.2e})"
        )

    logger.info(
        "Predicted probabilities: shape=%s, range=[%.4f, %.4f]",
        proba.shape,
        proba.min(),
        proba.max(),
    )
    return proba


def compute_expected_values(probabilities: np.ndarray) -> np.ndarray:
    """Compute expected value E = sum(i * P(y=i)) for each sample.

    Parameters
    ----------
    probabilities : ndarray of shape (n_samples, n_classes)
        Class probability matrix.

    Returns
    -------
    np.ndarray of shape (n_samples,)
        Expected values in [0, n_classes - 1].
    """
    classes = np.arange(probabilities.shape[1])
    ev = probabilities @ classes

    if np.any(ev < 0) or np.any(ev > probabilities.shape[1] - 1):
        raise ValueError(
            f"Expected values outside valid range: [{ev.min():.4f}, {ev.max():.4f}]"
        )

    logger.info(
        "Expected values: mean=%.4f, std=%.4f, range=[%.4f, %.4f]",
        ev.mean(),
        ev.std(),
        ev.min(),
        ev.max(),
    )
    return ev
