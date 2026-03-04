"""
Ensemble model construction utilities.

Provides soft-voting and stacking ensemble wrappers compatible with
the project's cross-validation and evaluation framework.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    """Soft-voting ensemble that averages predicted probabilities.

    Parameters
    ----------
    models : list[BaseEstimator]
        Fitted classifiers.
    weights : list[float] | None
        Per-model weights for averaging.  If None, equal weights.
    """

    def __init__(
        self,
        models: list[BaseEstimator] | None = None,
        weights: list[float] | None = None,
    ):
        self.models = models or []
        self.weights = weights

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SoftVotingEnsemble":
        """No-op: assumes models are already fitted."""
        self.classes_ = np.arange(5)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average predicted probabilities across models."""
        probas = []
        for m in self.models:
            p = m.predict_proba(X)
            if p.shape[1] < 5:
                full_p = np.zeros((p.shape[0], 5))
                for i, cls in enumerate(m.classes_):
                    full_p[:, int(cls)] = p[:, i]
                p = full_p
            probas.append(p)

        if self.weights is not None:
            w = np.array(self.weights) / np.sum(self.weights)
            avg = sum(p * wi for p, wi in zip(probas, w))
        else:
            avg = np.mean(probas, axis=0)

        return avg

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return argmax of averaged probabilities."""
        return np.argmax(self.predict_proba(X), axis=1)


def create_soft_voting_ensemble(
    models: list[BaseEstimator],
    weights: list[float] | None = None,
) -> SoftVotingEnsemble:
    """Create a soft-voting ensemble from fitted models.

    Parameters
    ----------
    models : list[BaseEstimator]
        Pre-fitted classifiers.
    weights : list[float] | None
        Optional per-model weights.

    Returns
    -------
    SoftVotingEnsemble
    """
    ensemble = SoftVotingEnsemble(models=models, weights=weights)
    ensemble.classes_ = np.arange(5)
    logger.info("Created SoftVotingEnsemble with %d models", len(models))
    return ensemble


def create_stacking_ensemble(
    base_models: list[tuple[str, BaseEstimator]],
    meta_learner: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[BaseEstimator, np.ndarray]:
    """Create a stacking ensemble using out-of-fold predictions.

    Parameters
    ----------
    base_models : list[tuple[str, BaseEstimator]]
        Named unfitted base classifiers (will be cloned per fold).
    meta_learner : BaseEstimator
        Meta-learner to train on stacked OOF predictions.
    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector.
    folds : list[tuple[ndarray, ndarray]]
        Pre-computed fold indices.

    Returns
    -------
    tuple[BaseEstimator, ndarray]
        Fitted meta-learner and the OOF meta-features matrix.
    """
    from sklearn.base import clone

    n_samples = X.shape[0]
    n_classes = 5
    n_base = len(base_models)

    # Generate OOF predictions for each base model
    oof_meta = np.zeros((n_samples, n_base * n_classes))

    for model_idx, (name, base_model) in enumerate(base_models):
        col_start = model_idx * n_classes
        col_end = col_start + n_classes

        for train_idx, val_idx in folds:
            fold_model = clone(base_model)
            fold_model.fit(X[train_idx], y[train_idx])
            proba = fold_model.predict_proba(X[val_idx])

            if proba.shape[1] < n_classes:
                full_proba = np.zeros((proba.shape[0], n_classes))
                for i, cls in enumerate(fold_model.classes_):
                    full_proba[:, int(cls)] = proba[:, i]
                proba = full_proba

            oof_meta[val_idx, col_start:col_end] = proba

        logger.info("Stacking: OOF predictions generated for %s", name)

    # Train meta-learner on OOF meta-features
    meta = clone(meta_learner)
    meta.fit(oof_meta, y)
    logger.info("Stacking: meta-learner fitted on %d meta-features", oof_meta.shape[1])

    return meta, oof_meta
