"""
SHAP explainer fitting and value computation.

Provides a thin wrapper around SHAP's TreeExplainer for consistent
fitting, serialization, and SHAP value computation across the project.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def fit_shap_explainer(
    model: Any,
    X_background: np.ndarray | None = None,
) -> Any:
    """Fit a SHAP TreeExplainer on the given model.

    Parameters
    ----------
    model : fitted estimator
        A tree-based model (LightGBM, XGBoost, CatBoost) or a
        :class:`SoftVotingEnsemble` wrapping tree-based estimators.
    X_background : ndarray | None
        Optional background dataset for the explainer.  For
        ``TreeExplainer`` this is usually not required but can
        improve approximation quality for certain model types.

    Returns
    -------
    shap.TreeExplainer
        Fitted SHAP explainer instance.
    """
    import shap

    # For ensemble models, we extract the first base model.
    # The notebook should handle per-model SHAP separately if needed.
    actual_model = model
    if hasattr(model, "models"):
        # SoftVotingEnsemble -- use the first base model
        actual_model = model.models[0]
        logger.info(
            "Ensemble detected; fitting TreeExplainer on first base model (%s).",
            type(actual_model).__name__,
        )

    if X_background is not None:
        explainer = shap.TreeExplainer(actual_model, data=X_background)
    else:
        explainer = shap.TreeExplainer(actual_model)

    logger.info("SHAP TreeExplainer fitted for %s.", type(actual_model).__name__)
    return explainer


def compute_shap_values(
    explainer: Any,
    X: np.ndarray,
    feature_names: list[str] | None = None,
) -> Any:
    """Compute SHAP values for a set of samples.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        Pre-fitted SHAP explainer.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix to explain.
    feature_names : list[str] | None
        Feature names for the Explanation object.

    Returns
    -------
    shap.Explanation
        SHAP Explanation object with ``.values``, ``.base_values``,
        and ``.data`` populated.
    """
    import shap

    shap_values = explainer.shap_values(X)

    # TreeExplainer may return a list (one array per class) or a 3-D array.
    if isinstance(shap_values, list):
        # Shape: list of (n_samples, n_features) -> stack to (n_samples, n_features, n_classes)
        shap_array = np.stack(shap_values, axis=-1)
    else:
        shap_array = np.asarray(shap_values)

    # Build an Explanation object for downstream compatibility.
    explanation = shap.Explanation(
        values=shap_array,
        base_values=np.asarray(explainer.expected_value),
        data=X,
        feature_names=feature_names,
    )

    n_samples = X.shape[0]
    logger.info(
        "SHAP values computed for %d samples; shape=%s.",
        n_samples,
        shap_array.shape,
    )
    return explanation


def save_explainer(explainer: Any, path: str | Path) -> Path:
    """Persist a fitted SHAP explainer to disk.

    Parameters
    ----------
    explainer : shap.TreeExplainer
        Fitted explainer to serialize.
    path : str | Path
        Destination file path (e.g. ``explainer.joblib``).

    Returns
    -------
    Path
        Resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, path)
    logger.info("SHAP explainer saved to %s.", path)
    return path


def load_explainer(path: str | Path) -> Any:
    """Load a previously saved SHAP explainer.

    Parameters
    ----------
    path : str | Path
        Path to the serialized explainer file.

    Returns
    -------
    shap.TreeExplainer
        Loaded explainer.
    """
    path = Path(path)
    explainer = joblib.load(path)
    logger.info("SHAP explainer loaded from %s.", path)
    return explainer
