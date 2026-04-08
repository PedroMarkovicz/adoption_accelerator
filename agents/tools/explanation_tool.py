"""
Explanation tool adapter.

Stateless adapter wrapping SHAP computation + modality classification +
top-K extraction + Feature Interpretation Layer.

Accepts a feature vector and feature names, returns both a raw
``ExplanationResult`` (for audit) and an ``InterpretedExplanation``
(for LLM nodes).

Consumed by: ``agents/nodes/inference.py``
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import numpy as np

from adoption_accelerator.features.registry import build_feature_registry
from adoption_accelerator.inference.explain import (
    _load_default_provenance_map,
    build_explanation_result,
    build_modality_map,
)
from adoption_accelerator.inference.serving import get_explainer
from adoption_accelerator.interpretability.contracts import ExplanationResult
from adoption_accelerator.interpretability.explainer import compute_shap_values
from adoption_accelerator.interpretability.translator import (
    InterpretedExplanation,
    translate_to_interpreted,
)

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_feature_registry(
    feature_names_tuple: tuple[str, ...],
) -> dict[str, Any]:
    """Build and cache the feature registry from a tuple of feature names."""
    provenance_map = _load_default_provenance_map()
    return build_feature_registry(list(feature_names_tuple), provenance_map=provenance_map)


def run_explanation(
    feature_vector: np.ndarray | list[float],
    feature_names: list[str],
    predicted_class: int,
    probabilities: list[float],
    confidence: float,
    explainer_path: str | None = None,
    top_k: int = 10,
    modality_available: dict[str, bool] | None = None,
) -> dict[str, Any]:
    """Compute SHAP explanation and interpreted translation for a single sample.

    Parameters
    ----------
    feature_vector : ndarray or list[float]
        940-dim feature vector (1D or 2D with shape (1, 940)).
    feature_names : list[str]
        Ordered feature names from the model schema.
    predicted_class : int
        Predicted class (0-4).
    probabilities : list[float]
        Per-class probabilities.
    confidence : float
        Probability of the predicted class (Issue #1 semantics).
    explainer_path : str | None
        Optional override for the explainer joblib path.
    top_k : int
        Number of top features to include.
    modality_available : dict[str, bool] or None
        Which modalities were present in the request.  Passed to the
        interpretation layer so baseline entries (absent modalities)
        are excluded from modality contributions and top factors.
        When ``None``, all modalities are assumed present.

    Returns
    -------
    dict
        Keys: ``explanation_result`` (ExplanationResult),
        ``interpreted_explanation`` (InterpretedExplanation).
    """
    fv = np.asarray(feature_vector, dtype=np.float64)
    if fv.ndim == 1:
        fv = fv.reshape(1, -1)

    # Load cached explainer
    explainer = get_explainer(explainer_path)

    # Compute raw SHAP values
    shap_result = compute_shap_values(explainer, fv, feature_names)

    # Build modality map for the raw ExplanationResult
    modality_map = build_modality_map(feature_names)

    # Build raw ExplanationResult (for audit trail)
    explanation_result = build_explanation_result(
        shap_values=shap_result.values,
        feature_values=fv,
        feature_names=feature_names,
        predicted_class=predicted_class,
        probabilities=probabilities,
        confidence=confidence,
        modality_map=modality_map,
        top_k=top_k,
    )

    # Build InterpretedExplanation via the Feature Interpretation Layer
    registry = _get_feature_registry(tuple(feature_names))

    # Extract SHAP values for the predicted class
    shap_vals = np.asarray(shap_result.values)
    if shap_vals.ndim == 3:
        # (1, n_features, n_classes) -> predicted class slice
        shap_1d = shap_vals[0, :, predicted_class]
    elif shap_vals.ndim == 2:
        shap_1d = shap_vals[0] if shap_vals.shape[0] == 1 else shap_vals[predicted_class]
    else:
        shap_1d = shap_vals

    interpreted = translate_to_interpreted(
        shap_values=shap_1d,
        feature_names=feature_names,
        registry=registry,
        top_k=top_k,
        modality_available=modality_available,
        raw_explanation=explanation_result,
    )

    logger.info(
        "Explanation tool: %d top factors, %d modalities",
        len(interpreted.top_factors),
        len(interpreted.modality_contributions),
    )

    return {
        "explanation_result": explanation_result,
        "interpreted_explanation": interpreted,
    }
