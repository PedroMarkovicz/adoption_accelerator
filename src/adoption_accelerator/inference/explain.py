"""
Combined inference + explanation interface for the agent system.

Provides the single-prediction interface that produces both a
``PredictionResult`` and an ``ExplanationResult`` in one call,
along with modality classification, top-K extraction, and modality
contribution computation.

This module bridges the inference and interpretability layers,
giving agent nodes a single entry point for the complete
prediction + explanation flow.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import numpy as np

from adoption_accelerator.inference.contracts import PredictionResult
from adoption_accelerator.inference.pipeline import CLASS_LABELS, InferencePipeline
from adoption_accelerator.interpretability.contracts import ExplanationResult
from adoption_accelerator.interpretability.explainer import compute_shap_values

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provenance map loading (cached)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _load_default_provenance_map() -> dict[str, str]:
    """Load the provenance map from the integrated feature schema.

    This is schema-driven: each feature's modality comes from the
    ``source`` field in the per-modality ``schema.json`` files that
    were merged during feature integration.
    """
    from adoption_accelerator import config as cfg

    integrated_schema_path = cfg.DATA_FEATURES / "integrated" / "v1" / "schema.json"
    try:
        import json

        with open(integrated_schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        return {col["name"]: col["source"] for col in schema["columns"]}
    except Exception as exc:
        logger.warning("Could not load integrated schema for provenance map: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Modality classification
# ---------------------------------------------------------------------------

# Naming convention patterns for modality classification.
# Used as a fallback when no provenance map is available, or to
# reclassify metadata features from their parent modality.
_METADATA_PREFIXES = ("meta_", "entity_", "mean_image_", "mean_blur_", "image_size")
_SENTIMENT_PREFIXES = (
    "doc_sentiment_",
    "mean_sentence_",
    "min_sentence_",
    "max_sentence_",
    "sentiment_variance",
)
_IMAGE_EMB_PREFIX = "img_emb_"
_TEXT_EMB_PREFIX = "text_emb_"


def classify_feature_modality(
    feature_name: str,
    provenance_map: dict[str, str] | None = None,
) -> str:
    """Classify a single feature into its modality.

    Resolution order:
    1. Reclassify metadata features (``meta_*``, ``entity_*``,
       ``sentence_count_sentiment``) regardless of provenance source.
    2. Provenance map (schema-driven, authoritative) if available.
    3. Naming convention patterns as final fallback.

    The provenance map stores features in 3 source modalities
    (tabular, text, image).  This function further distinguishes
    ``metadata`` from ``image`` for features with ``meta_*`` /
    ``entity_*`` prefixes, matching the 4-modality schema used
    by the agent system.

    Parameters
    ----------
    feature_name : str
        Raw feature name from the schema.
    provenance_map : dict | None
        Column-to-modality mapping from ``build_provenance_map()``.

    Returns
    -------
    str
        One of ``"tabular"``, ``"text"``, ``"image"``, ``"metadata"``.
    """
    # Check provenance map first
    source = None
    if provenance_map:
        source = provenance_map.get(feature_name)

    # Reclassify metadata features out of their parent modality
    if feature_name.startswith(_METADATA_PREFIXES):
        return "metadata"
    if feature_name in ("entity_count", "entity_type_count"):
        return "metadata"
    if feature_name.startswith(("sentence_count_sentiment",)):
        return "metadata"

    # If provenance map gave us a source, use it
    if source:
        return source

    # Fallback: naming convention
    if feature_name.startswith(_IMAGE_EMB_PREFIX):
        return "image"
    if feature_name.startswith(_TEXT_EMB_PREFIX):
        return "text"
    if feature_name.startswith(_SENTIMENT_PREFIXES):
        return "text"

    # Default to tabular for unrecognized features
    return "tabular"


def build_modality_map(
    feature_names: list[str],
    provenance_map: dict[str, str] | None = None,
) -> dict[str, str]:
    """Classify all features into modalities.

    When no provenance map is provided, the integrated feature schema
    is loaded automatically as the authoritative source.

    Parameters
    ----------
    feature_names : list[str]
        Ordered feature names from the model schema.
    provenance_map : dict | None
        Schema-driven column-to-modality mapping.  If ``None``, the
        default provenance map is loaded from the integrated schema.

    Returns
    -------
    dict[str, str]
        Mapping of feature name to modality.
    """
    if provenance_map is None:
        provenance_map = _load_default_provenance_map()

    return {
        name: classify_feature_modality(name, provenance_map) for name in feature_names
    }


# ---------------------------------------------------------------------------
# SHAP modality grouping and top-K extraction
# ---------------------------------------------------------------------------


def compute_modality_contributions(
    shap_values_1d: np.ndarray,
    feature_names: list[str],
    modality_map: dict[str, str],
) -> dict[str, float]:
    """Compute per-modality SHAP contribution percentages.

    Each modality's contribution is ``sum(|SHAP|) / total(|SHAP|)``
    for the given sample.

    Parameters
    ----------
    shap_values_1d : ndarray of shape (n_features,)
        SHAP values for one sample and one class (the predicted class).
    feature_names : list[str]
        Feature names matching the SHAP values.
    modality_map : dict[str, str]
        Feature name to modality mapping.

    Returns
    -------
    dict[str, float]
        Modality -> contribution proportion (sums to ~1.0).
    """
    abs_shap = np.abs(shap_values_1d)
    total = float(abs_shap.sum())
    if total == 0:
        return {m: 0.0 for m in ("tabular", "text", "image", "metadata")}

    contributions: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        modality = modality_map.get(name, "tabular")
        contributions[modality] = contributions.get(modality, 0.0) + float(abs_shap[i])

    # Normalize to proportions
    return {m: round(v / total, 6) for m, v in sorted(contributions.items())}


def extract_top_k_per_modality(
    shap_values_1d: np.ndarray,
    feature_values_1d: np.ndarray,
    feature_names: list[str],
    modality_map: dict[str, str],
    k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Extract top-K features by |SHAP| for each modality.

    Parameters
    ----------
    shap_values_1d : ndarray of shape (n_features,)
        SHAP values for one sample / one class.
    feature_values_1d : ndarray of shape (n_features,)
        Actual feature values for the sample.
    feature_names : list[str]
        Feature names.
    modality_map : dict[str, str]
        Feature name to modality mapping.
    k : int
        Maximum features per modality.

    Returns
    -------
    dict[str, list[dict]]
        Modality -> list of dicts with keys ``feature``, ``value``,
        ``shap_value``, ``modality``.
    """
    # Group indices by modality
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(feature_names):
        modality = modality_map.get(name, "tabular")
        groups.setdefault(modality, []).append(i)

    result: dict[str, list[dict[str, Any]]] = {}
    for modality, indices in sorted(groups.items()):
        abs_vals = np.abs(shap_values_1d[indices])
        # Argsort descending within this modality
        top_local = np.argsort(abs_vals)[::-1][:k]

        entries = []
        for local_idx in top_local:
            global_idx = indices[local_idx]
            entries.append(
                {
                    "feature": feature_names[global_idx],
                    "value": float(feature_values_1d[global_idx]),
                    "shap_value": float(shap_values_1d[global_idx]),
                    "modality": modality,
                }
            )
        result[modality] = entries

    return result


def extract_top_k_overall(
    shap_values_1d: np.ndarray,
    feature_values_1d: np.ndarray,
    feature_names: list[str],
    modality_map: dict[str, str],
    k: int = 20,
) -> list[dict[str, Any]]:
    """Extract the global top-K features by |SHAP| across all modalities.

    Parameters
    ----------
    shap_values_1d : ndarray of shape (n_features,)
        SHAP values for one sample / one class.
    feature_values_1d : ndarray of shape (n_features,)
        Actual feature values.
    feature_names : list[str]
        Feature names.
    modality_map : dict[str, str]
        Feature name to modality mapping.
    k : int
        Number of top features to return.

    Returns
    -------
    list[dict]
        Top-K features sorted by descending |SHAP|.
    """
    abs_shap = np.abs(shap_values_1d)
    top_indices = np.argsort(abs_shap)[::-1][:k]

    return [
        {
            "feature": feature_names[i],
            "value": float(feature_values_1d[i]),
            "shap_value": float(shap_values_1d[i]),
            "modality": modality_map.get(feature_names[i], "tabular"),
        }
        for i in top_indices
    ]


# ---------------------------------------------------------------------------
# ExplanationResult assembly
# ---------------------------------------------------------------------------


def build_explanation_result(
    shap_values: np.ndarray,
    feature_values: np.ndarray,
    feature_names: list[str],
    predicted_class: int,
    probabilities: list[float],
    confidence: float,
    modality_map: dict[str, str],
    pet_id: str = "",
    top_k: int = 10,
) -> ExplanationResult:
    """Assemble a complete ExplanationResult from raw SHAP output.

    Wires together modality classification, modality contributions,
    and top-K extraction into a single structured result.

    Parameters
    ----------
    shap_values : ndarray
        SHAP values.  Shape ``(1, n_features, n_classes)`` or
        ``(1, n_features)``.
    feature_values : ndarray of shape (1, n_features) or (n_features,)
        Feature values for the sample.
    feature_names : list[str]
        Feature names matching the second axis.
    predicted_class : int
        Predicted class (0-4).
    probabilities : list[float]
        Per-class probabilities.
    confidence : float
        Max probability.
    modality_map : dict[str, str]
        Feature name to modality mapping.
    pet_id : str
        Optional pet identifier.
    top_k : int
        Number of top features per modality.

    Returns
    -------
    ExplanationResult
    """
    vals = np.asarray(shap_values)
    fv = np.asarray(feature_values)

    # Flatten to single sample
    if vals.ndim == 3:
        # (1, n_features, n_classes) -> use predicted class
        shap_1d = vals[0, :, predicted_class]
    elif vals.ndim == 2:
        shap_1d = vals[0] if vals.shape[0] == 1 else vals[predicted_class]
    else:
        shap_1d = vals

    if fv.ndim == 2:
        fv = fv[0]

    # Modality contributions
    modality_contributions = compute_modality_contributions(
        shap_1d, feature_names, modality_map
    )

    # Top-K per modality
    top_per_modality = extract_top_k_per_modality(
        shap_1d, fv, feature_names, modality_map, k=top_k
    )

    # Flatten into a single top_features list (overall top-K)
    top_features = extract_top_k_overall(
        shap_1d, fv, feature_names, modality_map, k=top_k * 2
    )

    return ExplanationResult(
        pet_id=pet_id,
        predicted_class=predicted_class,
        probabilities=probabilities,
        top_features=top_features,
        modality_contributions=modality_contributions,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Combined predict + explain
# ---------------------------------------------------------------------------


def predict_single_with_explanation(
    pipeline: InferencePipeline,
    explainer: Any,
    features: np.ndarray,
    feature_names: list[str],
    provenance_map: dict[str, str] | None = None,
    pet_id: str = "",
    top_k: int = 10,
) -> tuple[PredictionResult, ExplanationResult]:
    """Run prediction and SHAP explanation for a single sample.

    This is the primary entry point for the agent inference node.
    It combines ``predict_single()`` and ``compute_shap_values()``
    into a single call, avoiding separate tool invocations.

    Parameters
    ----------
    pipeline : InferencePipeline
        Loaded inference pipeline (use ``get_inference_pipeline()``).
    explainer : shap.TreeExplainer
        Loaded SHAP explainer (use ``get_explainer()``).
    features : ndarray of shape (1, n_features) or (n_features,)
        940-dimensional feature vector.
    feature_names : list[str]
        Ordered feature names matching the model schema.
    provenance_map : dict | None
        Schema-driven column-to-modality mapping.
    pet_id : str
        Optional pet identifier for the explanation result.
    top_k : int
        Number of top features per modality in the explanation.

    Returns
    -------
    tuple[PredictionResult, ExplanationResult]
        The prediction result and its corresponding explanation.
    """
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # Step 1: Prediction
    prediction_result = pipeline.predict_single(features)

    # Step 2: SHAP values
    shap_explanation = compute_shap_values(explainer, features, feature_names)

    # Step 3: Build modality map
    modality_map = build_modality_map(feature_names, provenance_map)

    # Step 4: Assemble explanation
    explanation_result = build_explanation_result(
        shap_values=shap_explanation.values,
        feature_values=features,
        feature_names=feature_names,
        predicted_class=prediction_result.prediction,
        probabilities=[
            prediction_result.probabilities[i]
            for i in range(len(prediction_result.probabilities))
        ],
        confidence=prediction_result.confidence,
        modality_map=modality_map,
        pet_id=pet_id,
        top_k=top_k,
    )

    return prediction_result, explanation_result
