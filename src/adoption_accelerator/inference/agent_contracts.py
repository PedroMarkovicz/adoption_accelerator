"""
Standardized output contracts aligning inference layer with agent state.

Defines the adapter structures that bridge the core ML inference outputs
(``PredictionResult``, ``ExplanationResult``) with the agent state schema
defined in ``docs/agent_architecture.md``.

This module ensures a predictable, consistent output structure that
downstream agent nodes can safely consume.  It also defines the error
response contract for the inference layer.

Consumed by:
  - ``agents/nodes/inference.py`` (writes to AgentState)
  - ``agents/nodes/aggregator.py`` (assembles final AgentResponse)
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field

from adoption_accelerator.inference.contracts import PredictionResult
from adoption_accelerator.interpretability.contracts import ExplanationResult


# ---------------------------------------------------------------------------
# Feature factor (individual feature contribution)
# ---------------------------------------------------------------------------


class FeatureFactor(BaseModel):
    """A single feature's contribution to the prediction.

    Maps to ``AgentState`` top_positive_factors / top_negative_factors
    and the FeatureFactor schema in agent_architecture.md.
    """

    feature: str = Field(..., description="Raw feature name")
    display_name: str = Field("", description="Human-readable feature name")
    value: str = Field("", description="Current feature value as string")
    shap_value: float = Field(..., description="SHAP value for this feature")
    modality: str = Field(..., description="Feature modality: tabular/text/image/metadata")
    direction: str = Field(..., description="positive or negative")


# ---------------------------------------------------------------------------
# Inference output (aligned with AgentState fields)
# ---------------------------------------------------------------------------


class InferenceOutput(BaseModel):
    """Standardized output from the inference layer.

    This structure maps directly to the Phase 1 fields in AgentState:
    - ``prediction`` -> ``AgentState.prediction``
    - ``explanation`` -> ``AgentState.explanation``
    - ``feature_vector`` -> ``AgentState.feature_vector``

    It also provides pre-computed fields that the aggregator node
    uses to build the final AgentResponse.
    """

    # Core prediction (maps to AgentState.prediction)
    prediction: PredictionResult

    # SHAP explanation (maps to AgentState.explanation)
    explanation: Optional[ExplanationResult] = None

    # Raw feature vector (maps to AgentState.feature_vector)
    # Stored as list for JSON serialization; the agent can reconstruct
    # the numpy array when needed for SHAP or counterfactual analysis.
    feature_vector: Optional[list[float]] = None

    # Feature names (for downstream SHAP / interpretation use)
    feature_names: Optional[list[str]] = None

    # Pre-computed modality contributions (convenience for aggregator)
    modality_contributions: dict[str, float] = Field(default_factory=dict)

    # Pre-computed top factors split by direction (convenience for aggregator)
    top_positive_factors: list[FeatureFactor] = Field(default_factory=list)
    top_negative_factors: list[FeatureFactor] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Error contract (aligned with NodeError in agent_architecture.md)
# ---------------------------------------------------------------------------


class InferenceError(BaseModel):
    """Error response from the inference layer.

    Aligns with the ``NodeError`` schema from the agent architecture.
    """

    node: str = "inference"
    error_type: str = Field(..., description="Error category")
    message: str = Field(..., description="Human-readable error description")
    timestamp: str = Field("", description="ISO 8601 UTC timestamp")
    recoverable: bool = True


# ---------------------------------------------------------------------------
# Conversion utilities
# ---------------------------------------------------------------------------


def build_inference_output(
    prediction_result: PredictionResult,
    explanation_result: ExplanationResult | None = None,
    feature_vector: np.ndarray | None = None,
    feature_names: list[str] | None = None,
) -> InferenceOutput:
    """Build a standardized InferenceOutput from inference layer results.

    Extracts modality contributions and splits top features into
    positive / negative factor lists for direct consumption by the
    agent aggregator node.

    Parameters
    ----------
    prediction_result : PredictionResult
        Output from ``InferencePipeline.predict_single()``.
    explanation_result : ExplanationResult | None
        Output from ``build_explanation_result()``.
    feature_vector : np.ndarray | None
        The 940-dim feature vector used for prediction.
    feature_names : list[str] | None
        Ordered feature names from the model schema.

    Returns
    -------
    InferenceOutput
    """
    # Serialize feature vector
    fv_list = None
    if feature_vector is not None:
        fv = np.asarray(feature_vector)
        if fv.ndim == 2:
            fv = fv[0]
        fv_list = [float(v) for v in fv]

    # Extract modality contributions and top factors from explanation
    modality_contributions: dict[str, float] = {}
    top_positive: list[FeatureFactor] = []
    top_negative: list[FeatureFactor] = []

    if explanation_result is not None:
        modality_contributions = explanation_result.modality_contributions

        for feat_dict in explanation_result.top_features:
            shap_val = feat_dict.get("shap_value", 0.0)
            factor = FeatureFactor(
                feature=feat_dict.get("feature", ""),
                display_name=feat_dict.get("feature", ""),
                value=str(feat_dict.get("value", "")),
                shap_value=shap_val,
                modality=feat_dict.get("modality", "tabular"),
                direction="positive" if shap_val >= 0 else "negative",
            )
            if shap_val >= 0:
                top_positive.append(factor)
            else:
                top_negative.append(factor)

        # Sort by absolute SHAP value descending
        top_positive.sort(key=lambda f: abs(f.shap_value), reverse=True)
        top_negative.sort(key=lambda f: abs(f.shap_value), reverse=True)

    return InferenceOutput(
        prediction=prediction_result,
        explanation=explanation_result,
        feature_vector=fv_list,
        feature_names=feature_names,
        modality_contributions=modality_contributions,
        top_positive_factors=top_positive,
        top_negative_factors=top_negative,
    )


def explanation_to_agent_dict(explanation: ExplanationResult) -> dict[str, Any]:
    """Convert an ExplanationResult to the dict structure expected by AgentState.

    The output matches the ``AgentState.explanation`` field structure
    defined in the agent architecture.

    Parameters
    ----------
    explanation : ExplanationResult
        Explanation from the inference layer.

    Returns
    -------
    dict
        Serialized explanation compatible with AgentState.
    """
    return explanation.to_dict()
