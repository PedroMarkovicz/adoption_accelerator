"""
Aggregator node --- response assembly and final validation.

Collects all outputs from upstream nodes (prediction, explanation,
narrative, recommendations, improved description), builds the final
``AgentResponse``, and attaches execution trace and metadata.

This is a deterministic node with no LLM dependency.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from agents.state import (
    AgentResponse,
    AgentState,
    FeatureFactor,
    NodeError,
    ResponseMetadata,
    TraceEntry,
)

from adoption_accelerator.features.display_names import USER_DISPLAY_NAMES

logger = logging.getLogger(__name__)


def _format_value(raw_value: str, feature_name: str) -> str:
    """Format a raw feature value string for human readability.

    Rounds floats, removes trailing zeros, and returns empty string
    for missing/empty values.
    """
    if not raw_value or raw_value.strip() == "":
        return ""

    try:
        num = float(raw_value)
    except (ValueError, TypeError):
        return raw_value

    # Integer-like values (counts, codes, binary flags)
    if num == int(num) and abs(num) < 1_000_000:
        return str(int(num))

    # Very small values (frequencies, ratios < 0.01) - show 4 decimals
    if abs(num) < 0.01:
        return f"{num:.4f}"

    # Normal range - 2 decimal places
    return f"{num:.2f}"


def aggregator_node(state: AgentState) -> dict:
    """Assemble the final AgentResponse from all state outputs.

    Collects Phase 1 (deterministic) and Phase 2 (LLM, when available)
    outputs, validates completeness, and builds the response payload.

    Parameters
    ----------
    state : AgentState
        Graph state with outputs from all upstream nodes.

    Returns
    -------
    dict
        State update with ``response`` (AgentResponse) and ``trace``.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    prediction = state.get("prediction")
    explanation = state.get("explanation")
    interpreted = state.get("interpreted_explanation")
    narrative = state.get("narrative_explanation", "")
    recommendations = state.get("recommendations", [])
    improved_description = state.get("improved_description")
    errors = state.get("errors", [])
    warnings = state.get("warnings", [])
    trace_entries = state.get("trace", [])
    # Issue #2: modality availability computed by the inference node
    modality_available: dict[str, bool] = state.get("modality_available") or {}

    # Validate that we have at least the prediction
    if prediction is None:
        return _build_error_response(state, started_at, t0)

    # Build modality contributions from explanation or interpreted
    modality_contributions: dict[str, float] = {}
    if interpreted is not None:
        modality_contributions = interpreted.modality_contributions
    elif explanation is not None:
        modality_contributions = explanation.modality_contributions

    # Build top positive/negative factors from interpreted explanation
    top_positive: list[FeatureFactor] = []
    top_negative: list[FeatureFactor] = []

    if interpreted is not None:
        for factor in interpreted.top_factors:
            display_name = USER_DISPLAY_NAMES.get(factor.name, factor.description)
            ff = FeatureFactor(
                feature=factor.name,
                display_name=display_name,
                value=_format_value(factor.value, factor.name),
                shap_value=factor.shap_magnitude,
                modality=factor.modality,
                direction=factor.direction,
            )
            if factor.direction == "positive":
                top_positive.append(ff)
            else:
                top_negative.append(ff)
    elif explanation is not None:
        # Fallback: use raw top_features from ExplanationResult
        for feat_dict in explanation.top_features:
            shap_val = feat_dict.get("shap_value", 0.0)
            feat_name = feat_dict.get("feature", "")
            display_name = USER_DISPLAY_NAMES.get(feat_name, feat_name)
            ff = FeatureFactor(
                feature=feat_name,
                display_name=display_name,
                value=_format_value(str(feat_dict.get("value", "")), feat_name),
                shap_value=shap_val,
                modality=feat_dict.get("modality", "tabular"),
                direction="positive" if shap_val >= 0 else "negative",
            )
            if shap_val >= 0:
                top_positive.append(ff)
            else:
                top_negative.append(ff)

    # Sort by absolute SHAP magnitude
    top_positive.sort(key=lambda f: abs(f.shap_value), reverse=True)
    top_negative.sort(key=lambda f: abs(f.shap_value), reverse=True)

    # Compute timing metadata from trace
    inference_time_ms = 0.0
    nodes_executed = []
    for entry in trace_entries:
        nodes_executed.append(entry.node)
        if entry.node == "inference":
            inference_time_ms = entry.duration_ms

    total_time_ms = sum(e.duration_ms for e in trace_entries)

    # Build response metadata
    metadata = ResponseMetadata(
        session_id=state.get("session_id", ""),
        model_version="tuned_v1",
        model_type="SoftVotingEnsemble",
        inference_time_ms=round(inference_time_ms, 2),
        total_time_ms=round(total_time_ms, 2),
        timestamp=state.get("timestamp", ""),
        nodes_executed=nodes_executed,
        errors=[e for e in errors if not e.recoverable],
    )

    # Build the final response
    response = AgentResponse(
        prediction=prediction.prediction,
        prediction_label=prediction.prediction_label,
        probabilities=prediction.probabilities,
        confidence=prediction.confidence,
        # Issue #1: pass through diagnostic max-class probability
        max_class_probability=getattr(prediction, "max_class_probability", None),
        narrative_explanation=narrative or "",
        modality_contributions=modality_contributions,
        top_positive_factors=top_positive,
        top_negative_factors=top_negative,
        # Issue #2: include modality availability for frontend
        modality_available=modality_available,
        recommendations=recommendations or [],
        improved_description=improved_description,
        warnings=warnings,
        metadata=metadata,
    )

    duration_ms = (time.perf_counter() - t0) * 1000
    completed_at = datetime.now(timezone.utc).isoformat()

    trace = TraceEntry(
        node="aggregator",
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=round(duration_ms, 2),
        status="success",
        metadata={
            "has_narrative": bool(narrative),
            "n_recommendations": len(recommendations or []),
            "has_improved_description": improved_description is not None,
        },
    )

    logger.info(
        "Aggregator: prediction=%d (%s), confidence=%.4f, "
        "total_pipeline=%.1fms",
        prediction.prediction,
        prediction.prediction_label,
        prediction.confidence,
        total_time_ms + duration_ms,
    )

    return {
        "response": response,
        "trace": [trace],
    }


def _build_error_response(state: AgentState, started_at: str, t0: float) -> dict:
    """Build an error response when prediction is missing."""
    duration_ms = (time.perf_counter() - t0) * 1000
    completed_at = datetime.now(timezone.utc).isoformat()

    error = NodeError(
        node="aggregator",
        error_type="missing_prediction",
        message="Cannot build response: no prediction available in state",
        timestamp=state.get("timestamp", ""),
        recoverable=False,
    )

    trace = TraceEntry(
        node="aggregator",
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=round(duration_ms, 2),
        status="error",
    )

    return {
        "errors": [error],
        "trace": [trace],
    }
