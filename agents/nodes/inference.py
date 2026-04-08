"""
Inference node --- deterministic ML prediction + SHAP explanation.

Loads the cached model bundle, builds the 940-dim feature vector from
the ``PredictionRequest``, runs prediction and SHAP explanation, and
applies the Feature Interpretation Layer to translate raw SHAP into
an ``InterpretedExplanation``.

This is a deterministic node with no LLM dependency.
Target latency: < 1 second.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from agents.state import AgentState, NodeError, TraceEntry

logger = logging.getLogger(__name__)


def inference_node(state: AgentState) -> dict:
    """Execute prediction + explanation for the request in state.

    Orchestrates the full deterministic inference pipeline:
    1. Build feature vector from PredictionRequest
    2. Run model prediction (predict_single)
    3. Compute SHAP values
    4. Apply Feature Interpretation Layer (aggregate + translate)
    5. Write results to state

    Parameters
    ----------
    state : AgentState
        Must contain a valid ``request`` field.

    Returns
    -------
    dict
        State updates: ``prediction``, ``explanation``,
        ``interpreted_explanation``, ``feature_vector``,
        ``feature_names``, ``trace``, and ``errors`` on failure.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timestamp = state.get("timestamp", datetime.now(timezone.utc).isoformat())

    request = state.get("request")
    if request is None:
        return _error_result(
            "missing_request",
            "No PredictionRequest in state",
            started_at,
            timestamp,
        )

    # Check for fatal orchestrator errors
    errors = state.get("errors", [])
    fatal = [e for e in errors if not e.recoverable]
    if fatal:
        return _error_result(
            "upstream_error",
            f"Fatal error from upstream node: {fatal[0].message}",
            started_at,
            timestamp,
        )

    try:
        # Step 1: Run prediction tool (builds feature vector + predicts)
        from agents.tools.prediction_tool import run_prediction

        pred_output = run_prediction(request)
        prediction_result = pred_output["prediction_result"]
        feature_vector = pred_output["feature_vector"]
        feature_names = pred_output["feature_names"]

        prediction_ms = (time.perf_counter() - t0) * 1000

        # Compute modality availability from the raw request.
        # This must be done before the explanation tool so that absent
        # modalities are excluded from SHAP contribution percentages.
        from adoption_accelerator.interpretability.modality_utils import (
            detect_modality_availability,
        )
        modality_available = detect_modality_availability(request)

        # Step 2: Run explanation tool (SHAP + interpretation)
        from agents.tools.explanation_tool import run_explanation

        probabilities = [
            prediction_result.probabilities[i]
            for i in range(len(prediction_result.probabilities))
        ]

        expl_output = run_explanation(
            feature_vector=feature_vector,
            feature_names=feature_names,
            predicted_class=prediction_result.prediction,
            probabilities=probabilities,
            confidence=prediction_result.confidence,
            modality_available=modality_available,
        )

        explanation_result = expl_output["explanation_result"]
        interpreted_explanation = expl_output["interpreted_explanation"]

        total_ms = (time.perf_counter() - t0) * 1000
        completed_at = datetime.now(timezone.utc).isoformat()

        trace = TraceEntry(
            node="inference",
            started_at=started_at,
            completed_at=completed_at,
            duration_ms=round(total_ms, 2),
            status="success",
            metadata={
                "prediction_ms": round(prediction_ms, 2),
                "total_ms": round(total_ms, 2),
                "predicted_class": prediction_result.prediction,
                "confidence": round(prediction_result.confidence, 4),
                "n_features": len(feature_names),
            },
        )

        logger.info(
            "Inference node: class=%d, confidence=%.4f, "
            "prediction=%.1fms, total=%.1fms",
            prediction_result.prediction,
            prediction_result.confidence,
            prediction_ms,
            total_ms,
        )

        return {
            "prediction": prediction_result,
            "explanation": explanation_result,
            "interpreted_explanation": interpreted_explanation,
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "modality_available": modality_available,
            "trace": [trace],
            "errors": [],
        }

    except Exception as exc:
        logger.exception("Inference node failed: %s", exc)
        return _error_result(
            "inference_failure",
            str(exc),
            started_at,
            timestamp,
        )


def _error_result(
    error_type: str,
    message: str,
    started_at: str,
    timestamp: str,
) -> dict[str, Any]:
    """Build an error return dict for the inference node."""
    completed_at = datetime.now(timezone.utc).isoformat()
    return {
        "errors": [
            NodeError(
                node="inference",
                error_type=error_type,
                message=message,
                timestamp=timestamp,
                recoverable=False,
            )
        ],
        "trace": [
            TraceEntry(
                node="inference",
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=0.0,
                status="error",
                metadata={"error_type": error_type},
            )
        ],
    }
