"""
Recommender node --- LLM-powered actionable recommendation generation.

Reads the prediction, interpreted explanation, and request from state,
formats a prompt with actionable features and SHAP context, calls the
OpenAI API with structured output, validates recommendations, and
writes them to state.

On failure, falls back to rule-based recommendations.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.guardrails.fallbacks import generate_fallback_recommendations
from agents.guardrails.validators import validate_recommendations
from agents.state import AgentState, NodeError, Recommendation, TraceEntry

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# JSON Schema for structured output enforcement
_RECOMMENDATION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "current_value": {"type": "string"},
                    "suggested_value": {"type": "string"},
                    "expected_impact": {"type": "string"},
                    "priority": {"type": "integer"},
                    "category": {
                        "type": "string",
                        "enum": ["photo", "description", "health", "listing_details"],
                    },
                    "actionable": {"type": "boolean"},
                },
                "required": [
                    "feature", "current_value", "suggested_value",
                    "expected_impact", "priority", "category", "actionable",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["recommendations"],
    "additionalProperties": False,
}


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_current_values(request: Any) -> dict[str, Any]:
    """Extract current actionable feature values from the request."""
    t = request.tabular
    care_map = {1: "Yes", 2: "No", 3: "Not Sure"}

    return {
        "photo_count": str(len(request.images)) if request.images else "0",
        "video_count": str(t.video_amt),
        "fee": str(t.fee),
        "vaccinated": care_map.get(t.vaccinated, str(t.vaccinated)),
        "dewormed": care_map.get(t.dewormed, str(t.dewormed)),
        "sterilized": care_map.get(t.sterilized, str(t.sterilized)),
        "has_name": "Yes" if t.name and t.name.strip() else "No",
        "description_length": str(len(request.description)) if request.description else "0",
    }


def _build_negative_factors_text(interpreted: Any) -> str:
    """Format negative factors for the prompt, excluding non-actionable latent features."""
    lines = []
    for f in interpreted.top_factors:
        if f.direction != "negative":
            continue
        # Skip latent embedding factors (not actionable)
        if f.group in ("image_embedding", "text_embedding"):
            continue
        lines.append(
            f"- {f.description} (modality: {f.modality}, importance: {f.shap_magnitude:.4f})"
        )

    return "\n".join(lines) if lines else "- No specific negative factors identified"


def _build_modality_breakdown(modality_contributions: dict[str, float]) -> str:
    lines = []
    for mod, pct in sorted(modality_contributions.items(), key=lambda x: x[1], reverse=True):
        if pct > 0.01:
            label = {
                "tabular": "Listing attributes",
                "text": "Description content",
                "image": "Photo characteristics",
                "metadata": "Image metadata",
            }.get(mod, mod)
            lines.append(f"- {label}: {pct:.1%}")
    return "\n".join(lines) if lines else "- No modality data available"


def recommender_node(state: AgentState) -> dict:
    """Generate actionable recommendations for improving adoption speed.

    Parameters
    ----------
    state : AgentState
        Must contain ``prediction``, ``interpreted_explanation``,
        and ``request``.

    Returns
    -------
    dict
        State updates: ``recommendations``, ``trace``, ``errors``.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timestamp = state.get("timestamp", "")

    prediction = state.get("prediction")
    interpreted = state.get("interpreted_explanation")
    request = state.get("request")

    if prediction is None or interpreted is None or request is None:
        recommendations = _generate_fallback(state)
        return _build_result(recommendations, started_at, t0, used_fallback=True)

    try:
        current_values = _extract_current_values(request)
        negative_factors = _build_negative_factors_text(interpreted)
        modality_breakdown = _build_modality_breakdown(interpreted.modality_contributions)

        system_prompt = _load_prompt("recommender_system.txt")
        user_template = _load_prompt("recommender_user.txt")

        user_prompt = user_template.format(
            predicted_class=prediction.prediction,
            prediction_label=prediction.prediction_label,
            negative_factors=negative_factors,
            modality_breakdown=modality_breakdown,
            **current_values,
        )

        # Run counterfactual analysis
        try:
            from adoption_accelerator.interpretability.counterfactual import generate_counterfactuals
            # Tool takes args as dict if called as a LangChain tool, or directly as a python func. Provide kwargs:
            cf_results = generate_counterfactuals.invoke({
                "request": request,
                "target_class": max(0, prediction.prediction - 1),
                "current_class": prediction.prediction,
            })
            if cf_results:
                cf_text = "\n".join([
                    f"- If {r['feature']} changed to '{r['suggested_value']}': {r['expected_impact']}"
                    for r in cf_results
                ])
                user_prompt += f"\n\nQuantitative Counterfactual Impacts (USE THESE EXACT NUMBERS IF APPLICABLE):\n{cf_text}\n"
        except Exception as e:
            logger.warning("Counterfactual analysis failed: %s", e)

        # Call the LLM with structured output
        from agents.llm import call_llm

        llm_result = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            node_name="recommender",
            verbosity="low",
            json_schema=_RECOMMENDATION_JSON_SCHEMA,
        )

        raw_text = llm_result["text"].strip()
        usage = llm_result.get("usage", {})

        # Parse JSON response
        parsed = json.loads(raw_text)
        raw_recs = parsed.get("recommendations", parsed) if isinstance(parsed, dict) else parsed

        # Validate recommendations
        valid_recs = validate_recommendations(raw_recs, max_count=5)

        # Convert to Recommendation objects, filter no-ops, deduplicate
        recommendations = []
        seen_features: set[str] = set()
        rank = 1
        for r in valid_recs:
            curr = str(r.get("current_value", "")).strip().lower()
            sugg = str(r.get("suggested_value", "")).strip().lower()
            impact = str(r.get("expected_impact", "")).strip().lower()
            feature_key = str(r.get("feature", "")).strip().lower()

            if curr == sugg or "no change" in impact or "no impact" in impact:
                continue

            # Deduplicate: keep only the first (highest-priority) recommendation per feature
            if feature_key in seen_features:
                continue
            seen_features.add(feature_key)

            recommendations.append(
                Recommendation(
                    feature=r.get("feature", ""),
                    current_value=r.get("current_value", ""),
                    suggested_value=r.get("suggested_value", ""),
                    expected_impact=r.get("expected_impact", ""),
                    priority=rank,
                    category=r.get("category", "listing_details"),
                    actionable=r.get("actionable", True),
                )
            )
            rank += 1

        if not recommendations:
            logger.warning("No valid recommendations after filtering. Using fallback.")
            recommendations = _generate_fallback(state)
            return _build_result(
                recommendations, started_at, t0,
                used_fallback=True,
                metadata={"validation_note": "all LLM recs filtered", "llm_usage": usage},
            )

        return _build_result(
            recommendations, started_at, t0,
            metadata={"llm_usage": usage, "llm_latency_ms": llm_result.get("latency_ms", 0)},
        )

    except Exception as exc:
        logger.warning("Recommender LLM call failed: %s. Using fallback.", exc)
        recommendations = _generate_fallback(state)
        return _build_result(
            recommendations, started_at, t0,
            used_fallback=True,
            errors=[
                NodeError(
                    node="recommender",
                    error_type="llm_failure",
                    message=str(exc),
                    timestamp=timestamp,
                    recoverable=True,
                )
            ],
        )


def _generate_fallback(state: AgentState) -> list[Recommendation]:
    """Generate rule-based fallback recommendations."""
    prediction = state.get("prediction")
    interpreted = state.get("interpreted_explanation")
    request = state.get("request")

    predicted_class = prediction.prediction if prediction else 4

    top_negative: list[dict[str, Any]] = []
    if interpreted is not None:
        top_negative = [
            {
                "name": f.name,
                "description": f.description,
                "shap_magnitude": f.shap_magnitude,
                "direction": f.direction,
                "modality": f.modality,
            }
            for f in interpreted.top_factors
            if f.direction == "negative"
        ]

    current_values: dict[str, Any] = {}
    if request is not None:
        current_values = _extract_current_values(request)

    return generate_fallback_recommendations(
        predicted_class=predicted_class,
        top_negative_factors=top_negative,
        current_values=current_values,
    )


def _build_result(
    recommendations: list[Recommendation],
    started_at: str,
    t0: float,
    *,
    used_fallback: bool = False,
    metadata: dict[str, Any] | None = None,
    errors: list[NodeError] | None = None,
) -> dict:
    duration_ms = (time.perf_counter() - t0) * 1000
    completed_at = datetime.now(timezone.utc).isoformat()

    trace_meta: dict[str, Any] = {
        "used_fallback": used_fallback,
        "n_recommendations": len(recommendations),
    }
    if metadata:
        trace_meta.update(metadata)

    trace = TraceEntry(
        node="recommender",
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=round(duration_ms, 2),
        status="success",
        metadata=trace_meta,
    )

    return {
        "recommendations": recommendations,
        "trace": [trace],
        "errors": errors or [],
    }
