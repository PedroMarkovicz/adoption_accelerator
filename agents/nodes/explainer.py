"""
Explainer node --- LLM-powered natural-language explanation generation.

Reads the prediction, interpreted explanation, and request from state,
formats a prompt using the template, calls the OpenAI API, validates
the output, and writes the narrative explanation to state.

On failure, falls back to a deterministic template-based explanation.

Issue #3 fix: filters baseline factors (absent-modality SHAP offsets)
from the top factors text before sending to the LLM.  Adds explicit
prompt guardrails and post-generation hallucination detection so that
the narrative cannot reference non-existent description or photo content.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.guardrails.fallbacks import generate_fallback_explanation
from agents.guardrails.validators import validate_explanation
from agents.state import AgentState, NodeError, TraceEntry

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

# Phrases that indicate the LLM is hallucinating about absent modalities.
# If has_description is False and these appear, the explanation is invalid.
_DESCRIPTION_HALLUCINATION_PATTERNS = [
    r"description content",
    r"text pattern",
    r"description pattern",
    r"write-up",
    r"listing text",
    r"text-based",
    r"semantic pattern",
    r"wording of the listing",
]

# Same for absent image modality
_IMAGE_HALLUCINATION_PATTERNS = [
    r"photo quality",
    r"image pattern",
    r"visual pattern",
    r"photo content",
    r"image quality",
    r"picture quality",
]


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    path = _PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _format_pet_profile(request: Any) -> dict[str, str]:
    """Extract pet profile fields from a PredictionRequest."""
    t = request.tabular
    type_map = {1: "Dog", 2: "Cat"}
    gender_map = {1: "Male", 2: "Female", 3: "Mixed"}
    care_map = {1: "Yes", 2: "No", 3: "Not Sure"}

    return {
        "pet_type": type_map.get(t.type, str(t.type)),
        "age": str(t.age),
        "gender": gender_map.get(t.gender, str(t.gender)),
        "fee": str(t.fee),
        "photo_count": str(len(request.images)) if request.images else "0",
        "has_description": "Yes" if request.description else "No",
        "vaccinated": care_map.get(t.vaccinated, str(t.vaccinated)),
        "dewormed": care_map.get(t.dewormed, str(t.dewormed)),
        "sterilized": care_map.get(t.sterilized, str(t.sterilized)),
        "health": {1: "Healthy", 2: "Minor Injury", 3: "Serious Injury"}.get(
            t.health, str(t.health)
        ),
        "breed": str(t.breed1),
    }


def _build_modality_breakdown(
    modality_contributions: dict[str, float],
    modality_available: dict[str, bool] | None = None,
) -> str:
    """Format modality contributions for the prompt.

    Issue #2/3 fix: only present modalities with real contributions
    are included.  Absent modalities are excluded from percentages
    by the aggregation layer, so they simply won't appear here.
    """
    lines = []
    for mod, pct in sorted(
        modality_contributions.items(), key=lambda x: x[1], reverse=True
    ):
        if pct > 0.01:
            label = {
                "tabular": "Listing attributes",
                "text": "Description content",
                "image": "Photo characteristics",
                "metadata": "Image metadata",
            }.get(mod, mod)
            lines.append(f"- {label}: {pct:.1%}")

    # Explicitly mention absent modalities so the LLM can reason about
    # their absence rather than hallucinating their presence.
    if modality_available is not None:
        if not modality_available.get("text", True):
            lines.append("- Description content: NOT PROVIDED (no text submitted)")
        if not modality_available.get("image", True):
            lines.append("- Photo characteristics: NOT PROVIDED (no images submitted)")

    return "\n".join(lines) if lines else "- No modality data available"


def _build_top_factors_text(
    top_factors: list,
    modality_available: dict[str, bool] | None = None,
) -> str:
    """Format interpreted factors for the prompt.

    Issue #3 fix: factors from absent modalities (is_baseline=True) are
    excluded before being sent to the LLM, preventing the generation of
    hallucinated references to non-existent content.
    """
    lines = []
    for i, f in enumerate(top_factors, 1):
        if isinstance(f, dict):
            name = f.get("name", "")
            desc = f.get("description", "")
            magnitude = f.get("shap_magnitude", 0)
            direction = f.get("direction", "")
            modality = f.get("modality", "")
            is_baseline = f.get("is_baseline", False)
        else:
            name = getattr(f, "name", "")
            desc = getattr(f, "description", "")
            magnitude = getattr(f, "shap_magnitude", 0)
            direction = getattr(f, "direction", "")
            modality = getattr(f, "modality", "")
            is_baseline = getattr(f, "is_baseline", False)

        # Issue #3: skip baseline factors entirely — they represent
        # zero-vector offsets from absent modalities, not real signals.
        if is_baseline:
            continue

        # Additional safety gate using modality_available
        if modality_available is not None:
            if modality == "text" and not modality_available.get("text", True):
                continue
            if modality == "image" and not modality_available.get("image", True):
                continue

        dir_arrow = "+" if direction == "positive" else "-"
        lines.append(
            f"{i}. [{dir_arrow}] {desc} (modality: {modality}, importance: {magnitude:.4f})"
        )

    if not lines:
        return "- Only listing attributes influenced this prediction (no description or photos were provided)"
    return "\n".join(lines)


def _detect_hallucination(
    narrative: str,
    modality_available: dict[str, bool],
) -> tuple[bool, str]:
    """Check if the narrative hallucinates about absent modalities.

    Returns (is_hallucination, reason).
    """
    text_lower = narrative.lower()

    if not modality_available.get("text", True):
        for pattern in _DESCRIPTION_HALLUCINATION_PATTERNS:
            if re.search(pattern, text_lower):
                return (
                    True,
                    f"Narrative references description content (pattern: '{pattern}') "
                    f"but has_description=False",
                )

    if not modality_available.get("image", True):
        for pattern in _IMAGE_HALLUCINATION_PATTERNS:
            if re.search(pattern, text_lower):
                return (
                    True,
                    f"Narrative references image content (pattern: '{pattern}') "
                    f"but n_images=0",
                )

    return False, ""


def _detect_coherence_issues(
    narrative: str,
    modality_contributions: dict[str, float],
    modality_available: dict[str, bool],
) -> tuple[bool, str]:
    """Detect logical contradictions in the narrative.

    Catches cases where the narrative simultaneously attributes prediction
    influence to a modality AND claims that modality was absent.

    Returns (has_issue, reason).
    """
    text_lower = narrative.lower()

    # Check: if text modality contributes significantly, narrative should
    # not claim description was absent/not provided
    text_contribution = modality_contributions.get("text", 0.0)
    text_present = modality_available.get("text", True)

    if text_present and text_contribution > 0.3:
        absence_patterns = [
            r"no description",
            r"no text",
            r"without a description",
            r"no description content",
            r"missing description",
            r"description was not",
            r"no listing text",
        ]
        for pattern in absence_patterns:
            if re.search(pattern, text_lower):
                # Allow phrases like "no description or photos were needed"
                # but catch contradictions like "strongest factor is description
                # ... no description content to reference"
                return (
                    True,
                    f"Narrative claims description is absent ('{pattern}') "
                    f"but text modality contributes {text_contribution:.0%}",
                )

    # Same check for images
    image_contribution = modality_contributions.get("image", 0.0)
    image_present = modality_available.get("image", True)

    if image_present and image_contribution > 0.3:
        absence_patterns = [
            r"no photos",
            r"no images",
            r"without photos",
            r"photos were not",
            r"missing photos",
        ]
        for pattern in absence_patterns:
            if re.search(pattern, text_lower):
                return (
                    True,
                    f"Narrative claims photos are absent ('{pattern}') "
                    f"but image modality contributes {image_contribution:.0%}",
                )

    return False, ""


def explainer_node(state: AgentState) -> dict:
    """Generate a natural-language explanation of the prediction.

    Parameters
    ----------
    state : AgentState
        Must contain ``prediction``, ``interpreted_explanation``,
        and ``request``.

    Returns
    -------
    dict
        State updates: ``narrative_explanation``, ``trace``, ``errors``.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timestamp = state.get("timestamp", "")

    prediction = state.get("prediction")
    interpreted = state.get("interpreted_explanation")
    request = state.get("request")
    # Issue #2/3: read modality availability from state
    modality_available: dict[str, bool] = state.get("modality_available") or {}

    # If missing critical data, generate fallback immediately
    if prediction is None or interpreted is None:
        narrative = _generate_fallback(state)
        return _build_result(narrative, started_at, t0, used_fallback=True)

    try:
        # Build prompt context
        profile = _format_pet_profile(request)
        modality_breakdown = _build_modality_breakdown(
            interpreted.modality_contributions,
            modality_available=modality_available,
        )
        # Issue #3: pass modality_available so baseline factors are excluded
        top_factors_text = _build_top_factors_text(
            interpreted.top_factors,
            modality_available=modality_available,
        )

        class_probs = ", ".join(
            f"Class {k}: {v:.1%}" for k, v in prediction.probabilities.items()
        )

        # Load and format prompts
        system_prompt = _load_prompt("explainer_system.txt")
        user_template = _load_prompt("explainer_user.txt")

        user_prompt = user_template.format(
            predicted_class=prediction.prediction,
            prediction_label=prediction.prediction_label,
            confidence=prediction.confidence,
            class_probabilities=class_probs,
            modality_breakdown=modality_breakdown,
            top_factors=top_factors_text,
            **profile,
        )

        # Call the LLM
        from agents.llm import call_llm

        llm_result = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            node_name="explainer",
            verbosity="low",
        )

        narrative = llm_result["text"].strip()
        usage = llm_result.get("usage", {})

        # Validate the output (schema-level check)
        is_valid, reason = validate_explanation(narrative, prediction.prediction)
        if not is_valid:
            logger.warning("Explanation validation failed: %s. Using fallback.", reason)
            narrative = _generate_fallback(state)
            return _build_result(
                narrative, started_at, t0,
                used_fallback=True,
                metadata={"validation_failure": reason, "llm_usage": usage},
            )

        # Issue #3: post-generation hallucination check
        if modality_available:
            is_hallucination, hallucination_reason = _detect_hallucination(
                narrative, modality_available
            )
            if is_hallucination:
                logger.warning(
                    "Hallucination detected in explanation: %s. Using fallback.",
                    hallucination_reason,
                )
                narrative = _generate_fallback(state)
                return _build_result(
                    narrative, started_at, t0,
                    used_fallback=True,
                    metadata={
                        "validation_failure": hallucination_reason,
                        "llm_usage": usage,
                    },
                )

        # Coherence check: detect contradictions between attribution and
        # claims about modality absence (e.g., "description is strongest
        # factor" followed by "no description was provided").
        if modality_available and interpreted is not None:
            has_issue, coherence_reason = _detect_coherence_issues(
                narrative,
                interpreted.modality_contributions,
                modality_available,
            )
            if has_issue:
                logger.warning(
                    "Coherence issue in explanation: %s. Using fallback.",
                    coherence_reason,
                )
                narrative = _generate_fallback(state)
                return _build_result(
                    narrative, started_at, t0,
                    used_fallback=True,
                    metadata={
                        "validation_failure": coherence_reason,
                        "llm_usage": usage,
                    },
                )

        return _build_result(
            narrative, started_at, t0,
            metadata={"llm_usage": usage, "llm_latency_ms": llm_result.get("latency_ms", 0)},
        )

    except Exception as exc:
        logger.warning("Explainer LLM call failed: %s. Using fallback.", exc)
        narrative = _generate_fallback(state)
        return _build_result(
            narrative, started_at, t0,
            used_fallback=True,
            errors=[
                NodeError(
                    node="explainer",
                    error_type="llm_failure",
                    message=str(exc),
                    timestamp=timestamp,
                    recoverable=True,
                )
            ],
        )


def _generate_fallback(state: AgentState) -> str:
    """Generate a deterministic fallback explanation."""
    prediction = state.get("prediction")
    interpreted = state.get("interpreted_explanation")

    if prediction is None:
        return "Unable to generate explanation: prediction data is unavailable."

    modality_contributions = {}
    top_factors_dicts: list[dict[str, Any]] = []

    if interpreted is not None:
        modality_contributions = interpreted.modality_contributions
        top_factors_dicts = [
            {
                "name": f.name,
                "description": f.description,
                "shap_magnitude": f.shap_magnitude,
                "direction": f.direction,
                "modality": f.modality,
                "is_baseline": f.is_baseline,
            }
            for f in interpreted.top_factors
            if not f.is_baseline  # exclude baseline from fallback too
        ]

    return generate_fallback_explanation(
        predicted_class=prediction.prediction,
        prediction_label=prediction.prediction_label,
        confidence=prediction.confidence,
        modality_contributions=modality_contributions,
        top_factors=top_factors_dicts,
    )


def _build_result(
    narrative: str,
    started_at: str,
    t0: float,
    *,
    used_fallback: bool = False,
    metadata: dict[str, Any] | None = None,
    errors: list[NodeError] | None = None,
) -> dict:
    """Build the node return dict."""
    duration_ms = (time.perf_counter() - t0) * 1000
    completed_at = datetime.now(timezone.utc).isoformat()

    trace_meta = {"used_fallback": used_fallback}
    if metadata:
        trace_meta.update(metadata)

    trace = TraceEntry(
        node="explainer",
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=round(duration_ms, 2),
        status="success",
        metadata=trace_meta,
    )

    result: dict[str, Any] = {
        "narrative_explanation": narrative,
        "trace": [trace],
        "errors": errors or [],
    }
    return result
