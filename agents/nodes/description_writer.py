"""
Description writer node --- LLM-powered pet description optimization.

Only runs when the request has a non-empty description. Reads the
original description, text-modality SHAP data, and pet profile from
state, then generates an improved version optimized for adoption.

On failure, returns ``None`` (the original description remains).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.guardrails.validators import validate_description
from agents.state import AgentState, NodeError, TraceEntry

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _build_text_insights(interpreted: Any) -> str:
    """Build text-modality SHAP insights from aggregated embeddings."""
    lines = []

    if interpreted is None:
        return "No text analysis data available."

    # Check aggregated embeddings for text semantic patterns
    agg = interpreted.aggregated_embeddings
    if "text_semantic_patterns" in agg:
        entry = agg["text_semantic_patterns"]
        lines.append(f"- {entry.get('description', 'Text patterns influenced the prediction')}")

    # Include text-related factors from top factors
    for f in interpreted.top_factors:
        if f.modality == "text" and not f.group.endswith("_embedding"):
            dir_word = "positively" if f.direction == "positive" else "negatively"
            lines.append(f"- {f.description} {dir_word} influenced the prediction")

    return "\n".join(lines) if lines else "Text analysis suggests the description could be improved."


def description_writer_node(state: AgentState) -> dict:
    """Generate an improved pet description.

    Only produces output when the request has a non-empty description.
    On failure, returns ``None`` for ``improved_description``.

    Parameters
    ----------
    state : AgentState
        Must contain ``request`` with a non-empty description.

    Returns
    -------
    dict
        State updates: ``improved_description``, ``trace``, ``errors``.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timestamp = state.get("timestamp", "")

    request = state.get("request")
    interpreted = state.get("interpreted_explanation")

    # Skip if no description to improve
    if request is None or not request.description or not request.description.strip():
        logger.info("Description writer: skipping (no description in request)")
        return _build_result(None, started_at, t0, skipped=True)

    try:
        # Build pet profile
        t = request.tabular
        type_map = {1: "Dog", 2: "Cat"}
        gender_map = {1: "Male", 2: "Female", 3: "Mixed"}
        care_map = {1: "Yes", 2: "No", 3: "Not Sure"}
        health_map = {1: "Healthy", 2: "Minor Injury", 3: "Serious Injury"}

        pet_type = type_map.get(t.type, str(t.type))
        text_insights = _build_text_insights(interpreted)

        system_prompt = _load_prompt("writer_system.txt")
        user_template = _load_prompt("writer_user.txt")

        user_prompt = user_template.format(
            pet_type=pet_type,
            age=str(t.age),
            gender=gender_map.get(t.gender, str(t.gender)),
            breed=str(t.breed1),
            health=health_map.get(t.health, str(t.health)),
            vaccinated=care_map.get(t.vaccinated, str(t.vaccinated)),
            sterilized=care_map.get(t.sterilized, str(t.sterilized)),
            original_description=request.description,
            text_insights=text_insights,
        )

        # Call the LLM with medium verbosity for richer text
        from agents.llm import call_llm

        llm_result = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            node_name="description_writer",
            verbosity="medium",
        )

        improved = llm_result["text"].strip()
        usage = llm_result.get("usage", {})

        # Validate the output
        is_valid, reason = validate_description(improved, pet_type=pet_type)
        if not is_valid:
            logger.warning(
                "Description validation failed: %s. Returning None.", reason
            )
            return _build_result(
                None, started_at, t0,
                metadata={"validation_failure": reason, "llm_usage": usage},
                warnings=[f"Description improvement was generated but failed quality validation: {reason}. The original description is preserved."],
            )

        return _build_result(
            improved, started_at, t0,
            metadata={"llm_usage": usage, "llm_latency_ms": llm_result.get("latency_ms", 0)},
        )

    except Exception as exc:
        logger.warning("Description writer LLM call failed: %s. Returning None.", exc)
        return _build_result(
            None, started_at, t0,
            errors=[
                NodeError(
                    node="description_writer",
                    error_type="llm_failure",
                    message=str(exc),
                    timestamp=timestamp,
                    recoverable=True,
                )
            ],
        )


def _build_result(
    improved_description: str | None,
    started_at: str,
    t0: float,
    *,
    skipped: bool = False,
    metadata: dict[str, Any] | None = None,
    errors: list[NodeError] | None = None,
    warnings: list[str] | None = None,
) -> dict:
    duration_ms = (time.perf_counter() - t0) * 1000
    completed_at = datetime.now(timezone.utc).isoformat()

    trace_meta: dict[str, Any] = {
        "skipped": skipped,
        "has_output": improved_description is not None,
    }
    if metadata:
        trace_meta.update(metadata)

    trace = TraceEntry(
        node="description_writer",
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=round(duration_ms, 2),
        status="skipped" if skipped else "success",
        metadata=trace_meta,
    )

    return {
        "improved_description": improved_description,
        "trace": [trace],
        "errors": errors or [],
        "warnings": warnings or [],
    }
