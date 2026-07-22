"""synthesizer node — fuses the evidence board into narrative, headline,
and a grounded optimized description."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from adoption_accelerator.agents.contracts import NodeError, TraceEntry
from adoption_accelerator.agents.llm.client import extract_usage, get_chat_model
from adoption_accelerator.agents.llm.registry import resolve_role
from adoption_accelerator.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_TIMEOUT_SECONDS = 20.0

# Visual-trait phrases that must be grounded in observed_traits when a
# description mentions them. Deliberately narrow: colors/eyes/coat.
_VISUAL_TRAIT_PATTERNS = [
    r"blue eyes", r"green eyes", r"amber eyes",
    r"black and white", r"brown coat", r"white coat", r"black coat",
    r"golden coat", r"spotted", r"striped", r"fluffy coat",
]


class SynthesisOutput(BaseModel):
    narrative: str
    headline: str
    optimized_description: Optional[str] = None


def _violates_grounding(description: str, observed_traits: list[str]) -> str | None:
    """Return the offending trait phrase when the description claims a visual
    trait whose significant words are not all present in observed_traits."""
    text = description.lower()
    observed = " ".join(observed_traits).lower()
    for pattern in _VISUAL_TRAIT_PATTERNS:
        # Does the description actually claim this trait? (word-boundary, so
        # "spotted" does not match inside "unspotted")
        if not re.search(rf"\b{re.escape(pattern)}\b", text):
            continue
        # Grounded if every significant word of the trait appears (word-boundary)
        # in observed_traits. "and" is not significant.
        words = [w for w in pattern.split() if w != "and"]
        if all(re.search(rf"\b{re.escape(w)}\b", observed) for w in words):
            continue  # grounded -> not a violation
        return pattern
    return None


def _build_user_prompt(state: AgentState) -> str:
    ev = state["prediction_evidence"]
    lines = [
        "PREDICTION EVIDENCE:",
        f"- class {ev.predicted_class} ({ev.prediction_label}), "
        f"confidence {ev.class_confidence:.1%}",
        f"- uncertainty: {ev.uncertainty_reading}",
    ]
    for d in ev.key_drivers:
        lines.append(f"- [{d.direction}] {d.reading}")

    visual = state.get("visual_evidence")
    if visual is not None:
        lines.append("VISUAL EVIDENCE:")
        lines.append(f"- overall appeal: {visual.overall_visual_appeal}/10")
        lines.append(f"- observed traits: {', '.join(visual.observed_traits) or 'none'}")
        lines.append(f"- strategy: {visual.photo_strategy_summary}")
        for flag in visual.consistency_flags:
            lines.append(f"- consistency flag: {flag}")
    else:
        lines.append("VISUAL EVIDENCE: none (no photos were provided/analyzed)")

    recs = state.get("recommendation_evidence")
    if recs is not None and recs.recommendations:
        lines.append("VALIDATED RECOMMENDATIONS (measured by the real model):")
        for r in recs.recommendations:
            lines.append(
                f"- P{r.priority} {r.action}: {r.measured_impact.expected_speedup}"
            )
    else:
        lines.append("VALIDATED RECOMMENDATIONS: none")

    request = state.get("request")
    t = request.tabular
    lines.append("LISTING DATA:")
    lines.append(
        f"- {'Dog' if t.type == 1 else 'Cat'}, age {t.age} months, "
        f"gender code {t.gender}, fee {t.fee}, name: {t.name or '(none)'}"
    )
    lines.append(f"- original description: {request.description or '(none)'}")
    return "\n".join(lines)


def _fallback_narrative(state: AgentState) -> str:
    ev = state["prediction_evidence"]
    parts = [
        f"The model predicts: {ev.prediction_label} "
        f"(confidence {ev.class_confidence:.0%})."
    ]
    recs = state.get("recommendation_evidence")
    if recs is not None and recs.recommendations:
        top = recs.recommendations[0]
        parts.append(f"Top validated action: {top.action} "
                     f"({top.measured_impact.expected_speedup}).")
    return " ".join(parts)


async def synthesizer_node(state: AgentState) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timestamp = state.get("timestamp", "")

    if state.get("prediction_evidence") is None:
        return {
            "narrative": "Prediction unavailable.", "headline": "",
            "optimized_description": None,
            "errors": [NodeError(node="synthesizer", error_type="missing_input",
                                 message="prediction_evidence missing",
                                 timestamp=timestamp, recoverable=True)],
            "trace": [_trace(started_at, t0, "error", {})],
        }

    errors: list[NodeError] = []
    meta: dict = {}
    try:
        system = (_PROMPTS_DIR / "synthesizer_system.txt").read_text(
            encoding="utf-8"
        )
        model = get_chat_model("synthesizer").with_structured_output(
            SynthesisOutput, include_raw=True
        )
        result = await asyncio.wait_for(
            model.ainvoke([("system", system),
                           ("user", _build_user_prompt(state))]),
            timeout=_TIMEOUT_SECONDS,
        )
        output: SynthesisOutput = result["parsed"]
        raw = result["raw"]

        narrative = output.narrative
        headline = output.headline
        description = output.optimized_description

        # Grounding gate for the description
        if description:
            visual = state.get("visual_evidence")
            observed = visual.observed_traits if visual is not None else []
            offending = _violates_grounding(description, observed)
            if offending is not None:
                logger.warning(
                    "Dropping ungrounded description (claims '%s')", offending
                )
                meta["description_dropped"] = offending
                description = None

        resolved = resolve_role("synthesizer")
        meta["model"] = resolved.api_model
        meta["llm_usage"] = extract_usage(raw, resolved.model_key)
        return {
            "narrative": narrative, "headline": headline,
            "optimized_description": description,
            "errors": errors,
            "trace": [_trace(started_at, t0, "success", meta)],
        }

    except Exception as exc:
        logger.warning("synthesizer failed: %s. Template fallback.", exc)
        errors.append(NodeError(node="synthesizer", error_type="llm_failure",
                                message=str(exc), timestamp=timestamp,
                                recoverable=True))
        return {
            "narrative": _fallback_narrative(state),
            "headline": state["prediction_evidence"].prediction_label,
            "optimized_description": None,
            "errors": errors,
            "trace": [_trace(started_at, t0, "success", {"used_fallback": True})],
        }


def _trace(started_at: str, t0: float, status: str, metadata: dict) -> TraceEntry:
    return TraceEntry(
        node="synthesizer",
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        status=status,
        metadata=metadata,
    )
