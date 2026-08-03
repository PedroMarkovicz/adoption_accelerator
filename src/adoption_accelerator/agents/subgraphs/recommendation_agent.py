"""Bounded ReAct recommendation agent.

Loop: the LLM proposes hypotheses and tests them through tools that
re-run the real ensemble (max ``MAX_TOOL_CALLS``). A finalize step then
asks for structured recommendations in which every item must cite a
``measurement_id``; ``MeasuredImpact`` is built exclusively from the
``MeasurementLog`` — the LLM can choose and rank, but never invent
numbers. On total LLM failure, a deterministic counterfactual sweep
still produces measured recommendations (``generated_by="deterministic"``)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field

from adoption_accelerator.agents.contracts import (
    MeasuredImpact,
    NodeError,
    RecommendationEvidence,
    TraceEntry,
    ValidatedRecommendation,
)
from adoption_accelerator.agents.llm.client import extract_usage, get_chat_model
from adoption_accelerator.agents.llm.registry import resolve_role
from adoption_accelerator.agents.runtime_config import (
    MAX_TOOL_CALLS,
    SPEEDUP_EPSILON,
    node_timeout,
)
from adoption_accelerator.agents.state import AgentState
from adoption_accelerator.agents.tools.actionable_features import (
    SWEEP_MAX_RECOMMENDATIONS,
    SWEEP_MIN_SHIFT,
    current_value,
    sweep_candidates,
)
from adoption_accelerator.agents.tools.recommendation_tools import (
    ACTIONABLE_FEATURES,
    MeasurementLog,
    make_recommendation_tools,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

from adoption_accelerator.target_labels import labels

_CLASS_LABELS = labels("inline")


class FinalRecommendationItem(BaseModel):
    measurement_id: str = Field(..., description="Id of the validating measurement")
    action: str
    feature: str
    suggested_value: str
    priority: int = 1
    category: str = ""
    rationale: str = ""


class FinalRecommendations(BaseModel):
    items: list[FinalRecommendationItem] = Field(default_factory=list)
    rejected_hypotheses: list[str] = Field(default_factory=list)


def _expected_class_delta(probability_shift: dict[int, float]) -> float:
    """Change in expected class value implied by a probability shift.

    Classes are ordinal and ascending in slowness (0 = same-day,
    4 = not adopted), so a negative delta means probability mass moved
    toward faster adoption.
    """
    return sum(int(k) * float(v) for k, v in probability_shift.items())


def _speedup_text(
    class_before: int,
    class_after: int,
    probability_shift: dict[int, float],
) -> str:
    """Describe a measured impact, using only what the measurement shows.

    The predicted class alone is not enough: a change can leave the class
    untouched while moving probability mass the wrong way, or move nothing
    at all. Both cases were previously reported as improvements.
    """
    if class_after < class_before:
        return (f"moves the prediction from '{_CLASS_LABELS[class_before]}' "
                f"to '{_CLASS_LABELS[class_after]}'")
    if class_after > class_before:
        return (f"moves the prediction from '{_CLASS_LABELS[class_before]}' "
                f"to '{_CLASS_LABELS[class_after]}', which is slower")

    delta = _expected_class_delta(probability_shift)
    if delta < -SPEEDUP_EPSILON:
        return "improves class probabilities without changing the predicted class"
    if delta > SPEEDUP_EPSILON:
        return ("shifts probability toward slower adoption without changing "
                "the predicted class")
    return "no measurable change in the predicted probabilities"


def _build_context(state: AgentState) -> str:
    ev = state["prediction_evidence"]
    lines = [
        f"Current prediction: class {ev.predicted_class} ({ev.prediction_label}), "
        f"confidence {ev.class_confidence:.1%}.",
        "Probabilities: " + ", ".join(
            f"class {k}: {v:.1%}" for k, v in ev.probabilities.items()),
        "Key drivers:",
    ]
    for d in ev.key_drivers:
        lines.append(f"- [{d.direction}] {d.display_name or d.feature}: {d.reading}")
    visual = state.get("visual_evidence")
    if visual is not None:
        lines.append(
            f"Visual evidence: overall appeal {visual.overall_visual_appeal}/10. "
            f"Strategy: {visual.photo_strategy_summary}"
        )
        for flag in visual.consistency_flags:
            lines.append(f"- consistency flag: {flag}")
    else:
        lines.append("Visual evidence: none (no photos analyzed).")
    lines.append(f"Actionable features: {sorted(ACTIONABLE_FEATURES)}")
    return "\n".join(lines)


async def _react_loop(
    model, tools, log: MeasurementLog, context: str, resolved,
) -> tuple[int, list, dict]:
    """Run the bounded tool loop. Returns (tool_calls_used, message_history,
    accumulated_llm_usage)."""
    system = (_PROMPTS_DIR / "recommendation_agent_system.txt").read_text(
        encoding="utf-8"
    ).format(max_tool_calls=MAX_TOOL_CALLS)
    tool_map = {t.name: t for t in tools}
    bound = model.bind_tools(tools)
    messages: list = [SystemMessage(content=system), HumanMessage(content=context)]
    used = 0
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    while used < MAX_TOOL_CALLS:
        response = await bound.ainvoke(messages)
        messages.append(response)
        u = extract_usage(response, resolved.model_key)
        total_usage["input_tokens"] += u["input_tokens"]
        total_usage["output_tokens"] += u["output_tokens"]
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break
        for call in tool_calls:
            if used >= MAX_TOOL_CALLS:
                messages.append(ToolMessage(
                    content=json.dumps({"error": "tool budget exhausted"}),
                    tool_call_id=call["id"],
                ))
                continue
            tool = tool_map.get(call["name"])
            if tool is None:
                result = json.dumps({"error": f"unknown tool {call['name']}"})
            else:
                result = tool.invoke(call["args"])
            used += 1
            messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return used, messages, total_usage


def _finalize_items(
    final: FinalRecommendations, log: MeasurementLog, notes: list[str], request,
) -> list[ValidatedRecommendation]:
    validated = []
    for item in final.items:
        measurement = log.measurements.get(item.measurement_id)
        if measurement is None:
            notes.append(
                f"dropped recommendation citing unknown measurement "
                f"'{item.measurement_id}'"
            )
            continue
        shift = {
            int(k): v for k, v in measurement["probability_shift"].items()
        }
        validated.append(ValidatedRecommendation(
            action=item.action,
            feature=item.feature,
            current_value=current_value(request, item.feature),
            suggested_value=item.suggested_value,
            measured_impact=MeasuredImpact(
                class_before=measurement["class_before"],
                class_after=measurement["class_after"],
                probability_shift=shift,
                expected_speedup=_speedup_text(
                    measurement["class_before"],
                    measurement["class_after"],
                    shift,
                ),
            ),
            priority=item.priority,
            category=item.category,
            rationale=item.rationale,
        ))
    validated.sort(key=lambda r: r.priority)
    return validated


def _deterministic_sweep(request, baseline) -> list[ValidatedRecommendation]:
    """Fallback: measure the standard candidates without LLM curation."""
    tools, log = make_recommendation_tools(request, baseline)
    counterfactual = next(t for t in tools if t.name == "run_counterfactual")
    results = []
    for feature, value in sweep_candidates():
        raw = json.loads(counterfactual.invoke({"feature": feature, "value": value}))
        if "error" in raw:
            continue
        shift_to_faster = sum(
            v for k, v in raw["probability_shift"].items()
            if int(k) < raw["class_before"]
        )
        if raw["class_after"] < raw["class_before"] or shift_to_faster > SWEEP_MIN_SHIFT:
            results.append((shift_to_faster, feature, value, raw))
    results.sort(key=lambda r: r[0], reverse=True)
    recs = []
    for priority, (_, feature, value, raw) in enumerate(
        results[:SWEEP_MAX_RECOMMENDATIONS], start=1
    ):
        shift = {int(k): v for k, v in raw["probability_shift"].items()}
        recs.append(ValidatedRecommendation(
            action=f"Set {feature} to {value}",
            feature=feature, current_value=current_value(request, feature),
            suggested_value=value,
            measured_impact=MeasuredImpact(
                class_before=raw["class_before"], class_after=raw["class_after"],
                probability_shift=shift,
                expected_speedup=_speedup_text(raw["class_before"],
                                               raw["class_after"], shift),
            ),
            priority=priority, category="listing_details",
            rationale="Deterministic counterfactual sweep (LLM unavailable).",
        ))
    return recs


async def recommendation_agent_node(state: AgentState) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timestamp = state.get("timestamp", "")

    request = state.get("request")
    prediction = state.get("prediction")
    if prediction is None or state.get("prediction_evidence") is None:
        return {
            "recommendation_evidence": None,
            "errors": [NodeError(node="recommendation_agent",
                                 error_type="missing_input",
                                 message="prediction evidence missing",
                                 timestamp=timestamp, recoverable=True)],
            "trace": [_trace(started_at, t0, "error", {})],
        }

    notes: list[str] = []
    errors: list[NodeError] = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}
    try:
        tools, log = make_recommendation_tools(request, prediction)
        model = get_chat_model("recommendation_agent")
        context = _build_context(state)
        resolved = resolve_role("recommendation_agent")

        used, messages, loop_usage = await asyncio.wait_for(
            _react_loop(model, tools, log, context, resolved),
            timeout=node_timeout("recommendation_agent"),
        )
        total_usage["input_tokens"] += loop_usage["input_tokens"]
        total_usage["output_tokens"] += loop_usage["output_tokens"]
        if used >= MAX_TOOL_CALLS:
            notes.append(f"tool budget of {MAX_TOOL_CALLS} exhausted")

        structured = model.with_structured_output(FinalRecommendations, include_raw=True)
        result = await asyncio.wait_for(
            structured.ainvoke(messages + [HumanMessage(content=(
                "Provide your final recommendations now. Every item must cite "
                "the measurement_id that validated it."
            ))]),
            timeout=node_timeout("recommendation_agent"),
        )
        final: FinalRecommendations = result["parsed"]
        final_usage = extract_usage(result["raw"], resolved.model_key)
        total_usage["input_tokens"] += final_usage["input_tokens"]
        total_usage["output_tokens"] += final_usage["output_tokens"]

        recommendations = _finalize_items(final, log, notes, request)
        evidence = RecommendationEvidence(
            source="recommendation_agent",
            confidence="high" if recommendations else "low",
            generated_by=resolved.api_model,
            notes=notes,
            recommendations=recommendations,
            rejected_hypotheses=final.rejected_hypotheses,
            iterations_used=used,
        )
        meta = {"model": resolved.api_model, "tool_calls": used,
                "n_recommendations": len(recommendations),
                "llm_usage": {"input_tokens": total_usage["input_tokens"],
                             "output_tokens": total_usage["output_tokens"],
                             "model_key": resolved.model_key}}
        return {"recommendation_evidence": evidence, "errors": errors,
                "trace": [_trace(started_at, t0, "success", meta)]}

    except Exception as exc:
        logger.warning("recommendation_agent failed: %s. Deterministic sweep.", exc)
        recs = _deterministic_sweep(request, prediction)
        evidence = RecommendationEvidence(
            source="recommendation_agent", confidence="medium",
            generated_by="deterministic",
            notes=notes + ["LLM unavailable; deterministic sweep used"],
            recommendations=recs, rejected_hypotheses=[], iterations_used=0,
        )
        errors.append(NodeError(node="recommendation_agent",
                                error_type="llm_failure", message=str(exc),
                                timestamp=timestamp, recoverable=True))
        return {"recommendation_evidence": evidence, "errors": errors,
                "trace": [_trace(started_at, t0, "success",
                                 {"used_fallback": True})]}


def _trace(started_at: str, t0: float, status: str, metadata: dict) -> TraceEntry:
    return TraceEntry(
        node="recommendation_agent",
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        status=status,
        metadata=metadata,
    )
