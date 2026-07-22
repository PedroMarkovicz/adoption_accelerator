"""aggregator node — assembles the final AdoptionReport and estimates
LLM cost from trace metadata."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from adoption_accelerator.agents.contracts import (
    AdoptionReport,
    NodeError,
    ReportMetadata,
    TraceEntry,
)
from adoption_accelerator.agents.llm.registry import load_models_config
from adoption_accelerator.agents.state import AgentState

logger = logging.getLogger(__name__)


def estimate_cost_usd(trace: list[TraceEntry]) -> float:
    """Sum LLM cost from ``llm_usage`` entries in trace metadata."""
    catalog = load_models_config().catalog
    total = 0.0
    for entry in trace:
        usage = entry.metadata.get("llm_usage")
        if not usage:
            continue
        spec = catalog.get(usage.get("model_key", ""))
        if spec is None:
            continue
        total += usage.get("input_tokens", 0) / 1e6 * spec.pricing.input_usd_per_1m
        total += usage.get("output_tokens", 0) / 1e6 * spec.pricing.output_usd_per_1m
    return round(total, 6)


def aggregator_node(state: AgentState) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    prediction_evidence = state.get("prediction_evidence")
    if prediction_evidence is None:
        # Fatal upstream failure: no report can be built.
        return {
            "report": None,
            "errors": [NodeError(node="aggregator", error_type="missing_evidence",
                                 message="no prediction evidence in state",
                                 timestamp=state.get("timestamp", ""),
                                 recoverable=False)],
            "trace": [],
        }

    trace = list(state.get("trace", []))
    timing = {t.node: t.duration_ms for t in trace}
    llm_models = {
        t.node: t.metadata["model"] for t in trace if "model" in t.metadata
    }

    report = AdoptionReport(
        prediction=prediction_evidence,
        visual=state.get("visual_evidence"),
        recommendations=state.get("recommendation_evidence"),
        narrative=state.get("narrative") or "",
        optimized_description=state.get("optimized_description"),
        headline=state.get("headline") or "",
        metadata=ReportMetadata(
            session_id=state.get("session_id", ""),
            ml_model_version="tuned_v1",
            llm_models=llm_models,
            timing_ms=timing,
            estimated_cost_usd=estimate_cost_usd(trace),
            errors=list(state.get("errors", [])),
            timestamp=state.get("timestamp", ""),
        ),
    )

    duration_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {
        "report": report,
        "errors": [],
        "trace": [TraceEntry(node="aggregator", started_at=started_at,
                             completed_at=datetime.now(timezone.utc).isoformat(),
                             duration_ms=duration_ms, status="success")],
    }
