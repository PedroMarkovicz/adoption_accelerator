"""data_analyst node — turns deterministic ML outputs into
``PredictionEvidence``. All structured fields are assembled
deterministically; the LLM contributes only the natural-language
``reading`` per driver and the ``uncertainty_reading``. On any LLM
failure the evidence degrades to deterministic readings.

Precondition: evidence is always non-None WHEN inference succeeded and
populated ``state["prediction"]`` and ``state["interpreted_explanation"]``.
If those are missing (a fatal upstream failure), the node returns
``prediction_evidence: None`` with a non-recoverable ``NodeError`` so the
aggregator can surface the fatal error instead of fabricating evidence
from nothing."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from adoption_accelerator.agents.contracts import (
    FactorInsight,
    NodeError,
    PredictionEvidence,
    TraceEntry,
)
from adoption_accelerator.agents.llm.client import extract_usage, get_chat_model
from adoption_accelerator.agents.llm.registry import resolve_role
from adoption_accelerator.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_TIMEOUT_SECONDS = 15.0


class DataAnalystOutput(BaseModel):
    """Structured output the LLM must return."""

    driver_readings: list[str] = Field(
        ..., description="One plain-language sentence per key driver, in order"
    )
    uncertainty_reading: str = Field(
        ..., description="One sentence on how confident the model is"
    )


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _build_factors(interpreted) -> list[FactorInsight]:
    factors = []
    for f in interpreted.top_factors:
        if f.is_baseline:
            continue
        factors.append(
            FactorInsight(
                feature=f.name,
                display_name=f.description,
                value=f.value,
                direction=f.direction,
                shap_magnitude=f.shap_magnitude,
                modality=f.modality,
                reading="",  # filled by LLM or fallback
            )
        )
    return factors


def _deterministic_readings(factors: list[FactorInsight]) -> None:
    for f in factors:
        arrow = "speeds up" if f.direction == "positive" else "slows down"
        f.reading = f"{f.display_name or f.feature} currently {arrow} the predicted adoption."


def _assemble_evidence(
    prediction, interpreted, generated_by: str, confidence: str,
    factors: list[FactorInsight], uncertainty: str, notes: list[str],
) -> PredictionEvidence:
    return PredictionEvidence(
        source="data_analyst",
        confidence=confidence,
        generated_by=generated_by,
        notes=notes,
        predicted_class=prediction.prediction,
        prediction_label=prediction.prediction_label,
        probabilities=prediction.probabilities,
        class_confidence=prediction.confidence,
        modality_contributions=interpreted.modality_contributions,
        modality_available=interpreted.modality_available,
        key_drivers=factors,
        uncertainty_reading=uncertainty,
    )


async def data_analyst_node(state: AgentState) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timestamp = state.get("timestamp", "")

    prediction = state.get("prediction")
    interpreted = state.get("interpreted_explanation")

    if prediction is None or interpreted is None:
        return {
            "prediction_evidence": None,
            "errors": [NodeError(node="data_analyst", error_type="missing_input",
                                 message="prediction or interpretation missing",
                                 timestamp=timestamp, recoverable=False)],
            "trace": [_trace(started_at, t0, "error", {})],
        }

    factors = _build_factors(interpreted)
    errors: list[NodeError] = []
    notes: list[str] = []
    trace_meta: dict = {}

    try:
        drivers_text = "\n".join(
            f"{i}. [{f.direction}] {f.display_name or f.feature} "
            f"(modality: {f.modality}, magnitude: {f.shap_magnitude:.4f})"
            for i, f in enumerate(factors, 1)
        ) or "(none - only listing attributes were provided)"
        modality_breakdown = "\n".join(
            f"- {mod}: {pct:.1%}"
            for mod, pct in sorted(interpreted.modality_contributions.items(),
                                   key=lambda x: x[1], reverse=True)
        )
        user_prompt = _load_prompt("data_analyst_user.txt").format(
            predicted_class=prediction.prediction,
            prediction_label=prediction.prediction_label,
            class_probabilities=", ".join(
                f"class {k}: {v:.1%}" for k, v in prediction.probabilities.items()
            ),
            confidence=prediction.confidence,
            modality_breakdown=modality_breakdown,
            drivers=drivers_text,
        )

        model = get_chat_model("data_analyst").with_structured_output(
            DataAnalystOutput, include_raw=True
        )
        result = await asyncio.wait_for(
            model.ainvoke(
                [("system", _load_prompt("data_analyst_system.txt")),
                 ("user", user_prompt)]
            ),
            timeout=_TIMEOUT_SECONDS,
        )
        output: DataAnalystOutput = result["parsed"]
        raw = result["raw"]

        readings = output.driver_readings
        if len(readings) != len(factors):
            notes.append("driver_readings count mismatch; padded deterministically")
        for f, reading in zip(factors, readings):
            f.reading = reading
        _deterministic_fill = [f for f in factors if not f.reading]
        if _deterministic_fill:
            _deterministic_readings(_deterministic_fill)

        resolved = resolve_role("data_analyst")
        evidence = _assemble_evidence(
            prediction, interpreted, resolved.api_model, "high",
            factors, output.uncertainty_reading, notes,
        )
        trace_meta["model"] = resolved.api_model
        trace_meta["llm_usage"] = extract_usage(raw, resolved.model_key)

    except Exception as exc:
        logger.warning("data_analyst LLM failed: %s. Using fallback.", exc)
        _deterministic_readings(factors)
        evidence = _assemble_evidence(
            prediction, interpreted, "deterministic", "medium",
            factors,
            f"The model assigns {prediction.confidence:.0%} probability to the "
            f"predicted class.",
            notes + ["LLM unavailable; deterministic readings used"],
        )
        errors.append(NodeError(node="data_analyst", error_type="llm_failure",
                                message=str(exc), timestamp=timestamp,
                                recoverable=True))
        trace_meta["used_fallback"] = True

    return {
        "prediction_evidence": evidence,
        "errors": errors,
        "trace": [_trace(started_at, t0, "success", trace_meta)],
    }


def _trace(started_at: str, t0: float, status: str, metadata: dict) -> TraceEntry:
    return TraceEntry(
        node="data_analyst",
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        status=status,
        metadata=metadata,
    )
