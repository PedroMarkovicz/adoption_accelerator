"""
Validation logic for E2E test outputs.

Performs comprehensive checks on the full agent graph output,
verifying schema compliance against the ``AdoptionReport`` contract,
state propagation, guardrails, and scenario-specific expectations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from adoption_accelerator.agents.contracts import AdoptionReport, ValidatedRecommendation
from adoption_accelerator.agents.tools.recommendation_tools import ACTIONABLE_FEATURES
from adoption_accelerator.inference.pipeline import CLASS_LABELS

# Raw embedding pattern that must never appear in user-facing outputs
_RAW_EMBEDDING_RE = re.compile(r"(img_emb_|text_emb_)\d+")

# Valid categories for recommendations
_VALID_CATEGORIES = {"photo", "description", "health", "listing_details"}


@dataclass
class ValidationResult:
    """Result of validating a single scenario execution."""

    scenario_name: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.failed) == 0

    @property
    def total_checks(self) -> int:
        return len(self.passed) + len(self.failed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "is_valid": self.is_valid,
            "total_checks": self.total_checks,
            "passed_count": len(self.passed),
            "failed_count": len(self.failed),
            "warning_count": len(self.warnings),
            "passed": self.passed,
            "failed": self.failed,
            "warnings": self.warnings,
        }


def validate_scenario_output(
    scenario_name: str,
    result: dict[str, Any],
    expected_behavior: dict[str, Any],
) -> ValidationResult:
    """Run all validations on a completed scenario execution.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario for reporting.
    result : dict
        Raw LangGraph state output after graph invocation (``AgentState``).
    expected_behavior : dict
        Scenario-specific expected behavior flags (currently informational
        only; the new graph topology has no conditional per-scenario
        routing to assert against).

    Returns
    -------
    ValidationResult
        Aggregated validation results.
    """
    v = ValidationResult(scenario_name=scenario_name)

    _validate_pipeline_execution(v, result)
    _validate_report_schema(v, result)
    _validate_prediction_integrity(v, result)
    _validate_narrative(v, result)
    _validate_recommendation_output(v, result)
    _validate_metadata(v, result)
    _validate_trace_entries(v, result)
    _validate_state_propagation(v, result)
    _validate_guardrails(v, result)
    _validate_no_raw_embeddings(v, result)
    _validate_modality_awareness(v, result)

    return v


# -----------------------------------------------------------------------
# Validation checks
# -----------------------------------------------------------------------


def _check(v: ValidationResult, condition: bool, description: str) -> None:
    """Record a pass/fail check."""
    if condition:
        v.passed.append(description)
    else:
        v.failed.append(description)


def _warn(v: ValidationResult, condition: bool, description: str) -> None:
    """Record a warning (non-fatal) if condition is false."""
    if not condition:
        v.warnings.append(description)


def _validate_pipeline_execution(v: ValidationResult, result: dict) -> None:
    """Verify the full pipeline executed successfully."""
    _check(v, result is not None, "Pipeline returned a result")
    _check(v, "report" in result, "'report' field present in state")
    _check(v, result.get("report") is not None, "report is not None")

    report = result.get("report")
    if report is not None:
        _check(v, isinstance(report, AdoptionReport),
               "Report is an AdoptionReport instance")


def _validate_report_schema(v: ValidationResult, result: dict) -> None:
    """Verify the AdoptionReport follows the expected schema."""
    report: AdoptionReport | None = result.get("report")
    if report is None:
        v.failed.append("Cannot validate schema: report is None")
        return

    prediction = report.prediction

    # Core prediction fields (the deterministic core, always present)
    _check(v, isinstance(prediction.predicted_class, int),
           "prediction.predicted_class is an integer")
    _check(v, prediction.predicted_class in range(5),
           f"prediction.predicted_class is valid class (0-4), got {prediction.predicted_class}")
    _check(v, isinstance(prediction.prediction_label, str) and len(prediction.prediction_label) > 0,
           "prediction.prediction_label is a non-empty string")
    _check(v, isinstance(prediction.class_confidence, float),
           "prediction.class_confidence is a float")
    _check(v, 0.0 < prediction.class_confidence <= 1.0,
           f"prediction.class_confidence in valid range (0,1], got {prediction.class_confidence:.4f}")

    # Probabilities
    _check(v, isinstance(prediction.probabilities, dict),
           "prediction.probabilities is a dict")
    _check(v, len(prediction.probabilities) == 5,
           f"prediction.probabilities has 5 entries, got {len(prediction.probabilities)}")

    prob_sum = sum(prediction.probabilities.values())
    _check(v, 0.99 <= prob_sum <= 1.01,
           f"probabilities sum to ~1.0, got {prob_sum:.4f}")

    for cls, prob in prediction.probabilities.items():
        _check(v, 0.0 <= prob <= 1.0,
               f"probability for class {cls} in [0,1], got {prob:.4f}")

    # Modality contributions
    _check(v, isinstance(prediction.modality_contributions, dict),
           "prediction.modality_contributions is a dict")
    if prediction.modality_contributions:
        mod_sum = sum(prediction.modality_contributions.values())
        _check(v, 0.95 <= mod_sum <= 1.05,
               f"modality contributions sum to ~1.0, got {mod_sum:.4f}")

    # Narrative & metadata presence
    _check(v, isinstance(report.narrative, str), "report.narrative is a string")
    _check(v, isinstance(report.metadata.session_id, str) and len(report.metadata.session_id) > 0,
           "report.metadata.session_id is present")


def _validate_prediction_integrity(v: ValidationResult, result: dict) -> None:
    """Verify prediction is consistent and meaningful."""
    report: AdoptionReport | None = result.get("report")
    if report is None:
        return

    prediction = report.prediction

    expected_label = CLASS_LABELS.get(prediction.predicted_class, "")
    _check(v, prediction.prediction_label == expected_label,
           f"prediction_label matches class: '{prediction.prediction_label}' "
           f"== '{expected_label}'")

    if prediction.probabilities:
        pred_prob = prediction.probabilities.get(prediction.predicted_class, 0.0)
        _check(v, abs(prediction.class_confidence - pred_prob) < 0.001,
               f"class_confidence equals predicted-class probability: "
               f"{prediction.class_confidence:.4f} ~= "
               f"probabilities[{prediction.predicted_class}]={pred_prob:.4f}")


def _validate_narrative(v: ValidationResult, result: dict) -> None:
    """Validate the narrative explanation."""
    report: AdoptionReport | None = result.get("report")
    if report is None:
        return

    narrative = report.narrative
    _check(v, len(narrative) > 0, "narrative is non-empty")
    _warn(v, len(narrative) >= 50,
          f"narrative has meaningful length (>= 50 chars, got {len(narrative)})")

    # Check we have at least some key drivers from the deterministic core
    _check(v, len(report.prediction.key_drivers) > 0,
           f"At least 1 key driver present, got {len(report.prediction.key_drivers)}")


def _validate_recommendation_output(v: ValidationResult, result: dict) -> None:
    """Validate recommendations, if the evidence is present."""
    report: AdoptionReport | None = result.get("report")
    if report is None:
        return

    if report.recommendations is None:
        v.warnings.append("No recommendation evidence present (recommendations is None)")
        return

    recs = report.recommendations.recommendations
    _check(v, len(recs) <= 5,
           f"At most 5 recommendations, got {len(recs)}")

    for i, rec in enumerate(recs):
        _check(v, isinstance(rec, ValidatedRecommendation),
               f"Recommendation {i+1} is a ValidatedRecommendation instance")
        _check(v, len(rec.feature) > 0,
               f"Recommendation {i+1} has a feature name")
        _check(v, rec.category in _VALID_CATEGORIES,
               f"Recommendation {i+1} category '{rec.category}' is valid")
        _check(v, isinstance(rec.priority, int) and rec.priority >= 1,
               f"Recommendation {i+1} priority is a positive int")

        # measured_impact must be measured against the real ensemble, never estimated
        impact = rec.measured_impact
        _check(v, impact.class_before in range(5),
               f"Recommendation {i+1} measured_impact.class_before is valid (0-4), "
               f"got {impact.class_before}")
        _check(v, impact.class_after in range(5),
               f"Recommendation {i+1} measured_impact.class_after is valid (0-4), "
               f"got {impact.class_after}")
        _check(v, isinstance(impact.probability_shift, dict) and len(impact.probability_shift) > 0,
               f"Recommendation {i+1} measured_impact.probability_shift is a non-empty dict")


def _validate_metadata(v: ValidationResult, result: dict) -> None:
    """Validate report metadata."""
    report: AdoptionReport | None = result.get("report")
    if report is None:
        return

    meta = report.metadata
    _check(v, len(meta.session_id) > 0, "session_id is non-empty")
    _check(v, len(meta.timestamp) > 0, "timestamp is non-empty")
    _check(v, meta.ml_model_version == "tuned_v1",
           f"ml_model_version is 'tuned_v1', got '{meta.ml_model_version}'")
    _check(v, isinstance(meta.timing_ms, dict) and len(meta.timing_ms) >= 3,
           f"At least 3 node timings recorded, got {len(meta.timing_ms)}")
    _check(v, meta.estimated_cost_usd >= 0.0,
           f"estimated_cost_usd >= 0, got {meta.estimated_cost_usd}")

    expected_nodes = {"orchestrator", "inference"}
    actual_nodes = set(meta.timing_ms)
    for node in expected_nodes:
        _check(v, node in actual_nodes,
               f"'{node}' in metadata.timing_ms")


def _validate_trace_entries(v: ValidationResult, result: dict) -> None:
    """Validate execution trace."""
    trace_entries = result.get("trace", [])

    _check(v, len(trace_entries) >= 3,
           f"At least 3 trace entries, got {len(trace_entries)}")

    for entry in trace_entries:
        _check(v, len(entry.started_at) > 0,
               f"Trace '{entry.node}' has started_at")
        _check(v, entry.duration_ms >= 0,
               f"Trace '{entry.node}' duration >= 0 ({entry.duration_ms:.1f}ms)")
        _check(v, entry.status in ("success", "error", "skipped"),
               f"Trace '{entry.node}' has valid status: {entry.status}")

    # Verify execution order matches the Evidence Board graph topology:
    # orchestrator -> inference -> [visual_analyst, data_analyst]
    #              -> recommendation_agent -> synthesizer -> aggregator
    node_order = [e.node for e in trace_entries]

    def _idx(name: str) -> int | None:
        return node_order.index(name) if name in node_order else None

    orch_idx, inf_idx = _idx("orchestrator"), _idx("inference")
    if orch_idx is not None and inf_idx is not None:
        _check(v, orch_idx < inf_idx, "orchestrator executes before inference")

    for analyst in ("visual_analyst", "data_analyst"):
        analyst_idx = _idx(analyst)
        if inf_idx is not None and analyst_idx is not None:
            _check(v, inf_idx < analyst_idx, f"inference executes before {analyst}")

    rec_idx = _idx("recommendation_agent")
    if rec_idx is not None:
        for analyst in ("visual_analyst", "data_analyst"):
            analyst_idx = _idx(analyst)
            if analyst_idx is not None:
                _check(v, analyst_idx < rec_idx,
                       f"{analyst} executes before recommendation_agent")

    synth_idx = _idx("synthesizer")
    if rec_idx is not None and synth_idx is not None:
        _check(v, rec_idx < synth_idx, "recommendation_agent executes before synthesizer")

    agg_idx = _idx("aggregator")
    if synth_idx is not None and agg_idx is not None:
        _check(v, synth_idx < agg_idx, "synthesizer executes before aggregator")


def _validate_state_propagation(v: ValidationResult, result: dict) -> None:
    """Verify critical state fields were propagated correctly."""
    _check(v, result.get("session_id") is not None and len(result.get("session_id", "")) > 0,
           "session_id propagated in state")
    _check(v, result.get("timestamp") is not None and len(result.get("timestamp", "")) > 0,
           "timestamp propagated in state")
    _check(v, result.get("prediction") is not None,
           "prediction (raw PredictionResult) propagated in state")
    _check(v, result.get("explanation") is not None,
           "explanation propagated in state")
    _check(v, result.get("interpreted_explanation") is not None,
           "interpreted_explanation propagated in state")
    _check(v, result.get("feature_vector") is not None,
           "feature_vector propagated in state")
    _check(v, result.get("feature_names") is not None,
           "feature_names propagated in state")
    _check(v, result.get("prediction_evidence") is not None,
           "prediction_evidence propagated in state")


def _validate_guardrails(v: ValidationResult, result: dict) -> None:
    """Validate guardrail behavior."""
    report: AdoptionReport | None = result.get("report")
    if report is None or report.recommendations is None:
        return

    recs = report.recommendations.recommendations

    # Recommendations must only target the actionable-feature whitelist
    # enforced inside the recommendation tools.
    for rec in recs:
        _check(v, rec.feature in ACTIONABLE_FEATURES,
               f"Recommendation '{rec.feature}' is actionable "
               f"(in ACTIONABLE_FEATURES whitelist)")

    # Recommendations capped at 5
    _check(v, len(recs) <= 5,
           f"Recommendations capped at 5, got {len(recs)}")


def _validate_no_raw_embeddings(v: ValidationResult, result: dict) -> None:
    """Verify no raw embedding dimension names leak into user-facing outputs."""
    report: AdoptionReport | None = result.get("report")
    if report is None:
        return

    if report.narrative:
        _check(v, not _RAW_EMBEDDING_RE.search(report.narrative),
               "No raw embedding names in narrative")

    if report.optimized_description:
        _check(v, not _RAW_EMBEDDING_RE.search(report.optimized_description),
               "No raw embedding names in optimized_description")

    if report.recommendations is not None:
        for rec in report.recommendations.recommendations:
            for fld in (rec.feature, rec.current_value, rec.suggested_value, rec.rationale):
                _check(v, not _RAW_EMBEDDING_RE.search(str(fld)),
                       f"No raw embedding names in recommendation field for '{rec.feature}'")

    # Check key-driver display names
    for factor in report.prediction.key_drivers:
        _check(v, not _RAW_EMBEDDING_RE.search(factor.display_name),
               f"No raw embedding names in key driver display_name '{factor.display_name}'")

        jargon_words = ["log1p", "train-fitted", "recoded", "variance", "crop-hint"]
        bad_jargon = [j for j in jargon_words if j in factor.display_name.lower()]
        _check(v, len(bad_jargon) == 0,
               f"No technical jargon in key driver display_name '{factor.display_name}'")


def _validate_modality_awareness(v: ValidationResult, result: dict) -> None:
    """Verify absent modalities are correctly excluded from contributions."""
    report: AdoptionReport | None = result.get("report")
    if report is None:
        return

    prediction = report.prediction

    _check(v, isinstance(prediction.modality_available, dict),
           "modality_available is a dict")
    _check(v, len(prediction.modality_available) > 0,
           "modality_available is non-empty")

    available = prediction.modality_available
    contributions = prediction.modality_contributions

    # tabular is always present
    _check(v, available.get("tabular", False),
           "modality_available['tabular'] is True")

    if not available.get("text", True):
        text_contrib = contributions.get("text", 0.0)
        _check(v, text_contrib < 0.01,
               f"text modality absent -> contribution should be <1%, got {text_contrib:.1%}")

    if not available.get("image", True):
        image_contrib = contributions.get("image", 0.0)
        _check(v, image_contrib < 0.01,
               f"image modality absent -> contribution should be <1%, got {image_contrib:.1%}")

    present_sum = sum(
        contrib for mod, contrib in contributions.items()
        if available.get(mod, True)
    )
    if contributions:
        _check(v, 0.95 <= present_sum <= 1.05,
               f"Present modality contributions sum to ~1.0, got {present_sum:.4f}")
