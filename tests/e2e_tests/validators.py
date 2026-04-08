"""
Validation logic for E2E test outputs.

Performs comprehensive checks on the full agent pipeline output,
verifying schema compliance, agent contracts, state propagation,
guardrails, and scenario-specific expectations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from agents.state import (
    AgentResponse,
    FeatureFactor,
    NodeError,
    Recommendation,
    ResponseMetadata,
    TraceEntry,
)

# Raw embedding pattern that must never appear in user-facing outputs
_RAW_EMBEDDING_RE = re.compile(r"(img_emb_|text_emb_)\d+")

# Valid categories for recommendations
_VALID_CATEGORIES = {"photo", "description", "health", "listing_details"}

# Expected class labels
_CLASS_LABELS = {
    0: "Same-day adoption",
    1: "Adopted within 1 week",
    2: "Adopted within 1 month",
    3: "Adopted within 1-3 months",
    4: "Not adopted (100+ days)",
}


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
        Raw LangGraph state output after graph invocation.
    expected_behavior : dict
        Scenario-specific expected behavior flags.

    Returns
    -------
    ValidationResult
        Aggregated validation results.
    """
    v = ValidationResult(scenario_name=scenario_name)

    _validate_pipeline_execution(v, result)
    _validate_agent_response_schema(v, result)
    _validate_prediction_integrity(v, result)
    _validate_explanation_output(v, result)
    _validate_recommendation_output(v, result)
    _validate_description_writer_output(v, result, expected_behavior)
    _validate_metadata(v, result)
    _validate_trace_entries(v, result, expected_behavior)
    _validate_state_propagation(v, result)
    _validate_guardrails(v, result)
    _validate_no_raw_embeddings(v, result)
    # Phase 1 correctness checks
    _validate_modality_awareness(v, result)
    _validate_no_hallucination(v, result)

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
    _check(v, "response" in result, "Response field present in state")
    _check(v, result.get("response") is not None, "Response is not None")

    response = result.get("response")
    if response is not None:
        _check(v, isinstance(response, AgentResponse),
               "Response is an AgentResponse instance")


def _validate_agent_response_schema(v: ValidationResult, result: dict) -> None:
    """Verify the AgentResponse follows the expected schema."""
    response: AgentResponse | None = result.get("response")
    if response is None:
        v.failed.append("Cannot validate schema: response is None")
        return

    # Core prediction fields
    _check(v, isinstance(response.prediction, int),
           "prediction is an integer")
    _check(v, response.prediction in range(5),
           f"prediction is valid class (0-4), got {response.prediction}")
    _check(v, isinstance(response.prediction_label, str) and len(response.prediction_label) > 0,
           "prediction_label is a non-empty string")
    _check(v, isinstance(response.confidence, float),
           "confidence is a float")
    _check(v, 0.0 < response.confidence <= 1.0,
           f"confidence in valid range (0,1], got {response.confidence:.4f}")

    # Probabilities
    _check(v, isinstance(response.probabilities, dict),
           "probabilities is a dict")
    _check(v, len(response.probabilities) == 5,
           f"probabilities has 5 entries, got {len(response.probabilities)}")

    prob_sum = sum(response.probabilities.values())
    _check(v, 0.99 <= prob_sum <= 1.01,
           f"probabilities sum to ~1.0, got {prob_sum:.4f}")

    for cls, prob in response.probabilities.items():
        _check(v, 0.0 <= prob <= 1.0,
               f"probability for class {cls} in [0,1], got {prob:.4f}")

    # Modality contributions
    _check(v, isinstance(response.modality_contributions, dict),
           "modality_contributions is a dict")
    if response.modality_contributions:
        mod_sum = sum(response.modality_contributions.values())
        _check(v, 0.95 <= mod_sum <= 1.05,
               f"modality contributions sum to ~1.0, got {mod_sum:.4f}")

    # Factor lists
    _check(v, isinstance(response.top_positive_factors, list),
           "top_positive_factors is a list")
    _check(v, isinstance(response.top_negative_factors, list),
           "top_negative_factors is a list")

    for factor in response.top_positive_factors + response.top_negative_factors:
        _check(v, isinstance(factor, FeatureFactor),
               f"Factor '{factor.feature}' is a FeatureFactor")

    # Recommendations
    _check(v, isinstance(response.recommendations, list),
           "recommendations is a list")

    # Metadata
    _check(v, isinstance(response.metadata, ResponseMetadata),
           "metadata is a ResponseMetadata")


def _validate_prediction_integrity(v: ValidationResult, result: dict) -> None:
    """Verify prediction is consistent and meaningful."""
    response: AgentResponse | None = result.get("response")
    if response is None:
        return

    # prediction_label should match the class
    expected_label = _CLASS_LABELS.get(response.prediction, "")
    _check(v, response.prediction_label == expected_label,
           f"prediction_label matches class: '{response.prediction_label}' "
           f"== '{expected_label}'")

    # Issue #1 fix: confidence must equal the probability of the PREDICTED
    # class, not the global max probability.
    if response.probabilities:
        pred_prob = response.probabilities.get(response.prediction, 0.0)
        _check(v, abs(response.confidence - pred_prob) < 0.001,
               f"confidence equals predicted-class probability: "
               f"{response.confidence:.4f} ~= probabilities[{response.prediction}]={pred_prob:.4f}")


def _validate_explanation_output(v: ValidationResult, result: dict) -> None:
    """Validate the narrative explanation."""
    response: AgentResponse | None = result.get("response")
    if response is None:
        return

    narrative = response.narrative_explanation
    _check(v, isinstance(narrative, str), "narrative_explanation is a string")
    _check(v, len(narrative) > 0, "narrative_explanation is non-empty")
    _check(v, len(narrative) >= 50,
           f"narrative_explanation has meaningful length "
           f"(>= 50 chars, got {len(narrative)})")

    # Check modality contributions are populated
    _check(v, len(response.modality_contributions) > 0,
           "modality_contributions is populated")

    # Check we have at least some factors
    total_factors = len(response.top_positive_factors) + len(response.top_negative_factors)
    _check(v, total_factors > 0,
           f"At least 1 top factor present, got {total_factors}")

    # Validate factor structure
    tabular_count = 0
    valued_count = 0
    for factor in response.top_positive_factors:
        _check(v, factor.direction == "positive",
               f"Positive factor '{factor.feature}' has direction='positive'")
        _check(v, factor.modality in ("tabular", "text", "image", "metadata"),
               f"Factor '{factor.feature}' has valid modality: {factor.modality}")
        if factor.modality == "tabular":
            tabular_count += 1
            if factor.value.strip() != "":
                valued_count += 1

    for factor in response.top_negative_factors:
        _check(v, factor.direction == "negative",
               f"Negative factor '{factor.feature}' has direction='negative'")
        if factor.modality == "tabular":
            tabular_count += 1
            if factor.value.strip() != "":
                valued_count += 1

    if tabular_count > 0:
        _check(v, valued_count > 0,
               f"At least some tabular factors have populated value fields ({valued_count}/{tabular_count})")


def _validate_recommendation_output(v: ValidationResult, result: dict) -> None:
    """Validate recommendations."""
    response: AgentResponse | None = result.get("response")
    if response is None:
        return

    recs = response.recommendations
    _check(v, len(recs) > 0, "At least 1 recommendation generated")
    _check(v, len(recs) <= 5,
           f"At most 5 recommendations, got {len(recs)}")

    for i, rec in enumerate(recs):
        _check(v, isinstance(rec, Recommendation),
               f"Recommendation {i+1} is a Recommendation instance")
        _check(v, len(rec.feature) > 0,
               f"Recommendation {i+1} has a feature name")
        _check(v, rec.category in _VALID_CATEGORIES,
               f"Recommendation {i+1} category '{rec.category}' is valid")
        _check(v, isinstance(rec.actionable, bool),
               f"Recommendation {i+1} actionable is bool")
        _check(v, isinstance(rec.priority, int) and rec.priority >= 1,
               f"Recommendation {i+1} priority is a positive int")


def _validate_description_writer_output(
    v: ValidationResult,
    result: dict,
    expected: dict[str, Any],
) -> None:
    """Validate description writer behavior based on scenario expectations."""
    response: AgentResponse | None = result.get("response")
    trace_entries: list[TraceEntry] = result.get("trace", [])

    should_run = expected.get("description_writer_runs", None)

    dw_traces = [e for e in trace_entries if e.node == "description_writer"]

    if should_run is True:
        _check(v, len(dw_traces) > 0,
               "description_writer executed (expected: yes)")
        if response is not None and response.improved_description is not None:
            _check(v, len(response.improved_description) > 0,
                   "improved_description is non-empty")
        elif response is not None and response.improved_description is None:
            has_warning = any("failed quality validation" in w for w in response.warnings)
            # If it failed validation or failed LLM, a warning or error should be present
            _check(v, has_warning or len(result.get("errors", [])) > 0,
                   "If description writer ran but returned None, a warning or error is recorded")
    elif should_run is False:
        # It might be in trace as skipped, or not present at all
        if dw_traces:
            _check(v, dw_traces[0].status == "skipped",
                   "description_writer was skipped (expected: not run)")
        else:
            v.passed.append("description_writer not in trace (expected: not run)")


def _validate_metadata(v: ValidationResult, result: dict) -> None:
    """Validate response metadata."""
    response: AgentResponse | None = result.get("response")
    if response is None:
        return

    meta = response.metadata
    _check(v, len(meta.session_id) > 0, "session_id is non-empty")
    _check(v, len(meta.timestamp) > 0, "timestamp is non-empty")
    _check(v, meta.model_version == "tuned_v1",
           f"model_version is 'tuned_v1', got '{meta.model_version}'")
    _check(v, meta.model_type == "SoftVotingEnsemble",
           f"model_type is 'SoftVotingEnsemble', got '{meta.model_type}'")
    _check(v, meta.inference_time_ms > 0,
           f"inference_time_ms > 0, got {meta.inference_time_ms}")
    _check(v, len(meta.nodes_executed) >= 4,
           f"At least 4 nodes executed, got {len(meta.nodes_executed)}")

    # The aggregator builds nodes_executed from upstream trace entries,
    # so it doesn't include itself.  Only check for upstream nodes.
    expected_nodes = {"orchestrator", "inference"}
    actual_nodes = set(meta.nodes_executed)
    for node in expected_nodes:
        _check(v, node in actual_nodes,
               f"'{node}' in nodes_executed")


def _validate_trace_entries(
    v: ValidationResult,
    result: dict,
    expected: dict[str, Any],
) -> None:
    """Validate execution trace."""
    trace_entries: list[TraceEntry] = result.get("trace", [])

    _check(v, len(trace_entries) >= 4,
           f"At least 4 trace entries, got {len(trace_entries)}")

    for entry in trace_entries:
        _check(v, isinstance(entry, TraceEntry),
               f"Trace entry for '{entry.node}' is a TraceEntry")
        _check(v, len(entry.started_at) > 0,
               f"Trace '{entry.node}' has started_at")
        _check(v, len(entry.completed_at) > 0,
               f"Trace '{entry.node}' has completed_at")
        _check(v, entry.duration_ms >= 0,
               f"Trace '{entry.node}' duration >= 0 ({entry.duration_ms:.1f}ms)")
        _check(v, entry.status in ("success", "error", "skipped"),
               f"Trace '{entry.node}' has valid status: {entry.status}")

    # Verify execution order: orchestrator and inference must come first
    node_order = [e.node for e in trace_entries]
    if "orchestrator" in node_order and "inference" in node_order:
        orch_idx = node_order.index("orchestrator")
        inf_idx = node_order.index("inference")
        _check(v, orch_idx < inf_idx,
               "orchestrator executes before inference")

    if "inference" in node_order and "aggregator" in node_order:
        inf_idx = node_order.index("inference")
        agg_idx = node_order.index("aggregator")
        _check(v, inf_idx < agg_idx,
               "inference executes before aggregator")


def _validate_state_propagation(v: ValidationResult, result: dict) -> None:
    """Verify critical state fields were propagated correctly."""
    _check(v, result.get("session_id") is not None and len(result.get("session_id", "")) > 0,
           "session_id propagated in state")
    _check(v, result.get("timestamp") is not None and len(result.get("timestamp", "")) > 0,
           "timestamp propagated in state")
    _check(v, result.get("prediction") is not None,
           "prediction propagated in state")
    _check(v, result.get("explanation") is not None,
           "explanation propagated in state")
    _check(v, result.get("interpreted_explanation") is not None,
           "interpreted_explanation propagated in state")
    _check(v, result.get("feature_vector") is not None,
           "feature_vector propagated in state")
    _check(v, result.get("feature_names") is not None,
           "feature_names propagated in state")


def _validate_guardrails(v: ValidationResult, result: dict) -> None:
    """Validate guardrail behavior."""
    response: AgentResponse | None = result.get("response")
    if response is None:
        return

    # Recommendations must only target actionable features
    non_actionable = {"Age", "Breed", "Color", "State", "Gender", "Type",
                      "MaturitySize", "FurLength", "Health"}
    for rec in response.recommendations:
        _check(v, rec.feature not in non_actionable,
               f"Recommendation '{rec.feature}' is actionable "
               f"(not in non-actionable set)")

    # Recommendations capped at 5
    _check(v, len(response.recommendations) <= 5,
           f"Recommendations capped at 5, got {len(response.recommendations)}")


def _validate_no_raw_embeddings(v: ValidationResult, result: dict) -> None:
    """Verify no raw embedding dimension names leak into user-facing outputs."""
    response: AgentResponse | None = result.get("response")
    if response is None:
        return

    # Check narrative
    if response.narrative_explanation:
        _check(v, not _RAW_EMBEDDING_RE.search(response.narrative_explanation),
               "No raw embedding names in narrative_explanation")

    # Check improved description
    if response.improved_description:
        _check(v, not _RAW_EMBEDDING_RE.search(response.improved_description),
               "No raw embedding names in improved_description")

    # Check recommendation fields
    for rec in response.recommendations:
        for fld in (rec.feature, rec.current_value, rec.suggested_value, rec.expected_impact):
            _check(v, not _RAW_EMBEDDING_RE.search(str(fld)),
                   f"No raw embedding names in recommendation field for '{rec.feature}'")

    # Check factor display names
    for factor in response.top_positive_factors + response.top_negative_factors:
        _check(v, not _RAW_EMBEDDING_RE.search(factor.display_name),
               f"No raw embedding names in factor display_name '{factor.display_name}'")
        
        jargon_words = ["log1p", "train-fitted", "recoded", "variance", "crop-hint"]
        # Skip checking encoding specifically because it might appear in standard English words reasonably sometimes,
        # but the specific bad ones are quite clear.
        bad_jargon = [j for j in jargon_words if j in factor.display_name.lower()]
        _check(v, len(bad_jargon) == 0,
               f"No technical jargon in factor display_name '{factor.display_name}'")


# Issue #2 correctness checks
_DESCRIPTION_HALLUCINATION_RE = re.compile(
    r"description content|text pattern|description pattern|write-up|"
    r"listing text|text-based|semantic pattern|wording of the listing",
    re.IGNORECASE,
)
_IMAGE_HALLUCINATION_RE = re.compile(
    r"photo quality|image pattern|visual pattern|photo content|"
    r"image quality|picture quality",
    re.IGNORECASE,
)


def _validate_modality_awareness(v: ValidationResult, result: dict) -> None:
    """Issue #2: verify absent modalities are correctly excluded."""
    response: AgentResponse | None = result.get("response")
    if response is None:
        return

    # modality_available should be present and populated
    _check(v, isinstance(response.modality_available, dict),
           "modality_available is a dict")
    _check(v, len(response.modality_available) > 0,
           "modality_available is non-empty")

    available = response.modality_available
    contributions = response.modality_contributions

    # tabular is always present
    _check(v, available.get("tabular", False),
           "modality_available['tabular'] is True")

    # When text is absent, its contribution should be 0 or missing
    if not available.get("text", True):
        text_contrib = contributions.get("text", 0.0)
        _check(v, text_contrib < 0.01,
               f"text modality absent → contribution should be <1%, got {text_contrib:.1%}")

    # When image is absent, its contribution should be 0 or missing
    if not available.get("image", True):
        image_contrib = contributions.get("image", 0.0)
        _check(v, image_contrib < 0.01,
               f"image modality absent → contribution should be <1%, got {image_contrib:.1%}")

    # Present modalities should sum to ~1.0
    present_sum = sum(
        contrib for mod, contrib in contributions.items()
        if available.get(mod, True)
    )
    if contributions:
        _check(v, 0.95 <= present_sum <= 1.05,
               f"Present modality contributions sum to ~1.0, got {present_sum:.4f}")


def _validate_no_hallucination(v: ValidationResult, result: dict) -> None:
    """Issue #3: verify narrative does not hallucinate about absent modalities."""
    response: AgentResponse | None = result.get("response")
    if response is None:
        return

    available = getattr(response, "modality_available", {}) or {}
    narrative = response.narrative_explanation or ""

    # Check for text hallucination when description is absent
    if not available.get("text", True) and narrative:
        _check(v, not _DESCRIPTION_HALLUCINATION_RE.search(narrative),
               "Narrative does not hallucinate about absent description content")

    # Check for image hallucination when no images provided
    if not available.get("image", True) and narrative:
        _check(v, not _IMAGE_HALLUCINATION_RE.search(narrative),
               "Narrative does not hallucinate about absent image content")
