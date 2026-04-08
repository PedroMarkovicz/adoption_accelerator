"""
Agent state and contract definitions for the LangGraph agent graph.

All shared data structures that flow between nodes are defined here.
This module is the single source of truth for the agent state schema,
response contracts, and supporting models.

References:
  - ``docs/agent_architecture.md`` Section 3.1 (AgentState)
  - ``docs/agent_architecture.md`` Section 10.1 (AgentResponse)
  - ``docs/next_steps_v02.md`` Section 5.1
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Supporting models
# ---------------------------------------------------------------------------


class NodeError(BaseModel):
    """Error captured during node execution."""

    node: str = Field(..., description="Name of the node that produced the error")
    error_type: str = Field(..., description="Error category")
    message: str = Field(..., description="Human-readable error description")
    timestamp: str = Field("", description="ISO 8601 UTC timestamp")
    recoverable: bool = True


class TraceEntry(BaseModel):
    """Execution trace for a single node."""

    node: str = Field(..., description="Node name")
    started_at: str = Field(..., description="ISO 8601 start time")
    completed_at: str = Field("", description="ISO 8601 completion time")
    duration_ms: float = Field(0.0, description="Execution duration in milliseconds")
    status: str = Field("success", description="success, error, or skipped")
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureFactor(BaseModel):
    """A single feature's contribution to the prediction.

    Used in AgentResponse for top_positive_factors / top_negative_factors.
    """

    feature: str = Field(..., description="Feature name")
    display_name: str = Field("", description="Human-readable feature name")
    value: str = Field("", description="Current feature value as string")
    shap_value: float = Field(..., description="SHAP contribution value")
    modality: str = Field(..., description="Feature modality: tabular/text/image/metadata")
    direction: str = Field(..., description="positive or negative")


class Recommendation(BaseModel):
    """A single actionable recommendation for improving adoption speed."""

    feature: str = Field(..., description="Target feature name")
    current_value: str = Field("", description="Current value of the feature")
    suggested_value: str = Field("", description="Recommended new value")
    expected_impact: str = Field(
        "", description="Expected outcome, e.g. 'Could improve from class 3 to class 2'"
    )
    priority: int = Field(1, description="Priority rank (1 = highest)")
    category: str = Field(
        "", description="Recommendation category: photo, description, health, listing_details"
    )
    actionable: bool = True


class ResponseMetadata(BaseModel):
    """Execution metadata attached to the final agent response."""

    session_id: str = ""
    model_version: str = "tuned_v1"
    model_type: str = "SoftVotingEnsemble"
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0
    timestamp: str = ""
    nodes_executed: list[str] = Field(default_factory=list)
    errors: list[NodeError] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# InterpretedExplanation (agent-facing semantic SHAP output)
# ---------------------------------------------------------------------------
# The canonical InterpretedExplanation and InterpretedFactor classes live
# in src/adoption_accelerator/interpretability/translator.py.  We re-export
# them here for convenience so that agent code can import from a single
# location.

from adoption_accelerator.interpretability.translator import (  # noqa: E402
    InterpretedExplanation,
    InterpretedFactor,
)


# ---------------------------------------------------------------------------
# AgentResponse (final output contract)
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Final structured output from the agent system.

    This is the top-level response returned to callers after the
    aggregator node assembles all Phase 1 (deterministic) and
    Phase 2 (LLM) outputs.

    Defined in ``agent_architecture.md`` Section 10.1.
    """

    # Core prediction
    prediction: int = Field(..., description="Predicted AdoptionSpeed class 0-4")
    prediction_label: str = Field(..., description="Human-readable adoption speed label")
    probabilities: dict[int, float] = Field(..., description="Per-class probabilities")
    # Issue #1 fix: confidence = P(predicted_class), not max(all_probs).
    confidence: float = Field(
        ..., description="Probability of the thresholded predicted class"
    )
    # Diagnostic: probability of the argmax class (may differ from confidence
    # when threshold optimization selects a non-argmax class).
    max_class_probability: Optional[float] = Field(
        None, description="Global argmax probability (diagnostic)"
    )

    # Explanation
    narrative_explanation: str = Field(
        "", description="Natural-language explanation of the prediction"
    )
    modality_contributions: dict[str, float] = Field(
        default_factory=dict,
        description="Per-modality SHAP contribution percentages (only present modalities)",
    )
    top_positive_factors: list[FeatureFactor] = Field(
        default_factory=list,
        description="Top features contributing to faster adoption",
    )
    top_negative_factors: list[FeatureFactor] = Field(
        default_factory=list,
        description="Top features contributing to slower adoption",
    )

    # Issue #2: indicate which data modalities were actually present for this
    # prediction so the frontend can display contextual warnings.
    modality_available: dict[str, bool] = Field(
        default_factory=dict,
        description=(
            "Which data modalities were actually present in this request. "
            "Absent modalities are excluded from modality_contributions."
        ),
    )

    # Recommendations
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Actionable suggestions to improve adoption speed",
    )

    # Improved description (optional, only when description writer runs)
    improved_description: Optional[str] = Field(
        None, description="AI-improved listing description"
    )

    # Warnings (Issue #6)
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings during execution"
    )

    # Metadata
    metadata: ResponseMetadata = Field(default_factory=ResponseMetadata)


# ---------------------------------------------------------------------------
# AgentState (LangGraph shared state)
# ---------------------------------------------------------------------------
# LangGraph requires a TypedDict for state with Annotated reducers for
# accumulating fields.  We define the state as a TypedDict so that
# LangGraph can properly handle channel merging for parallel fan-out.

from typing import TypedDict  # noqa: E402

from adoption_accelerator.inference.contracts import (  # noqa: E402
    PredictionRequest,
    PredictionResult,
)
from adoption_accelerator.interpretability.contracts import (  # noqa: E402
    ExplanationResult,
)


def _replace(existing: Any, new: Any) -> Any:
    """Reducer: last writer wins (replace)."""
    return new


class AgentState(TypedDict, total=False):
    """Shared state passed between nodes in the agent graph.

    Fields annotated with ``operator.add`` are accumulating (appended
    on each node write).  All other fields use last-writer-wins semantics.

    Defined in ``agent_architecture.md`` Section 3.1.
    """

    # --- Input (set by Orchestrator) ---
    request: PredictionRequest
    session_id: str
    timestamp: str

    # --- Phase 1 outputs (set by Inference Node) ---
    prediction: Optional[PredictionResult]
    explanation: Optional[ExplanationResult]
    interpreted_explanation: Optional[InterpretedExplanation]
    feature_vector: Optional[list[float]]
    feature_names: Optional[list[str]]
    # Issue #2: which data modalities were present in the request.
    modality_available: Optional[dict[str, bool]]

    # --- Phase 2 outputs (set by LLM nodes) ---
    narrative_explanation: Optional[str]
    recommendations: Optional[list[Recommendation]]
    improved_description: Optional[str]

    # --- Final output (set by Aggregator) ---
    response: Optional[AgentResponse]

    # --- Observability (accumulating) ---
    errors: Annotated[list[NodeError], operator.add]
    trace: Annotated[list[TraceEntry], operator.add]
    warnings: Annotated[list[str], operator.add]
