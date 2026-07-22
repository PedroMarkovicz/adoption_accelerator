"""Evidence contracts and the AdoptionReport — the agent layer's single
source of truth for every boundary type.

Replaces the old ``AgentResponse``. Design rule: the deterministic core
(``prediction``) is always present; every other evidence field is
optional so that any failure path still yields a valid report.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observability primitives (ported semantics from the old agents/state.py)
# ---------------------------------------------------------------------------


class NodeError(BaseModel):
    node: str
    error_type: str
    message: str
    timestamp: str = ""
    recoverable: bool = True


class TraceEntry(BaseModel):
    node: str
    started_at: str
    completed_at: str = ""
    duration_ms: float = 0.0
    status: str = "success"  # success, error, skipped
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evidence base
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """Base for all typed evidence emitted by analyst nodes."""

    source: str
    confidence: Literal["high", "medium", "low"]
    generated_by: str  # model id or "deterministic"
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# PredictionEvidence (data_analyst)
# ---------------------------------------------------------------------------


class FactorInsight(BaseModel):
    feature: str
    display_name: str = ""
    value: str = ""
    direction: Literal["positive", "negative"]
    shap_magnitude: float
    modality: str
    reading: str = ""  # natural-language interpretation


class PredictionEvidence(Evidence):
    predicted_class: int
    prediction_label: str
    probabilities: dict[int, float]
    class_confidence: float  # P(predicted_class)
    modality_contributions: dict[str, float]  # present modalities only
    modality_available: dict[str, bool]
    key_drivers: list[FactorInsight] = Field(default_factory=list)
    uncertainty_reading: str = ""


# ---------------------------------------------------------------------------
# VisualEvidence (visual_analyst)
# ---------------------------------------------------------------------------


class PhotoQuality(BaseModel):
    sharpness: int = Field(..., ge=1, le=5)
    lighting: int = Field(..., ge=1, le=5)
    framing: int = Field(..., ge=1, le=5)
    background: int = Field(..., ge=1, le=5)
    issues: list[str] = Field(default_factory=list)


class PhotoContent(BaseModel):
    pet_visible: bool
    expression: str = ""
    setting: str = ""
    distinctive_traits: list[str] = Field(default_factory=list)


class PhotoAssessment(BaseModel):
    image_index: int
    quality: PhotoQuality
    content: PhotoContent
    appeal_score: int = Field(..., ge=1, le=10)
    improvement_suggestions: list[str] = Field(default_factory=list)


class VisualEvidence(Evidence):
    photos: list[PhotoAssessment] = Field(default_factory=list)
    overall_visual_appeal: int = Field(..., ge=1, le=10)
    best_photo_index: Optional[int] = None
    observed_traits: list[str] = Field(default_factory=list)
    consistency_flags: list[str] = Field(default_factory=list)
    photo_strategy_summary: str = ""


# ---------------------------------------------------------------------------
# RecommendationEvidence (recommendation_agent)
# ---------------------------------------------------------------------------


class MeasuredImpact(BaseModel):
    class_before: int
    class_after: int
    probability_shift: dict[int, float]  # per-class delta from the real ensemble
    expected_speedup: str


class ValidatedRecommendation(BaseModel):
    action: str
    feature: str
    current_value: str
    suggested_value: str
    measured_impact: MeasuredImpact  # measured, never estimated
    priority: int = 1
    category: str = ""  # photo / description / health / listing_details
    rationale: str = ""


class RecommendationEvidence(Evidence):
    recommendations: list[ValidatedRecommendation] = Field(default_factory=list)
    rejected_hypotheses: list[str] = Field(default_factory=list)
    iterations_used: int = 0


# ---------------------------------------------------------------------------
# AdoptionReport (final contract)
# ---------------------------------------------------------------------------


class ReportMetadata(BaseModel):
    session_id: str
    ml_model_version: str = "tuned_v1"
    llm_models: dict[str, str] = Field(default_factory=dict)  # role -> model id
    timing_ms: dict[str, float] = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0
    errors: list[NodeError] = Field(default_factory=list)
    timestamp: str = ""


class AdoptionReport(BaseModel):
    """Final structured output of the agent graph."""

    # Deterministic core — always present, even if every LLM fails
    prediction: PredictionEvidence
    # Evidence — optional depending on modalities/failures
    visual: Optional[VisualEvidence] = None
    recommendations: Optional[RecommendationEvidence] = None
    # Synthesis
    narrative: str = ""
    optimized_description: Optional[str] = None
    headline: str = ""
    metadata: ReportMetadata
