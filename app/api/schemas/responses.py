"""
Response schemas for the Adoption Accelerator API.

These Pydantic models define the exact contract between the FastAPI
server and any frontend client.  They are derived from the e2e test
output JSONs and the AgentResponse defined in agents/state.py.

Mapping reference (agents/state.py -> API):
  AgentResponse.probabilities (dict[int, float])  -> dict[str, float]
  FeatureFactor                                   -> FeatureFactorOut
  Recommendation                                  -> RecommendationOut
  ResponseMetadata                                -> ResponseMetadataOut
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel


class FeatureFactorOut(BaseModel):
    """A single feature's SHAP contribution, frontend-ready."""

    feature: str
    display_name: str
    value: str
    shap_value: float
    modality: Literal["tabular", "text", "image", "metadata"]
    direction: Literal["positive", "negative"]


class RecommendationOut(BaseModel):
    """A single actionable recommendation."""

    feature: str
    current_value: str
    suggested_value: str
    expected_impact: str
    priority: int
    category: Literal["photo", "description", "health", "listing_details"]
    actionable: bool


class ResponseMetadataOut(BaseModel):
    """Execution metadata for the prediction response."""

    session_id: str
    model_version: str
    model_type: str
    inference_time_ms: float
    total_time_ms: float
    timestamp: str
    nodes_executed: list[str]
    errors: list[dict]


class Phase1Response(BaseModel):
    """Deterministic ML results — available immediately after inference."""

    prediction: int
    prediction_label: str
    probabilities: dict[str, float]
    confidence: float
    modality_contributions: dict[str, float]
    modality_available: dict[str, bool]
    top_positive_factors: list[FeatureFactorOut]
    top_negative_factors: list[FeatureFactorOut]


class Phase2Response(BaseModel):
    """LLM-generated results — narrative, recommendations, improved description."""

    narrative_explanation: Optional[str] = None
    recommendations: list[RecommendationOut] = []
    improved_description: Optional[str] = None


class PredictionStatusResponse(BaseModel):
    """Full prediction response returned by POST /predict."""

    session_id: str
    status: Literal["pending", "phase1_ready", "complete", "error"]
    phase1: Optional[Phase1Response] = None
    phase2: Optional[Phase2Response] = None
    metadata: Optional[ResponseMetadataOut] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Response for GET /health."""

    model_status: Literal["healthy", "degraded", "offline"]
    model_version: str
    model_type: str
    feature_count: int
    agent_status: Literal["connected", "degraded", "offline"]


# ---------------------------------------------------------------------------
# Phase 5: Explore Data & System Status
# ---------------------------------------------------------------------------


class ModelInfoResponse(BaseModel):
    """Detailed model metadata for GET /health/model."""

    model_name: str
    model_version: str
    model_family: str
    base_models: list[str]
    feature_count: int
    training_qwk: float
    modality_breakdown: dict[str, int]


class DistributionEntry(BaseModel):
    """Histogram/bar data for a single feature."""

    feature: str
    display_name: str
    type: Literal["numeric", "categorical"]
    bins: list[float] | None = None
    categories: list[str] | None = None
    counts: list[int]
    by_class: dict[str, list[int]]


class DistributionsResponse(BaseModel):
    """Response for GET /explore/distributions."""

    feature: str
    data: DistributionEntry
    class_labels: dict[str, str]


class PerClassMetric(BaseModel):
    """Per-class precision/recall/F1 entry."""

    class_id: int  # renamed from 'class' to avoid Python keyword
    label: str
    precision: float
    recall: float
    f1: float
    support: int


class GlobalFeatureImportance(BaseModel):
    """A single entry in global feature importance ranking."""

    rank: int
    feature: str
    display_name: str
    mean_abs_shap: float


class PerformanceResponse(BaseModel):
    """Response for GET /explore/performance."""

    confusion_matrix: list[list[int]]
    class_labels: list[str]
    per_class_metrics: list[PerClassMetric]
    aggregate_metrics: dict[str, float]
    global_importance: list[GlobalFeatureImportance]


class RecentPredictionEntry(BaseModel):
    """A single entry in the recent predictions log."""

    session_id: str
    timestamp: str
    pet_type: str
    prediction: int
    prediction_label: str
    confidence: float
    response_time_ms: float
    status: str


class RecentPredictionsResponse(BaseModel):
    """Response for GET /predictions/recent."""

    predictions: list[RecentPredictionEntry]
    total_today: int
