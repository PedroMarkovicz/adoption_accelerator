"""
Response schemas for the Adoption Accelerator API.

These Pydantic models define the exact contract between the FastAPI
server and any frontend client.

The prediction flow is served by ``ReportStatusResponse``, a thin wrapper
around the agent graph's ``AdoptionReport`` (see
``adoption_accelerator.agents.contracts``).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from adoption_accelerator.agents.contracts import AdoptionReport


class ReportStatusResponse(BaseModel):
    """Status/result wrapper returned by POST /predict and its status poll."""

    session_id: str
    status: str  # queued / running / done / error
    report: Optional[AdoptionReport] = None
    error: Optional[str] = None


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


# ---------------------------------------------------------------------------
# GET /meta: reference data for the frontend
# ---------------------------------------------------------------------------


class SpeedClassEntry(BaseModel):
    index: int
    label: str


class BreedOption(BaseModel):
    id: int
    type: int
    name: str


class LabeledOption(BaseModel):
    id: int
    name: str


class IdLabel(BaseModel):
    id: int
    label: str


class MetaResponse(BaseModel):
    """Response for GET /meta: labels and categorical reference options."""

    model_version: str
    modality_breakdown: dict[str, int]
    adoption_speed_classes: list[SpeedClassEntry]
    breeds: list[BreedOption]
    colors: list[LabeledOption]
    states: list[LabeledOption]
    maturity_sizes: list[IdLabel]
    fur_lengths: list[IdLabel]
