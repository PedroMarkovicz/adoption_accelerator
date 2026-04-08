"""
Health check router.

GET /health -- returns model status, version, feature count,
and agent graph connectivity.  The inference pipeline singleton
must be loaded and pass a basic attribute check to be considered
"healthy".
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request

from app.api.schemas.responses import HealthResponse, ModelInfoResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Return the current health status of the ML model and agent graph."""
    pipeline = getattr(request.app.state, "pipeline", None)
    graph = getattr(request.app.state, "graph", None)

    # Check ML pipeline
    if pipeline is None:
        model_status: str = "offline"
        feature_count = 0
        model_version = "unknown"
        model_type = "unknown"
    else:
        try:
            feature_count = len(pipeline.feature_schema.get("features", []))
            # model_version is stored in the bundle metadata; fall back to
            # the known value used throughout the pipeline.
            bundle_meta = pipeline.bundle.get("metadata", {})
            model_version = bundle_meta.get("model_version", "tuned_v1")
            model_type = bundle_meta.get("model_type", "SoftVotingEnsemble")
            model_status = "healthy"
        except Exception as exc:
            logger.warning("Pipeline health check failed: %s", exc)
            model_status = "degraded"
            feature_count = 0
            model_version = "unknown"
            model_type = "unknown"

    agent_status: str = "connected" if graph is not None else "offline"

    return HealthResponse(
        model_status=model_status,
        model_version=model_version,
        model_type=model_type,
        feature_count=feature_count,
        agent_status=agent_status,
    )


@router.get("/health/model", response_model=ModelInfoResponse)
def model_info(request: Request) -> ModelInfoResponse:
    """Return detailed model metadata."""
    model_meta = getattr(request.app.state, "model_meta", None)
    pipeline = getattr(request.app.state, "pipeline", None)

    if model_meta is None:
        model_meta = {}

    feature_count = 0
    if pipeline is not None:
        try:
            feature_count = len(pipeline.feature_schema.get("features", []))
        except Exception:
            pass

    # Modality breakdown from feature schema
    modality_breakdown = getattr(request.app.state, "modality_breakdown", {})

    return ModelInfoResponse(
        model_name=model_meta.get("model_name", "SoftVotingEnsemble"),
        model_version=model_meta.get("model_version", "tuned_v1"),
        model_family=model_meta.get("model_family", "ensemble"),
        base_models=model_meta.get("base_models", ["LightGBM", "XGBoost", "CatBoost"]),
        feature_count=feature_count,
        training_qwk=model_meta.get("training_qwk", 0.0),
        modality_breakdown=modality_breakdown if modality_breakdown else {
            "tabular": feature_count, "text": 0, "image": 0, "metadata": 0
        },
    )
