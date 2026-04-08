"""
Explore Data API router.

GET /explore/distributions  -- precomputed histogram/bar data for a given feature
GET /explore/performance    -- confusion matrix, per-class metrics, global importance
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from app.api.schemas.responses import (
    DistributionEntry,
    DistributionsResponse,
    GlobalFeatureImportance,
    PerClassMetric,
    PerformanceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explore", tags=["explore"])

CLASS_LABELS = {
    "0": "Same-day",
    "1": "Within 1 week",
    "2": "Within 1 month",
    "3": "Within 1-3 months",
    "4": "100+ days",
}


@router.get("/distributions", response_model=DistributionsResponse)
def get_distributions(
    request: Request,
    feature: str = Query(..., description="Feature name to get distribution for"),
    color_by_class: bool = Query(False, description="Include per-class breakdown"),
) -> DistributionsResponse:
    """Return precomputed histogram/bar data for the requested feature."""
    distributions: dict[str, Any] = getattr(request.app.state, "distributions", None) or {}

    if not distributions:
        raise HTTPException(503, detail="Distribution data not loaded.")

    if feature not in distributions:
        available = sorted(distributions.keys())
        raise HTTPException(
            404,
            detail=f"Feature '{feature}' not found. Available: {available}",
        )

    raw = distributions[feature]

    by_class = raw.get("by_class", {}) if color_by_class else {}

    entry = DistributionEntry(
        feature=raw["feature"],
        display_name=raw["display_name"],
        type=raw["type"],
        bins=raw.get("bins"),
        categories=raw.get("categories"),
        counts=raw["counts"],
        by_class=by_class,
    )

    return DistributionsResponse(
        feature=feature,
        data=entry,
        class_labels=CLASS_LABELS,
    )


@router.get("/features")
def get_available_features(request: Request) -> dict:
    """Return the list of features available for distribution exploration."""
    distributions: dict[str, Any] = getattr(request.app.state, "distributions", None) or {}
    features = []
    for key, val in distributions.items():
        features.append({
            "feature": key,
            "display_name": val["display_name"],
            "type": val["type"],
        })
    return {"features": features}


@router.get("/performance", response_model=PerformanceResponse)
def get_performance(request: Request) -> PerformanceResponse:
    """Return confusion matrix, per-class metrics, and global feature importance."""
    performance: dict[str, Any] = getattr(request.app.state, "performance", None) or {}
    global_importance_raw: list[dict] = getattr(
        request.app.state, "global_importance", None
    ) or []
    display_names: dict[str, str] = getattr(
        request.app.state, "display_names", None
    ) or {}

    if not performance:
        raise HTTPException(503, detail="Performance data not loaded.")

    per_class = [
        PerClassMetric(
            class_id=m["class"],
            label=m["label"],
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            support=m["support"],
        )
        for m in performance.get("per_class_metrics", [])
    ]

    # Top 30 global importance with display names
    importance = []
    for entry in global_importance_raw[:30]:
        feat = entry["feature"]
        importance.append(
            GlobalFeatureImportance(
                rank=entry["rank"],
                feature=feat,
                display_name=display_names.get(feat, feat),
                mean_abs_shap=round(entry["mean_abs_shap"], 6),
            )
        )

    return PerformanceResponse(
        confusion_matrix=performance["confusion_matrix"],
        class_labels=performance["class_labels"],
        per_class_metrics=per_class,
        aggregate_metrics=performance["aggregate_metrics"],
        global_importance=importance,
    )


@router.get("/patterns")
def get_adoption_patterns(request: Request) -> dict:
    """Return precomputed adoption pattern data."""
    patterns: dict[str, Any] = getattr(request.app.state, "adoption_patterns", None) or {}
    if not patterns:
        raise HTTPException(503, detail="Adoption patterns data not loaded.")
    return patterns
