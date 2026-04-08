"""
Recent predictions router.

GET /predictions/recent -- list of recent prediction entries from the job store.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query

from app.api.schemas.responses import (
    RecentPredictionEntry,
    RecentPredictionsResponse,
)
from app.api.services.job_store import job_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predictions", tags=["predictions"])

CLASS_LABELS = {
    0: "Same-day",
    1: "Within 1 week",
    2: "Within 1 month",
    3: "Within 1-3 months",
    4: "100+ days",
}


@router.get("/recent", response_model=RecentPredictionsResponse)
def get_recent_predictions(
    limit: int = Query(20, ge=1, le=100, description="Max entries to return"),
) -> RecentPredictionsResponse:
    """Return recent predictions from the in-memory job store."""
    all_jobs = job_store.get_all()

    # Sort by created_at descending (most recent first)
    sorted_jobs = sorted(all_jobs, key=lambda j: j[1].created_at, reverse=True)

    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp()

    entries: list[RecentPredictionEntry] = []
    total_today = 0

    for session_id, job in sorted_jobs:
        if job.created_at >= today_start:
            total_today += 1

        if len(entries) >= limit:
            continue

        # Extract prediction details from phase1_result
        phase1 = job.phase1_result or {}
        metadata = job.metadata or {}

        prediction = phase1.get("prediction", -1)
        prediction_label = phase1.get("prediction_label", "N/A")
        confidence = phase1.get("confidence", 0.0)

        # Try to determine pet type from the stored data
        pet_type = "N/A"
        # The modality_available can hint at data, but pet_type is not stored
        # in job store directly. We store it in metadata if available.
        pet_type = metadata.get("pet_type", "N/A")

        # Timestamp from metadata or created_at
        timestamp_str = metadata.get(
            "timestamp",
            datetime.fromtimestamp(job.created_at, tz=timezone.utc).isoformat(),
        )

        response_time_ms = metadata.get("total_time_ms", 0.0)

        # Map status
        if job.status == "error":
            display_status = "Error"
        elif job.status == "complete":
            display_status = "Success"
        elif job.status == "phase1_ready":
            display_status = "Partial"
        else:
            display_status = "Pending"

        entries.append(
            RecentPredictionEntry(
                session_id=session_id,
                timestamp=timestamp_str,
                pet_type=pet_type,
                prediction=prediction,
                prediction_label=prediction_label,
                confidence=round(confidence, 4),
                response_time_ms=round(response_time_ms, 1),
                status=display_status,
            )
        )

    return RecentPredictionsResponse(
        predictions=entries,
        total_today=total_today,
    )
