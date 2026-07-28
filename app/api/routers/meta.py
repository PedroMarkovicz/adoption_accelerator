# app/api/routers/meta.py
"""Reference metadata router. GET /meta returns class labels and the
categorical options (breeds/colors/states) consumed by the frontend wizard."""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.schemas.responses import MetaResponse
from app.api.services import meta_service

router = APIRouter()


@router.get("/meta", response_model=MetaResponse)
def get_meta(request: Request) -> MetaResponse:
    model_meta = getattr(request.app.state, "model_meta", {}) or {}
    modality = getattr(request.app.state, "modality_breakdown", {}) or {}
    payload = meta_service.build_meta(
        model_version=model_meta.get("model_version", "tuned_v1"),
        modality_breakdown=modality,
    )
    return MetaResponse.model_validate(payload)
