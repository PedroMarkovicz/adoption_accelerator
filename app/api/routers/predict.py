"""
Prediction router -- single-graph background architecture.

POST /predict              -- Create a job, run the Evidence Board graph in
                               a background thread, return 202 immediately.
GET  /predict/{id}/status  -- Poll job status and retrieve the AdoptionReport.

The POST endpoint accepts **two content types**:
  - ``application/json``: JSON body parsed as PetProfileRequest (no images).
  - ``multipart/form-data``: ``profile`` form field (JSON string) +
    optional ``images`` file parts.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile
from fastapi.responses import FileResponse, JSONResponse

from adoption_accelerator.agents.contracts import AdoptionReport

from app.api.schemas.requests import PetProfileRequest
from app.api.schemas.responses import ReportStatusResponse
from app.api.services.job_store import job_store
from app.api.services.prediction_service import run_report_background, translate_request
from app.api.services import session_storage

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 5 * 1024 * 1024


async def _parse_request(
    request: Request,
    session_id: str,
) -> tuple[PetProfileRequest, list[str]]:
    """Parse the incoming request body, handling both JSON and multipart.

    Uploaded images are persisted under the session's directory as
    ``{index}{ext}`` so the report can serve them back later.

    Returns:
        (pet, image_paths)
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type:
        form = await request.form()
        profile_raw = form.get("profile")
        if profile_raw is None:
            raise HTTPException(
                status_code=422,
                detail="Multipart request must include a 'profile' form field.",
            )
        pet = PetProfileRequest.model_validate_json(str(profile_raw))

        raw_images: list[UploadFile] = [
            value
            for value in form.getlist("images")
            if isinstance(value, UploadFile)
        ]

        if len(raw_images) > MAX_IMAGES:
            raise HTTPException(
                status_code=413,
                detail="Too many images. Upload at most 8.",
            )

        image_paths: list[str] = []
        for img_file in raw_images:
            content = await img_file.read()
            if not content:
                continue
            if len(content) > MAX_IMAGE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail="Image too large. Each image must be under 5 MB.",
                )
            path = session_storage.save_image(
                session_id,
                len(image_paths),
                img_file.filename or "",
                content,
            )
            image_paths.append(str(path))

        return pet, image_paths

    body = await request.json()
    pet = PetProfileRequest.model_validate(body)
    return pet, []


@router.post("/predict", status_code=202)
async def predict(request: Request) -> JSONResponse:
    """Create a prediction job and run the Evidence Board graph in the
    background.

    Returns 202 Accepted with the session_id and a "running" status. The
    client should poll GET /predict/{session_id}/status for the report.

    Accepts both ``application/json`` and ``multipart/form-data``.
    """
    graph_app = getattr(request.app.state, "graph", None)
    if graph_app is None:
        raise HTTPException(
            status_code=503,
            detail="Agent graph is not available. Server may still be starting up.",
        )

    session_id = str(uuid.uuid4())
    try:
        pet, image_paths = await _parse_request(request, session_id)
    except Exception:
        session_storage.delete_session(session_id)
        raise

    try:
        prediction_request = translate_request(pet, image_paths=image_paths)
        job_store.create(session_id)

        run_report_background(session_id, prediction_request, graph_app)

        logger.info("Report generation started for session %s", session_id)

        response = ReportStatusResponse(session_id=session_id, status="running")
        return JSONResponse(status_code=202, content=response.model_dump())

    except Exception as exc:
        session_storage.delete_session(session_id)
        logger.exception("Unexpected error while starting prediction")
        raise HTTPException(
            status_code=500, detail=f"Pipeline error: {exc}"
        ) from exc


@router.get("/predict/{session_id}/status", response_model=ReportStatusResponse)
def get_prediction_status(session_id: str) -> ReportStatusResponse:
    """Poll the current status of a prediction job.

    Returns the job state from the in-memory store:
    - ``running``: the graph is still executing.
    - ``done``: the AdoptionReport is available.
    - ``error``: an error occurred during processing.
    """
    job = job_store.get(session_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction job found for session_id={session_id}",
        )

    if job.status == "error":
        return ReportStatusResponse(
            session_id=session_id,
            status="error",
            error=job.error,
        )

    if job.status == "complete":
        report = (
            AdoptionReport.model_validate(job.phase1_result)
            if job.phase1_result is not None
            else None
        )
        return ReportStatusResponse(
            session_id=session_id,
            status="done",
            report=report,
        )

    # pending / phase1_ready (legacy) -- still running
    return ReportStatusResponse(session_id=session_id, status="running")


@router.get("/predict/{session_id}/images/{index}")
def get_prediction_image(session_id: str, index: int) -> FileResponse:
    """Serve one uploaded image for a session, addressed by upload index.

    The path is resolved inside the session's directory from the integer
    index alone, so no user-controlled path segment is ever joined.
    """
    path = session_storage.find_image(session_id, index)
    if path is None:
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(path, headers={"X-Content-Type-Options": "nosniff"})
