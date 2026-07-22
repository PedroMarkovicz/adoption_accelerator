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
import os
import tempfile
import uuid

from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile
from fastapi.responses import JSONResponse

from adoption_accelerator.agents.contracts import AdoptionReport

from app.api.schemas.requests import PetProfileRequest
from app.api.schemas.responses import ReportStatusResponse
from app.api.services.job_store import job_store
from app.api.services.prediction_service import run_report_background, translate_request

logger = logging.getLogger(__name__)

router = APIRouter()


async def _parse_request(
    request: Request,
) -> tuple[PetProfileRequest, list[str], str | None]:
    """Parse the incoming request body, handling both JSON and multipart.

    Returns:
        (pet, image_paths, temp_dir)
        - pet: validated PetProfileRequest
        - image_paths: list of temp file paths for uploaded images
        - temp_dir: path to the temp directory (for cleanup), or None
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

        # Save uploaded images to a temp directory
        image_paths: list[str] = []
        temp_dir: str | None = None

        raw_images: list[UploadFile] = []
        for key in form:
            if key == "images":
                val = form.getlist("images")
                raw_images = [v for v in val if isinstance(v, UploadFile)]
                break

        if raw_images:
            temp_dir = tempfile.mkdtemp(prefix="adopt_img_")
            for i, img_file in enumerate(raw_images):
                content = await img_file.read()
                if not content:
                    continue
                fname = img_file.filename or f"image_{i}.jpg"
                # Sanitize filename to avoid path traversal
                fname = os.path.basename(fname)
                fpath = os.path.join(temp_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(content)
                image_paths.append(fpath)

        return pet, image_paths, temp_dir
    else:
        # Standard JSON body
        body = await request.json()
        pet = PetProfileRequest.model_validate(body)
        return pet, [], None


@router.post("/predict", status_code=202)
async def predict(request: Request) -> JSONResponse:
    """Create a prediction job and run the Evidence Board graph in the
    background.

    Returns 202 Accepted with the session_id and a "running" status. The
    client should poll GET /predict/{session_id}/status for the report.

    Accepts both ``application/json`` and ``multipart/form-data``.
    """
    pet, image_paths, temp_dir = await _parse_request(request)

    graph_app = getattr(request.app.state, "graph", None)
    if graph_app is None:
        raise HTTPException(
            status_code=503,
            detail="Agent graph is not available. Server may still be starting up.",
        )

    session_id = str(uuid.uuid4())

    try:
        prediction_request = translate_request(pet, image_paths=image_paths)
        job_store.create(session_id)

        run_report_background(
            session_id,
            prediction_request,
            graph_app,
            temp_dir=temp_dir,
        )

        logger.info("Report generation started for session %s", session_id)

        response = ReportStatusResponse(session_id=session_id, status="running")
        return JSONResponse(status_code=202, content=response.model_dump())

    except Exception as exc:
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
