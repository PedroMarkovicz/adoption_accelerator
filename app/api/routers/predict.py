"""
Prediction router -- async two-phase architecture.

POST /predict              -- Run Phase 1 sync, spawn Phase 2 background, return 202.
GET  /predict/{id}/status  -- Poll job status and retrieve results.

The POST endpoint accepts **two content types**:
  - ``application/json``: JSON body parsed as PetProfileRequest (no images).
  - ``multipart/form-data``: ``profile`` form field (JSON string) +
    optional ``images`` file parts.
"""

from __future__ import annotations

import logging
import os
import tempfile

from fastapi import APIRouter, HTTPException, Request
from starlette.datastructures import UploadFile
from fastapi.responses import JSONResponse

from app.api.schemas.requests import PetProfileRequest
from app.api.schemas.responses import PredictionStatusResponse
from app.api.services.job_store import job_store
from app.api.services.prediction_service import run_phase1, spawn_phase2

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
    """Run Phase 1 synchronously and spawn Phase 2 in a background thread.

    Returns 202 Accepted with the Phase 1 results immediately available.
    The client should poll GET /predict/{session_id}/status for Phase 2.

    Accepts both ``application/json`` and ``multipart/form-data``.
    """
    pet, image_paths, temp_dir = await _parse_request(request)

    deterministic_graph = getattr(request.app.state, "deterministic_graph", None)
    full_graph = getattr(request.app.state, "graph", None)

    if deterministic_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Deterministic graph is not available. Server may still be starting up.",
        )

    if full_graph is None:
        raise HTTPException(
            status_code=503,
            detail="Full agent graph is not available. Server may still be starting up.",
        )

    try:
        phase1_response = run_phase1(
            pet, deterministic_graph, image_paths=image_paths
        )
        session_id = phase1_response.session_id

        # Spawn Phase 2 in a background thread (owns temp_dir cleanup)
        spawn_phase2(
            session_id,
            pet,
            full_graph,
            image_paths=image_paths,
            temp_dir=temp_dir,
        )

        logger.info(
            "Phase 1 ready for session %s, Phase 2 spawned in background",
            session_id,
        )

        return JSONResponse(
            status_code=202,
            content=phase1_response.model_dump(),
        )

    except ValueError as exc:
        logger.error("Phase 1 failed (bad response from graph): %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during Phase 1 prediction")
        raise HTTPException(
            status_code=500, detail=f"Pipeline error: {exc}"
        ) from exc


@router.get("/predict/{session_id}/status", response_model=PredictionStatusResponse)
def get_prediction_status(session_id: str) -> PredictionStatusResponse:
    """Poll the current status of a prediction job.

    Returns the job state from the in-memory store:
    - ``phase1_ready``: Phase 1 results available, Phase 2 still running.
    - ``complete``: Both Phase 1 and Phase 2 results available.
    - ``error``: An error occurred during processing.
    """
    job = job_store.get(session_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No prediction job found for session_id={session_id}",
        )

    if job.status == "error":
        return PredictionStatusResponse(
            session_id=session_id,
            status="error",
            phase1=job.phase1_result,
            error_message=job.error,
        )

    if job.status == "phase1_ready":
        return PredictionStatusResponse(
            session_id=session_id,
            status="phase1_ready",
            phase1=job.phase1_result,
            metadata=job.metadata,
        )

    if job.status == "complete":
        return PredictionStatusResponse(
            session_id=session_id,
            status="complete",
            phase1=job.phase1_result,
            phase2=job.phase2_result,
            metadata=job.metadata,
        )

    # Fallback: pending
    return PredictionStatusResponse(
        session_id=session_id,
        status="pending",
    )
