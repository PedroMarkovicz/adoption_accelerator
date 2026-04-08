"""
Prediction service: bridges the frontend API schemas with the backend
agent graph and ML pipeline.

Public functions:

  translate_request(pet)          -- PetProfileRequest -> PredictionRequest
  translate_phase1_response(state) -- deterministic graph state -> Phase1 dict
  translate_response(state)       -- full graph state -> PredictionStatusResponse
  run_phase1(pet, det_graph)      -- sync Phase 1; stores in job_store
  run_phase2_background(session_id, pet, full_graph) -- background thread for Phase 2
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Any

from adoption_accelerator.inference.contracts import PredictionRequest, TabularInput

from app.api.schemas.requests import PetProfileRequest
from app.api.schemas.responses import (
    FeatureFactorOut,
    Phase1Response,
    Phase2Response,
    PredictionStatusResponse,
    RecommendationOut,
    ResponseMetadataOut,
)
from app.api.services.job_store import job_store

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup tables: frontend strings -> backend integer codes
# ---------------------------------------------------------------------------

_PET_TYPE: dict[str, int] = {"Dog": 1, "Cat": 2}
_GENDER: dict[str, int] = {"Male": 1, "Female": 2, "Mixed": 3}
_TRISTATE: dict[str, int] = {"Yes": 1, "No": 2, "Not Sure": 3}
_HEALTH: dict[str, int] = {"Healthy": 1, "Minor Injury": 2, "Serious Injury": 3}


# ---------------------------------------------------------------------------
# Request translation
# ---------------------------------------------------------------------------


def translate_request(
    pet: PetProfileRequest,
    image_paths: list[str] | None = None,
) -> PredictionRequest:
    """Map a frontend PetProfileRequest to the backend PredictionRequest.

    Parameters
    ----------
    pet : PetProfileRequest
        The validated frontend request.
    image_paths : list[str] or None
        File-system paths to uploaded images (temp files saved by the
        router).  Passed straight through to ``PredictionRequest.images``.
    """
    tabular = TabularInput(
        type=_PET_TYPE[pet.pet_type],
        name=pet.name if pet.name.strip() else None,
        age=pet.age_months,
        breed1=pet.breed1,
        breed2=pet.breed2 if pet.breed2 != 0 else None,
        gender=_GENDER[pet.gender],
        color1=pet.color1,
        color2=pet.color2 if pet.color2 != 0 else None,
        color3=pet.color3 if pet.color3 != 0 else None,
        maturity_size=pet.maturity_size,
        fur_length=pet.fur_length,
        vaccinated=_TRISTATE[pet.vaccinated],
        dewormed=_TRISTATE[pet.dewormed],
        sterilized=_TRISTATE[pet.sterilized],
        health=_HEALTH[pet.health],
        quantity=pet.quantity,
        fee=pet.fee,
        state=pet.state,
        video_amt=pet.video_amt,
    )
    return PredictionRequest(
        tabular=tabular,
        description=pet.description,
        images=image_paths or [],
    )


# ---------------------------------------------------------------------------
# Response translation helpers
# ---------------------------------------------------------------------------


def _extract_phase1(response: Any) -> Phase1Response:
    """Build a Phase1Response from an AgentResponse object."""
    return Phase1Response(
        prediction=response.prediction,
        prediction_label=response.prediction_label,
        probabilities={str(k): v for k, v in response.probabilities.items()},
        confidence=response.confidence,
        modality_contributions=response.modality_contributions,
        modality_available=response.modality_available,
        top_positive_factors=[
            FeatureFactorOut(**f.model_dump())
            for f in response.top_positive_factors
        ],
        top_negative_factors=[
            FeatureFactorOut(**f.model_dump())
            for f in response.top_negative_factors
        ],
    )


def _extract_phase2(response: Any) -> Phase2Response:
    """Build a Phase2Response from an AgentResponse object.

    Applies defensive checks so that partial LLM failures produce
    None/empty values instead of raising exceptions.
    """
    # narrative_explanation: treat empty strings as None
    narrative = getattr(response, "narrative_explanation", None)
    if not narrative or not str(narrative).strip():
        narrative = None

    # recommendations: safely handle None or non-list values
    raw_recs = getattr(response, "recommendations", None) or []
    recommendations = []
    for r in raw_recs:
        try:
            recommendations.append(RecommendationOut(**r.model_dump()))
        except Exception:
            logger.warning("Skipping malformed recommendation: %s", r)

    # improved_description: treat empty strings as None
    improved = getattr(response, "improved_description", None)
    if improved is not None and not str(improved).strip():
        improved = None

    return Phase2Response(
        narrative_explanation=narrative,
        recommendations=recommendations,
        improved_description=improved,
    )


def _extract_metadata(response: Any) -> ResponseMetadataOut:
    """Build a ResponseMetadataOut from an AgentResponse object.

    Ensures node-level errors from metadata.errors are always propagated
    so the frontend can display degradation warnings.
    """
    meta = getattr(response, "metadata", None)
    if meta is None:
        logger.warning("AgentResponse has no metadata; using defaults.")
        return ResponseMetadataOut(
            session_id="",
            model_version="unknown",
            model_type="unknown",
            inference_time_ms=0.0,
            total_time_ms=0.0,
            timestamp="",
            nodes_executed=[],
            errors=[],
        )

    errors = []
    for e in getattr(meta, "errors", []):
        try:
            errors.append(e.model_dump())
        except Exception:
            errors.append({"node": "unknown", "error_type": "unknown", "message": str(e)})

    return ResponseMetadataOut(
        session_id=meta.session_id,
        model_version=meta.model_version,
        model_type=meta.model_type,
        inference_time_ms=meta.inference_time_ms,
        total_time_ms=meta.total_time_ms,
        timestamp=meta.timestamp,
        nodes_executed=meta.nodes_executed,
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Full response translation (kept for backward compatibility)
# ---------------------------------------------------------------------------


def translate_response(state: dict) -> PredictionStatusResponse:
    """Map the agent graph output state to a complete PredictionStatusResponse."""
    response = state.get("response")
    if response is None:
        raise ValueError(
            "Agent graph produced no response. Check for errors in the state."
        )

    phase1 = _extract_phase1(response)
    phase2 = _extract_phase2(response)
    metadata = _extract_metadata(response)

    return PredictionStatusResponse(
        session_id=metadata.session_id,
        status="complete",
        phase1=phase1,
        phase2=phase2,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Phase 1: synchronous deterministic graph
# ---------------------------------------------------------------------------


def run_phase1(
    pet: PetProfileRequest,
    deterministic_graph: Any,
    image_paths: list[str] | None = None,
) -> PredictionStatusResponse:
    """Run the deterministic graph synchronously and store Phase 1 in the job store.

    Returns a PredictionStatusResponse with status="phase1_ready" and
    only phase1 populated.
    """
    prediction_request = translate_request(pet, image_paths=image_paths)
    state = deterministic_graph.invoke({"request": prediction_request})

    response = state.get("response")
    if response is None:
        raise ValueError(
            "Deterministic graph produced no response. Check for errors in the state."
        )

    phase1 = _extract_phase1(response)
    metadata = _extract_metadata(response)
    session_id = metadata.session_id

    # Store in job store
    job_store.set_phase1_ready(
        session_id,
        phase1=phase1.model_dump(),
        metadata=metadata.model_dump(),
    )

    return PredictionStatusResponse(
        session_id=session_id,
        status="phase1_ready",
        phase1=phase1,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Phase 2: background thread for full graph
# ---------------------------------------------------------------------------


def _cleanup_temp_dir(temp_dir: str | None) -> None:
    """Remove a temp directory created for uploaded images."""
    if temp_dir and os.path.isdir(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as exc:
            logger.warning("Failed to clean up temp dir %s: %s", temp_dir, exc)


def run_phase2_background(
    session_id: str,
    pet: PetProfileRequest,
    full_graph: Any,
    image_paths: list[str] | None = None,
    temp_dir: str | None = None,
) -> None:
    """Run the full graph in a background thread and update the job store.

    This function is the target for ``threading.Thread``. It runs the
    complete pipeline (including LLM agents) and stores both Phase 1
    and Phase 2 results upon completion.  The temp directory created
    for uploaded images is cleaned up in the ``finally`` block.
    """
    try:
        prediction_request = translate_request(pet, image_paths=image_paths)
        start = time.time()
        state = full_graph.invoke({"request": prediction_request})
        elapsed_ms = (time.time() - start) * 1000

        response = state.get("response")
        if response is None:
            job_store.set_error(session_id, "Full graph produced no response.")
            return

        phase1 = _extract_phase1(response)
        phase2 = _extract_phase2(response)
        metadata = _extract_metadata(response)

        # Even if all Phase 2 fields are None/empty (LLM failure), mark as
        # complete so the frontend renders fallback UI instead of an error.
        job_store.set_complete(
            session_id,
            phase1=phase1.model_dump(),
            phase2=phase2.model_dump(),
            metadata=metadata.model_dump(),
        )

        logger.info(
            "Phase 2 complete for session %s (%.0f ms)", session_id, elapsed_ms
        )

    except Exception as exc:
        logger.exception("Phase 2 failed for session %s", session_id)
        # Mark as complete with empty Phase 2 so the frontend degrades
        # gracefully rather than showing a hard error.
        job_store.set_complete(
            session_id,
            phase1=None,  # Phase 1 already stored from run_phase1
            phase2=Phase2Response().model_dump(),
            metadata=None,
        )
        logger.info(
            "Session %s marked complete with empty Phase 2 after error: %s",
            session_id,
            exc,
        )
    finally:
        _cleanup_temp_dir(temp_dir)


def spawn_phase2(
    session_id: str,
    pet: PetProfileRequest,
    full_graph: Any,
    image_paths: list[str] | None = None,
    temp_dir: str | None = None,
) -> threading.Thread:
    """Spawn a daemon thread to run Phase 2 in the background."""
    t = threading.Thread(
        target=run_phase2_background,
        args=(session_id, pet, full_graph),
        kwargs={"image_paths": image_paths, "temp_dir": temp_dir},
        daemon=True,
    )
    t.start()
    return t
