"""
Prediction service: bridges the frontend API schemas with the backend
Evidence Board agent graph.

Public functions:

  translate_request(pet)                          -- PetProfileRequest -> PredictionRequest
  run_report_background(session_id, request, app)  -- background thread running the graph
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import threading
from typing import Any

from adoption_accelerator.agents.graph import get_graph_config
from adoption_accelerator.agents.observability.audit import write_audit_record
from adoption_accelerator.inference.contracts import PredictionRequest, TabularInput

from app.api.schemas.requests import PetProfileRequest
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
# Background graph run
# ---------------------------------------------------------------------------


def _cleanup_temp_dir(temp_dir: str | None) -> None:
    """Remove a temp directory created for uploaded images."""
    if temp_dir and os.path.isdir(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as exc:
            logger.warning("Failed to clean up temp dir %s: %s", temp_dir, exc)


def run_report_background(
    session_id: str,
    request: PredictionRequest,
    graph_app: Any,
    temp_dir: str | None = None,
) -> threading.Thread:
    """Run the Evidence Board graph in a background thread.

    Invokes ``graph_app.ainvoke`` with the standard graph entry state,
    extracts the resulting ``AdoptionReport`` from ``state["report"]``,
    and stores it in the job store. On failure (exception or missing
    report) the job is marked as errored instead.

    This function is the target for ``threading.Thread``; it returns the
    started (daemon) thread. Any temp directory created for uploaded
    images is cleaned up once the run finishes.
    """

    def _worker() -> None:
        try:
            state = asyncio.run(
                graph_app.ainvoke(
                    {"request": request, "session_id": session_id, "errors": [], "trace": []},
                    config=get_graph_config(session_id) or None,
                )
            )
            report = state.get("report")
            if report is None:
                job_store.set_error(session_id, "Graph produced no report.")
            else:
                job_store.set_complete(
                    session_id,
                    phase1=report.model_dump(mode="json"),
                    phase2=None,
                )
                logger.info("Report complete for session %s", session_id)

            try:
                write_audit_record(state)
            except Exception as audit_exc:
                logger.warning(
                    "Audit record write failed for session %s: %s", session_id, audit_exc
                )
        except Exception as exc:
            logger.exception("Report graph failed for session %s", session_id)
            job_store.set_error(session_id, str(exc))
        finally:
            _cleanup_temp_dir(temp_dir)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread
