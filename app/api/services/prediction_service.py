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
import threading
from typing import Any

from adoption_accelerator.agents.graph import get_graph_config
from adoption_accelerator.agents.observability.audit import write_audit_record
from adoption_accelerator.inference.contracts import (
    ListingLabels,
    PredictionRequest,
    TabularInput,
)

from app.api.schemas.requests import PetProfileRequest
from app.api.services.job_store import job_store
from app.api.services.meta_service import load_breeds, load_colors, load_states

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lookup tables: frontend strings -> backend integer codes
# ---------------------------------------------------------------------------

_PET_TYPE: dict[str, int] = {"Dog": 1, "Cat": 2}
_GENDER: dict[str, int] = {"Male": 1, "Female": 2, "Mixed": 3}
_TRISTATE: dict[str, int] = {"Yes": 1, "No": 2, "Not Sure": 3}
_HEALTH: dict[str, int] = {"Healthy": 1, "Minor Injury": 2, "Serious Injury": 3}


def _resolve_labels(pet: PetProfileRequest) -> ListingLabels:
    """Resolve reference IDs into names for the generative layer.

    The agent layer cannot read the reference CSVs itself, so the names
    travel with the request. Unspecified IDs (0) are left out rather than
    resolved to a placeholder.
    """
    breeds = {b["id"]: b["name"] for b in load_breeds()}
    colors = {c["id"]: c["name"] for c in load_colors()}
    states = {s["id"]: s["name"] for s in load_states()}

    breed_names: list[str] = []
    for breed_id in (pet.breed1, pet.breed2):
        name = breeds.get(breed_id) if breed_id else None
        if name and name not in breed_names:
            breed_names.append(name)

    color_names: list[str] = []
    for color_id in (pet.color1, pet.color2, pet.color3):
        name = colors.get(color_id) if color_id else None
        if name and name not in color_names:
            color_names.append(name)

    return ListingLabels(
        breed=" / ".join(breed_names) or None,
        colors=color_names,
        state=states.get(pet.state) if pet.state else None,
    )


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
        labels=_resolve_labels(pet),
    )


# ---------------------------------------------------------------------------
# Background graph run
# ---------------------------------------------------------------------------


def run_report_background(
    session_id: str,
    request: PredictionRequest,
    graph_app: Any,
) -> threading.Thread:
    """Run the Evidence Board graph in a background thread.

    Invokes ``graph_app.ainvoke`` with the standard graph entry state,
    extracts the resulting ``AdoptionReport`` from ``state["report"]``,
    and stores it in the job store. On failure (exception or missing
    report) the job is marked as errored instead.

    Uploaded images are NOT removed here: they live under the session's
    directory and are deleted when the job expires.
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

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread
