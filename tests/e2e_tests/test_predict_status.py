"""Tests for the /predict router's profile echo.

GET /predict/{id}/status returns the PetProfileRequest submitted at
POST /predict time as a `listing` field, stored on the JobRecord so it
shares the job's TTL.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.main import app

_SAMPLE_PROFILE = {
    "pet_type": "Dog",
    "name": "Milo",
    "age_months": 8,
    "gender": "Male",
    "breed1": 307,
    "maturity_size": 2,
    "fur_length": 1,
    "vaccinated": "Yes",
    "dewormed": "Yes",
    "sterilized": "No",
    "health": "Healthy",
    "fee": 50.0,
    "description": "Friendly puppy looking for a home.",
}


def test_status_echoes_the_submitted_profile(monkeypatch):
    """The status response carries back the profile exactly as submitted.

    The real agent graph is never invoked: run_report_background is
    stubbed out so this test cannot trigger a live LLM call. Only the
    /predict -> job_store.create -> /status round trip is exercised.
    """
    import app.api.routers.predict as predict_router
    from app.api.services.job_store import job_store

    def _skip_graph_run(session_id, request, graph_app):
        return None

    monkeypatch.setattr(predict_router, "run_report_background", _skip_graph_run)

    with TestClient(app) as client:
        resp = client.post("/predict", json=_SAMPLE_PROFILE)
        assert resp.status_code == 202
        session_id = resp.json()["session_id"]
        try:
            status = client.get(f"/predict/{session_id}/status")
        finally:
            job_store._jobs.pop(session_id, None)

    assert status.status_code == 200
    listing = status.json()["listing"]
    assert listing is not None
    assert listing["name"] == _SAMPLE_PROFILE["name"]
    assert listing["pet_type"] == _SAMPLE_PROFILE["pet_type"]
    assert listing["age_months"] == _SAMPLE_PROFILE["age_months"]
    assert listing["vaccinated"] == _SAMPLE_PROFILE["vaccinated"]


def test_status_listing_is_null_when_no_profile_was_stored():
    """A job created without a profile reports listing as null, not an error."""
    from app.api.services.job_store import job_store

    job_store.create("no-profile-session")
    try:
        with TestClient(app) as client:
            resp = client.get("/predict/no-profile-session/status")
        assert resp.status_code == 200
        assert resp.json()["listing"] is None
    finally:
        job_store._jobs.pop("no-profile-session", None)


def test_profile_is_evicted_with_the_job(monkeypatch):
    """The stored profile does not outlive its job, and is present in both
    the single-job lookup (get) and the bulk lookup (get_all) while the
    job is still live.
    """
    import app.api.services.job_store as js

    monkeypatch.setattr(js, "JOB_TTL_SECONDS", -1)  # everything is already expired
    store = js.JobStore()
    store.create("expiring", profile={"name": "Milo"})
    assert store.get("expiring").profile == {"name": "Milo"}

    all_jobs = dict(store.get_all())
    assert all_jobs["expiring"].profile == {"name": "Milo"}

    store.cleanup_expired()

    assert store.get("expiring") is None
