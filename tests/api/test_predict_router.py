"""Tests for the /predict router: job creation and status polling."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_profile():
    return {
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


def test_status_echoes_the_submitted_profile(client, sample_profile):
    """The status response carries back the profile exactly as submitted."""
    resp = client.post("/predict", json=sample_profile)
    assert resp.status_code == 202
    session_id = resp.json()["session_id"]

    status = client.get(f"/predict/{session_id}/status")
    assert status.status_code == 200
    listing = status.json()["listing"]
    assert listing is not None
    assert listing["name"] == sample_profile["name"]
    assert listing["pet_type"] == sample_profile["pet_type"]
    assert listing["age_months"] == sample_profile["age_months"]
    assert listing["vaccinated"] == sample_profile["vaccinated"]


def test_status_listing_is_null_when_no_profile_was_stored(client):
    """A job created without a profile reports listing as null, not an error."""
    from app.api.services.job_store import job_store

    job_store.create("no-profile-session")
    try:
        resp = client.get("/predict/no-profile-session/status")
        assert resp.status_code == 200
        assert resp.json()["listing"] is None
    finally:
        job_store._jobs.pop("no-profile-session", None)


def test_profile_is_evicted_with_the_job():
    """The stored profile does not outlive its job."""
    import time
    from app.api.services import job_store as job_store_module

    store = job_store_module.JobStore()
    store.create("expiring", profile={"name": "Milo"})
    assert store.get("expiring").profile == {"name": "Milo"}

    original_ttl = job_store_module.JOB_TTL_SECONDS
    job_store_module.JOB_TTL_SECONDS = -1
    try:
        store.cleanup_expired()
    finally:
        job_store_module.JOB_TTL_SECONDS = original_ttl

    assert store.get("expiring") is None
