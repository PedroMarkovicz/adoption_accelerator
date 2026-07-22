"""API service flow test: background report generation into the job store."""

from __future__ import annotations

from adoption_accelerator.agents.contracts import (
    AdoptionReport,
    PredictionEvidence,
    ReportMetadata,
)
from adoption_accelerator.contracts_test_helpers import make_request


def make_report() -> AdoptionReport:
    return AdoptionReport(
        prediction=PredictionEvidence(
            source="data_analyst",
            confidence="medium",
            generated_by="deterministic",
            predicted_class=2,
            prediction_label="Adopted within 1 month",
            probabilities={0: 0.1, 1: 0.2, 2: 0.4, 3: 0.2, 4: 0.1},
            class_confidence=0.4,
            modality_contributions={"tabular": 1.0},
            modality_available={"tabular": True},
        ),
        narrative="n",
        headline="h",
        metadata=ReportMetadata(session_id="s1", timestamp="t"),
    )


class FakeGraphApp:
    async def ainvoke(self, state, config=None):
        return {"report": make_report()}


class FakeGraphAppNoReport:
    async def ainvoke(self, state, config=None):
        return {"report": None}


class FakeGraphAppRaises:
    async def ainvoke(self, state, config=None):
        raise RuntimeError("boom")


def test_background_report_fills_job_store():
    from app.api.services import job_store as job_store_mod
    from app.api.services.prediction_service import run_report_background

    session_id = "test-session-report"
    job_store_mod.job_store.create(session_id)

    thread = run_report_background(session_id, make_request(), FakeGraphApp())
    thread.join(timeout=10)

    stored = job_store_mod.job_store.get(session_id)
    assert stored is not None
    assert stored.status == "complete"
    assert stored.phase1_result is not None
    # Round-trips through the AdoptionReport contract.
    report = AdoptionReport.model_validate(stored.phase1_result)
    assert report.prediction.predicted_class == 2
    assert report.metadata.session_id == "s1"


def test_background_report_missing_report_sets_error():
    from app.api.services import job_store as job_store_mod
    from app.api.services.prediction_service import run_report_background

    session_id = "test-session-no-report"
    job_store_mod.job_store.create(session_id)

    thread = run_report_background(session_id, make_request(), FakeGraphAppNoReport())
    thread.join(timeout=10)

    stored = job_store_mod.job_store.get(session_id)
    assert stored is not None
    assert stored.status == "error"
    assert stored.error is not None


def test_background_report_exception_sets_error():
    from app.api.services import job_store as job_store_mod
    from app.api.services.prediction_service import run_report_background

    session_id = "test-session-raises"
    job_store_mod.job_store.create(session_id)

    thread = run_report_background(session_id, make_request(), FakeGraphAppRaises())
    thread.join(timeout=10)

    stored = job_store_mod.job_store.get(session_id)
    assert stored is not None
    assert stored.status == "error"
    assert "boom" in stored.error


def test_background_report_writes_audit_record(monkeypatch):
    """A successful background run must call write_audit_record with the
    final graph state (Spec 9: every prediction is audit-logged)."""
    from app.api.services import job_store as job_store_mod
    from app.api.services import prediction_service

    calls: list[dict] = []

    def fake_write_audit_record(state, *args, **kwargs):
        calls.append(state)
        return None

    monkeypatch.setattr(prediction_service, "write_audit_record", fake_write_audit_record)

    session_id = "test-session-audit"
    job_store_mod.job_store.create(session_id)

    thread = prediction_service.run_report_background(
        session_id, make_request(), FakeGraphApp()
    )
    thread.join(timeout=10)

    assert len(calls) == 1
    assert calls[0]["report"].metadata.session_id == "s1"


def test_background_report_audit_failure_does_not_break_job(monkeypatch):
    """An audit-write failure must not leave the job stuck or errored."""
    from app.api.services import job_store as job_store_mod
    from app.api.services import prediction_service

    def raising_write_audit_record(state, *args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        prediction_service, "write_audit_record", raising_write_audit_record
    )

    session_id = "test-session-audit-failure"
    job_store_mod.job_store.create(session_id)

    thread = prediction_service.run_report_background(
        session_id, make_request(), FakeGraphApp()
    )
    thread.join(timeout=10)

    stored = job_store_mod.job_store.get(session_id)
    assert stored is not None
    assert stored.status == "complete"
