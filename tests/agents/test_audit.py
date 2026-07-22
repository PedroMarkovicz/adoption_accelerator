"""Unit tests for the audit record port (report -> audit dict, no I/O)."""

from __future__ import annotations

import json

from adoption_accelerator.agents.contracts import (
    AdoptionReport,
    PredictionEvidence,
    ReportMetadata,
)
from adoption_accelerator.agents.observability.audit import (
    build_audit_record,
    write_audit_record,
)


def _make_report(
    *,
    predicted_class: int = 2,
    class_confidence: float = 0.4321,
    session_id: str = "sess-abc",
    timestamp: str = "2026-07-21T12:00:00Z",
) -> AdoptionReport:
    prediction = PredictionEvidence(
        source="data_analyst",
        confidence="high",  # qualitative Evidence.confidence -- must NOT leak
        # into the audit record's numeric "confidence" field.
        generated_by="deterministic",
        predicted_class=predicted_class,
        prediction_label="2-4 weeks",
        probabilities={0: 0.1, 1: 0.1, 2: class_confidence, 3: 0.2, 4: 0.2},
        class_confidence=class_confidence,
        modality_contributions={"tabular": 1.0},
        modality_available={"tabular": True, "text": False, "image": False},
    )
    metadata = ReportMetadata(session_id=session_id, timestamp=timestamp)
    return AdoptionReport(prediction=prediction, metadata=metadata, narrative="hi")


def test_build_audit_record_from_report():
    report = _make_report(predicted_class=3, class_confidence=0.777)
    state = {
        "report": report,
        "trace": [],
        "errors": [],
        "session_id": report.metadata.session_id,
        "timestamp": report.metadata.timestamp,
    }

    record = build_audit_record(state)

    # Predicted class comes from report.prediction.predicted_class, not the
    # qualitative Evidence.confidence literal ("high"/"medium"/"low").
    assert record["prediction"]["prediction"] == 3
    assert record["prediction"]["confidence"] == 0.777
    assert record["prediction"]["confidence"] != "high"
    assert record["prediction"]["prediction_label"] == "2-4 weeks"
    assert record["session_id"] == "sess-abc"
    assert record["timestamp"] == "2026-07-21T12:00:00Z"


def test_build_audit_record_handles_missing_report():
    state = {
        "trace": [],
        "errors": [],
        "session_id": "sess-no-report",
        "timestamp": "2026-07-21T00:00:00Z",
    }

    record = build_audit_record(state)

    assert record["prediction"] == {}
    assert record["session_id"] == "sess-no-report"
    assert record["timestamp"] == "2026-07-21T00:00:00Z"
    assert record["error_count"] == 0
    assert record["errors"] == []


def test_write_audit_record_writes_jsonl_to_given_dir(tmp_path):
    """write_audit_record() must append a JSON line to
    <audit_dir>/YYYY-MM-DD.jsonl. Uses a tmp_path so the test never writes
    into the repo working tree."""
    report = _make_report(session_id="sess-write-test")
    state = {
        "report": report,
        "trace": [],
        "errors": [],
        "session_id": report.metadata.session_id,
        "timestamp": report.metadata.timestamp,
    }

    audit_dir = tmp_path / "audit"
    file_path = write_audit_record(state, audit_dir=audit_dir)

    assert file_path is not None
    assert file_path.exists()
    assert file_path.parent == audit_dir

    lines = file_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["session_id"] == "sess-write-test"
    assert record["prediction"]["prediction"] == report.prediction.predicted_class
