"""
Prediction audit logging for the agent system.

Provides append-only audit records that capture the essential details
of every prediction for compliance and debugging.  Each audit entry
records the session, request summary, prediction, node durations,
LLM token usage, and any errors.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.state import AgentState

logger = logging.getLogger(__name__)

# Default audit log location (project_root/logs/audit/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_AUDIT_DIR = _PROJECT_ROOT / "logs" / "audit"


def build_audit_record(state: AgentState) -> dict[str, Any]:
    """Build an audit record from a completed agent state.

    Parameters
    ----------
    state : AgentState
        Completed graph state.

    Returns
    -------
    dict
        Audit record with session, prediction, timing, and error data.
    """
    response = state.get("response")
    trace_entries = state.get("trace", [])
    errors = state.get("errors", [])
    request = state.get("request")

    # Request summary (no PII, no raw data)
    request_summary: dict[str, Any] = {}
    if request is not None:
        t = request.tabular
        request_summary = {
            "pet_type": t.type,
            "age": t.age,
            "has_description": bool(request.description and request.description.strip()),
            "n_images": len(request.images) if request.images else 0,
        }

    # Node durations
    node_durations = {e.node: e.duration_ms for e in trace_entries}

    # LLM token usage from trace metadata
    llm_usage: dict[str, dict[str, int]] = {}
    for entry in trace_entries:
        usage = entry.metadata.get("llm_usage")
        if usage:
            llm_usage[entry.node] = usage

    # Prediction info
    prediction_info: dict[str, Any] = {}
    if response is not None:
        prediction_info = {
            "prediction": response.prediction,
            "prediction_label": response.prediction_label,
            "confidence": round(response.confidence, 4),
            "n_recommendations": len(response.recommendations),
            "has_narrative": bool(response.narrative_explanation),
            "has_improved_description": response.improved_description is not None,
        }

    return {
        "audit_version": "1.0",
        "session_id": state.get("session_id", ""),
        "timestamp": state.get("timestamp", ""),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "request_summary": request_summary,
        "prediction": prediction_info,
        "node_durations_ms": node_durations,
        "total_duration_ms": sum(node_durations.values()),
        "llm_token_usage": llm_usage,
        "error_count": len(errors),
        "errors": [
            {
                "node": e.node,
                "error_type": e.error_type,
                "message": e.message,
                "recoverable": e.recoverable,
            }
            for e in errors
        ],
    }


def write_audit_record(
    state: AgentState,
    audit_dir: str | Path | None = None,
) -> Path | None:
    """Write an audit record to the append-only audit log.

    Each record is written as a single JSON line to a date-partitioned
    file: ``<audit_dir>/YYYY-MM-DD.jsonl``.

    Parameters
    ----------
    state : AgentState
        Completed graph state.
    audit_dir : str | Path | None
        Directory for audit files.  Defaults to ``logs/audit/``.

    Returns
    -------
    Path | None
        Path to the audit file, or None on failure.
    """
    audit_path = Path(audit_dir) if audit_dir else _DEFAULT_AUDIT_DIR

    try:
        audit_path.mkdir(parents=True, exist_ok=True)

        record = build_audit_record(state)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        file_path = audit_path / f"{date_str}.jsonl"

        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

        logger.info(
            "Audit record written: session=%s, file=%s",
            record["session_id"],
            file_path,
        )
        return file_path

    except Exception as exc:
        logger.error("Failed to write audit record: %s", exc)
        return None
