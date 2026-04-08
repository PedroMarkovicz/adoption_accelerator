"""
Orchestrator node --- entry point of the agent graph.

Validates the ``PredictionRequest``, generates session metadata
(UUID v4 session_id, ISO 8601 UTC timestamp), and determines the
execution plan for downstream nodes.

This is a deterministic node with no LLM dependency.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

from pydantic import ValidationError

from agents.state import AgentState, NodeError, TraceEntry

logger = logging.getLogger(__name__)


def orchestrator_node(state: AgentState) -> dict:
    """Validate request and initialize session metadata.

    Parameters
    ----------
    state : AgentState
        Current graph state containing the ``request`` field.

    Returns
    -------
    dict
        State updates: ``session_id``, ``timestamp``, ``trace``,
        and ``errors`` if the request is malformed.
    """
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    # Generate session metadata
    session_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc).isoformat()

    # Validate the request
    request = state.get("request")
    errors: list[NodeError] = []

    if request is None:
        errors.append(
            NodeError(
                node="orchestrator",
                error_type="missing_request",
                message="No PredictionRequest found in state",
                timestamp=timestamp,
                recoverable=False,
            )
        )
    else:
        # Validate via Pydantic re-construction (catches schema issues)
        try:
            from adoption_accelerator.inference.contracts import PredictionRequest

            if isinstance(request, dict):
                PredictionRequest(**request)
            # If it's already a PredictionRequest, validation passed at construction
        except (ValidationError, TypeError) as exc:
            errors.append(
                NodeError(
                    node="orchestrator",
                    error_type="invalid_request",
                    message=f"PredictionRequest validation failed: {exc}",
                    timestamp=timestamp,
                    recoverable=False,
                )
            )

    duration_ms = (time.perf_counter() - t0) * 1000
    completed_at = datetime.now(timezone.utc).isoformat()

    trace = TraceEntry(
        node="orchestrator",
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=round(duration_ms, 2),
        status="error" if errors else "success",
        metadata={"session_id": session_id},
    )

    logger.info(
        "Orchestrator: session=%s, errors=%d, duration=%.1fms",
        session_id,
        len(errors),
        duration_ms,
    )

    return {
        "session_id": session_id,
        "timestamp": timestamp,
        "errors": errors,
        "trace": [trace],
    }
