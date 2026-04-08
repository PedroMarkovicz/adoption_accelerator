"""
Execution tracing and structured logging for the agent system.

Provides:
- ``configure_logging`` --- sets up structured JSON logging via structlog.
- ``extract_trace_summary`` --- extracts a trace summary from AgentState.
- ``log_node_execution`` --- structured log helper for node entry/exit.
- LangSmith integration (optional, enabled via LANGCHAIN_TRACING_V2 env var).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

from agents.state import AgentState, TraceEntry

logger = logging.getLogger(__name__)


def configure_logging(
    level: int = logging.INFO,
    json_format: bool = True,
) -> None:
    """Configure structured logging for the agent system.

    When ``json_format`` is True and structlog is available, configures
    structlog with JSON rendering.  Otherwise falls back to standard
    library logging with a structured format string.

    Parameters
    ----------
    level : int
        Logging level.
    json_format : bool
        Whether to use JSON output format.
    """
    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        logger.info("Structured logging configured via structlog (JSON=%s)", json_format)

    except ImportError:
        # Fallback to standard library logging with structured format
        fmt = (
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
            if not json_format
            else '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        )
        logging.basicConfig(
            level=level,
            format=fmt,
            stream=sys.stderr,
            force=True,
        )
        logger.info("Structured logging configured via stdlib (JSON=%s)", json_format)


def extract_trace_summary(state: AgentState) -> dict[str, Any]:
    """Extract a trace summary from the completed agent state.

    Collects all ``TraceEntry`` objects and produces a summary dict
    suitable for structured logging or audit records.

    Parameters
    ----------
    state : AgentState
        Completed graph state with trace entries.

    Returns
    -------
    dict
        Summary with node timings, total duration, and status.
    """
    trace_entries: list[TraceEntry] = state.get("trace", [])
    errors = state.get("errors", [])

    node_timings: dict[str, dict[str, Any]] = {}
    for entry in trace_entries:
        node_timings[entry.node] = {
            "duration_ms": entry.duration_ms,
            "status": entry.status,
            "started_at": entry.started_at,
            "completed_at": entry.completed_at,
            "metadata": entry.metadata,
        }

    total_duration_ms = sum(e.duration_ms for e in trace_entries)
    nodes_executed = [e.node for e in trace_entries]
    failed_nodes = [e.node for e in trace_entries if e.status == "error"]
    skipped_nodes = [e.node for e in trace_entries if e.status == "skipped"]

    return {
        "session_id": state.get("session_id", ""),
        "timestamp": state.get("timestamp", ""),
        "nodes_executed": nodes_executed,
        "failed_nodes": failed_nodes,
        "skipped_nodes": skipped_nodes,
        "total_duration_ms": round(total_duration_ms, 2),
        "node_timings": node_timings,
        "error_count": len(errors),
        "has_fatal_errors": any(not e.recoverable for e in errors),
    }


def log_node_execution(
    node_name: str,
    status: str,
    duration_ms: float,
    metadata: dict[str, Any] | None = None,
    session_id: str = "",
) -> None:
    """Emit a structured log entry for a node execution.

    Parameters
    ----------
    node_name : str
        Name of the executed node.
    status : str
        Execution status (success, error, skipped).
    duration_ms : float
        Execution duration in milliseconds.
    metadata : dict | None
        Additional metadata to include.
    session_id : str
        Session ID for correlation.
    """
    log_data = {
        "event": "node_execution",
        "node": node_name,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "session_id": session_id,
    }
    if metadata:
        log_data["metadata"] = metadata

    if status == "error":
        logger.error("Node execution: %s", log_data)
    else:
        logger.info("Node execution: %s", log_data)


def log_graph_completion(state: AgentState) -> None:
    """Log the completion of a full graph execution.

    Parameters
    ----------
    state : AgentState
        Completed graph state.
    """
    summary = extract_trace_summary(state)
    response = state.get("response")

    log_data = {
        "event": "graph_completion",
        "session_id": summary["session_id"],
        "nodes_executed": summary["nodes_executed"],
        "total_duration_ms": summary["total_duration_ms"],
        "error_count": summary["error_count"],
        "has_fatal_errors": summary["has_fatal_errors"],
    }

    if response is not None:
        log_data["prediction"] = response.prediction
        log_data["prediction_label"] = response.prediction_label
        log_data["confidence"] = round(response.confidence, 4)
        log_data["n_recommendations"] = len(response.recommendations)
        log_data["has_narrative"] = bool(response.narrative_explanation)
        log_data["has_improved_description"] = response.improved_description is not None

    logger.info("Graph completed: %s", log_data)


def is_langsmith_enabled() -> bool:
    """Check whether LangSmith tracing is enabled via environment."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true"


def configure_langsmith() -> bool:
    """Configure LangSmith tracing if the environment is set.

    Requires ``LANGCHAIN_TRACING_V2=true`` and a valid
    ``LANGCHAIN_API_KEY`` in the environment.

    Returns
    -------
    bool
        True if LangSmith was configured, False otherwise.
    """
    if not is_langsmith_enabled():
        return False

    api_key = os.environ.get("LANGCHAIN_API_KEY", "")
    if not api_key:
        logger.warning("LANGCHAIN_TRACING_V2 is set but LANGCHAIN_API_KEY is missing")
        return False

    project = os.environ.get("LANGCHAIN_PROJECT", "adoption-accelerator")
    logger.info("LangSmith tracing enabled: project=%s", project)
    return True
