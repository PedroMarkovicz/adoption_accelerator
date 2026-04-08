"""
Metric emission for the agent system.

Tracks key operational metrics from agent graph executions:
- inference_latency_ms
- llm_latency_ms (per node)
- total_latency_ms
- prediction_distribution
- confidence_distribution
- node_error_rate
- fallback_rate
- llm_token_usage

Metrics are stored in-memory for lightweight tracking and can be
exported or queried for monitoring purposes.
"""

from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from agents.state import AgentState

logger = logging.getLogger(__name__)


@dataclass
class MetricStore:
    """In-memory metric store for the agent system.

    Thread-safe via a reentrant lock.  Designed for single-process
    monitoring; for production, metrics should be exported to a
    dedicated backend (Prometheus, Datadog, etc.).
    """

    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # Latency histograms (lists of values in ms)
    inference_latency_ms: list[float] = field(default_factory=list)
    llm_latency_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    total_latency_ms: list[float] = field(default_factory=list)

    # Distribution counters
    prediction_distribution: Counter = field(default_factory=Counter)
    confidence_values: list[float] = field(default_factory=list)

    # Error and fallback rates
    node_errors: Counter = field(default_factory=Counter)
    node_executions: Counter = field(default_factory=Counter)
    fallback_count: Counter = field(default_factory=Counter)

    # Token usage
    llm_token_usage: dict[str, dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    )

    # Total request count
    request_count: int = 0

    def record_execution(self, state: AgentState) -> None:
        """Record metrics from a completed graph execution.

        Parameters
        ----------
        state : AgentState
            Completed graph state with trace and response.
        """
        with self._lock:
            self.request_count += 1

            trace_entries = state.get("trace", [])
            response = state.get("response")
            errors = state.get("errors", [])

            # Node-level metrics from trace
            total_ms = 0.0
            for entry in trace_entries:
                self.node_executions[entry.node] += 1
                total_ms += entry.duration_ms

                if entry.status == "error":
                    self.node_errors[entry.node] += 1

                # Inference latency
                if entry.node == "inference":
                    self.inference_latency_ms.append(entry.duration_ms)

                # LLM latency and token usage
                if entry.node in ("explainer", "recommender", "description_writer"):
                    llm_ms = entry.metadata.get("llm_latency_ms", 0)
                    if llm_ms:
                        self.llm_latency_ms[entry.node].append(llm_ms)

                    usage = entry.metadata.get("llm_usage")
                    if usage:
                        stored = self.llm_token_usage[entry.node]
                        stored["input_tokens"] += usage.get("input_tokens", 0)
                        stored["output_tokens"] += usage.get("output_tokens", 0)
                        stored["total_tokens"] += usage.get("total_tokens", 0)

                    if entry.metadata.get("used_fallback"):
                        self.fallback_count[entry.node] += 1

            self.total_latency_ms.append(total_ms)

            # Prediction distribution
            if response is not None:
                self.prediction_distribution[response.prediction] += 1
                self.confidence_values.append(response.confidence)

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of all collected metrics.

        Returns
        -------
        dict
            Metric summary suitable for logging or API response.
        """
        with self._lock:
            return {
                "request_count": self.request_count,
                "inference_latency_ms": _stats(self.inference_latency_ms),
                "total_latency_ms": _stats(self.total_latency_ms),
                "llm_latency_ms": {
                    node: _stats(vals) for node, vals in self.llm_latency_ms.items()
                },
                "prediction_distribution": dict(self.prediction_distribution),
                "confidence": _stats(self.confidence_values),
                "node_error_rate": {
                    node: (
                        self.node_errors.get(node, 0) / count if count > 0 else 0.0
                    )
                    for node, count in self.node_executions.items()
                },
                "fallback_rate": {
                    node: (
                        self.fallback_count.get(node, 0) / self.node_executions.get(node, 1)
                    )
                    for node in ("explainer", "recommender", "description_writer")
                    if self.node_executions.get(node, 0) > 0
                },
                "llm_token_usage": dict(self.llm_token_usage),
            }

    def reset(self) -> None:
        """Reset all metrics to initial state."""
        with self._lock:
            self.inference_latency_ms.clear()
            self.llm_latency_ms.clear()
            self.total_latency_ms.clear()
            self.prediction_distribution.clear()
            self.confidence_values.clear()
            self.node_errors.clear()
            self.node_executions.clear()
            self.fallback_count.clear()
            self.llm_token_usage.clear()
            self.request_count = 0


def _stats(values: list[float]) -> dict[str, float]:
    """Compute basic statistics for a list of numeric values."""
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    return {
        "count": n,
        "mean": round(sum(sorted_vals) / n, 2),
        "min": round(sorted_vals[0], 2),
        "max": round(sorted_vals[-1], 2),
        "p50": round(sorted_vals[n // 2], 2),
    }


# Module-level singleton metric store
_metric_store = MetricStore()


def get_metric_store() -> MetricStore:
    """Return the module-level metric store singleton."""
    return _metric_store
