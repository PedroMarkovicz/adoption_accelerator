"""Runtime configuration for the agent graph.

Loads and validates ``configs/agents/timeouts.yaml``. Mirrors the loader
pattern in ``agents/llm/registry.py``: pydantic validation, a cache keyed
by resolved path, and a clear hook for tests.

Why this module exists: the YAML predates it by months and nothing read
it. Every node carried its own ``_TIMEOUT_SECONDS`` literal that happened
to match, so editing the config was a no-op that looked like a change.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from adoption_accelerator import config

# The node set compiled in agents/graph.py. A timeout config that does not
# cover exactly these names is a configuration error, not a default.
GRAPH_NODES: frozenset[str] = frozenset({
    "orchestrator",
    "inference",
    "visual_analyst",
    "data_analyst",
    "recommendation_agent",
    "synthesizer",
    "aggregator",
})

# Bounded ReAct budget for the recommendation agent. Raising this raises
# cost and latency roughly linearly; both recorded runs exhausted it.
MAX_TOOL_CALLS: int = 8

# Expected-class-value moves smaller than this are noise, not effect.
# Matches the scale of the ordinal target (0-4).
SPEEDUP_EPSILON: float = 1e-3

_DEFAULT_TIMEOUTS_PATH = config.PROJECT_ROOT / "configs" / "agents" / "timeouts.yaml"

_timeouts_cache: dict[Path, "NodeTimeouts"] = {}


class NodeTimeouts(BaseModel):
    """Per-node timeout budget in seconds."""

    node_timeouts: dict[str, float]

    @model_validator(mode="after")
    def _validate_node_coverage(self) -> "NodeTimeouts":
        configured = set(self.node_timeouts)
        unknown = configured - GRAPH_NODES
        if unknown:
            raise ValueError(
                f"unknown node(s) in timeouts config: {sorted(unknown)}"
            )
        missing = GRAPH_NODES - configured
        if missing:
            raise ValueError(
                f"missing timeout for graph node(s): {sorted(missing)}"
            )
        for node, seconds in self.node_timeouts.items():
            if seconds <= 0:
                raise ValueError(
                    f"node '{node}' has a non-positive timeout: {seconds}"
                )
        return self


def load_node_timeouts(path: Path | None = None) -> NodeTimeouts:
    """Load and validate the node timeout config (cached per path)."""
    resolved = (path or _DEFAULT_TIMEOUTS_PATH).resolve()
    if resolved not in _timeouts_cache:
        with open(resolved, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _timeouts_cache[resolved] = NodeTimeouts.model_validate(raw)
    return _timeouts_cache[resolved]


def node_timeout(node: str) -> float:
    """Timeout in seconds for one graph node.

    Raises KeyError for a name that is not a graph node, so a typo fails
    loudly instead of silently picking a default.
    """
    timeouts = load_node_timeouts().node_timeouts
    return timeouts[node]


def clear_runtime_config_cache() -> None:
    """Clear the config cache (tests)."""
    _timeouts_cache.clear()
