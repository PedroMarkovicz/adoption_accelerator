"""
Agent configuration loader and LLM client factory.

Loads YAML configuration files from ``configs/agents/`` and exposes
typed Pydantic config objects for the agent system.  Also provides
a singleton OpenAI client factory for use by LLM-powered nodes.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Resolve project root (parent of the agents/ package)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs" / "agents"


# ---------------------------------------------------------------------------
# Typed configuration models
# ---------------------------------------------------------------------------


class LLMReasoningConfig(BaseModel):
    """LLM reasoning effort configuration."""

    effort: str = "minimal"


class LLMMaxTokensConfig(BaseModel):
    """Per-node max output token limits."""

    explainer: int = 500
    recommender: int = 600
    description_writer: int = 800


class LLMConfig(BaseModel):
    """LLM provider and model configuration."""

    provider: str = "openai"
    model: str = "gpt-5-nano"
    reasoning: LLMReasoningConfig = Field(default_factory=LLMReasoningConfig)
    verbosity: str = "low"
    temperature: float = 0.3
    max_tokens: LLMMaxTokensConfig = Field(default_factory=LLMMaxTokensConfig)


class TimeoutConfig(BaseModel):
    """Timeout settings for the agent graph."""

    global_timeout: int = 30
    node_timeouts: dict[str, int] = Field(
        default_factory=lambda: {
            "orchestrator": 2,
            "inference": 5,
            "explainer": 10,
            "recommender": 10,
            "description_writer": 15,
            "aggregator": 2,
        }
    )

    def get_node_timeout(self, node_name: str) -> int:
        """Return the timeout for a specific node, with a sensible default."""
        return self.node_timeouts.get(node_name, 10)


class GraphNodeDef(BaseModel):
    """Definition of a single node in the graph topology."""

    name: str
    phase: int = 1
    deterministic: bool = True
    conditional: bool = False


class GraphEdgeDef(BaseModel):
    """Definition of a single edge in the graph topology."""

    from_node: str = Field(..., alias="from")
    to: Any = None  # str or list[str]
    routing: Optional[str] = None

    model_config = {"populate_by_name": True}


class GraphConfig(BaseModel):
    """Graph topology configuration."""

    entry_point: str = "orchestrator"
    nodes: list[GraphNodeDef] = Field(default_factory=list)
    edges: list[GraphEdgeDef] = Field(default_factory=list)
    routing_rules: dict[str, dict[str, str]] = Field(default_factory=dict)


class AgentConfig(BaseModel):
    """Top-level agent configuration combining all sub-configs."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    timeouts: TimeoutConfig = Field(default_factory=TimeoutConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)


# ---------------------------------------------------------------------------
# YAML loading utilities
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict if the file is missing."""
    if not path.exists():
        logger.warning("Config file not found: %s (using defaults)", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data else {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def load_agent_config(config_dir: str | None = None) -> AgentConfig:
    """Load and cache the full agent configuration.

    Reads ``llm.yaml``, ``timeouts.yaml``, and ``graph.yaml`` from
    the given directory (defaulting to ``configs/agents/``).

    Parameters
    ----------
    config_dir : str | None
        Path to the agent config directory.  Defaults to
        ``<project_root>/configs/agents/``.

    Returns
    -------
    AgentConfig
        Fully validated configuration object.
    """
    base = Path(config_dir) if config_dir else _CONFIGS_DIR

    llm_data = _load_yaml(base / "llm.yaml")
    timeout_data = _load_yaml(base / "timeouts.yaml")
    graph_data = _load_yaml(base / "graph.yaml")

    config = AgentConfig(
        llm=LLMConfig(**llm_data) if llm_data else LLMConfig(),
        timeouts=TimeoutConfig(**timeout_data) if timeout_data else TimeoutConfig(),
        graph=GraphConfig(**graph_data) if graph_data else GraphConfig(),
    )

    logger.info(
        "Agent config loaded: provider=%s, model=%s, global_timeout=%ds",
        config.llm.provider,
        config.llm.model,
        config.timeouts.global_timeout,
    )
    return config


def load_api_key(env_file: str | None = None) -> str | None:
    """Load the OpenAI API key from a ``.env`` file.

    Uses ``python-dotenv`` if available; otherwise falls back to
    ``os.environ``.  Returns ``None`` if the key is not set.

    Parameters
    ----------
    env_file : str | None
        Path to the ``.env`` file.  Defaults to ``<project_root>/.env``.

    Returns
    -------
    str | None
        The API key, or ``None`` if not found.
    """
    import os

    env_path = Path(env_file) if env_file else _PROJECT_ROOT / ".env"

    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
    except ImportError:
        logger.debug("python-dotenv not installed; reading OPENAI_API_KEY from env only")

    return os.environ.get("OPENAI_API_KEY")


# ---------------------------------------------------------------------------
# OpenAI client factory (singleton)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_openai_client() -> Any:
    """Return a cached OpenAI client instance.

    Loads the API key from the environment (via ``.env`` if
    ``python-dotenv`` is installed) and creates an ``openai.OpenAI``
    client configured with sensible defaults.

    Returns
    -------
    openai.OpenAI
        Configured OpenAI client.

    Raises
    ------
    RuntimeError
        If ``OPENAI_API_KEY`` is not set.
    """
    import httpx
    from openai import OpenAI

    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or set it as an environment variable."
        )

    cfg = load_agent_config()
    timeout_seconds = cfg.timeouts.global_timeout

    client = OpenAI(
        api_key=api_key,
        timeout=httpx.Timeout(timeout_seconds, connect=5.0),
    )

    logger.info(
        "OpenAI client created: model=%s, timeout=%ds",
        cfg.llm.model,
        timeout_seconds,
    )
    return client
