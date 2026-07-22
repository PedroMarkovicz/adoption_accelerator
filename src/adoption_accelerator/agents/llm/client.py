"""Provider-agnostic chat model factory.

Thin wrapper over LangChain's ``init_chat_model``: the provider is data
in ``configs/agents/models.yaml``, never code. Reasoning-effort is only
forwarded to providers whose catalog entry declares
``reasoning_kind: effort``.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from adoption_accelerator.agents.llm.registry import (
    load_models_config,
    resolve_role,
)

logger = logging.getLogger(__name__)

_model_cache: dict[str, BaseChatModel] = {}


def get_chat_model(role: str) -> BaseChatModel:
    """Return the chat model configured for a role (cached)."""
    if role not in _model_cache:
        resolved = resolve_role(role)
        spec = load_models_config().catalog[resolved.model_key]

        kwargs: dict[str, Any] = {
            "model": resolved.api_model,
            "model_provider": resolved.provider,
            "max_tokens": resolved.max_output_tokens,
        }
        if spec.reasoning_kind == "effort":
            kwargs["reasoning_effort"] = resolved.reasoning_effort

        logger.info(
            "Initializing chat model for role=%s: %s/%s",
            role, resolved.provider, resolved.api_model,
        )
        _model_cache[role] = init_chat_model(**kwargs)
    return _model_cache[role]


def extract_usage(message: Any, model_key: str) -> dict[str, Any]:
    """Extract token usage from a LangChain AIMessage (zeros when absent)."""
    usage = getattr(message, "usage_metadata", None) or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0)),
        "output_tokens": int(usage.get("output_tokens", 0)),
        "model_key": model_key,
    }


def clear_model_cache() -> None:
    """Clear cached model instances (tests)."""
    _model_cache.clear()
