"""Startup validation for every config the running system depends on.

Loading each config at boot turns a malformed or drifted file into a
failed startup rather than a runtime surprise on the first request that
happens to touch it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, TypeVar

from adoption_accelerator.agents.llm.registry import (
    _DEFAULT_CONFIG_PATH as _MODELS_CONFIG_PATH,
    load_models_config,
)
from adoption_accelerator.agents.runtime_config import (
    _DEFAULT_TIMEOUTS_PATH,
    load_node_timeouts,
)
from adoption_accelerator.target_labels import (
    _DEFAULT_SERVING_PATH,
    load_target_config,
)

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


def _load_or_raise(loader: Callable[[], _T], path: Path) -> _T:
    """Run a config loader, re-raising any failure as a RuntimeError that
    names the offending file instead of a bare pydantic traceback.
    """
    try:
        return loader()
    except Exception as exc:
        raise RuntimeError(f"Invalid config at {path}: {exc}") from exc


def validate_all_configs() -> None:
    """Load and validate every runtime config. Raises on the first problem.

    Call this before any expensive startup work so a bad config fails
    fast instead of after the model bundle has loaded.
    """
    _load_or_raise(load_models_config, _MODELS_CONFIG_PATH)
    _load_or_raise(load_node_timeouts, _DEFAULT_TIMEOUTS_PATH)
    _load_or_raise(load_target_config, _DEFAULT_SERVING_PATH)
    logger.info("Startup: all configs validated.")
