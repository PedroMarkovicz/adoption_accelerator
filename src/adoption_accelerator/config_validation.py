"""Startup validation for every config the running system depends on.

Loading each config at boot turns a malformed or drifted file into a
failed startup rather than a runtime surprise on the first request that
happens to touch it.
"""

from __future__ import annotations

import logging

from adoption_accelerator.agents.llm.registry import load_models_config
from adoption_accelerator.agents.runtime_config import load_node_timeouts

logger = logging.getLogger(__name__)


def validate_all_configs() -> None:
    """Load and validate every runtime config. Raises on the first problem.

    Call this before any expensive startup work so a bad config fails
    fast instead of after the model bundle has loaded.
    """
    load_models_config()
    load_node_timeouts()
    logger.info("Startup: all configs validated.")
