"""Optional Langfuse tracing. Without keys the system runs untraced."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_langfuse_handler(session_id: str) -> Optional[Any]:
    """Return a Langfuse CallbackHandler, or None when not configured.

    Requires both ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` to be
    set. Any import or construction failure (missing package, bad
    credentials, network issues) degrades to ``None`` rather than raising,
    so tracing is always best-effort.
    """
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "")
    if not public or not secret:
        return None
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except Exception as exc:
        logger.warning("Langfuse unavailable: %s", exc)
        return None
