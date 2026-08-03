"""visual_analyst node — a VLM looks at the actual pet photos.

Deliberately independent: receives only declared type/breed/age, never
the ML prediction or SHAP, so its conclusions can be cross-referenced
downstream without contamination. Skips itself when the request has no
images (the spec's conditional-skip, implemented in-node so the graph
topology stays linear)."""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from adoption_accelerator.agents.contracts import (
    NodeError,
    PhotoAssessment,
    TraceEntry,
    VisualEvidence,
)
from adoption_accelerator.agents.llm.client import extract_usage, get_chat_model
from adoption_accelerator.agents.llm.registry import resolve_role
from adoption_accelerator.agents.runtime_config import node_timeout
from adoption_accelerator.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
MAX_IMAGES = 3


class VisualAnalysisOutput(BaseModel):
    """Structured output the VLM must return (Evidence base fields are
    added by the node)."""

    photos: list[PhotoAssessment] = Field(default_factory=list)
    overall_visual_appeal: int = Field(..., ge=1, le=10)
    best_photo_index: Optional[int] = None
    observed_traits: list[str] = Field(default_factory=list)
    appeal_hooks: list[str] = Field(default_factory=list)
    consistency_flags: list[str] = Field(default_factory=list)
    photo_strategy_summary: str = ""


def load_images_base64(
    paths: list[str], cap: int = MAX_IMAGES
) -> list[tuple[int, str, str]]:
    """Load up to ``cap`` readable images as (index, mime, base64)."""
    loaded: list[tuple[int, str, str]] = []
    for i, path in enumerate(paths):
        if len(loaded) >= cap:
            break
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            logger.warning("Skipping unreadable image %s: %s", path, exc)
            continue
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        loaded.append((i, mime, base64.b64encode(data).decode("ascii")))
    return loaded


async def visual_analyst_node(state: AgentState) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timestamp = state.get("timestamp", "")
    request = state.get("request")

    images = load_images_base64(list(getattr(request, "images", []) or []))
    if not images:
        return {
            "visual_evidence": None,
            "errors": [],
            "trace": [_trace(started_at, t0, "skipped", {"reason": "no images"})],
        }

    try:
        t = request.tabular
        breed = request.labels.breed if request.labels else None
        pet_context = (
            f"Declared pet type: {'Dog' if t.type == 1 else 'Cat'}. "
            f"Declared breed: {breed or 'not specified'}. "
            f"Declared age: {t.age} months. "
            f"{len(images)} photo(s) attached."
        )
        content: list[dict] = [{"type": "text", "text": pet_context}]
        for index, mime, b64 in images:
            content.append({"type": "text", "text": f"Photo index {index}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })

        system_prompt = (_PROMPTS_DIR / "visual_analyst_system.txt").read_text(
            encoding="utf-8"
        )
        model = get_chat_model("visual_analyst").with_structured_output(
            VisualAnalysisOutput, include_raw=True
        )
        result = await asyncio.wait_for(
            model.ainvoke(
                [("system", system_prompt), ("user", content)]
            ),
            timeout=node_timeout("visual_analyst"),
        )
        output: VisualAnalysisOutput = result["parsed"]
        raw = result["raw"]

        resolved = resolve_role("visual_analyst")
        notes = []
        if len(images) < len(request.images):
            notes.append(
                f"analyzed {len(images)} of {len(request.images)} images (cap/unreadable)"
            )
        evidence = VisualEvidence(
            source="visual_analyst",
            confidence="medium",
            generated_by=resolved.api_model,
            notes=notes,
            **output.model_dump(),
        )
        return {
            "visual_evidence": evidence,
            "errors": [],
            "trace": [_trace(started_at, t0, "success",
                             {"model": resolved.api_model,
                              "n_images": len(images),
                              "llm_usage": extract_usage(raw, resolved.model_key)})],
        }

    except Exception as exc:
        logger.warning("visual_analyst failed: %s", exc)
        return {
            "visual_evidence": None,
            "errors": [NodeError(node="visual_analyst", error_type="llm_failure",
                                 message=str(exc), timestamp=timestamp,
                                 recoverable=True)],
            "trace": [_trace(started_at, t0, "error", {})],
        }


def _trace(started_at: str, t0: float, status: str, metadata: dict) -> TraceEntry:
    return TraceEntry(
        node="visual_analyst",
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        status=status,
        metadata=metadata,
    )
