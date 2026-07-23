"""Filesystem storage for per-session uploaded images.

Uploads live under ``artifacts/sessions/{session_id}/images/{index}{ext}``.
This module owns that directory: the predict router writes into it, the image
endpoint reads from it, and the job store's expiry hook deletes it.

Files are named by upload index so that ``PhotoAssessment.image_index`` maps
directly onto a file, and so no user-supplied filename ever becomes part of a
path.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SESSIONS_DIR = _ROOT / "artifacts" / "sessions"

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

_ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def is_valid_session_id(session_id: str) -> bool:
    """True when session_id is a canonical UUID string."""
    return bool(_UUID_RE.match(session_id or ""))


def images_dir(session_id: str) -> Path:
    """Directory holding one session's uploaded images."""
    return SESSIONS_DIR / session_id / "images"


def save_image(session_id: str, index: int, filename: str, content: bytes) -> Path:
    """Persist one upload as ``{index}{ext}``.

    The user's filename is used only to derive a suffix; it never becomes part
    of the stored path.
    """
    ext = Path(filename or "").suffix.lower()
    if ext == ".jpeg":
        ext = ".jpg"
    if ext not in _ALLOWED_EXT:
        ext = ".jpg"
    target_dir = images_dir(session_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{index}{ext}"
    path.write_bytes(content)
    return path


def find_image(session_id: str, index: int) -> Path | None:
    """Return the stored file for ``index``, or None when it does not exist."""
    if not is_valid_session_id(session_id) or index < 0:
        return None
    directory = images_dir(session_id)
    if not directory.is_dir():
        return None
    for candidate in sorted(directory.glob(f"{index}.*")):
        if candidate.is_file():
            return candidate
    return None


def delete_session(session_id: str) -> None:
    """Remove a session's directory. Safe when it does not exist."""
    if not is_valid_session_id(session_id):
        return
    directory = SESSIONS_DIR / session_id
    if directory.is_dir():
        shutil.rmtree(directory, ignore_errors=True)


def prune_all() -> int:
    """Delete every session directory and return how many were removed.

    Called at startup: the job store is in-memory, so no job survives a restart
    and every existing directory is an orphan by definition.
    """
    if not SESSIONS_DIR.is_dir():
        return 0
    removed = 0
    for directory in SESSIONS_DIR.iterdir():
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
    return removed
