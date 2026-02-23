"""
Path resolution and directory management utilities.

Functions
---------
get_project_root()
    Return the absolute path to the project root directory.
get_data_path(stage, subset)
    Resolve data paths by stage and subset.
ensure_directories()
    Create the full directory scaffold idempotently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    Walks up from this file's location until it finds ``pyproject.toml``.
    """
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not locate project root (no pyproject.toml found).")


def get_data_path(stage: str, subset: Optional[str] = None) -> Path:
    """Resolve data paths by stage and optional subset.

    Parameters
    ----------
    stage : str
        One of ``"raw"``, ``"cleaned"``, ``"features"``, ``"submissions"``.
    subset : str, optional
        Sub-directory within the stage (e.g. ``"train"``, ``"test"``).

    Returns
    -------
    Path
        Absolute path to the requested data directory.
    """
    valid_stages = {"raw", "cleaned", "features", "submissions"}
    if stage not in valid_stages:
        raise ValueError(f"Invalid stage '{stage}'. Must be one of {valid_stages}.")

    root = get_project_root()
    path = root / "data" / stage

    if subset is not None:
        path = path / subset

    return path


def ensure_directories() -> list[Path]:
    """Create the full project directory scaffold idempotently.

    Returns
    -------
    list[Path]
        List of all directories that were ensured to exist.
    """
    root = get_project_root()

    directories = [
        # Data directories
        root / "data" / "raw",
        root / "data" / "cleaned",
        root / "data" / "features",
        root / "data" / "submissions",
        # Report directories
        root / "reports" / "figures",
        root / "reports" / "metrics",
        # Artifact directories
        root / "artifacts" / "models",
    ]

    for d in directories:
        d.mkdir(parents=True, exist_ok=True)

    return directories
