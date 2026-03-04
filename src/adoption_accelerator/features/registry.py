"""
Feature registry utilities for schema registration and validation.

Provides functions to save and load ``schema.json`` files that accompany
every versioned feature set in ``data/features/{modality}/v{N}/``.

Functions
---------
save_feature_schema(columns, metadata, path)
    Generate and persist a schema.json for a feature set.
load_feature_schema(path)
    Load and validate a schema.json.
compute_config_hash(config_dict)
    Compute a SHA-256 hash for a feature generation configuration.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("adoption_accelerator")


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute a SHA-256 hash for a feature generation configuration.

    Parameters
    ----------
    config_dict : dict
        Dictionary with configuration parameters used to generate the
        feature set (e.g., version, seed, feature list).

    Returns
    -------
    str
        Hex-encoded SHA-256 hash string (first 16 characters).
    """
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def save_feature_schema(
    columns: list[dict[str, str]],
    metadata: dict[str, Any],
    path: Path | str,
) -> Path:
    """Generate and persist a ``schema.json`` for a feature set.

    Parameters
    ----------
    columns : list of dict
        Column descriptors, each with keys ``name``, ``dtype``,
        ``source``, and ``description``.
    metadata : dict
        Top-level metadata for the schema. Expected keys include
        ``version``, ``modality``, ``config_hash``, ``n_rows_train``,
        ``n_rows_test``, ``n_features``, ``seed``.
    path : Path | str
        Destination path (e.g., ``data/features/tabular/v1/schema.json``).

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = {
        "version": metadata.get("version", "v1"),
        "modality": metadata.get("modality", "unknown"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": metadata.get("config_hash", ""),
        "model_name": metadata.get("model_name"),
        "columns": columns,
        "n_rows_train": metadata.get("n_rows_train", 0),
        "n_rows_test": metadata.get("n_rows_test", 0),
        "n_features": metadata.get("n_features", len(columns)),
        "seed": metadata.get("seed", 42),
        "notes": metadata.get("notes", ""),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    logger.info(
        "Saved feature schema: %s (%d columns, modality=%s)",
        path,
        len(columns),
        schema["modality"],
    )
    return path


def load_feature_schema(path: Path | str) -> dict[str, Any]:
    """Load and validate a ``schema.json``.

    Parameters
    ----------
    path : Path | str
        Path to the ``schema.json`` file.

    Returns
    -------
    dict
        Parsed schema dictionary.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist.
    ValueError
        If the schema is missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    required_keys = {"version", "modality", "columns", "n_rows_train", "n_rows_test"}
    missing = required_keys - set(schema.keys())
    if missing:
        raise ValueError(f"Schema is missing required keys: {missing}")

    logger.info(
        "Loaded feature schema: %s (v=%s, modality=%s, %d cols)",
        path,
        schema["version"],
        schema["modality"],
        len(schema["columns"]),
    )
    return schema


def build_column_descriptors(
    df: pd.DataFrame,
    source: str = "tabular",
    descriptions: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build column descriptor dicts from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to describe.
    source : str
        Modality tag (e.g., ``"tabular"``, ``"text"``, ``"image"``).
    descriptions : dict, optional
        Mapping of column name to human-readable description. Columns
        absent from this mapping receive a generic description.

    Returns
    -------
    list of dict
        One dict per column with keys ``name``, ``dtype``, ``source``,
        ``description``.
    """
    descriptions = descriptions or {}
    cols: list[dict[str, str]] = []
    for col in df.columns:
        cols.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "source": source,
                "description": descriptions.get(col, ""),
            }
        )
    return cols
