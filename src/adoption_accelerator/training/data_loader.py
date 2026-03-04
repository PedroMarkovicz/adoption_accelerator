"""
Data loader for the modeling phase.

Loads the versioned integrated feature matrix and validates it against
the schema contract. This module is the *only* ingestion point for
modeling notebooks -- they never touch raw or modality-specific data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adoption_accelerator import config

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────
TARGET_COL = "AdoptionSpeed"
INDEX_COL = "PetID"
VALID_CLASSES = {0, 1, 2, 3, 4}


def _feature_dir(feature_version: str) -> Path:
    """Return the path to a versioned integrated feature directory."""
    return config.DATA_FEATURES / "integrated" / feature_version


def load_feature_schema(feature_version: str = "v1") -> dict[str, Any]:
    """Load and return the integrated feature schema JSON."""
    schema_path = _feature_dir(feature_version) / "schema.json"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def load_modeling_data(
    feature_version: str = "v1",
    split: str = "train",
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load integrated feature Parquet for a given version and split.

    Parameters
    ----------
    feature_version : str
        Version tag (e.g. ``"v1"``).
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series | None]
        ``(X, y)`` for train split; ``(X, None)`` for test split.
        ``X`` uses ``PetID`` as the index, ``AdoptionSpeed`` is excluded.
    """
    parquet_path = _feature_dir(feature_version) / f"{split}.parquet"
    logger.info("Loading %s features from %s", split, parquet_path)
    df = pd.read_parquet(parquet_path)

    # Ensure PetID is the index
    if INDEX_COL in df.columns:
        df = df.set_index(INDEX_COL)
    if df.index.name != INDEX_COL:
        raise ValueError(f"Expected index '{INDEX_COL}', got '{df.index.name}'")

    y: pd.Series | None = None
    if split == "train":
        if TARGET_COL not in df.columns:
            raise ValueError(f"Target column '{TARGET_COL}' not found in train data")
        y = df[TARGET_COL].copy()
        df = df.drop(columns=[TARGET_COL])
    else:
        # Test set must not contain the target
        if TARGET_COL in df.columns:
            df = df.drop(columns=[TARGET_COL])

    logger.info(
        "Loaded %s split: %d rows x %d features",
        split,
        df.shape[0],
        df.shape[1],
    )
    return df, y


def validate_modeling_inputs(
    X: pd.DataFrame,
    y: pd.Series | None,
    schema_path: str | Path | None = None,
    expected_rows: int | None = None,
    expected_features: int | None = None,
) -> dict[str, Any]:
    """Validate feature matrix and target vector for modeling readiness.

    Returns a dict of gate results compatible with the validation framework.
    """
    report: dict[str, Any] = {"gates": {}, "passed": True}

    def _gate(gate_id: str, condition: bool, message: str, critical: bool = True):
        status = "PASS" if condition else "FAIL"
        report["gates"][gate_id] = {
            "status": status,
            "message": message,
            "critical": critical,
        }
        if not condition and critical:
            report["passed"] = False

    # G11-1: Shape
    if expected_rows is not None:
        _gate(
            "G11-1a",
            X.shape[0] == expected_rows,
            f"Row count: {X.shape[0]} (expected {expected_rows})",
        )
    if expected_features is not None:
        _gate(
            "G11-1b",
            X.shape[1] == expected_features,
            f"Feature count: {X.shape[1]} (expected {expected_features})",
        )

    # G11-3: Target not in features
    _gate(
        "G11-3",
        TARGET_COL not in X.columns,
        f"'{TARGET_COL}' absent from feature matrix",
    )

    # G11-4: No NaN / inf
    nan_count = int(X.isna().sum().sum())
    inf_count = int(np.isinf(X.select_dtypes(include=[np.number])).sum().sum())
    _gate("G11-4a", nan_count == 0, f"NaN count: {nan_count}")
    _gate("G11-4b", inf_count == 0, f"Inf count: {inf_count}")

    # PetID uniqueness
    _gate(
        "PetID-unique",
        not X.index.duplicated().any(),
        f"Duplicate PetIDs: {int(X.index.duplicated().sum())}",
    )

    if y is not None:
        # G11-2: Target domain
        unique_classes = set(y.unique())
        _gate(
            "G11-2",
            unique_classes.issubset(VALID_CLASSES)
            and len(unique_classes) == len(VALID_CLASSES),
            f"Target classes: {sorted(unique_classes)}",
        )

        # Shape alignment
        _gate(
            "shape-align",
            X.shape[0] == len(y),
            f"X rows ({X.shape[0]}) == y length ({len(y)})",
        )

    logger.info("Validation report: %s", "PASS" if report["passed"] else "FAIL")
    return report
