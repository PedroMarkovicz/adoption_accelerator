"""
Inference data loader.

Loads versioned integrated feature Parquets for inference.  Unlike the
training data loader, this module never handles target columns and
supports both test-set and new-data scenarios.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from adoption_accelerator import config

logger = logging.getLogger(__name__)

INDEX_COL = "PetID"
TARGET_COL = "AdoptionSpeed"


def _feature_dir(feature_version: str) -> Path:
    """Return path to a versioned integrated feature directory."""
    return config.DATA_FEATURES / "integrated" / feature_version


def load_inference_data(
    feature_version: str = "v1",
    split: str = "test",
) -> tuple[pd.DataFrame, pd.Series | None, pd.Index]:
    """Load integrated feature Parquet for inference.

    Parameters
    ----------
    feature_version : str
        Version tag (e.g. ``"v1"``).
    split : str
        ``"test"`` or ``"train"``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series | None, pd.Index]
        ``(X, y, pet_ids)`` -- ``y`` is ``None`` for test split.
    """
    parquet_path = _feature_dir(feature_version) / f"{split}.parquet"
    logger.info("Loading %s features from %s", split, parquet_path)
    df = pd.read_parquet(parquet_path)

    # Ensure PetID is the index
    if INDEX_COL in df.columns:
        df = df.set_index(INDEX_COL)
    if df.index.name != INDEX_COL:
        raise ValueError(f"Expected index '{INDEX_COL}', got '{df.index.name}'")

    pet_ids = df.index.copy()

    y: pd.Series | None = None
    if TARGET_COL in df.columns:
        if split == "train":
            y = df[TARGET_COL].copy()
        df = df.drop(columns=[TARGET_COL])

    logger.info(
        "Loaded %s split: %d rows x %d features",
        split,
        df.shape[0],
        df.shape[1],
    )
    return df, y, pet_ids


def load_feature_schema(feature_version: str = "v1") -> dict[str, Any]:
    """Load the integrated feature schema JSON."""
    schema_path = _feature_dir(feature_version) / "schema.json"
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)
