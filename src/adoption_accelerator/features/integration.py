"""
Feature integration utilities for multimodal feature fusion.

Provides functions for loading, validating, merging, and persisting
modality-specific feature sets into a unified feature matrix.
Consumed by Notebook 10 and the deterministic pipeline.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adoption_accelerator import config
from adoption_accelerator.features.registry import (
    build_column_descriptors,
    compute_config_hash,
    load_feature_schema,
    save_feature_schema,
)

logger = logging.getLogger("adoption_accelerator")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_modality_features(
    modality: str,
    version: str,
    split: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a modality's feature Parquet and validate against its schema.json.

    Parameters
    ----------
    modality : str
        Name of the modality directory (e.g. ``"tabular"``, ``"text"``, ``"images"``).
    version : str
        Version directory name (e.g. ``"v1"``).
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The loaded DataFrame and its parsed schema dictionary.
    """
    base = config.DATA_FEATURES / modality / version
    parquet_path = base / f"{split}.parquet"
    schema_path = base / "schema.json"

    if not parquet_path.exists():
        raise FileNotFoundError(f"Feature file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    schema = load_feature_schema(schema_path)

    # Validate column count against schema
    schema_cols = [c["name"] for c in schema["columns"]]
    if set(df.columns) != set(schema_cols):
        extra = set(df.columns) - set(schema_cols)
        missing = set(schema_cols) - set(df.columns)
        logger.warning(
            "Column mismatch for %s/%s/%s: extra=%s, missing=%s",
            modality,
            version,
            split,
            extra,
            missing,
        )

    logger.info(
        "Loaded %s/%s/%s: %d rows x %d cols",
        modality,
        version,
        split,
        len(df),
        len(df.columns),
    )
    return df, schema


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_petid_alignment(
    dfs: dict[str, pd.DataFrame],
    split: str,
) -> dict[str, Any]:
    """Verify that all provided DataFrames share an identical PetID set.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Mapping of modality name to DataFrame (PetID as index).
    split : str
        ``"train"`` or ``"test"`` (for logging).

    Returns
    -------
    dict
        Alignment report with per-modality PetID counts and any mismatches.
    """
    modalities = list(dfs.keys())
    id_sets = {m: set(dfs[m].index) for m in modalities}
    reference = id_sets[modalities[0]]

    report: dict[str, Any] = {
        "split": split,
        "aligned": True,
        "per_modality_count": {m: len(ids) for m, ids in id_sets.items()},
        "mismatches": [],
    }

    for m in modalities[1:]:
        only_ref = reference - id_sets[m]
        only_m = id_sets[m] - reference
        if only_ref or only_m:
            report["aligned"] = False
            report["mismatches"].append(
                {
                    "pair": (modalities[0], m),
                    f"only_in_{modalities[0]}": len(only_ref),
                    f"only_in_{m}": len(only_m),
                }
            )

    return report


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------


def merge_modality_dataframes(
    dfs: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Horizontally concatenate multiple DataFrames on their index (PetID).

    Uses ``pd.concat`` on ``axis=1``. Detects NaN columns introduced by
    misaligned indices and logs them.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Mapping of modality name to DataFrame (PetID as index).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    frames = list(dfs.values())
    merged = pd.concat(frames, axis=1, join="outer")

    # Detect NaN columns introduced by outer join
    nan_counts = merged.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0]
    if len(cols_with_nans) > 0:
        logger.warning(
            "Columns with NaN after merge (%d): %s",
            len(cols_with_nans),
            cols_with_nans.to_dict(),
        )

    return merged


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def build_provenance_map(
    schemas: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Construct a column-to-modality mapping from individual modality schemas.

    Parameters
    ----------
    schemas : dict[str, dict]
        Mapping of modality name to its parsed schema dict.

    Returns
    -------
    dict[str, str]
        Mapping of column name to source modality label.
    """
    provenance: dict[str, str] = {}
    for modality_name, schema in schemas.items():
        source_label = schema.get("modality", modality_name)
        for col_info in schema["columns"]:
            provenance[col_info["name"]] = source_label
    return provenance


# ---------------------------------------------------------------------------
# Scaling (conditional)
# ---------------------------------------------------------------------------


def apply_cross_modality_scaling(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    method: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Fit a scaler on train, apply to both splits.

    Parameters
    ----------
    train_df, test_df : pd.DataFrame
        Feature matrices.
    columns : list[str]
        Columns to scale.
    method : str
        ``"standard"`` or ``"robust"``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, scaler]
        Scaled DataFrames and the fitted scaler object.
    """
    from sklearn.preprocessing import RobustScaler, StandardScaler

    scaler_cls = StandardScaler if method == "standard" else RobustScaler
    scaler = scaler_cls()

    train_out = train_df.copy()
    test_out = test_df.copy()

    train_out[columns] = scaler.fit_transform(train_df[columns])
    test_out[columns] = scaler.transform(test_df[columns])

    logger.info("Applied %s scaling to %d columns.", method, len(columns))
    return train_out, test_out, scaler


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


def audit_integrated_matrix(
    df: pd.DataFrame,
    expected_rows: int,
    expected_cols: int,
    split: str = "train",
) -> dict[str, Any]:
    """Run comprehensive data quality checks on the integrated feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        The integrated feature matrix to audit.
    expected_rows : int
        Expected number of rows.
    expected_cols : int
        Expected number of feature columns (excluding target).
    split : str
        ``"train"`` or ``"test"`` (for reporting).

    Returns
    -------
    dict
        Audit report with per-check pass/fail results.
    """
    checks: dict[str, Any] = {}

    # 1. Row count
    checks["row_count"] = {
        "expected": expected_rows,
        "actual": len(df),
        "pass": len(df) == expected_rows,
    }

    # 2. Column count (feature columns only, exclude AdoptionSpeed if present)
    feature_cols = [c for c in df.columns if c != "AdoptionSpeed"]
    checks["feature_count"] = {
        "expected": expected_cols,
        "actual": len(feature_cols),
        "pass": len(feature_cols) == expected_cols,
    }

    # 3. NaN check
    nan_total = int(df[feature_cols].isna().sum().sum())
    checks["no_nans"] = {
        "nan_count": nan_total,
        "pass": nan_total == 0,
    }

    # 4. Infinite values
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns
    inf_total = int(np.isinf(df[numeric_cols]).sum().sum())
    checks["no_infs"] = {
        "inf_count": inf_total,
        "pass": inf_total == 0,
    }

    # 5. Duplicate PetIDs
    dup_count = int(df.index.duplicated().sum())
    checks["no_duplicate_petids"] = {
        "duplicate_count": dup_count,
        "pass": dup_count == 0,
    }

    # 6. Memory footprint (MB)
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    checks["memory_mb"] = {
        "value": round(mem_mb, 2),
        "under_2gb": mem_mb < 2048,
    }

    all_pass = all(v.get("pass", True) for v in checks.values() if isinstance(v, dict))
    checks["all_pass"] = all_pass

    return checks


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def integrate_features(
    feature_versions: dict[str, dict[str, str]],
    target_version: str = "v1",
    apply_scaling: bool = False,
    scaling_method: str = "standard",
) -> dict[str, Any]:
    """Orchestrate loading, validation, merging, and persistence of all
    modality features into a unified integrated feature matrix.

    Parameters
    ----------
    feature_versions : dict
        Mapping of modality name to ``{"dir": "<directory_name>", "version": "<vN>"}``.
        Example: ``{"tabular": {"dir": "tabular", "version": "v1"}, ...}``
    target_version : str
        Version string for the integrated output directory.
    apply_scaling : bool
        Whether to apply cross-modality normalization.
    scaling_method : str
        ``"standard"`` or ``"robust"``.

    Returns
    -------
    dict
        Integration report with metrics and audit results.
    """
    output_dir = config.DATA_FEATURES / "integrated" / target_version
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "target_version": target_version,
        "source_versions": {},
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # -- 1. Load all modality features -----------------------------------
    train_dfs: dict[str, pd.DataFrame] = {}
    test_dfs: dict[str, pd.DataFrame] = {}
    schemas: dict[str, dict[str, Any]] = {}

    for modality, info in feature_versions.items():
        dir_name = info["dir"]
        version = info["version"]

        train_df, schema = load_modality_features(dir_name, version, "train")
        test_df, _ = load_modality_features(dir_name, version, "test")

        train_dfs[modality] = train_df
        test_dfs[modality] = test_df
        schemas[modality] = schema

        report["source_versions"][modality] = {
            "dir": dir_name,
            "version": version,
            "config_hash": schema.get("config_hash", ""),
            "n_features": schema.get("n_features", len(train_df.columns)),
            "model_name": schema.get("model_name"),
        }

    # -- 2. Validate PetID alignment -----------------------------------
    train_alignment = validate_petid_alignment(train_dfs, "train")
    test_alignment = validate_petid_alignment(test_dfs, "test")
    report["alignment"] = {"train": train_alignment, "test": test_alignment}

    if not train_alignment["aligned"] or not test_alignment["aligned"]:
        logger.error("PetID alignment failure detected!")

    # -- 3. Merge features -----------------------------------------------
    train_merged = merge_modality_dataframes(train_dfs)
    test_merged = merge_modality_dataframes(test_dfs)

    # -- 4. Optional scaling -----------------------------------------------
    scaler = None
    if apply_scaling:
        numeric_cols = train_merged.select_dtypes(include=[np.number]).columns.tolist()
        train_merged, test_merged, scaler = apply_cross_modality_scaling(
            train_merged,
            test_merged,
            numeric_cols,
            scaling_method,
        )

    # -- 5. Build provenance map -----------------------------------------
    provenance = build_provenance_map(schemas)
    report["provenance_map"] = provenance

    # -- 6. Persist --------------------------------------------------------
    train_merged.to_parquet(output_dir / "train.parquet")
    test_merged.to_parquet(output_dir / "test.parquet")

    total_features = len(train_merged.columns)
    report["n_features_total"] = total_features
    report["n_rows_train"] = len(train_merged)
    report["n_rows_test"] = len(test_merged)

    # Schema
    all_col_descriptors = []
    for col in train_merged.columns:
        source = provenance.get(col, "unknown")
        all_col_descriptors.append(
            {
                "name": col,
                "dtype": str(train_merged[col].dtype),
                "source": source,
                "description": "",
            }
        )

    integration_config = {
        "source_versions": report["source_versions"],
        "apply_scaling": apply_scaling,
        "scaling_method": scaling_method if apply_scaling else None,
    }

    schema_metadata = {
        "version": target_version,
        "modality": "integrated",
        "config_hash": compute_config_hash(integration_config),
        "n_rows_train": len(train_merged),
        "n_rows_test": len(test_merged),
        "n_features": total_features,
        "seed": config.SEED,
        "notes": (
            f"Integrated multimodal feature matrix from "
            f"{', '.join(feature_versions.keys())}."
        ),
    }

    save_feature_schema(
        all_col_descriptors, schema_metadata, output_dir / "schema.json"
    )

    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    return report
