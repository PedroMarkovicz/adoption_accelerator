"""
Feature importance computation from SHAP values.

Computes global importance, per-modality aggregation, per-class
importance, and builds the structured ``global_importance.json``
artifact consumed by the agent system and frontend.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute mean |SHAP| per feature across all samples.

    Parameters
    ----------
    shap_values : ndarray
        SHAP values.  Accepted shapes:

        * ``(n_samples, n_features)`` -- single-output or pre-aggregated.
        * ``(n_samples, n_features, n_classes)`` -- multiclass.
    feature_names : list[str]
        Feature names matching the second axis.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``mean_abs_shap``.
        Sorted descending by importance.
    """
    vals = np.asarray(shap_values)

    if vals.ndim == 3:
        # Average absolute SHAP across classes then samples
        mean_abs = np.mean(np.abs(vals), axis=(0, 2))
    elif vals.ndim == 2:
        mean_abs = np.mean(np.abs(vals), axis=0)
    else:
        raise ValueError(f"Unexpected SHAP values shape: {vals.shape}")

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    )
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "rank"
    return df


def compute_modality_importance(
    global_importance: pd.DataFrame,
    provenance_map: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Aggregate feature importance by modality.

    Parameters
    ----------
    global_importance : pd.DataFrame
        Output of :func:`compute_global_importance` with columns
        ``feature`` and ``mean_abs_shap``.
    provenance_map : dict[str, str]
        Maps feature name -> modality (e.g. ``"tabular"``, ``"text"``,
        ``"image"``).

    Returns
    -------
    dict[str, dict]
        Per-modality summary::

            {
              "tabular": {"total_importance": 0.45, "n_features": 45},
              "text":    {"total_importance": 0.35, "n_features": 784},
              ...
            }

        ``total_importance`` is the fraction of total mean |SHAP|
        attributed to that modality (sums to 1.0).
    """
    df = global_importance.copy()
    df["modality"] = df["feature"].map(provenance_map).fillna("unknown")

    total_shap = df["mean_abs_shap"].sum()
    if total_shap == 0:
        total_shap = 1.0  # avoid division by zero

    modality_agg = (
        df.groupby("modality")
        .agg(
            total_shap=("mean_abs_shap", "sum"),
            n_features=("mean_abs_shap", "count"),
        )
        .sort_values("total_shap", ascending=False)
    )
    modality_agg["total_importance"] = modality_agg["total_shap"] / total_shap

    result = {}
    for modality, row in modality_agg.iterrows():
        result[str(modality)] = {
            "total_importance": round(float(row["total_importance"]), 6),
            "n_features": int(row["n_features"]),
        }
    return result


def compute_per_class_importance(
    shap_values: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    provenance_map: dict[str, str] | None = None,
    top_k: int = 10,
) -> dict[str, list[dict[str, Any]]]:
    """Compute mean SHAP per feature for each target class.

    Parameters
    ----------
    shap_values : ndarray
        SHAP values of shape ``(n_samples, n_features)`` or
        ``(n_samples, n_features, n_classes)``.
    y : array-like
        True labels for each sample.
    feature_names : list[str]
        Feature names.
    provenance_map : dict | None
        Optional modality tags for each feature.
    top_k : int
        Number of top features to return per class.

    Returns
    -------
    dict[str, list[dict]]
        Maps class label (str) -> list of top-K feature dicts with
        keys ``feature``, ``mean_shap``, and optionally ``modality``.
    """
    vals = np.asarray(shap_values)
    y_arr = np.asarray(y)
    n_classes = 5

    result: dict[str, list[dict[str, Any]]] = {}

    for cls in range(n_classes):
        mask = y_arr == cls

        if vals.ndim == 3:
            # Use the SHAP values for this specific class output
            cls_shap = vals[mask, :, cls]
        else:
            cls_shap = vals[mask, :]

        if cls_shap.shape[0] == 0:
            result[str(cls)] = []
            continue

        mean_shap = cls_shap.mean(axis=0)
        # Rank by absolute contribution
        top_indices = np.argsort(np.abs(mean_shap))[::-1][:top_k]

        top_features = []
        for idx in top_indices:
            entry: dict[str, Any] = {
                "feature": feature_names[idx],
                "mean_shap": round(float(mean_shap[idx]), 6),
            }
            if provenance_map:
                entry["modality"] = provenance_map.get(feature_names[idx], "unknown")
            top_features.append(entry)

        result[str(cls)] = top_features

    return result


def build_global_importance_artifact(
    global_importance: pd.DataFrame,
    modality_importance: dict[str, dict[str, Any]],
    per_class_importance: dict[str, list[dict[str, Any]]],
    provenance_map: dict[str, str] | None = None,
    model_tag: str = "tuned_v1",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the structured ``global_importance.json`` artifact.

    Parameters
    ----------
    global_importance : pd.DataFrame
        Per-feature importance (from :func:`compute_global_importance`).
    modality_importance : dict
        Per-modality importance.
    per_class_importance : dict
        Per-class top-K features.
    provenance_map : dict | None
        Feature -> modality map.
    model_tag : str
        Model version identifier.
    metadata : dict | None
        Extra metadata (model version, n_samples, etc.).

    Returns
    -------
    dict
        The complete artifact structure ready for JSON serialization.
    """
    feature_list = []
    for rank, (_, row) in enumerate(global_importance.iterrows(), start=1):
        entry: dict[str, Any] = {
            "rank": rank,
            "feature": row["feature"],
            "mean_abs_shap": round(float(row["mean_abs_shap"]), 6),
        }
        if provenance_map:
            entry["modality"] = provenance_map.get(row["feature"], "unknown")
        feature_list.append(entry)

    artifact: dict[str, Any] = {
        "model_version": model_tag,
        "n_samples_evaluated": (metadata or {}).get("n_samples_evaluated", 0),
        "global_top_k": feature_list,
        "modality_importance": modality_importance,
        "per_class_top_k": per_class_importance,
    }

    if metadata:
        for k, v in metadata.items():
            if k not in artifact:
                artifact[k] = v

    return artifact


def save_global_importance(
    artifact: dict[str, Any],
    path: str | Path,
) -> Path:
    """Persist the global importance artifact as JSON.

    Parameters
    ----------
    artifact : dict
        Output of :func:`build_global_importance_artifact`.
    path : str | Path
        Destination file path.

    Returns
    -------
    Path
        Resolved output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, default=_convert)

    logger.info("Global importance artifact saved to %s.", path)
    return path
