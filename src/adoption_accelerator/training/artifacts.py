"""
Model artifact bundle persistence and loading.

A bundle is the atomic deployable unit containing: trained model,
thresholds, metrics, config, and feature schema.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to native Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_model_bundle(
    model: Any,
    run_config: dict[str, Any],
    metrics: dict[str, Any],
    thresholds: dict[str, Any] | None,
    feature_columns: list[str],
    path: str | Path,
) -> Path:
    """Persist a model artifact bundle to a versioned directory.

    Parameters
    ----------
    model : fitted estimator
        Trained classifier.
    run_config : dict
        Full training configuration snapshot.
    metrics : dict
        Aggregated evaluation metrics.
    thresholds : dict | None
        Threshold optimization results (boundaries + QWK).
    feature_columns : list[str]
        Ordered list of expected feature column names.
    path : str | Path
        Directory to write the bundle into.

    Returns
    -------
    Path
        The bundle directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Model object
    joblib.dump(model, path / "model.joblib")

    # Metrics
    with open(path / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_to_native(metrics), f, indent=2, default=str)

    # Config
    with open(path / "config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(_to_native(run_config), f, default_flow_style=False, sort_keys=False)

    # Thresholds
    if thresholds is not None:
        with open(path / "thresholds.json", "w", encoding="utf-8") as f:
            json.dump(_to_native(thresholds), f, indent=2)

    # Feature schema
    with open(path / "feature_schema.json", "w", encoding="utf-8") as f:
        json.dump(
            {"features": feature_columns, "n_features": len(feature_columns)},
            f,
            indent=2,
        )

    logger.info("Model bundle saved to %s", path)
    return path


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a model artifact bundle.

    Returns
    -------
    dict
        Keys: ``model``, ``config``, ``metrics``, ``thresholds``,
        ``feature_schema``.
    """
    path = Path(path)
    bundle: dict[str, Any] = {}

    bundle["model"] = joblib.load(path / "model.joblib")

    with open(path / "metrics.json", encoding="utf-8") as f:
        bundle["metrics"] = json.load(f)

    with open(path / "config.yaml", encoding="utf-8") as f:
        bundle["config"] = yaml.safe_load(f)

    thresholds_path = path / "thresholds.json"
    if thresholds_path.exists():
        with open(thresholds_path, encoding="utf-8") as f:
            bundle["thresholds"] = json.load(f)
    else:
        bundle["thresholds"] = None

    with open(path / "feature_schema.json", encoding="utf-8") as f:
        bundle["feature_schema"] = json.load(f)

    logger.info("Model bundle loaded from %s", path)
    return bundle
