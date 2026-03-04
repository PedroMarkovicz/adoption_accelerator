"""
Inference audit log generation.

Produces a structured JSON log capturing all metadata about an
inference run -- model version, data version, prediction summary,
validation gate results, latency metrics, and output file hashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _file_sha256(path: str | Path) -> str:
    """Compute the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_package_versions() -> dict[str, str]:
    """Collect versions of key packages."""
    versions: dict[str, str] = {}
    for pkg in (
        "lightgbm",
        "xgboost",
        "catboost",
        "scikit-learn",
        "pandas",
        "numpy",
        "joblib",
    ):
        try:
            import importlib.metadata

            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "N/A"
    return versions


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python."""
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def generate_audit_log(
    model_bundle: dict[str, Any],
    feature_version: str,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    diagnostics: dict[str, Any],
    latency_profile: dict[str, Any],
    config_snapshot: dict[str, Any],
    validation_results: dict[str, Any],
    submission_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Produce and persist the structured inference audit JSON log.

    Parameters
    ----------
    model_bundle : dict
        The loaded model bundle.
    feature_version : str
        Feature version tag (e.g. ``"v1"``).
    predictions : ndarray
        Final integer class predictions.
    probabilities : ndarray
        Class probability matrix.
    diagnostics : dict
        Output of ``compute_prediction_diagnostics``.
    latency_profile : dict
        Output of ``profile_inference_latency``.
    config_snapshot : dict
        Full inference configuration.
    validation_results : dict
        All gate results.
    submission_path : str or Path
        Path to the submission CSV.
    predictions_path : str or Path
        Path to the predictions Parquet.
    output_path : str or Path
        Destination for the audit log JSON.

    Returns
    -------
    Path
        The written audit log path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission_path = Path(submission_path)
    predictions_path = Path(predictions_path)

    # Model metadata
    bundle_config = model_bundle.get("config", {})
    bundle_metrics = model_bundle.get("metrics", {})
    bundle_thresholds = model_bundle.get("thresholds", {})

    # Confidence
    confidence = probabilities.max(axis=1)

    now = datetime.now(timezone.utc).isoformat()
    run_id = f"inference_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    audit: dict[str, Any] = {
        "run_id": run_id,
        "timestamp": now,
        "model": {
            "version": "tuned_v1",
            "type": bundle_config.get(
                "model_family", bundle_config.get("model_name", "unknown")
            ),
            "bundle_path": config_snapshot.get("model", {}).get("bundle_path", ""),
            "cv_qwk": bundle_metrics.get("mean_qwk_threshold", None),
            "thresholds": bundle_thresholds.get("thresholds", []),
        },
        "data": {
            "feature_version": feature_version,
            "test_rows": int(len(predictions)),
            "n_features": config_snapshot.get("expected", {}).get("n_features", None),
            "source_path": f"data/features/integrated/{feature_version}/test.parquet",
        },
        "predictions": {
            "distribution": diagnostics.get("prediction_distribution", {}),
            "distribution_pct": diagnostics.get("prediction_distribution_pct", {}),
            "mean_confidence": round(float(np.mean(confidence)), 6),
            "median_confidence": round(float(np.median(confidence)), 6),
            "min_confidence": round(float(np.min(confidence)), 6),
            "max_confidence": round(float(np.max(confidence)), 6),
            "low_confidence_count": diagnostics.get("low_confidence_count", 0),
            "low_confidence_pct": diagnostics.get("low_confidence_pct", 0),
        },
        "validation_gates": validation_results,
        "diagnostics": {
            "reproducibility_test": diagnostics.get("reproducibility_test", "N/A"),
            "degradation_test": diagnostics.get("degradation_test", {}),
        },
        "latency": latency_profile,
        "output_files": {
            "submission": {
                "path": str(submission_path),
                "sha256": _file_sha256(submission_path)
                if submission_path.exists()
                else "N/A",
                "row_count": int(len(predictions)),
            },
            "predictions": {
                "path": str(predictions_path),
                "sha256": _file_sha256(predictions_path)
                if predictions_path.exists()
                else "N/A",
                "row_count": int(len(predictions)),
            },
        },
        "config_snapshot": config_snapshot,
        "environment": {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "key_packages": _get_package_versions(),
        },
    }

    # Serialize
    audit = _to_serializable(audit)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)

    logger.info("Inference audit log saved to %s", output_path)
    return output_path
