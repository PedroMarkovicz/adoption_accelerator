"""
Experiment logger for structured, reproducible experiment tracking.

Persists training run metadata, hyperparameters, fold-level metrics,
and aggregated results as JSON artifacts.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def log_experiment(
    run_config: dict[str, Any],
    metrics: dict[str, Any],
    artifacts_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and optionally persist a structured experiment record.

    Parameters
    ----------
    run_config : dict
        Model type, hyperparameters, feature version, seed, etc.
    metrics : dict
        Aggregated and per-fold metrics from :class:`CVResult`.
    artifacts_path : str | Path | None
        Path where the model artifact bundle is stored (if any).
    output_path : str | Path | None
        If provided, the record is appended/written to this JSON file.

    Returns
    -------
    dict
        The experiment record.
    """
    record: dict[str, Any] = {
        "run_id": f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": run_config,
        "metrics": metrics,
    }
    if artifacts_path is not None:
        record["artifacts_path"] = str(artifacts_path)

    if output_path is not None:
        _append_to_log(record, Path(output_path))

    return record


def _append_to_log(record: dict[str, Any], path: Path) -> None:
    """Append an experiment record to a JSON list file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[dict[str, Any]] = []
    if path.exists():
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)

    existing.append(record)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, default=str)

    logger.info("Experiment logged to %s (%d total records)", path, len(existing))


def create_experiment_comparison(
    experiment_logs: list[dict[str, Any]],
    primary_metric: str = "mean_qwk_threshold",
) -> pd.DataFrame:
    """Aggregate multiple experiment logs into a comparison DataFrame.

    Sorted descending by *primary_metric*.
    """
    rows = []
    for exp in experiment_logs:
        row = {
            "model_name": exp["config"].get("model_name", "unknown"),
            "model_type": exp["config"].get("model_type", "unknown"),
        }
        # Flatten aggregated metrics
        for k, v in exp["metrics"].items():
            if isinstance(v, (int, float)):
                row[k] = v
        rows.append(row)

    df = pd.DataFrame(rows)
    if primary_metric in df.columns:
        df = df.sort_values(primary_metric, ascending=False).reset_index(drop=True)
    return df


def save_experiment_logs(
    experiments: list[dict[str, Any]],
    path: str | Path,
) -> Path:
    """Write a list of experiment records to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2, default=str)
    logger.info("Saved %d experiment records to %s", len(experiments), path)
    return path
