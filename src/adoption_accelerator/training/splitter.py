"""
Cross-validation splitter utilities.

Provides deterministic stratified K-fold splitting with reproducibility
guarantees.  Fold indices can be persisted to JSON for reuse across
notebooks (11 -> 12).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from adoption_accelerator import config

logger = logging.getLogger(__name__)


def create_cv_splitter(
    n_splits: int = 5,
    seed: int | None = None,
) -> StratifiedKFold:
    """Create a deterministic StratifiedKFold splitter.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    seed : int | None
        Random state.  Defaults to ``config.SEED``.

    Returns
    -------
    StratifiedKFold
    """
    if seed is None:
        seed = config.SEED
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def get_fold_indices(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    splitter: StratifiedKFold,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate all (train_idx, val_idx) arrays from the splitter.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        One tuple per fold.
    """
    return list(splitter.split(X, y))


def persist_fold_indices(
    folds: list[tuple[np.ndarray, np.ndarray]],
    path: str | Path | None = None,
) -> Path:
    """Save fold indices to JSON for cross-notebook reproducibility.

    Parameters
    ----------
    folds : list[tuple[np.ndarray, np.ndarray]]
        Output of :func:`get_fold_indices`.
    path : str | Path | None
        Destination file.  Defaults to ``artifacts/cv_folds_v1.json``.

    Returns
    -------
    Path
        Written file path.
    """
    if path is None:
        path = config.ARTIFACTS_DIR / "cv_folds_v1.json"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "n_splits": len(folds),
        "seed": config.SEED,
        "folds": [
            {
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
            }
            for train_idx, val_idx in folds
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    logger.info("Fold indices persisted to %s", path)
    return path


def load_fold_indices(
    path: str | Path | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reload previously persisted fold indices.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        Fold index pairs, identical format to :func:`get_fold_indices`.
    """
    if path is None:
        path = config.ARTIFACTS_DIR / "cv_folds_v1.json"
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    folds = [
        (np.array(fold["train_idx"]), np.array(fold["val_idx"]))
        for fold in payload["folds"]
    ]
    logger.info("Loaded %d folds from %s", len(folds), path)
    return folds


def validate_fold_balance(
    y: pd.Series | np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    tolerance: float = 0.02,
) -> dict:
    """Check that each fold's target distribution is within tolerance of the
    overall distribution.

    Returns
    -------
    dict
        ``{"passed": bool, "details": list[dict]}``
    """
    y_arr = np.asarray(y)
    overall_dist = np.bincount(y_arr, minlength=5) / len(y_arr)

    details = []
    all_ok = True
    for i, (_, val_idx) in enumerate(folds):
        fold_dist = np.bincount(y_arr[val_idx], minlength=5) / len(val_idx)
        max_diff = float(np.max(np.abs(fold_dist - overall_dist)))
        ok = max_diff <= tolerance
        if not ok:
            all_ok = False
        details.append(
            {
                "fold": i,
                "max_proportion_diff": round(max_diff, 4),
                "within_tolerance": ok,
            }
        )

    return {"passed": all_ok, "tolerance": tolerance, "details": details}
