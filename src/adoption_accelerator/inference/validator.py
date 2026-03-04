"""
Inference-time validation logic.

Centralises model-bundle integrity checks, feature-schema parity
assertions, and input data quality validation.  These checks run
*before* any prediction to guarantee the pipeline operates on correct
inputs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Model bundle validation ─────────────────────────────────────────


def validate_model_bundle(
    bundle: dict[str, Any],
    expected_n_features: int = 940,
    expected_cv_qwk: float | None = None,
    qwk_tolerance: float = 1e-2,
) -> dict[str, Any]:
    """Assert model bundle integrity.

    Parameters
    ----------
    bundle : dict
        Output of ``load_model_bundle`` -- keys: ``model``, ``config``,
        ``metrics``, ``thresholds``, ``feature_schema``.
    expected_n_features : int
        Expected number of feature columns.
    expected_cv_qwk : float | None
        If provided, assert the bundle's CV QWK matches this value.
    qwk_tolerance : float
        Allowed absolute difference for the QWK comparison.

    Returns
    -------
    dict
        Validation report with per-gate results.
    """
    report: dict[str, Any] = {"gates": {}, "passed": True}

    def _gate(gate_id: str, ok: bool, detail: str, critical: bool = True) -> None:
        status = "PASS" if ok else "FAIL"
        report["gates"][gate_id] = {"status": status, "detail": detail}
        if not ok and critical:
            report["passed"] = False
            logger.error("[%s] FAIL -- %s", gate_id, detail)
        else:
            logger.info("[%s] %s -- %s", gate_id, status, detail)

    # G14-1: All required keys present and non-null
    required_keys = ["model", "config", "metrics", "thresholds", "feature_schema"]
    missing = [k for k in required_keys if bundle.get(k) is None]
    _gate(
        "G14-1",
        len(missing) == 0,
        f"Missing bundle components: {missing}"
        if missing
        else "All components present",
    )

    # Feature schema size
    schema = bundle.get("feature_schema") or {}
    features = schema.get("features", [])
    _gate(
        "G14-1b",
        len(features) == expected_n_features,
        f"Feature schema defines {len(features)} features (expected {expected_n_features})",
    )

    # G14-2: CV QWK matches expected
    if expected_cv_qwk is not None:
        metrics = bundle.get("metrics") or {}
        actual_qwk = metrics.get("mean_qwk_threshold", metrics.get("qwk"))
        if actual_qwk is not None:
            diff = abs(float(actual_qwk) - expected_cv_qwk)
            _gate(
                "G14-2",
                diff < qwk_tolerance,
                f"CV QWK = {actual_qwk:.6f} (expected ~{expected_cv_qwk:.4f}, diff={diff:.6f})",
            )
        else:
            _gate("G14-2", False, "CV QWK key not found in metrics.json")

    # G14-8: Thresholds validation
    thresholds_obj = bundle.get("thresholds") or {}
    thresh_vals = thresholds_obj.get("thresholds", [])
    _gate(
        "G14-8",
        len(thresh_vals) == 4,
        f"Thresholds count: {len(thresh_vals)} (expected 4)",
    )
    if len(thresh_vals) == 4:
        ascending = all(
            thresh_vals[i] < thresh_vals[i + 1] for i in range(len(thresh_vals) - 1)
        )
        _gate(
            "G14-8b",
            ascending,
            f"Thresholds ascending: {ascending} -- {thresh_vals}",
        )

    return report


# ── Feature schema parity ───────────────────────────────────────────


def validate_feature_schema_parity(
    X: pd.DataFrame,
    expected_schema: dict[str, Any],
) -> dict[str, Any]:
    """Compare feature matrix columns against the model bundle schema.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix to validate.
    expected_schema : dict
        ``feature_schema.json`` from the model bundle.

    Returns
    -------
    dict
        Parity report.
    """
    expected_features = expected_schema.get("features", [])
    actual_features = list(X.columns)

    report: dict[str, Any] = {"passed": True, "details": {}}

    # Column count
    count_ok = len(actual_features) == len(expected_features)
    report["details"]["column_count"] = {
        "expected": len(expected_features),
        "actual": len(actual_features),
        "match": count_ok,
    }

    # Column names (set comparison)
    expected_set = set(expected_features)
    actual_set = set(actual_features)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    report["details"]["missing_columns"] = missing
    report["details"]["extra_columns"] = extra

    # Column order
    order_ok = actual_features == expected_features
    report["details"]["order_match"] = order_ok

    report["passed"] = count_ok and len(missing) == 0 and len(extra) == 0 and order_ok

    if not report["passed"]:
        logger.error(
            "Feature schema parity FAILED: count=%s, missing=%d, extra=%d, order=%s",
            count_ok,
            len(missing),
            len(extra),
            order_ok,
        )
    else:
        logger.info(
            "Feature schema parity: PASS (%d columns match)", len(expected_features)
        )

    return report


# ── Data quality validation ─────────────────────────────────────────


def validate_data_quality(X: pd.DataFrame) -> dict[str, Any]:
    """Assert no NaN, no infinite values, all numeric.

    Returns
    -------
    dict
        Quality report with anomaly details.
    """
    report: dict[str, Any] = {"passed": True, "details": {}}

    # NaN check
    nan_total = int(X.isna().sum().sum())
    report["details"]["nan_count"] = nan_total
    if nan_total > 0:
        nan_cols = X.columns[X.isna().any()].tolist()
        report["details"]["nan_columns"] = nan_cols
        report["passed"] = False
        logger.error(
            "Data quality FAIL: %d NaN values in columns %s", nan_total, nan_cols
        )

    # Infinite check
    numeric_df = X.select_dtypes(include=[np.number])
    inf_total = int(np.isinf(numeric_df).sum().sum())
    report["details"]["inf_count"] = inf_total
    if inf_total > 0:
        inf_cols = numeric_df.columns[np.isinf(numeric_df).any()].tolist()
        report["details"]["inf_columns"] = inf_cols
        report["passed"] = False
        logger.error(
            "Data quality FAIL: %d Inf values in columns %s", inf_total, inf_cols
        )

    if report["passed"]:
        logger.info("Data quality: PASS (0 NaN, 0 Inf, %d columns)", X.shape[1])

    return report
