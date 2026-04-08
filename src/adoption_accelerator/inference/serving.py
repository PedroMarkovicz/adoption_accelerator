"""
Singleton factories for the inference layer.

Provides cached access to the InferencePipeline and SHAP explainer
so that the model bundle is loaded once and reused across agent
invocations.  On first load, runs artifact integrity checks and a
warm-up prediction to verify correctness before accepting real
requests.

Consumed by:
  - ``agents/tools/prediction_tool.py``
  - ``agents/tools/explanation_tool.py``
  - ``agents/nodes/inference.py``
"""

from __future__ import annotations

import functools
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from adoption_accelerator import config
from adoption_accelerator.inference.pipeline import InferencePipeline
from adoption_accelerator.interpretability.explainer import (
    compute_shap_values,
    load_explainer,
)

logger = logging.getLogger(__name__)

# Default serving config path
_DEFAULT_SERVING_CONFIG = config.PROJECT_ROOT / "configs" / "inference" / "serving.yaml"


def _load_serving_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load and return the serving configuration from YAML."""
    path = config_path or _DEFAULT_SERVING_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_bundle_path(serving_config: dict[str, Any]) -> Path:
    """Resolve the model bundle path from serving config."""
    raw = serving_config.get("model", {}).get("bundle_path", "artifacts/models/tuned_v1")
    bundle_path = Path(raw)
    if not bundle_path.is_absolute():
        bundle_path = config.PROJECT_ROOT / bundle_path
    return bundle_path


# ---------------------------------------------------------------------------
# Singleton: InferencePipeline
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_inference_pipeline(
    bundle_path: str | None = None,
) -> InferencePipeline:
    """Return a cached InferencePipeline instance.

    On first call the model bundle is loaded from disk, validated
    (via ``InferencePipeline.__init__``), and warmed up with a dummy
    prediction.  Subsequent calls return the same instance.

    Parameters
    ----------
    bundle_path : str | None
        Explicit path to the model bundle directory.  When ``None``
        the path is read from ``configs/inference/serving.yaml``.

    Returns
    -------
    InferencePipeline
        Cached pipeline instance.

    Raises
    ------
    RuntimeError
        If the model bundle fails validation or warm-up.
    """
    if bundle_path is None:
        serving_cfg = _load_serving_config()
        resolved = _resolve_bundle_path(serving_cfg)
    else:
        resolved = Path(bundle_path)

    logger.info("Loading InferencePipeline from %s (first load) ...", resolved)
    t0 = time.perf_counter()

    # InferencePipeline.__init__ already validates the bundle
    pipeline = InferencePipeline(resolved)

    load_ms = (time.perf_counter() - t0) * 1000
    logger.info("InferencePipeline loaded in %.1f ms.", load_ms)

    # Warm-up: run one dummy prediction to trigger JIT / lazy init paths
    _warm_up_pipeline(pipeline)

    return pipeline


def _warm_up_pipeline(pipeline: InferencePipeline) -> None:
    """Run a single dummy prediction to warm JIT-compiled paths."""
    n_features = len(pipeline.feature_schema.get("features", []))
    if n_features == 0:
        logger.warning("Feature schema is empty; skipping warm-up prediction.")
        return

    dummy = np.zeros((1, n_features), dtype=np.float64)
    t0 = time.perf_counter()
    result = pipeline.predict_single(dummy)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        "Warm-up prediction: class=%d, confidence=%.4f, latency=%.1f ms.",
        result.prediction,
        result.confidence,
        elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Singleton: SHAP Explainer
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def get_explainer(
    explainer_path: str | None = None,
) -> Any:
    """Return a cached SHAP explainer instance.

    On first call the explainer is loaded from disk and warmed up
    with a dummy SHAP computation.

    Parameters
    ----------
    explainer_path : str | None
        Explicit path to ``explainer.joblib``.  When ``None`` the
        path is derived from the serving config bundle path.

    Returns
    -------
    shap.TreeExplainer
        Cached SHAP explainer.
    """
    if explainer_path is None:
        serving_cfg = _load_serving_config()
        bundle_path = _resolve_bundle_path(serving_cfg)
        resolved = bundle_path / "explainer.joblib"
    else:
        resolved = Path(explainer_path)

    logger.info("Loading SHAP explainer from %s (first load) ...", resolved)
    t0 = time.perf_counter()
    explainer = load_explainer(resolved)
    load_ms = (time.perf_counter() - t0) * 1000
    logger.info("SHAP explainer loaded in %.1f ms.", load_ms)

    # Warm-up: compute SHAP for one dummy sample
    _warm_up_explainer(explainer)

    return explainer


def _warm_up_explainer(explainer: Any) -> None:
    """Run a single dummy SHAP computation to warm internal caches."""
    # Infer feature count from the explainer's expected input
    try:
        if hasattr(explainer, "expected_value"):
            ev = np.asarray(explainer.expected_value)
            # For tree explainers the data attribute may store background
            n_features = 940  # fallback to known schema size
            if hasattr(explainer, "data") and explainer.data is not None:
                data = np.asarray(explainer.data)
                if data.ndim == 2:
                    n_features = data.shape[1]
        else:
            n_features = 940

        dummy = np.zeros((1, n_features), dtype=np.float64)
        t0 = time.perf_counter()
        compute_shap_values(explainer, dummy)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Warm-up SHAP computation: latency=%.1f ms.", elapsed_ms)
    except Exception as exc:
        logger.warning("SHAP warm-up failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


def clear_caches() -> None:
    """Clear all singleton caches.  Useful for testing."""
    get_inference_pipeline.cache_clear()
    get_explainer.cache_clear()
    logger.info("Inference serving caches cleared.")
