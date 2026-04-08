"""
Prediction tool adapter.

Stateless adapter wrapping ``InferencePipeline.predict_single()``.
Accepts a ``PredictionRequest``, builds the feature vector, and
returns a ``PredictionResult``.  No LLM involvement.

Consumed by: ``agents/nodes/inference.py``
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from adoption_accelerator.inference.contracts import PredictionRequest, PredictionResult
from adoption_accelerator.inference.feature_builder import build_feature_vector
from adoption_accelerator.inference.serving import get_inference_pipeline

logger = logging.getLogger(__name__)


def run_prediction(
    request: PredictionRequest,
    bundle_path: str | None = None,
) -> dict[str, Any]:
    """Execute a prediction for a single request.

    Loads the cached InferencePipeline, builds the 940-dim feature
    vector from the request, and runs the model prediction.

    Parameters
    ----------
    request : PredictionRequest
        Input prediction request with tabular, description, and image data.
    bundle_path : str | None
        Optional override for the model bundle path.

    Returns
    -------
    dict
        Keys: ``prediction_result`` (PredictionResult),
        ``feature_vector`` (list[float]),
        ``feature_names`` (list[str]).
    """
    pipeline = get_inference_pipeline(bundle_path)
    feature_schema = pipeline.feature_schema.get("features", [])

    # Build feature vector from the raw request
    feature_vector = build_feature_vector(request, feature_schema)

    # Run prediction
    prediction_result = pipeline.predict_single(feature_vector)

    # Serialize feature vector for state storage
    fv = np.asarray(feature_vector)
    if fv.ndim == 2:
        fv = fv[0]

    logger.info(
        "Prediction tool: class=%d, confidence=%.4f",
        prediction_result.prediction,
        prediction_result.confidence,
    )

    return {
        "prediction_result": prediction_result,
        "feature_vector": [float(v) for v in fv],
        "feature_names": feature_schema,
    }
