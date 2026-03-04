"""
Inference pipeline orchestrator.

Single callable entry point for the frontend service layer and the
agent system.  Wires together: data loading, validation, prediction,
threshold application, and (optionally) explanation.

This module is consumed by:
  - ``app/services/ml_service.py`` (frontend)
  - ``agents/tools/prediction_tool.py`` (agent system)
  - ``pipelines/run_inference.py`` (CLI batch inference)

Notebook 14 tests this pipeline end-to-end but uses the individual
modules directly for finer diagnostic control.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from adoption_accelerator.inference.contracts import PredictionResult
from adoption_accelerator.inference.predictor import (
    compute_expected_values,
    predict_probabilities,
)
from adoption_accelerator.inference.validator import (
    validate_data_quality,
    validate_feature_schema_parity,
    validate_model_bundle,
)
from adoption_accelerator.training.artifacts import load_model_bundle
from adoption_accelerator.training.evaluation import apply_thresholds

logger = logging.getLogger(__name__)

CLASS_LABELS = {
    0: "Same-day adoption",
    1: "Adopted within 1 week",
    2: "Adopted within 1 month",
    3: "Adopted within 1-3 months",
    4: "Not adopted (100+ days)",
}


class InferencePipeline:
    """End-to-end inference pipeline wrapping all inference modules.

    Parameters
    ----------
    bundle_path : str or Path
        Path to the model artifact bundle directory.
    """

    def __init__(self, bundle_path: str | Path) -> None:
        self.bundle_path = Path(bundle_path)
        self.bundle = load_model_bundle(self.bundle_path)

        # Validate bundle
        report = validate_model_bundle(self.bundle)
        if not report["passed"]:
            raise RuntimeError(f"Model bundle validation failed: {report}")

        self.model = self.bundle["model"]
        self.thresholds = self.bundle["thresholds"]["thresholds"]
        self.feature_schema = self.bundle["feature_schema"]
        self.n_classes = 5

        logger.info("InferencePipeline initialized from %s", bundle_path)

    def predict_batch(
        self,
        X,
    ) -> dict[str, Any]:
        """Run batch prediction on a feature matrix.

        Parameters
        ----------
        X : pd.DataFrame or ndarray
            Feature matrix (n_samples, n_features).

        Returns
        -------
        dict
            Keys: ``predictions``, ``probabilities``, ``expected_values``,
            ``confidence``.
        """
        proba = predict_probabilities(self.model, X, n_classes=self.n_classes)
        ev = compute_expected_values(proba)
        preds = apply_thresholds(proba, self.thresholds, n_classes=self.n_classes)
        confidence = proba.max(axis=1)

        return {
            "predictions": preds,
            "probabilities": proba,
            "expected_values": ev,
            "confidence": confidence,
        }

    def predict_single(self, features: np.ndarray) -> PredictionResult:
        """Predict for a single sample.

        Parameters
        ----------
        features : ndarray of shape (1, n_features) or (n_features,)
            Feature vector.

        Returns
        -------
        PredictionResult
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        result = self.predict_batch(features)
        pred_class = int(result["predictions"][0])

        return PredictionResult(
            prediction=pred_class,
            prediction_label=CLASS_LABELS.get(pred_class, str(pred_class)),
            probabilities={
                i: float(result["probabilities"][0, i]) for i in range(self.n_classes)
            },
            confidence=float(result["confidence"][0]),
            metadata={
                "model_version": "tuned_v1",
                "bundle_path": str(self.bundle_path),
            },
        )
