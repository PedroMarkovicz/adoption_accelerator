"""
Counterfactual analysis tool adapter.

For each actionable feature, simulates alternative values and re-runs
the prediction to identify changes that could improve the adoption
speed class.

Consumed by: ``agents/nodes/recommender.py`` (future, LLM phase)
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from adoption_accelerator.inference.contracts import PredictionRequest, PredictionResult
from adoption_accelerator.inference.serving import get_inference_pipeline

logger = logging.getLogger(__name__)


# Actionable features and their candidate values for counterfactual analysis.
# Only features that a pet owner can realistically change are included.
ACTIONABLE_FEATURES: dict[str, dict[str, Any]] = {
    "PhotoAmt": {
        "field_path": ("tabular", "photo_amt"),
        "candidates": [1, 3, 5, 7, 10],
        "description": "Number of photos in the listing",
    },
    "VideoAmt": {
        "field_path": ("tabular", "video_amt"),
        "candidates": [0, 1, 2],
        "description": "Number of videos in the listing",
    },
    "Vaccinated": {
        "field_path": ("tabular", "vaccinated"),
        "candidates": [1],  # 1=Yes
        "description": "Pet vaccination status",
    },
    "Dewormed": {
        "field_path": ("tabular", "dewormed"),
        "candidates": [1],  # 1=Yes
        "description": "Pet deworming status",
    },
    "Sterilized": {
        "field_path": ("tabular", "sterilized"),
        "candidates": [1],  # 1=Yes
        "description": "Pet sterilization status",
    },
    "Fee": {
        "field_path": ("tabular", "fee"),
        "candidates": [0, 10, 25, 50],
        "description": "Adoption fee amount",
    },
    "Name": {
        "field_path": ("tabular", "name"),
        "candidates": ["Buddy"],  # any non-empty name
        "description": "Whether the pet has a name",
    },
    "Quantity": {
        "field_path": ("tabular", "quantity"),
        "candidates": [1],
        "description": "Number of pets in the listing",
    },
}


def _mutate_request(
    request: PredictionRequest,
    feature_name: str,
    new_value: Any,
) -> PredictionRequest:
    """Create a mutated copy of the request with one feature changed."""
    data = request.model_dump()

    if feature_name == "PhotoAmt":
        # PhotoAmt is derived from images list length; simulate by adjusting images
        data["images"] = [f"synthetic_{i}.jpg" for i in range(int(new_value))]
    elif feature_name == "Name":
        data["tabular"]["name"] = str(new_value) if new_value else None
    elif feature_name in ("Vaccinated", "Dewormed", "Sterilized", "Fee", "Quantity", "VideoAmt"):
        field = feature_name.lower()
        if field == "videoamt":
            field = "video_amt"
        data["tabular"][field] = new_value
    else:
        return request

    return PredictionRequest(**data)


def run_counterfactual(
    request: PredictionRequest,
    current_prediction: int,
    feature_vector: np.ndarray | list[float],
    feature_names: list[str],
    bundle_path: str | None = None,
) -> list[dict[str, Any]]:
    """Run counterfactual analysis for actionable features.

    For each actionable feature, tries candidate values and checks
    if the prediction improves (lower class = faster adoption).

    Parameters
    ----------
    request : PredictionRequest
        Original prediction request.
    current_prediction : int
        Current predicted class (0-4).
    feature_vector : ndarray or list[float]
        Current 940-dim feature vector (unused, kept for API consistency).
    feature_names : list[str]
        Ordered feature names from the model schema.
    bundle_path : str | None
        Optional override for the model bundle path.

    Returns
    -------
    list[dict]
        List of counterfactual results, each with keys:
        ``feature``, ``current_value``, ``new_value``,
        ``new_prediction``, ``improvement``, ``description``.
    """
    from adoption_accelerator.inference.feature_builder import build_feature_vector

    pipeline = get_inference_pipeline(bundle_path)
    feature_schema = pipeline.feature_schema.get("features", [])
    results: list[dict[str, Any]] = []

    for feat_name, spec in ACTIONABLE_FEATURES.items():
        # Get current value
        current_value = _get_current_value(request, feat_name)

        for candidate in spec["candidates"]:
            # Skip if candidate equals current value
            if str(candidate) == str(current_value):
                continue

            try:
                mutated = _mutate_request(request, feat_name, candidate)
                new_fv = build_feature_vector(mutated, feature_schema)
                new_result = pipeline.predict_single(new_fv)

                improvement = current_prediction - new_result.prediction
                if improvement > 0:
                    results.append({
                        "feature": feat_name,
                        "current_value": str(current_value),
                        "new_value": str(candidate),
                        "new_prediction": new_result.prediction,
                        "improvement": improvement,
                        "description": spec["description"],
                    })
            except Exception as exc:
                logger.warning(
                    "Counterfactual failed for %s=%s: %s",
                    feat_name, candidate, exc,
                )

    # Sort by improvement descending
    results.sort(key=lambda r: r["improvement"], reverse=True)

    logger.info(
        "Counterfactual tool: %d improvements found from %d candidates",
        len(results),
        sum(len(s["candidates"]) for s in ACTIONABLE_FEATURES.values()),
    )
    return results


def _get_current_value(request: PredictionRequest, feature_name: str) -> Any:
    """Extract the current value of a feature from the request."""
    t = request.tabular
    mapping = {
        "PhotoAmt": len(request.images),
        "VideoAmt": t.video_amt,
        "Vaccinated": t.vaccinated,
        "Dewormed": t.dewormed,
        "Sterilized": t.sterilized,
        "Fee": t.fee,
        "Name": t.name or "",
        "Quantity": t.quantity,
    }
    return mapping.get(feature_name, "")
