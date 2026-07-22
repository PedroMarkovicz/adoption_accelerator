"""LangChain tools for the recommendation agent.

The agent can only *measure* — every tool re-runs the real ensemble.
The actionable-feature whitelist is enforced here, inside the tools,
so the agent cannot test non-actionable features (Age, Breed, ...) no
matter what it asks for. Every successful measurement is stored in a
``MeasurementLog`` under an id; the finalize step later builds
``MeasuredImpact`` objects ONLY from this log, never from LLM text."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool, tool

from adoption_accelerator.inference.contracts import (
    PredictionRequest,
    PredictionResult,
)
from adoption_accelerator.inference.serving import get_inference_pipeline

logger = logging.getLogger(__name__)

# Copied from the old agents/tools/counterfactual_tool.py (same semantics).
ACTIONABLE_FEATURES: dict[str, dict[str, Any]] = {
    "PhotoAmt": {"description": "Number of photos in the listing"},
    "VideoAmt": {"description": "Number of videos in the listing"},
    "Vaccinated": {"description": "Pet vaccination status (1=Yes)"},
    "Dewormed": {"description": "Pet deworming status (1=Yes)"},
    "Sterilized": {"description": "Pet sterilization status (1=Yes)"},
    "Fee": {"description": "Adoption fee amount"},
    "Name": {"description": "Whether the pet has a name"},
    "Quantity": {"description": "Number of pets in the listing"},
}

_INT_FEATURES = {"PhotoAmt", "VideoAmt", "Vaccinated", "Dewormed",
                 "Sterilized", "Quantity"}


def _coerce(feature: str, value: str) -> Any:
    if feature in _INT_FEATURES:
        return int(float(value))
    if feature == "Fee":
        return float(value)
    return value  # Name


def _mutate_request(request: PredictionRequest, changes: dict[str, Any]) -> PredictionRequest:
    data = request.model_dump()
    for feature, value in changes.items():
        if feature == "PhotoAmt":
            data["images"] = [f"synthetic_{i}.jpg" for i in range(int(value))]
        elif feature == "Name":
            data["tabular"]["name"] = str(value) if value else None
        else:
            field = "video_amt" if feature == "VideoAmt" else feature.lower()
            data["tabular"][field] = value
    return PredictionRequest(**data)


def _current_value(request: PredictionRequest, feature: str) -> str:
    t = request.tabular
    mapping = {
        "PhotoAmt": len(request.images), "VideoAmt": t.video_amt,
        "Vaccinated": t.vaccinated, "Dewormed": t.dewormed,
        "Sterilized": t.sterilized, "Fee": t.fee,
        "Name": t.name or "", "Quantity": t.quantity,
    }
    return str(mapping.get(feature, ""))


class MeasurementLog:
    """Stores every real measurement made through the tools."""

    def __init__(self) -> None:
        self.measurements: dict[str, dict[str, Any]] = {}
        self._counter = 0

    def record(self, changes: dict[str, Any], baseline: PredictionResult,
               result: PredictionResult) -> str:
        self._counter += 1
        mid = f"m{self._counter}"
        self.measurements[mid] = {
            "changes": {k: str(v) for k, v in changes.items()},
            "class_before": baseline.prediction,
            "class_after": result.prediction,
            "probability_shift": {
                k: round(result.probabilities[k] - baseline.probabilities[k], 4)
                for k in baseline.probabilities
            },
        }
        return mid


def make_recommendation_tools(
    request: PredictionRequest, baseline: PredictionResult
) -> tuple[list[BaseTool], MeasurementLog]:
    """Build the tool belt bound to one specific request."""
    from adoption_accelerator.inference.feature_builder import build_feature_vector

    pipeline = get_inference_pipeline()
    feature_schema = pipeline.feature_schema.get("features", [])
    log = MeasurementLog()

    def _measure(changes: dict[str, Any]) -> dict[str, Any]:
        for feature in changes:
            if feature not in ACTIONABLE_FEATURES:
                return {"error": (
                    f"Feature '{feature}' is not actionable. Actionable "
                    f"features: {sorted(ACTIONABLE_FEATURES)}"
                )}
        try:
            coerced = {f: _coerce(f, str(v)) for f, v in changes.items()}
        except (ValueError, TypeError) as exc:
            return {"error": f"could not interpret value(s) for {list(changes)}: {exc}"}
        mutated = _mutate_request(request, coerced)
        fv = build_feature_vector(mutated, feature_schema)
        result = pipeline.predict_single(fv)
        # Log the raw, as-requested change values (not the coerced/typed
        # values used to mutate the request) so the log reflects exactly
        # what the caller asked for.
        mid = log.record(changes, baseline, result)
        entry = dict(log.measurements[mid])
        entry["measurement_id"] = mid
        return entry

    @tool
    def run_counterfactual(feature: str, value: str) -> str:
        """Re-run the real ML ensemble with ONE feature changed. Returns the
        measured class change and per-class probability shift as JSON."""
        return json.dumps(_measure({feature: value}))

    @tool
    def run_what_if(changes_json: str) -> str:
        """Re-run the real ML ensemble with MULTIPLE combined changes.
        `changes_json` is a JSON object mapping feature name to new value."""
        try:
            changes = json.loads(changes_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid JSON: {exc}"})
        if not isinstance(changes, dict) or not changes:
            return json.dumps({"error": "changes_json must be a non-empty object"})
        return json.dumps(_measure(changes))

    @tool
    def lookup_feature(feature: str) -> str:
        """Look up a feature: description, whether it is actionable, and its
        current value in this listing."""
        spec = ACTIONABLE_FEATURES.get(feature)
        return json.dumps({
            "feature": feature,
            "actionable": spec is not None,
            "description": spec["description"] if spec else "not actionable",
            "current_value": _current_value(request, feature),
        })

    return [run_counterfactual, run_what_if, lookup_feature], log
