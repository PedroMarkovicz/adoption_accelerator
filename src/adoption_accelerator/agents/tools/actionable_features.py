"""The single owner of what the recommendation agent may change.

Before this module the same eight names appeared in ACTIONABLE_FEATURES,
_INT_FEATURES, _coerce, _mutate_request, _current_value, and a separate
candidate list in the deterministic sweep. Adding a feature meant five
coordinated edits and missing one failed silently.

This table stays in Python rather than YAML on purpose. apply_change
needs real code (PhotoAmt synthesizes filenames, Name writes a nested
field), and expressing that in YAML would mean writing an interpreter
for it, which is the over-configuring that left timeouts.yaml dead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from adoption_accelerator.inference.contracts import PredictionRequest

# A counterfactual must move at least this much probability toward a
# faster class before the deterministic sweep will surface it. Below this
# the move is indistinguishable from float noise in the ensemble.
SWEEP_MIN_SHIFT: float = 0.01

# The sweep ranks by measured shift and keeps this many. Matches the
# number of recommendation cards the report renders without scrolling.
SWEEP_MAX_RECOMMENDATIONS: int = 5


@dataclass(frozen=True)
class ActionableFeature:
    name: str
    description: str
    kind: Literal["int", "float", "str"]
    sweep_default: str | None = None


ACTIONABLE_FEATURES: dict[str, ActionableFeature] = {
    f.name: f
    for f in (
        ActionableFeature(
            "PhotoAmt", "Number of photos in the listing", "int", "5"
        ),
        ActionableFeature("VideoAmt", "Number of videos in the listing", "int"),
        ActionableFeature(
            "Vaccinated", "Pet vaccination status (1=Yes)", "int", "1"
        ),
        ActionableFeature("Dewormed", "Pet deworming status (1=Yes)", "int", "1"),
        ActionableFeature(
            "Sterilized", "Pet sterilization status (1=Yes)", "int", "1"
        ),
        ActionableFeature("Fee", "Adoption fee amount", "float", "0"),
        ActionableFeature("Name", "Whether the pet has a name", "str"),
        ActionableFeature("Quantity", "Number of pets in the listing", "int"),
    )
}

_COERCERS = {
    "int": lambda v: int(float(v)),
    "float": float,
    "str": str,
}


def coerce(name: str, value: str) -> Any:
    """Coerce a raw string value to the feature's declared type.

    Raises KeyError for a feature that is not in the table.
    """
    spec = ACTIONABLE_FEATURES[name]
    return _COERCERS[spec.kind](value)


def apply_change(data: dict[str, Any], name: str, value: Any) -> None:
    """Apply one feature change to a ``PredictionRequest.model_dump()`` dict.

    Mutates in place. Raises KeyError for a feature not in the table.
    """
    if name not in ACTIONABLE_FEATURES:
        raise KeyError(name)
    if name == "PhotoAmt":
        data["images"] = [f"synthetic_{i}.jpg" for i in range(int(value))]
    elif name == "Name":
        data["tabular"]["name"] = str(value) if value else None
    elif name == "VideoAmt":
        data["tabular"]["video_amt"] = value
    else:
        data["tabular"][name.lower()] = value


def current_value(request: PredictionRequest, name: str) -> str:
    """Read a feature's current value off a request, as a string.

    Returns "" for a name that is not in the table, rather than raising:
    callers such as ``lookup_feature`` (reporting on a non-actionable
    feature) and ``_finalize_items`` (reading back an LLM-supplied,
    unconstrained feature name) rely on this graceful degradation.
    """
    if name not in ACTIONABLE_FEATURES:
        return ""
    t = request.tabular
    if name == "PhotoAmt":
        return str(len(request.images))
    if name == "VideoAmt":
        return str(t.video_amt)
    if name == "Name":
        return str(t.name or "")
    return str(getattr(t, name.lower()))


def sweep_candidates() -> list[tuple[str, str]]:
    """Feature/value pairs the deterministic fallback measures."""
    return [
        (f.name, f.sweep_default)
        for f in ACTIONABLE_FEATURES.values()
        if f.sweep_default is not None
    ]
