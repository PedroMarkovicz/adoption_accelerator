"""
Interpretability output contracts.

Defines the ``ExplanationResult`` dataclass that serves as the
formal interface between the interpretability layer and its
downstream consumers (inference pipeline, agent system, frontend).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExplanationResult:
    """Structured explanation for a single prediction.

    Attributes
    ----------
    pet_id : str
        Unique pet identifier.
    predicted_class : int
        Predicted AdoptionSpeed class (0-4).
    probabilities : list[float]
        Per-class predicted probabilities.
    top_features : list[dict[str, Any]]
        Top-K contributing features with SHAP values and feature values.
    modality_contributions : dict[str, float]
        Per-modality total SHAP proportion for this prediction.
    confidence : float
        Max predicted probability.
    """

    pet_id: str = ""
    predicted_class: int = 0
    probabilities: list[float] = field(default_factory=list)
    top_features: list[dict[str, Any]] = field(default_factory=list)
    modality_contributions: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "pet_id": self.pet_id,
            "predicted_class": self.predicted_class,
            "probabilities": self.probabilities,
            "top_features": self.top_features,
            "modality_contributions": self.modality_contributions,
            "confidence": self.confidence,
        }
