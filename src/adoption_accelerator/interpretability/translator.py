"""
SHAP → semantic translation for the Feature Interpretation Layer.

Provides the single entry-point function :func:`translate_to_interpreted`
that converts raw SHAP values into a structured ``InterpretedExplanation``
dict consumable by agent graph nodes and LLM prompt templates.

This module sits between the SHAP computation output and the agent
reasoning pipeline, ensuring that agents never see raw embedding
dimension names.

Functions
---------
translate_to_interpreted(shap_values, feature_names, registry, top_k, modality_available)
    Translate raw SHAP values into an interpreted explanation dict.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adoption_accelerator.features.registry import (
    FeatureEntry,
)
from adoption_accelerator.interpretability.aggregation import (
    AggregatedSHAP,
    aggregate_shap_values,
    build_top_k_mixed,
)

logger = logging.getLogger("adoption_accelerator")


@dataclass(frozen=True)
class InterpretedFactor:
    """A single interpreted feature factor for agent consumption.

    Attributes
    ----------
    name : str
        Semantic name (e.g. ``"rescuer_pet_count"`` for direct features
        or ``"image_visual_patterns"`` for aggregated embeddings).
    description : str
        Human-readable description suitable for user-facing text.
    shap_magnitude : float
        Absolute SHAP contribution.
    direction : str
        ``"positive"`` or ``"negative"``.
    modality : str
        Source modality (``"tabular"``, ``"text"``, ``"image"``).
    group : str
        Semantic group name.
    is_baseline : bool
        ``True`` when this factor's modality was absent from the request.
        Baseline factors are filtered out before being sent to LLM agents
        (Issue #2 fix) to prevent narrative hallucination (Issue #3 fix).
    """

    name: str
    description: str
    shap_magnitude: float
    direction: str
    modality: str
    group: str
    is_baseline: bool = False
    value: str = ""


@dataclass
class InterpretedExplanation:
    """Fully interpreted explanation for a single prediction.

    This is the semantic output consumed by LLM prompt templates
    and the agent state schema.  It contains **no raw embedding
    dimension names** and **no baseline-only factors** for absent
    modalities.

    Attributes
    ----------
    top_factors : list[InterpretedFactor]
        Top contributing factors sorted by absolute SHAP magnitude.
        Baseline entries (absent-modality offsets) are excluded.
    modality_contributions : dict[str, float]
        Fractional contribution per present modality (sums to 1.0).
        Absent modalities are excluded entirely.
    aggregated_embeddings : dict[str, dict]
        For each aggregation key, the collapsed magnitude, direction,
        and semantic template sentence.
    modality_available : dict[str, bool]
        Which modalities were actually present in the request.
    """

    top_factors: list[InterpretedFactor] = field(default_factory=list)
    modality_contributions: dict[str, float] = field(default_factory=dict)
    aggregated_embeddings: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    modality_available: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for agent state storage."""
        return {
            "top_factors": [
                {
                    "name": f.name,
                    "description": f.description,
                    "shap_magnitude": round(f.shap_magnitude, 6),
                    "direction": f.direction,
                    "modality": f.modality,
                    "group": f.group,
                    "is_baseline": f.is_baseline,
                    "value": f.value,
                }
                for f in self.top_factors
            ],
            "modality_contributions": self.modality_contributions,
            "aggregated_embeddings": self.aggregated_embeddings,
            "modality_available": self.modality_available,
        }


def translate_to_interpreted(
    shap_values: np.ndarray,
    feature_names: list[str],
    registry: dict[str, FeatureEntry],
    top_k: int = 10,
    modality_available: dict[str, bool] | None = None,
    raw_explanation: ExplanationResult | None = None,
) -> InterpretedExplanation:
    """Translate raw SHAP values into an interpreted explanation.

    This is the **single entry point** consumed by the inference node.
    It performs:

    1. SHAP aggregation (collapsing latent embedding dimensions).
    2. Modality-aware baseline detection (Issue #2 fix).
    3. Top-K mixed factor extraction (excluding baseline entries).
    4. Modality contribution computation (present modalities only).
    5. Assembly of the ``InterpretedExplanation`` object.

    Parameters
    ----------
    shap_values : np.ndarray
        Raw SHAP values for a single prediction, shape ``(n_features,)``.
        For multi-class SHAP of shape ``(n_features, n_classes)``, the
        caller should select the predicted-class slice before calling.
    feature_names : list[str]
        Ordered feature names matching the SHAP array.
    registry : dict[str, FeatureEntry]
        Built feature registry.
    top_k : int
        Number of top factors to include.
    modality_available : dict[str, bool] or None
        Which modalities were present in the request.  When ``None``,
        all modalities are assumed present (backward-compatible).

    Returns
    -------
    InterpretedExplanation
        Fully interpreted explanation with no raw embedding names and
        no phantom baseline entries for absent modalities.
    """
    from adoption_accelerator.interpretability.contracts import ExplanationResult
    
    # Step 1: aggregate SHAP values (with modality awareness)
    aggregated: AggregatedSHAP = aggregate_shap_values(
        shap_values, feature_names, registry,
        modality_available=modality_available,
    )

    # Step 2: build top-K mixed list (baseline entries filtered inside)
    top_k_dicts = build_top_k_mixed(aggregated, k=top_k)

    # Step 3: convert to InterpretedFactor objects
    raw_value_map = {}
    if raw_explanation is not None:
        raw_value_map = {f["feature"]: str(f.get("value", "")) for f in raw_explanation.top_features}

    top_factors = [
        InterpretedFactor(
            name=d["name"],
            description=d["description"],
            shap_magnitude=d["shap_magnitude"],
            direction=d["direction"],
            modality=d["modality"],
            group=d["group"],
            is_baseline=d.get("is_baseline", False),
            value=raw_value_map.get(d["name"], ""),
        )
        for d in top_k_dicts
    ]

    # Step 4: extract aggregated embedding summaries
    # Exclude baseline entries (absent-modality offsets) so downstream
    # consumers never see phantom modality artifacts like
    # image_visual_patterns when no images were provided.
    aggregated_embeddings: dict[str, dict[str, Any]] = {}
    for entry in aggregated.entries:
        if entry.is_aggregated and not entry.is_baseline:
            aggregated_embeddings[entry.name] = {
                "magnitude": round(entry.shap_magnitude, 6),
                "total_magnitude": round(entry.shap_magnitude, 6),
                "n_dimensions": entry.n_dimensions,
                "direction": entry.direction,
                "description": entry.description,
                "modality": entry.modality,
            }

    _modality_available = aggregated.modality_available

    explanation = InterpretedExplanation(
        top_factors=top_factors,
        modality_contributions=aggregated.modality_contributions,
        aggregated_embeddings=aggregated_embeddings,
        modality_available=_modality_available,
    )

    logger.info(
        "Translated SHAP to interpreted: %d top factors (after baseline filter), "
        "%d modalities, %d aggregated groups (%d baseline)",
        len(top_factors),
        len(aggregated.modality_contributions),
        len(aggregated_embeddings),
        sum(1 for e in aggregated.entries if e.is_aggregated and e.is_baseline),
    )
    return explanation
