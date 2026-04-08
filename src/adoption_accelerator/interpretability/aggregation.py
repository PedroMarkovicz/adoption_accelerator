"""
SHAP aggregation for the Feature Interpretation Layer.

Collapses latent embedding dimension SHAP values into modality-level
semantic summaries.  This ensures agents never reference individual
embedding dimensions (e.g. ``img_emb_3``) in user-facing explanations.

Functions
---------
aggregate_shap_values(shap_values, feature_names, registry, modality_available)
    Aggregate raw SHAP values using the feature registry.
build_top_k_mixed(aggregated, k)
    Build a mixed top-K list from aggregated SHAP output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adoption_accelerator.features.registry import (
    FeatureEntry,
    get_aggregation_groups,
)

logger = logging.getLogger("adoption_accelerator")


# -- Semantic explanation templates per aggregation key -------------------

SEMANTIC_TEMPLATES: dict[str, str] = {
    "image_visual_patterns": (
        "Visual characteristics of the pet photos {direction} "
        "influenced the prediction"
    ),
    "text_semantic_patterns": (
        "Semantic patterns in the pet description {direction} "
        "influenced the prediction"
    ),
}

# Mapping from aggregation key to which modality the group represents.
# Used to check whether a group's modality was present in the request.
_AGG_KEY_MODALITY: dict[str, str] = {
    "text_semantic_patterns": "text",
    "image_visual_patterns": "image",
}


@dataclass(frozen=True)
class AggregatedEntry:
    """A single entry in the aggregated SHAP output.

    For directly-interpretable features this is a 1:1 mapping from
    the raw SHAP value.  For latent embedding groups this is the
    collapsed aggregate across all member dimensions.

    Attributes
    ----------
    is_baseline : bool
        ``True`` when this entry's modality was absent from the
        request.  In that case the SHAP contribution reflects a
        zero-vector baseline offset rather than real user content.
        Baseline entries are excluded from the modality contribution
        percentages shown to users.
    """

    name: str
    description: str
    shap_magnitude: float
    direction: str  # "positive" or "negative"
    modality: str
    group: str
    is_aggregated: bool = False
    n_dimensions: int = 1
    # Issue #2: True when the modality was absent → contribution is a
    # baseline offset from the zero-vector, not real content signal.
    is_baseline: bool = False


@dataclass
class AggregatedSHAP:
    """Complete aggregated SHAP output for a single prediction.

    Attributes
    ----------
    entries : list[AggregatedEntry]
        All entries (individual + aggregated), sorted by magnitude descending.
    modality_contributions : dict[str, float]
        Fractional contribution per *present* modality (sums to 1.0).
        Absent modalities are excluded.
    total_shap : float
        Sum of all absolute SHAP values (conservation invariant).
    modality_available : dict[str, bool]
        Which modalities were present in the request.
    """

    entries: list[AggregatedEntry] = field(default_factory=list)
    modality_contributions: dict[str, float] = field(default_factory=dict)
    total_shap: float = 0.0
    modality_available: dict[str, bool] = field(default_factory=dict)


def _direction_label(value: float) -> str:
    """Convert a signed SHAP sum to a direction string."""
    return "positive" if value >= 0 else "negative"


def _format_direction(direction: str) -> str:
    """Convert ``"positive"``/``"negative"`` to an adverb."""
    return "positively" if direction == "positive" else "negatively"


def aggregate_shap_values(
    shap_values: np.ndarray,
    feature_names: list[str],
    registry: dict[str, FeatureEntry],
    modality_available: dict[str, bool] | None = None,
) -> AggregatedSHAP:
    """Aggregate raw SHAP values using the feature registry.

    Features whose ``FeatureEntry.aggregation_key`` is non-null are
    collapsed by summing absolute SHAP values within each aggregation
    group.  All other features pass through individually.

    Issue #2 fix: when ``modality_available`` is provided, aggregated
    embedding groups whose modality is absent are marked with
    ``is_baseline=True`` and their SHAP magnitudes are excluded from
    the modality contribution percentages.  This prevents zero-vector
    baseline offsets (inserted for absent modalities at feature build
    time) from dominating the explanation output.

    Parameters
    ----------
    shap_values : np.ndarray
        Raw SHAP values for a single prediction, shape ``(n_features,)``.
        For multi-class SHAP of shape ``(n_features, n_classes)``, the
        caller should select the predicted-class slice before calling.
    feature_names : list[str]
        Ordered feature names matching the SHAP array.
    registry : dict[str, FeatureEntry]
        Built feature registry from
        :func:`~adoption_accelerator.features.registry.build_feature_registry`.
    modality_available : dict[str, bool] or None
        Mapping of modality name → bool indicating whether real data was
        provided for that modality.  When ``None``, all modalities are
        assumed present (backward-compatible behaviour).

    Returns
    -------
    AggregatedSHAP
        Aggregated output with entries sorted by absolute magnitude.
    """
    vals = np.asarray(shap_values, dtype=np.float64).ravel()
    if vals.shape[0] != len(feature_names):
        raise ValueError(
            f"SHAP values length {vals.shape[0]} != "
            f"feature_names length {len(feature_names)}"
        )

    # Default: treat all modalities as present if not specified
    _modality_available: dict[str, bool] = modality_available or {
        "tabular": True,
        "text": True,
        "image": True,
        "metadata": True,
    }

    total_shap = float(np.abs(vals).sum())

    # Map feature index by name for fast lookup
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Identify aggregation groups from registry
    agg_groups = get_aggregation_groups(registry)

    # Track which features have been consumed by aggregation
    consumed: set[str] = set()
    entries: list[AggregatedEntry] = []

    # -- Step 1: aggregate latent embedding groups -----------------------
    for agg_key, member_names in agg_groups.items():
        member_indices = [
            name_to_idx[n] for n in member_names if n in name_to_idx
        ]
        if not member_indices:
            continue

        member_vals = vals[member_indices]
        magnitude = float(np.abs(member_vals).sum())
        net_sum = float(member_vals.sum())
        direction = _direction_label(net_sum)

        # Build semantic description from template
        template = SEMANTIC_TEMPLATES.get(agg_key, "{direction} influence")
        description = template.format(
            direction=_format_direction(direction)
        )

        # Use the first member's modality/group (all should be identical)
        first_entry = registry[member_names[0]]

        # Issue #2: determine if this group's modality was absent.
        # The group-to-modality mapping covers known embedding groups;
        # for other aggregation keys, default to present.
        group_modality = _AGG_KEY_MODALITY.get(agg_key, first_entry.modality)
        is_baseline = not _modality_available.get(group_modality, True)

        entries.append(
            AggregatedEntry(
                name=agg_key,
                description=description,
                shap_magnitude=magnitude,
                direction=direction,
                modality=first_entry.modality,
                group=first_entry.group,
                is_aggregated=True,
                n_dimensions=len(member_indices),
                is_baseline=is_baseline,
            )
        )
        consumed.update(member_names)

    # -- Step 2: pass through non-latent features individually -----------
    for i, name in enumerate(feature_names):
        if name in consumed:
            continue
        entry = registry[name]
        raw_val = float(vals[i])

        # Issue #2: mark non-latent features from absent modalities as baseline.
        # E.g. word_count, polarity_compound (text) or image_metadata features
        # are still zero-imputed when the modality is absent, so their SHAP
        # contributions are baseline offsets just like the latent embeddings.
        feature_modality = entry.modality
        feature_is_baseline = not _modality_available.get(feature_modality, True)

        entries.append(
            AggregatedEntry(
                name=name,
                description=entry.description,
                shap_magnitude=abs(raw_val),
                direction=_direction_label(raw_val),
                modality=entry.modality,
                group=entry.group,
                is_aggregated=False,
                is_baseline=feature_is_baseline,
            )
        )


    # Sort by magnitude descending
    entries.sort(key=lambda e: e.shap_magnitude, reverse=True)

    # -- Step 3: compute modality contributions (PRESENT modalities only) -
    # Issue #2: exclude baseline entries from the denominator so that
    # contributions from actually-present modalities sum to 100%.
    modality_abs: dict[str, float] = {}
    for e in entries:
        if e.is_baseline:
            # Skip absent-modality entries: their SHAP is a zero-vector
            # baseline offset, not real content signal.
            continue
        modality_abs[e.modality] = modality_abs.get(e.modality, 0.0) + e.shap_magnitude

    total_for_pct = sum(modality_abs.values()) or 1.0
    modality_contributions = {
        mod: round(val / total_for_pct, 6)
        for mod, val in sorted(
            modality_abs.items(), key=lambda x: x[1], reverse=True
        )
    }

    logger.debug(
        "aggregate_shap_values: %d entries (%d baseline), modalities=%s",
        len(entries),
        sum(1 for e in entries if e.is_baseline),
        list(modality_contributions.keys()),
    )

    return AggregatedSHAP(
        entries=entries,
        modality_contributions=modality_contributions,
        total_shap=total_shap,
        modality_available=_modality_available,
    )


def build_top_k_mixed(
    aggregated: AggregatedSHAP,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Build a mixed top-K list from aggregated SHAP output.

    The list contains both individual directly-interpretable features
    and aggregated modality entries, ranked by absolute magnitude.

    Baseline entries (from absent modalities) are excluded so that
    LLM nodes never receive phantom content signals.

    Parameters
    ----------
    aggregated : AggregatedSHAP
        Output of :func:`aggregate_shap_values`.
    k : int
        Maximum number of entries to return.

    Returns
    -------
    list[dict]
        Top-K entries as dicts with keys: ``name``, ``description``,
        ``shap_magnitude``, ``direction``, ``modality``, ``group``,
        ``is_aggregated``, ``is_baseline``.
    """
    # Always exclude baseline entries from the top-K list delivered to
    # LLM agents to prevent narrative hallucination (Issue #3 defence).
    non_baseline = [e for e in aggregated.entries if not e.is_baseline]
    return [
        {
            "name": e.name,
            "description": e.description,
            "shap_magnitude": round(e.shap_magnitude, 6),
            "direction": e.direction,
            "modality": e.modality,
            "group": e.group,
            "is_aggregated": e.is_aggregated,
            "is_baseline": e.is_baseline,
        }
        for e in non_baseline[:k]
    ]
