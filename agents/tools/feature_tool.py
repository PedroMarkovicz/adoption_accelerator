"""
Feature lookup tool adapter.

Adapter calling ``build_feature_registry()`` from the core library.
Returns ``FeatureEntry`` data (description, modality, group,
interpretability level) for queried features.  Caches the built
registry in memory.

Consumed by: ``agents/nodes/inference.py``, ``agents/nodes/explainer.py``
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from adoption_accelerator.features.registry import (
    FeatureEntry,
    build_feature_registry,
    get_aggregation_groups,
    get_group_members,
)
from adoption_accelerator.inference.explain import _load_default_provenance_map

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_cached_registry(
    feature_names_tuple: tuple[str, ...],
) -> dict[str, FeatureEntry]:
    """Build and cache the feature registry."""
    provenance_map = _load_default_provenance_map()
    registry = build_feature_registry(list(feature_names_tuple), provenance_map=provenance_map)
    logger.info("Feature registry built and cached: %d entries", len(registry))
    return registry


def lookup_features(
    feature_names: list[str],
    query_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Look up feature metadata from the registry.

    Parameters
    ----------
    feature_names : list[str]
        Full list of feature names from the model schema (used to
        build the registry on first call).
    query_names : list[str] | None
        Specific features to look up.  If ``None``, returns all entries.

    Returns
    -------
    list[dict]
        Feature entries as dicts with keys: ``name``, ``modality``,
        ``group``, ``description``, ``interpretability``,
        ``aggregation_key``.
    """
    registry = _get_cached_registry(tuple(feature_names))

    targets = query_names if query_names else list(registry.keys())

    results = []
    for name in targets:
        entry = registry.get(name)
        if entry is None:
            logger.warning("Feature '%s' not found in registry", name)
            continue
        results.append({
            "name": entry.name,
            "modality": entry.modality,
            "group": entry.group,
            "description": entry.description,
            "interpretability": entry.interpretability,
            "aggregation_key": entry.aggregation_key,
        })

    return results


def get_registry(feature_names: list[str]) -> dict[str, FeatureEntry]:
    """Return the cached feature registry for the given feature names.

    Convenience wrapper for direct registry access by other tools.
    """
    return _get_cached_registry(tuple(feature_names))


def get_modality_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """Return features grouped by semantic group."""
    registry = _get_cached_registry(tuple(feature_names))
    return get_group_members(registry)


def get_embedding_aggregation_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """Return features grouped by aggregation key (for latent embedding collapse)."""
    registry = _get_cached_registry(tuple(feature_names))
    return get_aggregation_groups(registry)
