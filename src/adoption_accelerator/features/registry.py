"""
Feature registry utilities for schema registration, validation, and
the **Feature Interpretation Layer**.

Part 1 — Schema Utilities
    Functions to save and load ``schema.json`` files that accompany
    every versioned feature set in ``data/features/{modality}/v{N}/``.

Part 2 — Feature Interpretation Layer (Phase 4)
    Dynamically maps every feature in the model's 940-feature schema to
    semantic metadata using pattern-based rules and explicit description
    dicts.  No static 940-entry artifact required.

Functions
---------
save_feature_schema(columns, metadata, path)
    Generate and persist a schema.json for a feature set.
load_feature_schema(path)
    Load and validate a schema.json.
compute_config_hash(config_dict)
    Compute a SHA-256 hash for a feature generation configuration.
build_feature_registry(feature_names, provenance_map)
    Build a dynamic feature registry from pattern rules and description dicts.
get_aggregation_groups(registry)
    Return aggregation_key → list[feature_name] mapping for latent features.
get_group_members(registry)
    Return group → list[feature_name] mapping.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("adoption_accelerator")


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute a SHA-256 hash for a feature generation configuration.

    Parameters
    ----------
    config_dict : dict
        Dictionary with configuration parameters used to generate the
        feature set (e.g., version, seed, feature list).

    Returns
    -------
    str
        Hex-encoded SHA-256 hash string (first 16 characters).
    """
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def save_feature_schema(
    columns: list[dict[str, str]],
    metadata: dict[str, Any],
    path: Path | str,
) -> Path:
    """Generate and persist a ``schema.json`` for a feature set.

    Parameters
    ----------
    columns : list of dict
        Column descriptors, each with keys ``name``, ``dtype``,
        ``source``, and ``description``.
    metadata : dict
        Top-level metadata for the schema. Expected keys include
        ``version``, ``modality``, ``config_hash``, ``n_rows_train``,
        ``n_rows_test``, ``n_features``, ``seed``.
    path : Path | str
        Destination path (e.g., ``data/features/tabular/v1/schema.json``).

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = {
        "version": metadata.get("version", "v1"),
        "modality": metadata.get("modality", "unknown"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": metadata.get("config_hash", ""),
        "model_name": metadata.get("model_name"),
        "columns": columns,
        "n_rows_train": metadata.get("n_rows_train", 0),
        "n_rows_test": metadata.get("n_rows_test", 0),
        "n_features": metadata.get("n_features", len(columns)),
        "seed": metadata.get("seed", 42),
        "notes": metadata.get("notes", ""),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    logger.info(
        "Saved feature schema: %s (%d columns, modality=%s)",
        path,
        len(columns),
        schema["modality"],
    )
    return path


def load_feature_schema(path: Path | str) -> dict[str, Any]:
    """Load and validate a ``schema.json``.

    Parameters
    ----------
    path : Path | str
        Path to the ``schema.json`` file.

    Returns
    -------
    dict
        Parsed schema dictionary.

    Raises
    ------
    FileNotFoundError
        If the schema file does not exist.
    ValueError
        If the schema is missing required keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    required_keys = {"version", "modality", "columns", "n_rows_train", "n_rows_test"}
    missing = required_keys - set(schema.keys())
    if missing:
        raise ValueError(f"Schema is missing required keys: {missing}")

    logger.info(
        "Loaded feature schema: %s (v=%s, modality=%s, %d cols)",
        path,
        schema["version"],
        schema["modality"],
        len(schema["columns"]),
    )
    return schema


def build_column_descriptors(
    df: pd.DataFrame,
    source: str = "tabular",
    descriptions: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    """Build column descriptor dicts from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame to describe.
    source : str
        Modality tag (e.g., ``"tabular"``, ``"text"``, ``"image"``).
    descriptions : dict, optional
        Mapping of column name to human-readable description. Columns
        absent from this mapping receive a generic description.

    Returns
    -------
    list of dict
        One dict per column with keys ``name``, ``dtype``, ``source``,
        ``description``.
    """
    descriptions = descriptions or {}
    cols: list[dict[str, str]] = []
    for col in df.columns:
        cols.append(
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "source": source,
                "description": descriptions.get(col, ""),
            }
        )
    return cols


# =====================================================================
#  Part 2 — Feature Interpretation Layer (Phase 4)
# =====================================================================


@dataclass(frozen=True)
class FeatureEntry:
    """Semantic metadata for a single model feature.

    Attributes
    ----------
    name : str
        Raw feature name (e.g. ``"rescuer_pet_count"``).
    modality : str
        Source modality: ``"tabular"``, ``"text"``, or ``"image"``.
    group : str
        Semantic group (e.g. ``"rescuer_statistics"``,
        ``"image_embedding"``).
    description : str
        Human-readable description for agent explanations.
    interpretability : str
        One of ``"direct"``, ``"engineered"``, or ``"latent"``.
    aggregation_key : str | None
        For latent features: key used by the SHAP aggregation function
        to collapse individual dimensions into a semantic group
        (e.g. ``"image_visual_patterns"``).  ``None`` for
        directly-interpretable features.
    """

    name: str
    modality: str
    group: str
    description: str
    interpretability: str
    aggregation_key: str | None = None


# -- Pattern rules (first-match-wins) ------------------------------------
#
# Each tuple is (regex_pattern, metadata_dict).  The metadata dict
# provides ``group``, ``modality`` (fallback if provenance map is not
# available), ``interpretability``, and optionally ``aggregation_key``
# and ``description_template`` (for embeddings).

PATTERN_RULES: list[tuple[str, dict[str, str]]] = [
    # --- Latent embedding dimensions (868 of 940) -----------------------
    (
        r"^img_emb_(\d+)$",
        {
            "group": "image_embedding",
            "modality": "image",
            "interpretability": "latent",
            "aggregation_key": "image_visual_patterns",
            "description_template": (
                "Dimension {0} of PCA-reduced EfficientNet image embedding"
            ),
        },
    ),
    (
        r"^text_emb_(\d+)$",
        {
            "group": "text_embedding",
            "modality": "text",
            "interpretability": "latent",
            "aggregation_key": "text_semantic_patterns",
            "description_template": (
                "Dimension {0} of sentence-transformer text embedding"
            ),
        },
    ),
    # --- Tabular naming conventions -------------------------------------
    (
        r"^rescuer_",
        {
            "group": "rescuer_statistics",
            "modality": "tabular",
            "interpretability": "engineered",
        },
    ),
    (
        r"^log_",
        {
            "group": "log_transforms",
            "modality": "tabular",
            "interpretability": "engineered",
        },
    ),
    (
        r"^is_",
        {
            "group": "binary_indicators",
            "modality": "tabular",
            "interpretability": "direct",
        },
    ),
    (
        r"_x_",
        {
            "group": "interactions",
            "modality": "tabular",
            "interpretability": "engineered",
        },
    ),
    (
        r"_missing$",
        {
            "group": "missing_indicators",
            "modality": "tabular",
            "interpretability": "engineered",
        },
    ),
    (
        r"^age_bin",
        {
            "group": "age_features",
            "modality": "tabular",
            "interpretability": "engineered",
        },
    ),
    (
        r"_freq$|_frequency$",
        {
            "group": "frequency_encoding",
            "modality": "tabular",
            "interpretability": "engineered",
        },
    ),
    # --- Text feature patterns ------------------------------------------
    (
        r"^doc_sentiment_",
        {
            "group": "sentiment_scores",
            "modality": "metadata",
            "interpretability": "engineered",
        },
    ),
    (
        r"_sentence_|^sentence_count_sentiment$",
        {
            "group": "sentiment_scores",
            "modality": "metadata",
            "interpretability": "engineered",
        },
    ),
    # --- Metadata / API features ----------------------------------------
    (
        r"^meta_|^entity_",
        {
            "group": "metadata_api",
            "modality": "metadata",
            "interpretability": "direct",
        },
    ),
    (
        r"^mean_image_|^mean_blur|^image_size",
        {
            "group": "image_quality",
            "modality": "metadata",
            "interpretability": "direct",
        },
    # --- Image feature patterns -----------------------------------------
    ),
    (
        r"^has_image|^actual_photo",
        {
            "group": "image_auxiliary",
            "modality": "image",
            "interpretability": "direct",
        },
    ),
]

# Pre-compile the regexes for performance
_COMPILED_RULES: list[tuple[re.Pattern[str], dict[str, str]]] = [
    (re.compile(pattern), meta) for pattern, meta in PATTERN_RULES
]


def _collect_all_descriptions() -> dict[str, str]:
    """Merge ``FEATURE_DESCRIPTIONS`` dicts from all feature modules.

    Returns a single flat dict mapping feature name → human-readable
    description.  Imports are deferred to avoid circular dependencies
    at module load time.
    """
    from adoption_accelerator.features.image import (
        FEATURE_DESCRIPTIONS as IMG_DESC,
    )
    from adoption_accelerator.features.tabular import (
        FEATURE_DESCRIPTIONS as TAB_DESC,
    )
    from adoption_accelerator.features.text import (
        FEATURE_DESCRIPTIONS as TXT_DESC,
    )

    merged: dict[str, str] = {}
    merged.update(TAB_DESC)
    merged.update(IMG_DESC)
    merged.update(TXT_DESC)
    return merged


def build_feature_registry(
    feature_names: list[str],
    provenance_map: dict[str, str] | None = None,
) -> dict[str, FeatureEntry]:
    """Build the feature registry dynamically from pattern rules and
    description dicts.

    For each feature name:

    1. Match against :data:`PATTERN_RULES` (first match wins) to obtain
       ``group``, ``interpretability``, and ``aggregation_key``.
    2. If *provenance_map* is provided, use it for modality
       (schema-driven, authoritative).  Otherwise fall back to the
       modality from the matched pattern rule.
    3. Look up a per-feature description from the merged
       ``FEATURE_DESCRIPTIONS`` dicts.  If found, use it.  Otherwise
       use the pattern's ``description_template`` (for embeddings) or
       a generic fallback.
    4. Return a dict with exactly ``len(feature_names)`` entries.

    Parameters
    ----------
    feature_names : list[str]
        Ordered feature names (e.g. from ``feature_schema.json``).
    provenance_map : dict[str, str] | None
        Optional column-to-modality mapping from
        :func:`~adoption_accelerator.features.integration.build_provenance_map`.

    Returns
    -------
    dict[str, FeatureEntry]
        Mapping of feature name → ``FeatureEntry``.

    Raises
    ------
    ValueError
        If any feature cannot be assigned a modality.
    """
    descriptions = _collect_all_descriptions()
    registry: dict[str, FeatureEntry] = {}

    for feat in feature_names:
        # -- Step 1: pattern matching ------------------------------------
        matched_meta: dict[str, str] | None = None
        match_groups: tuple[str, ...] = ()

        for compiled_re, meta in _COMPILED_RULES:
            m = compiled_re.search(feat)
            if m:
                matched_meta = meta
                match_groups = m.groups()
                break

        # -- Step 2: resolve modality ------------------------------------
        if provenance_map and feat in provenance_map:
            modality = provenance_map[feat]
        elif matched_meta and "modality" in matched_meta:
            modality = matched_meta["modality"]
        else:
            raise ValueError(
                f"Cannot determine modality for feature '{feat}'. "
                "Provide a provenance_map or add a matching pattern rule."
            )

        # -- Step 3: resolve group and interpretability ------------------
        if matched_meta:
            group = matched_meta.get("group", "ungrouped")
            interpretability = matched_meta.get("interpretability", "direct")
            aggregation_key = matched_meta.get("aggregation_key")
        else:
            group = "ungrouped"
            interpretability = "direct"
            aggregation_key = None

        # -- Step 4: resolve description ---------------------------------
        if feat in descriptions:
            description = descriptions[feat]
        elif matched_meta and "description_template" in matched_meta:
            description = matched_meta["description_template"].format(
                *match_groups
            )
        else:
            # Generic fallback — prefer a descriptive sentence
            description = f"Feature: {feat} ({modality})"

        registry[feat] = FeatureEntry(
            name=feat,
            modality=modality,
            group=group,
            description=description,
            interpretability=interpretability,
            aggregation_key=aggregation_key,
        )

    # -- Completeness check ----------------------------------------------
    if len(registry) != len(feature_names):
        missing = set(feature_names) - set(registry)
        raise ValueError(
            f"Registry incomplete: {len(missing)} features unclassified: "
            f"{sorted(missing)[:10]}..."
        )

    logger.info(
        "Feature registry built: %d features, %d groups, %d with aggregation_key",
        len(registry),
        len({e.group for e in registry.values()}),
        sum(1 for e in registry.values() if e.aggregation_key is not None),
    )
    return registry


def get_aggregation_groups(
    registry: dict[str, FeatureEntry],
) -> dict[str, list[str]]:
    """Return ``{aggregation_key: [feature_name, ...]}`` for latent features.

    Only features with a non-null ``aggregation_key`` are included.
    This is consumed by the SHAP aggregation function to collapse
    embedding dimensions into modality-level summaries.
    """
    groups: dict[str, list[str]] = {}
    for entry in registry.values():
        if entry.aggregation_key is not None:
            groups.setdefault(entry.aggregation_key, []).append(entry.name)
    return groups


def get_group_members(
    registry: dict[str, FeatureEntry],
) -> dict[str, list[str]]:
    """Return ``{group: [feature_name, ...]}`` for all features.

    Maps each semantic group to its member feature names.  Used for
    group-level SHAP reporting.
    """
    members: dict[str, list[str]] = {}
    for entry in registry.values():
        members.setdefault(entry.group, []).append(entry.name)
    return members
