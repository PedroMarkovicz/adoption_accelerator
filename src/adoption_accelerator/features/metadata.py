"""
Metadata feature extraction functions for Adoption Accelerator.

Provides aggregation of Google NLP sentiment JSONs and Google Vision API
metadata JSONs into per-PetID feature rows. These functions produce the
extended sentiment and metadata feature columns required by the text and
image feature extraction notebooks.

Functions
---------
aggregate_sentiment_features(split)
    Batch-load all sentiment JSONs for a split and aggregate at PetID
    level with extended statistics.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from adoption_accelerator import config as cfg

logger = logging.getLogger("adoption_accelerator")


# Default values for missing sentiment features
SENTIMENT_DEFAULTS: dict[str, float] = {
    "doc_sentiment_score": 0.0,
    "doc_sentiment_magnitude": 0.0,
    "sentence_count_sentiment": 0,
    "mean_sentence_score": 0.0,
    "min_sentence_score": 0.0,
    "max_sentence_score": 0.0,
    "sentiment_variance": 0.0,
    "entity_count": 0,
    "entity_type_count": 0,
}


def aggregate_sentiment_features(
    split: str = "train",
    *,
    progress: bool = True,
) -> pd.DataFrame:
    """Batch-load all sentiment JSONs and aggregate at PetID level.

    Produces extended sentiment features beyond what the basic ingestion
    parser provides, including min/max sentence scores, sentiment
    variance, and entity type counts.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    progress : bool
        If ``True``, log progress every 5,000 files.

    Returns
    -------
    pd.DataFrame
        One row per PetID with columns:
        ``PetID``, ``doc_sentiment_score``, ``doc_sentiment_magnitude``,
        ``sentence_count_sentiment``, ``mean_sentence_score``,
        ``min_sentence_score``, ``max_sentence_score``,
        ``sentiment_variance``, ``entity_count``, ``entity_type_count``.
    """
    base = cfg.RAW_TRAIN_SENTIMENT if split == "train" else cfg.RAW_TEST_SENTIMENT
    if not base.exists():
        raise FileNotFoundError(f"Sentiment directory not found: {base}")

    json_files = sorted(base.glob("*.json"))
    logger.info("Found %d sentiment JSON files in %s", len(json_files), base)

    rows: list[dict[str, Any]] = []
    for idx, fp in enumerate(json_files, 1):
        pet_id = fp.stem

        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Document-level sentiment
            doc_sent = data.get("documentSentiment", {})
            doc_score = float(doc_sent.get("score", 0.0))
            doc_magnitude = float(doc_sent.get("magnitude", 0.0))

            # Sentence-level aggregation
            sentences = data.get("sentences", [])
            sent_scores = [
                float(s.get("sentiment", {}).get("score", 0.0)) for s in sentences
            ]

            if sent_scores:
                mean_score = float(np.mean(sent_scores))
                min_score = float(np.min(sent_scores))
                max_score = float(np.max(sent_scores))
                var_score = float(np.var(sent_scores))
            else:
                mean_score = 0.0
                min_score = 0.0
                max_score = 0.0
                var_score = 0.0

            # Entity information
            entities = data.get("entities", [])
            entity_types = set(e.get("type", "UNKNOWN") for e in entities)

            rows.append(
                {
                    "PetID": pet_id,
                    "doc_sentiment_score": round(doc_score, 4),
                    "doc_sentiment_magnitude": round(doc_magnitude, 4),
                    "sentence_count_sentiment": len(sentences),
                    "mean_sentence_score": round(mean_score, 4),
                    "min_sentence_score": round(min_score, 4),
                    "max_sentence_score": round(max_score, 4),
                    "sentiment_variance": round(var_score, 6),
                    "entity_count": len(entities),
                    "entity_type_count": len(entity_types),
                }
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse %s: %s", fp, exc)

        if progress and idx % 5_000 == 0:
            logger.info("  ... loaded %d / %d sentiment JSONs", idx, len(json_files))

    df = pd.DataFrame(rows)
    logger.info(
        "Aggregated sentiment features for split='%s': %d rows x %d cols",
        split,
        *df.shape,
    )
    return df


# =====================================================================
#  Vision API Metadata Aggregation
# =====================================================================

# Default values for missing metadata features
METADATA_DEFAULTS: dict[str, float] = {
    "meta_label_count_mean": 0.0,
    "meta_top_label_score_mean": 0.0,
    "meta_dominant_color_count_mean": 0.0,
    "meta_avg_brightness_mean": 0.0,
    "meta_color_diversity_mean": 0.0,
    "meta_crop_confidence_mean": 0.0,
}


def aggregate_metadata_features(
    split: str = "train",
    *,
    progress: bool = True,
) -> pd.DataFrame:
    """Batch-load all Vision API metadata JSONs and aggregate at PetID level.

    Extracts per-image features (label count, top label score, dominant
    color count, brightness, color diversity, crop confidence) and
    aggregates them per PetID using mean pooling.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    progress : bool
        If ``True``, log progress every 10,000 files.

    Returns
    -------
    pd.DataFrame
        One row per PetID with aggregated metadata feature columns:
        ``PetID``, ``meta_label_count_mean``, ``meta_top_label_score_mean``,
        ``meta_dominant_color_count_mean``, ``meta_avg_brightness_mean``,
        ``meta_color_diversity_mean``, ``meta_crop_confidence_mean``.
    """
    base = cfg.RAW_TRAIN_METADATA if split == "train" else cfg.RAW_TEST_METADATA
    if not base.exists():
        raise FileNotFoundError(f"Metadata directory not found: {base}")

    json_files = sorted(base.glob("*.json"))
    logger.info("Found %d metadata JSON files in %s", len(json_files), base)

    # Collect per-image features
    image_rows: list[dict[str, Any]] = []
    for idx, fp in enumerate(json_files, 1):
        stem = fp.stem
        parts = stem.rsplit("-", 1)
        pet_id = parts[0]

        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Label annotations
            labels = data.get("labelAnnotations", [])
            label_scores = [la.get("score", 0.0) for la in labels]
            top_label_score = float(label_scores[0]) if label_scores else 0.0

            # Dominant colors
            color_info = (
                data.get("imagePropertiesAnnotation", {})
                .get("dominantColors", {})
                .get("colors", [])
            )
            dom_color_count = len(color_info)
            brightnesses: list[float] = []
            for ci in color_info:
                c = ci.get("color", {})
                r = float(c.get("red", 0))
                g = float(c.get("green", 0))
                b = float(c.get("blue", 0))
                brightnesses.append((r + g + b) / 3.0)
            avg_brightness = float(np.mean(brightnesses)) if brightnesses else 0.0
            color_diversity = (
                float(np.std(brightnesses)) if len(brightnesses) > 1 else 0.0
            )

            # Crop hints
            crop_hints = data.get("cropHintsAnnotation", {}).get("cropHints", [])
            crop_confidence = (
                float(crop_hints[0].get("confidence", 0.0)) if crop_hints else 0.0
            )

            image_rows.append(
                {
                    "PetID": pet_id,
                    "label_count": len(labels),
                    "top_label_score": round(top_label_score, 4),
                    "dominant_color_count": dom_color_count,
                    "avg_brightness": round(avg_brightness, 2),
                    "color_diversity": round(color_diversity, 2),
                    "crop_confidence": round(crop_confidence, 6),
                }
            )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse metadata %s: %s", fp, exc)

        if progress and idx % 10_000 == 0:
            logger.info("  ... loaded %d / %d metadata JSONs", idx, len(json_files))

    if not image_rows:
        logger.warning("No metadata files loaded for split='%s'", split)
        return pd.DataFrame(columns=["PetID"] + list(METADATA_DEFAULTS.keys()))

    img_df = pd.DataFrame(image_rows)

    # Aggregate per PetID (mean pooling)
    agg_cols = [
        "label_count",
        "top_label_score",
        "dominant_color_count",
        "avg_brightness",
        "color_diversity",
        "crop_confidence",
    ]
    agg_df = img_df.groupby("PetID")[agg_cols].mean().reset_index()

    # Rename to prefixed column names
    rename_map = {
        "label_count": "meta_label_count_mean",
        "top_label_score": "meta_top_label_score_mean",
        "dominant_color_count": "meta_dominant_color_count_mean",
        "avg_brightness": "meta_avg_brightness_mean",
        "color_diversity": "meta_color_diversity_mean",
        "crop_confidence": "meta_crop_confidence_mean",
    }
    agg_df = agg_df.rename(columns=rename_map)

    # Round for clean output
    for col in METADATA_DEFAULTS:
        if col in agg_df.columns:
            agg_df[col] = agg_df[col].round(4).astype(np.float32)

    logger.info(
        "Aggregated metadata features for split='%s': %d PetIDs x %d cols",
        split,
        len(agg_df),
        len(agg_df.columns),
    )
    return agg_df
