"""
Data ingestion utilities for the Adoption Accelerator project.

All I/O operations are centralized here so that path resolution, dtype
enforcement, and error handling are consistent across notebooks and
pipelines.

Functions
---------
load_tabular(split)
    Load train.csv or test.csv as a pandas DataFrame.
load_reference_table(name)
    Load a reference CSV (breed, color, state) as a DataFrame.
load_sentiment_json(pet_id, index, split)
    Load a single sentiment JSON file.
load_metadata_json(pet_id, index, split)
    Load a single metadata JSON file.
load_image(pet_id, index, split)
    Load a single pet image as a PIL Image.
load_all_sentiment_jsons(split)
    Batch-load all sentiment JSONs for a given split.
parse_sentiment_to_dataframe(records)
    Convert raw sentiment JSON records into a structured DataFrame.
get_file_inventory(directory)
    Recursively list files with counts and sizes.
save_parquet(df, path)
    Save a DataFrame to Parquet (PyArrow, snappy compression).
load_parquet(path)
    Load a Parquet file into a DataFrame.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

from adoption_accelerator import config as cfg

logger = logging.getLogger("adoption_accelerator")

# ── Expected column schema ──────────────────────────────────────────

EXPECTED_TRAIN_COLUMNS: list[str] = [
    "Type", "Name", "Age", "Breed1", "Breed2", "Gender",
    "Color1", "Color2", "Color3", "MaturitySize", "FurLength",
    "Vaccinated", "Dewormed", "Sterilized", "Health",
    "Quantity", "Fee", "State", "RescuerID", "VideoAmt",
    "Description", "PetID", "PhotoAmt", "AdoptionSpeed",
]

EXPECTED_TEST_COLUMNS: list[str] = [
    "Type", "Name", "Age", "Breed1", "Breed2", "Gender",
    "Color1", "Color2", "Color3", "MaturitySize", "FurLength",
    "Vaccinated", "Dewormed", "Sterilized", "Health",
    "Quantity", "Fee", "State", "RescuerID", "VideoAmt",
    "Description", "PetID", "PhotoAmt",
]


# ── Public API ──────────────────────────────────────────────────────


def load_tabular(split: str) -> pd.DataFrame:
    """Load ``train.csv`` or ``test.csv`` as a pandas DataFrame.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    pd.DataFrame
        Raw tabular data with original column names and dtypes.

    Raises
    ------
    ValueError
        If *split* is not ``"train"`` or ``"test"``.
    FileNotFoundError
        If the CSV file does not exist.
    """
    split = split.lower().strip()
    if split not in ("train", "test"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'.")

    csv_path = cfg.RAW_TRAIN_CSV if split == "train" else cfg.RAW_TEST_CSV

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    logger.info("Loading %s data from %s", split, csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded %s: %d rows × %d columns", split, *df.shape)
    return df


def load_reference_table(name: str) -> pd.DataFrame:
    """Load a reference CSV as a DataFrame.

    Parameters
    ----------
    name : str
        One of ``"breed"``, ``"color"``, ``"state"``.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If *name* is not recognised.
    FileNotFoundError
        If the CSV file does not exist.
    """
    name = name.lower().strip()
    path_map = {
        "breed": cfg.REF_BREED_LABELS,
        "color": cfg.REF_COLOR_LABELS,
        "state": cfg.REF_STATE_LABELS,
    }

    if name not in path_map:
        raise ValueError(
            f"Unknown reference table '{name}'. Must be one of {set(path_map)}."
        )

    path = path_map[name]
    if not path.exists():
        raise FileNotFoundError(f"Reference CSV not found: {path}")

    logger.info("Loading reference table '%s' from %s", name, path)
    df = pd.read_csv(path)
    logger.info("Loaded '%s': %d rows × %d columns", name, *df.shape)
    return df


def load_sentiment_json(
    pet_id: str,
    index: Optional[int] = None,
    split: str = "train",
) -> dict[str, Any]:
    """Load a single sentiment JSON file.

    Sentiment files are named ``{PetID}.json`` (one per listing).

    Parameters
    ----------
    pet_id : str
        Unique pet listing ID.
    index : int, optional
        Kept for API consistency but unused — sentiment is per-listing.
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    base = cfg.RAW_TRAIN_SENTIMENT if split == "train" else cfg.RAW_TEST_SENTIMENT
    path = base / f"{pet_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Sentiment JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata_json(
    pet_id: str,
    index: int = 1,
    split: str = "train",
) -> dict[str, Any]:
    """Load a single metadata (Google Vision API) JSON file.

    Parameters
    ----------
    pet_id : str
        Unique pet listing ID.
    index : int
        Image index (1-based).
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    dict
        Parsed JSON content.
    """
    base = cfg.RAW_TRAIN_METADATA if split == "train" else cfg.RAW_TEST_METADATA
    path = base / f"{pet_id}-{index}.json"
    if not path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image(
    pet_id: str,
    index: int = 1,
    split: str = "train",
) -> Image.Image:
    """Load a single pet image as a PIL Image.

    Parameters
    ----------
    pet_id : str
        Unique pet listing ID.
    index : int
        Image index (1-based).
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    PIL.Image.Image
        Loaded image object.
    """
    base = cfg.RAW_TRAIN_IMAGES if split == "train" else cfg.RAW_TEST_IMAGES
    path = base / f"{pet_id}-{index}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path)


def get_file_inventory(directory: Path | str) -> pd.DataFrame:
    """Recursively list files in a directory with counts and sizes.

    Parameters
    ----------
    directory : Path or str
        Root directory to inventory.

    Returns
    -------
    pd.DataFrame
        Columns: ``subdirectory``, ``extension``, ``file_count``,
        ``total_size_mb``.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            # Relative to the inventory root
            rel = path.relative_to(directory)
            subdir = str(rel.parent) if str(rel.parent) != "." else "(root)"
            records.append(
                {
                    "subdirectory": subdir,
                    "extension": path.suffix.lower(),
                    "size_bytes": path.stat().st_size,
                }
            )

    if not records:
        return pd.DataFrame(
            columns=["subdirectory", "extension", "file_count", "total_size_mb"]
        )

    df = pd.DataFrame(records)
    summary = (
        df.groupby(["subdirectory", "extension"])
        .agg(
            file_count=("size_bytes", "count"),
            total_size_mb=("size_bytes", lambda s: round(s.sum() / 1_048_576, 2)),
        )
        .reset_index()
        .sort_values(["subdirectory", "extension"])
        .reset_index(drop=True)
    )
    return summary


# ── Batch loaders ───────────────────────────────────────────────────


def load_all_sentiment_jsons(
    split: str = "train",
    *,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Batch-load all sentiment JSONs for a given split.

    Each returned dict contains the parsed JSON content plus
    a ``PetID`` key extracted from the filename.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    progress : bool
        If ``True``, log progress every 5 000 files.

    Returns
    -------
    list[dict[str, Any]]
        One dict per JSON file, enriched with ``PetID``.
    """
    base = cfg.RAW_TRAIN_SENTIMENT if split == "train" else cfg.RAW_TEST_SENTIMENT
    if not base.exists():
        raise FileNotFoundError(f"Sentiment directory not found: {base}")

    json_files = sorted(base.glob("*.json"))
    logger.info("Found %d sentiment JSON files in %s", len(json_files), base)

    records: list[dict[str, Any]] = []
    for idx, fp in enumerate(json_files, 1):
        pet_id = fp.stem  # e.g. "abc123"
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["PetID"] = pet_id
            records.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse %s: %s", fp, exc)

        if progress and idx % 5_000 == 0:
            logger.info("  … loaded %d / %d sentiment JSONs", idx, len(json_files))

    logger.info("Loaded %d sentiment records for split='%s'", len(records), split)
    return records


def parse_sentiment_to_dataframe(
    sentiment_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert raw sentiment JSON records into a structured DataFrame.

    Extracts document-level sentiment, sentence-level aggregates, and
    entity information from Google NLP API output.

    Parameters
    ----------
    sentiment_records : list[dict]
        Output of :func:`load_all_sentiment_jsons`.

    Returns
    -------
    pd.DataFrame
        One row per PetID with columns:
        ``PetID``, ``doc_sentiment_score``, ``doc_sentiment_magnitude``,
        ``sentence_count``, ``mean_sentence_score``, ``mean_sentence_magnitude``,
        ``entity_count``, ``language``.
    """
    rows: list[dict[str, Any]] = []
    for rec in sentiment_records:
        pet_id = rec.get("PetID", "")

        # ── Document-level sentiment ────────────────────────────────
        doc_sent = rec.get("documentSentiment", {})
        doc_score = doc_sent.get("score", 0.0)
        doc_magnitude = doc_sent.get("magnitude", 0.0)

        # ── Sentence-level aggregation ──────────────────────────────
        sentences = rec.get("sentences", [])
        sent_scores = [s.get("sentiment", {}).get("score", 0.0) for s in sentences]
        sent_magnitudes = [s.get("sentiment", {}).get("magnitude", 0.0) for s in sentences]
        mean_sent_score = float(np.mean(sent_scores)) if sent_scores else 0.0
        mean_sent_mag = float(np.mean(sent_magnitudes)) if sent_magnitudes else 0.0

        # ── Entity count ────────────────────────────────────────────
        entities = rec.get("entities", [])

        # ── Language ────────────────────────────────────────────────
        language = rec.get("language", "")

        rows.append(
            {
                "PetID": pet_id,
                "doc_sentiment_score": round(doc_score, 4),
                "doc_sentiment_magnitude": round(doc_magnitude, 4),
                "sentence_count": len(sentences),
                "mean_sentence_score": round(mean_sent_score, 4),
                "mean_sentence_magnitude": round(mean_sent_mag, 4),
                "entity_count": len(entities),
                "language": language,
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Parsed sentiment DataFrame: %d rows × %d cols", *df.shape)
    return df


def load_all_metadata_jsons(
    split: str = "train",
    *,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Batch-load all metadata JSONs for a given split.

    Each returned dict contains the parsed JSON content plus
    ``PetID`` and ``ImageIndex`` keys.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.
    progress : bool
        If ``True``, log progress every 10 000 files.

    Returns
    -------
    list[dict[str, Any]]
        One dict per JSON file, enriched with ``PetID`` and ``ImageIndex``.
    """
    base = cfg.RAW_TRAIN_METADATA if split == "train" else cfg.RAW_TEST_METADATA
    if not base.exists():
        raise FileNotFoundError(f"Metadata directory not found: {base}")

    json_files = sorted(base.glob("*.json"))
    logger.info("Found %d metadata JSON files in %s", len(json_files), base)

    records: list[dict[str, Any]] = []
    for idx, fp in enumerate(json_files, 1):
        stem = fp.stem  # e.g. "abc123-2"
        parts = stem.rsplit("-", 1)
        pet_id = parts[0]
        img_idx = int(parts[1]) if len(parts) == 2 else 1

        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            data["PetID"] = pet_id
            data["ImageIndex"] = img_idx
            records.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to parse %s: %s", fp, exc)

        if progress and idx % 10_000 == 0:
            logger.info("  … loaded %d / %d metadata JSONs", idx, len(json_files))

    logger.info("Loaded %d metadata records for split='%s'", len(records), split)
    return records


def parse_metadata_to_dataframe(
    metadata_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Convert raw metadata JSON records into a structured DataFrame.

    Extracts label annotations, dominant-color summaries, and crop-hint
    confidence from Google Vision API output.

    Parameters
    ----------
    metadata_records : list[dict]
        Output of :func:`load_all_metadata_jsons`.

    Returns
    -------
    pd.DataFrame
        One row per image with columns:
        ``PetID``, ``ImageIndex``, ``label_count``, ``top_label``,
        ``top_label_score``, ``labels_concat``, ``dominant_color_count``,
        ``avg_brightness``, ``color_diversity``, ``crop_confidence``.
    """
    rows: list[dict[str, Any]] = []
    for rec in metadata_records:
        pet_id = rec.get("PetID", "")
        img_idx = rec.get("ImageIndex", 1)

        # ── Label annotations ───────────────────────────────────────
        labels = rec.get("labelAnnotations", [])
        label_descs = [la.get("description", "") for la in labels]
        label_scores = [la.get("score", 0.0) for la in labels]
        top_label = label_descs[0] if label_descs else ""
        top_label_score = label_scores[0] if label_scores else 0.0

        # ── Dominant colors ─────────────────────────────────────────
        color_info = (
            rec.get("imagePropertiesAnnotation", {})
            .get("dominantColors", {})
            .get("colors", [])
        )
        dom_color_count = len(color_info)
        brightnesses: list[float] = []
        for ci in color_info:
            c = ci.get("color", {})
            r, g, b = c.get("red", 0), c.get("green", 0), c.get("blue", 0)
            brightnesses.append((r + g + b) / 3.0)
        avg_brightness = float(np.mean(brightnesses)) if brightnesses else 0.0
        color_diversity = float(np.std(brightnesses)) if len(brightnesses) > 1 else 0.0

        # ── Crop hints ──────────────────────────────────────────────
        crop_hints = (
            rec.get("cropHintsAnnotation", {})
            .get("cropHints", [])
        )
        crop_confidence = (
            crop_hints[0].get("confidence", 0.0) if crop_hints else 0.0
        )

        rows.append(
            {
                "PetID": pet_id,
                "ImageIndex": img_idx,
                "label_count": len(labels),
                "top_label": top_label,
                "top_label_score": top_label_score,
                "labels_concat": "|".join(label_descs),
                "dominant_color_count": dom_color_count,
                "avg_brightness": round(avg_brightness, 2),
                "color_diversity": round(color_diversity, 2),
                "crop_confidence": round(crop_confidence, 6),
            }
        )

    df = pd.DataFrame(rows)
    logger.info("Parsed metadata DataFrame: %d rows × %d cols", *df.shape)
    return df


def get_image_paths_for_pet(
    pet_id: str,
    split: str = "train",
) -> list[Path]:
    """Return a sorted list of image file paths for a given PetID.

    Parameters
    ----------
    pet_id : str
        Unique pet listing ID.
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    list[Path]
        Sorted paths to all matching JPEG files.
    """
    base = cfg.RAW_TRAIN_IMAGES if split == "train" else cfg.RAW_TEST_IMAGES
    return sorted(base.glob(f"{pet_id}-*.jpg"))


def compute_image_stats(
    image_paths: list[Path | str],
) -> pd.DataFrame:
    """Compute image statistics for a list of image paths.

    Statistics include width, height, aspect ratio, file size,
    average brightness, and blur score (Laplacian variance via NumPy).

    Parameters
    ----------
    image_paths : list[Path | str]
        Paths to JPEG image files.

    Returns
    -------
    pd.DataFrame
        One row per image with columns: ``path``, ``pet_id``,
        ``image_index``, ``width``, ``height``, ``aspect_ratio``,
        ``file_size_kb``, ``brightness``, ``blur_score``.
    """
    from scipy.ndimage import laplace  # local import to keep top-level light

    rows: list[dict[str, Any]] = []
    for p in image_paths:
        p = Path(p)
        stem = p.stem
        parts = stem.rsplit("-", 1)
        pet_id = parts[0]
        img_idx = int(parts[1]) if len(parts) == 2 else 1

        try:
            img = Image.open(p)
            w, h = img.size
            file_size_kb = round(p.stat().st_size / 1024, 2)

            arr = np.array(img.convert("L"), dtype=np.float64)
            brightness = float(arr.mean())
            lap = laplace(arr)
            blur_score = float(lap.var())

            rows.append(
                {
                    "path": str(p),
                    "pet_id": pet_id,
                    "image_index": img_idx,
                    "width": w,
                    "height": h,
                    "aspect_ratio": round(w / h, 4) if h > 0 else 0.0,
                    "file_size_kb": file_size_kb,
                    "brightness": round(brightness, 2),
                    "blur_score": round(blur_score, 2),
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to process image %s: %s", p, exc)

    df = pd.DataFrame(rows)
    logger.info("Computed image stats for %d images", len(df))
    return df


# ── Parquet I/O ─────────────────────────────────────────────────────


def save_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    """Save a DataFrame to Parquet using PyArrow with Snappy compression.

    Creates parent directories if they do not exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to persist.
    path : Path | str
        Destination path (should end in ``.parquet``).

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    size_mb = round(path.stat().st_size / 1_048_576, 2)
    logger.info("Saved Parquet: %s (%d rows, %.2f MB)", path, len(df), size_mb)
    return path


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load a Parquet file into a pandas DataFrame.

    Parameters
    ----------
    path : Path | str
        Path to the ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    logger.info("Loaded Parquet: %s (%d rows, %d cols)", path, *df.shape)
    return df
