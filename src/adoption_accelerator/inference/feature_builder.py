"""
Feature vector construction from a PredictionRequest at inference time.

Transforms a raw ``PredictionRequest`` into the 940-dimensional feature
vector expected by the trained model, reproducing the same feature
engineering logic used during training.

The module handles:
  - Tabular feature engineering from ``TabularInput`` fields
  - Text statistics and embedding extraction
  - Image embedding extraction and PCA reduction
  - Sentiment / metadata feature defaults
  - Graceful degradation when modalities are missing

Pre-computed training statistics (breed frequency maps, rescuer
aggregates, state frequency maps) are loaded from the training
feature Parquet and cached on first use.
"""

from __future__ import annotations

import functools
import logging
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from adoption_accelerator import config as cfg
from adoption_accelerator.inference.contracts import PredictionRequest

logger = logging.getLogger(__name__)


# =====================================================================
#  Default fill values for missing modalities
# =====================================================================

# Text defaults: used when description is empty or missing
TEXT_STATS_DEFAULTS: dict[str, float] = {
    "description_length": 0,
    "word_count": 0,
    "sentence_count": 0,
    "avg_word_length": 0.0,
    "uppercase_ratio": 0.0,
    "punctuation_ratio": 0.0,
    "digit_ratio": 0.0,
}

# Sentiment defaults: Google NLP is not available at inference time
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

# Image defaults: used when no images are provided
IMAGE_AUX_DEFAULTS: dict[str, float] = {
    "has_image_embedding": 0,
    "actual_photo_count": 0,
}

IMAGE_QUALITY_DEFAULTS: dict[str, float] = {
    "mean_image_brightness": 0.0,
    "mean_blur_score": 0.0,
    "image_size_std": 0.0,
}

IMAGE_METADATA_DEFAULTS: dict[str, float] = {
    "meta_label_count_mean": 0.0,
    "meta_top_label_score_mean": 0.0,
    "meta_dominant_color_count_mean": 0.0,
    "meta_avg_brightness_mean": 0.0,
    "meta_color_diversity_mean": 0.0,
    "meta_crop_confidence_mean": 0.0,
}


# =====================================================================
#  Training statistics (cached)
# =====================================================================


@functools.lru_cache(maxsize=1)
def _load_training_statistics() -> dict[str, Any]:
    """Load and cache training-set statistics needed for feature engineering.

    Computes breed frequency maps, state frequency maps, and rescuer
    aggregate defaults from the training data Parquet.  These are
    computed once and reused across all inference requests.
    """
    from adoption_accelerator.features.tabular import compute_rescuer_aggregates

    train_path = cfg.DATA_FEATURES / "tabular" / "v1" / "train.parquet"
    if not train_path.exists():
        logger.warning(
            "Training Parquet not found at %s; "
            "using global defaults for frequency features.",
            train_path,
        )
        return _fallback_statistics()

    train_df = pd.read_parquet(train_path)

    # Breed frequency map (from Breed1 column, if available)
    # The tabular train Parquet already has engineered features.
    # We need original Breed1 values to compute frequency encoding.
    # However, Breed1 is already in the Parquet as a raw column.
    breed_freq_map = train_df["Breed1"].value_counts(normalize=True)

    # State frequency: tabular Parquet has 'state_freq' already computed.
    # We reconstruct the map from state_freq values if State column isn't available.
    # Since the Parquet has state_freq already, we can extract unique mappings.
    # But PredictionRequest provides State as an int, so we need State -> freq map.
    # The tabular features Parquet may not have State; check first.
    state_freq_map = None
    if "State" in train_df.columns:
        state_freq_map = train_df["State"].value_counts(normalize=True)

    # Rescuer aggregates: use defaults (global means) since inference
    # requests don't come with RescuerID context.
    # Compute mean of the rescuer columns already in training data.
    rescuer_defaults = {}
    for col in ["rescuer_pet_count", "rescuer_mean_photo_amt", "rescuer_mean_fee"]:
        if col in train_df.columns:
            rescuer_defaults[col] = float(train_df[col].mean())
        else:
            rescuer_defaults[col] = 1.0  # safe fallback

    # breed1_frequency: if the column already exists, get its global mean as default
    breed_freq_default = 0.0
    if "breed1_frequency" in train_df.columns:
        breed_freq_default = float(train_df["breed1_frequency"].mean())

    # state_freq default
    state_freq_default = 0.0
    if "state_freq" in train_df.columns:
        state_freq_default = float(train_df["state_freq"].mean())

    stats = {
        "breed_freq_map": breed_freq_map,
        "state_freq_map": state_freq_map,
        "rescuer_defaults": rescuer_defaults,
        "breed_freq_default": breed_freq_default,
        "state_freq_default": state_freq_default,
    }

    logger.info("Training statistics loaded and cached.")
    return stats


def _fallback_statistics() -> dict[str, Any]:
    """Return safe global defaults when training data is not accessible."""
    return {
        "breed_freq_map": pd.Series(dtype=np.float64),
        "state_freq_map": None,
        "rescuer_defaults": {
            "rescuer_pet_count": 1.0,
            "rescuer_mean_photo_amt": 3.0,
            "rescuer_mean_fee": 0.0,
        },
        "breed_freq_default": 0.001,
        "state_freq_default": 0.02,
    }


# =====================================================================
#  Tabular features from PredictionRequest
# =====================================================================


def _build_tabular_features(request: PredictionRequest) -> dict[str, float]:
    """Transform PredictionRequest.tabular into all 45 tabular features.

    Reproduces the exact engineering steps from
    ``features/tabular.py::engineer_tabular_features``.
    """
    t = request.tabular
    stats = _load_training_statistics()

    features: dict[str, float] = {}

    # Binary / encoded
    features["is_dog"] = 1 if t.type == 1 else 0
    features["Gender"] = t.gender

    # Ordinal
    features["MaturitySize"] = t.maturity_size
    features["FurLength"] = t.fur_length
    features["Health"] = t.health

    # Care features (recoded: 1=Yes->1, 2=No->0, 3=Not Sure->-1)
    care_map = {1: 1, 2: 0, 3: -1}
    features["Vaccinated"] = care_map.get(t.vaccinated, 0)
    features["Dewormed"] = care_map.get(t.dewormed, 0)
    features["Sterilized"] = care_map.get(t.sterilized, 0)

    # Health care score (before recoding, count of Yes=1)
    features["health_care_score"] = sum(
        1 for v in [t.vaccinated, t.dewormed, t.sterilized] if v == 1
    )

    # Numeric transforms
    features["Age"] = t.age
    features["log_age"] = float(np.log1p(t.age))

    # Age bin
    age = t.age
    age_bins = [0, 1, 3, 6, 12, 24, 60, 255]
    age_bin = 6  # default: senior
    for i in range(len(age_bins) - 1):
        if age_bins[i] <= age <= age_bins[i + 1]:
            age_bin = i
            break
    features["age_bin"] = age_bin

    features["Fee"] = t.fee
    features["log_fee"] = float(np.log1p(t.fee))
    features["is_free"] = 1 if t.fee == 0 else 0
    features["fee_per_pet"] = float(t.fee / max(t.quantity, 1))

    features["Quantity"] = t.quantity
    features["log_quantity"] = float(np.log1p(t.quantity))
    features["is_single_pet"] = 1 if t.quantity == 1 else 0

    features["PhotoAmt"] = len(request.images) if request.images else 0
    features["log_photo_amt"] = float(np.log1p(features["PhotoAmt"]))
    features["has_photos"] = 1 if features["PhotoAmt"] > 0 else 0

    features["VideoAmt"] = t.video_amt
    features["has_video"] = 1 if t.video_amt > 0 else 0

    # Name features
    has_name = 1 if t.name and t.name.strip() else 0
    features["has_name"] = has_name
    if has_name:
        features["name_length"] = len(t.name)
        features["name_word_count"] = len(t.name.split())
    else:
        features["name_length"] = 0
        features["name_word_count"] = 0

    # Breed features
    breed2 = t.breed2 or 0
    features["Breed1"] = t.breed1
    features["Breed2"] = breed2
    features["is_mixed_breed"] = 1 if breed2 != 0 else 0
    features["breed_count"] = 1 + (1 if breed2 != 0 else 0)

    # Breed frequency encoding
    breed_freq_map = stats["breed_freq_map"]
    if len(breed_freq_map) > 0 and t.breed1 in breed_freq_map.index:
        features["breed1_frequency"] = float(breed_freq_map[t.breed1])
    else:
        features["breed1_frequency"] = stats["breed_freq_default"]

    # Color features
    color2 = t.color2 or 0
    color3 = t.color3 or 0
    features["Color1"] = t.color1
    features["Color2"] = color2
    features["Color3"] = color3
    features["color_count"] = sum(1 for c in [t.color1, color2, color3] if c != 0)
    features["breed2_missing"] = 1 if breed2 == 0 else 0
    features["color2_missing"] = 1 if color2 == 0 else 0
    features["color3_missing"] = 1 if color3 == 0 else 0

    # Interaction features
    features["age_x_type"] = t.age * features["is_dog"]
    features["health_x_vaccinated"] = t.health * features["Vaccinated"]

    # Rescuer aggregates: use global defaults (no RescuerID in PredictionRequest)
    rescuer_defaults = stats["rescuer_defaults"]
    features["rescuer_pet_count"] = rescuer_defaults["rescuer_pet_count"]
    features["rescuer_mean_photo_amt"] = rescuer_defaults["rescuer_mean_photo_amt"]
    features["rescuer_mean_fee"] = rescuer_defaults["rescuer_mean_fee"]

    # State frequency encoding
    state_freq_map = stats["state_freq_map"]
    if state_freq_map is not None and t.state in state_freq_map.index:
        features["state_freq"] = float(state_freq_map[t.state])
    else:
        features["state_freq"] = stats["state_freq_default"]

    return features


# =====================================================================
#  Text features from PredictionRequest
# =====================================================================


def _compute_text_statistics_single(description: str) -> dict[str, float]:
    """Compute text statistics for a single description string."""
    if not description or not description.strip():
        return dict(TEXT_STATS_DEFAULTS)

    text = description.strip()
    length = len(text)
    words = text.split()
    word_count = len(words)

    # Sentence count
    sentences = re.split(r"[.!?]+", text)
    sentence_count = max(len([s for s in sentences if s.strip()]), 1)

    # Average word length
    avg_word_length = np.mean([len(w) for w in words]) if words else 0.0

    # Character ratios
    total_chars = max(length, 1)
    uppercase_ratio = sum(1 for c in text if c.isupper()) / total_chars
    punctuation_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / total_chars
    digit_ratio = sum(1 for c in text if c.isdigit()) / total_chars

    return {
        "description_length": length,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_word_length": float(avg_word_length),
        "uppercase_ratio": float(uppercase_ratio),
        "punctuation_ratio": float(punctuation_ratio),
        "digit_ratio": float(digit_ratio),
    }


def _extract_text_embeddings(description: str) -> np.ndarray:
    """Extract 768-dim text embedding using sentence-transformers.

    Returns zeros if the model cannot be loaded or the description
    is empty.
    """
    if not description or not description.strip():
        return np.zeros(768, dtype=np.float32)

    try:
        from sentence_transformers import SentenceTransformer

        model = _get_text_embedding_model()
        embedding = model.encode([description], show_progress_bar=False)
        return np.asarray(embedding[0], dtype=np.float32)
    except Exception as exc:
        logger.warning("Text embedding extraction failed: %s", exc)
        return np.zeros(768, dtype=np.float32)


@functools.lru_cache(maxsize=1)
def _get_text_embedding_model() -> Any:
    """Load and cache the sentence-transformer model.

    Uses ``all-mpnet-base-v2`` to match the training pipeline
    (768-dimensional embeddings).
    """
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/all-mpnet-base-v2"
    logger.info("Loading text embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    return model


# =====================================================================
#  Image features from PredictionRequest
# =====================================================================


@functools.lru_cache(maxsize=1)
def _get_image_pca() -> Any:
    """Load and cache the image PCA reducer."""
    pca_path = cfg.ARTIFACTS_DIR / "models" / "image_pca_v1.joblib"
    if not pca_path.exists():
        logger.warning("Image PCA not found at %s", pca_path)
        return None
    pca = joblib.load(pca_path)
    logger.info("Image PCA loaded from %s", pca_path)
    return pca


def _extract_image_features(image_paths: list[str]) -> dict[str, Any]:
    """Extract image features from a list of image file paths.

    Returns a dict with:
    - ``embeddings``: 100-dim PCA-reduced mean embedding (or zeros)
    - ``quality``: image quality features
    - ``metadata``: Vision API metadata defaults (not available at inference)
    - ``aux``: auxiliary features (has_image_embedding, actual_photo_count)
    """
    result: dict[str, Any] = {
        "embeddings": np.zeros(100, dtype=np.float32),
        "quality": dict(IMAGE_QUALITY_DEFAULTS),
        "metadata": dict(IMAGE_METADATA_DEFAULTS),
        "aux": {
            "has_image_embedding": 0,
            "actual_photo_count": 0,
        },
    }

    if not image_paths:
        return result

    valid_paths = [p for p in image_paths if Path(p).exists()]
    if not valid_paths:
        result["aux"]["actual_photo_count"] = len(image_paths)
        return result

    result["aux"]["has_image_embedding"] = 1
    result["aux"]["actual_photo_count"] = len(valid_paths)

    try:
        # Extract raw embeddings
        raw_embeddings = _extract_raw_image_embeddings(valid_paths)
        if raw_embeddings is not None and len(raw_embeddings) > 0:
            # Mean pool across images
            mean_embedding = np.mean(raw_embeddings, axis=0, keepdims=True)

            # PCA reduce to 100 dims
            pca = _get_image_pca()
            if pca is not None:
                reduced = pca.transform(mean_embedding)
                result["embeddings"] = reduced[0].astype(np.float32)

            # Compute quality features
            result["quality"] = _compute_image_quality(valid_paths)
    except Exception as exc:
        logger.warning("Image feature extraction failed: %s", exc)

    return result


def _extract_raw_image_embeddings(image_paths: list[str]) -> np.ndarray | None:
    """Extract raw embeddings from images using pretrained backbone."""
    try:
        import torch
        from torchvision import models, transforms
        from PIL import Image

        # Load model (EfficientNet or ResNet as fallback)
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            model.classifier = torch.nn.Identity()
        except Exception:
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            model.fc = torch.nn.Identity()

        model.eval()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        embeddings = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB")
                tensor = transform(img).unsqueeze(0)
                with torch.no_grad():
                    emb = model(tensor).squeeze().numpy()
                embeddings.append(emb)
            except Exception as exc:
                logger.warning("Failed to process image %s: %s", path, exc)

        if embeddings:
            return np.stack(embeddings)
        return None
    except ImportError:
        logger.warning("torch/torchvision not available for image embedding extraction")
        return None


def _compute_image_quality(image_paths: list[str]) -> dict[str, float]:
    """Compute image quality features (brightness, blur, size)."""
    try:
        import cv2

        brightnesses = []
        blur_scores = []
        areas = []

        for path in image_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                areas.append(float(h * w))

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightnesses.append(float(np.mean(gray)))
                blur_scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
            except Exception:
                continue

        if brightnesses:
            return {
                "mean_image_brightness": float(np.mean(brightnesses)),
                "mean_blur_score": float(np.mean(blur_scores)),
                "image_size_std": float(np.std(areas)) if len(areas) > 1 else 0.0,
            }
    except ImportError:
        logger.warning("cv2 not available for image quality computation")

    return dict(IMAGE_QUALITY_DEFAULTS)


# =====================================================================
#  Feature vector assembly
# =====================================================================


def build_feature_vector(
    request: PredictionRequest,
    feature_schema: list[str],
) -> np.ndarray:
    """Transform a PredictionRequest into a 940-dim feature vector.

    Assembles features from all modalities (tabular, text, image,
    metadata/sentiment), applying the same engineering logic used
    during training.

    Parameters
    ----------
    request : PredictionRequest
        Raw prediction request with tabular, description, and image data.
    feature_schema : list[str]
        Ordered list of 940 feature names from the model bundle's
        ``feature_schema.json``.

    Returns
    -------
    np.ndarray of shape (1, 940)
        Feature vector ready for model prediction.

    Raises
    ------
    ValueError
        If the resulting vector does not match the expected schema size.
    """
    n_features = len(feature_schema)

    # Step 1: Build tabular features
    tabular = _build_tabular_features(request)

    # Step 2: Build text features
    text_stats = _compute_text_statistics_single(request.description)
    text_embeddings = _extract_text_embeddings(request.description)

    # Step 3: Build sentiment defaults (NLP API not available at inference time)
    sentiment = dict(SENTIMENT_DEFAULTS)

    # Step 4: Build image features
    image_result = _extract_image_features(request.images)
    image_embeddings = image_result["embeddings"]
    image_quality = image_result["quality"]
    image_metadata = image_result["metadata"]
    image_aux = image_result["aux"]

    # Step 5: Assemble all features into a dict
    all_features: dict[str, float] = {}
    all_features.update(tabular)
    all_features.update(text_stats)
    all_features.update(sentiment)
    all_features.update(image_aux)
    all_features.update(image_quality)
    all_features.update(image_metadata)

    # Add text embeddings
    for i in range(768):
        all_features[f"text_emb_{i}"] = float(text_embeddings[i])

    # Add image embeddings
    for i in range(100):
        all_features[f"img_emb_{i}"] = float(image_embeddings[i])

    # Step 6: Assemble in schema order
    vector = np.zeros((1, n_features), dtype=np.float64)
    missing_features = []
    for j, feat_name in enumerate(feature_schema):
        if feat_name in all_features:
            vector[0, j] = all_features[feat_name]
        else:
            missing_features.append(feat_name)

    if missing_features:
        logger.warning(
            "Missing %d features (filled with 0): %s",
            len(missing_features),
            missing_features[:10],
        )

    logger.info(
        "Feature vector built: shape=%s, non-zero=%d/%d",
        vector.shape,
        int(np.count_nonzero(vector)),
        n_features,
    )
    return vector


def build_feature_vector_degraded(
    request: PredictionRequest,
    feature_schema: list[str],
    exclude_modalities: set[str] | None = None,
) -> np.ndarray:
    """Build a feature vector with graceful degradation for missing modalities.

    This is a convenience wrapper around ``build_feature_vector`` that
    explicitly zeroes out features from specified modalities, simulating
    missing data scenarios.

    Parameters
    ----------
    request : PredictionRequest
        Raw prediction request.
    feature_schema : list[str]
        Ordered feature names from the model schema.
    exclude_modalities : set[str] | None
        Modalities to exclude (fill with zeros).  Valid values:
        ``"text"``, ``"image"``, ``"metadata"``.

    Returns
    -------
    np.ndarray of shape (1, 940)
    """
    # Build the full vector first
    vector = build_feature_vector(request, feature_schema)

    if not exclude_modalities:
        return vector

    # Zero out features from excluded modalities
    from adoption_accelerator.inference.explain import build_modality_map
    modality_map = build_modality_map(feature_schema)

    for j, feat_name in enumerate(feature_schema):
        modality = modality_map.get(feat_name, "tabular")
        if modality in exclude_modalities:
            vector[0, j] = 0.0

    return vector
