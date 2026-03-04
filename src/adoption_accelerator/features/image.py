"""
Image feature extraction functions for Adoption Accelerator.

Provides pretrained backbone loading, image preprocessing, batched
embedding extraction, per-PetID aggregation, image quality feature
computation, and optional dimensionality reduction via PCA.

All functions are designed for train/test parity and inference
reproducibility. Functions that require fitted artifacts (PCA) accept
those artifacts as explicit arguments -- they **never** compute them
internally from the input data.

Functions
---------
load_pretrained_backbone(model_name, device)
    Load a pretrained image backbone and return model + transform.
preprocess_image(image_path, transform)
    Load and preprocess a single image, returning a tensor.
extract_image_embeddings_batch(image_paths, model, transform, batch_size, device)
    Extract embeddings for a list of image paths in batches.
aggregate_embeddings_per_pet(embeddings_dict, pet_ids, embedding_dim, strategy)
    Aggregate per-image embeddings into per-PetID vectors.
compute_image_quality_features(pet_ids, split)
    Compute brightness, blur score, and size statistics per PetID.
reduce_image_dimensions(embeddings_train, embeddings_test, n_components)
    Fit PCA on train, transform both splits.
extract_image_features(pet_ids, split, config, fitted_artifacts)
    Orchestrate all image feature extraction steps.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from adoption_accelerator import config as cfg

logger = logging.getLogger("adoption_accelerator")


# =====================================================================
#  Constants & Column Definitions
# =====================================================================

# Image quality feature columns
IMAGE_QUALITY_COLUMNS: list[str] = [
    "mean_image_brightness",
    "mean_blur_score",
    "image_size_std",
]

# Metadata feature columns (from aggregate_metadata_features)
IMAGE_METADATA_COLUMNS: list[str] = [
    "meta_label_count_mean",
    "meta_top_label_score_mean",
    "meta_dominant_color_count_mean",
    "meta_avg_brightness_mean",
    "meta_color_diversity_mean",
    "meta_crop_confidence_mean",
]

# Auxiliary feature columns
IMAGE_AUX_COLUMNS: list[str] = [
    "has_image_embedding",
    "actual_photo_count",
]

# Feature descriptions for schema registration
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Auxiliary
    "has_image_embedding": "Binary flag: 1 if PetID has at least one image, 0 otherwise",
    "actual_photo_count": "Number of image files found on disk for PetID",
    # Quality
    "mean_image_brightness": "Mean brightness across all images for PetID (0-255 scale)",
    "mean_blur_score": "Mean Laplacian variance (blur score) across images for PetID",
    "image_size_std": "Standard deviation of image areas (width*height) across images",
    # Metadata
    "meta_label_count_mean": "Mean number of Vision API labels per image for PetID",
    "meta_top_label_score_mean": "Mean top-label confidence score per image for PetID",
    "meta_dominant_color_count_mean": "Mean dominant color count per image for PetID",
    "meta_avg_brightness_mean": "Mean Vision API brightness proxy per image for PetID",
    "meta_color_diversity_mean": "Mean color diversity (std of brightness) per image for PetID",
    "meta_crop_confidence_mean": "Mean crop-hint confidence per image for PetID",
}


# =====================================================================
#  Device Utilities
# =====================================================================


def _resolve_device(device: str | None = None) -> str:
    """Resolve the compute device for inference.

    Priority: CUDA > MPS > CPU.

    Parameters
    ----------
    device : str or None
        Explicit device string (``"cuda"``, ``"cpu"``, ``"mps"``).
        If ``None``, auto-detect.

    Returns
    -------
    str
        Resolved device string.
    """
    import torch

    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =====================================================================
#  Pretrained Backbone Loading
# =====================================================================


def load_pretrained_backbone(
    model_name: str = "efficientnet_v2_s",
    device: str | None = None,
) -> tuple[Any, Any, int]:
    """Load a pretrained image backbone for feature extraction.

    Removes the classification head and sets the model to evaluation
    mode. Returns the model, its preprocessing transform, and the
    output feature dimension.

    Parameters
    ----------
    model_name : str
        Model identifier. Supported: ``"efficientnet_v2_s"``,
        ``"efficientnet_b0"``, ``"resnet50"``.
    device : str or None
        Target device. If ``None``, auto-detects.

    Returns
    -------
    tuple of (nn.Module, transforms.Compose, int)
        - Model with classification head removed, in eval mode.
        - Preprocessing transform pipeline (resize, normalize).
        - Output feature dimension.

    Raises
    ------
    ValueError
        If ``model_name`` is not supported.
    """
    import torch
    import torchvision.models as models
    import torchvision.transforms as T

    device = _resolve_device(device)

    # -- Model registry --------------------------------------------------------
    model_registry: dict[str, dict[str, Any]] = {
        "efficientnet_v2_s": {
            "factory": lambda: models.efficientnet_v2_s(
                weights=models.EfficientNet_V2_S_Weights.DEFAULT,
            ),
            "input_size": 384,
            "feature_dim": 1280,
        },
        "efficientnet_b0": {
            "factory": lambda: models.efficientnet_b0(
                weights=models.EfficientNet_B0_Weights.DEFAULT,
            ),
            "input_size": 224,
            "feature_dim": 1280,
        },
        "resnet50": {
            "factory": lambda: models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT,
            ),
            "input_size": 224,
            "feature_dim": 2048,
        },
    }

    if model_name not in model_registry:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Must be one of: {list(model_registry.keys())}"
        )

    spec = model_registry[model_name]
    input_size = spec["input_size"]
    feature_dim = spec["feature_dim"]

    logger.info("Loading pretrained model: %s (device=%s)", model_name, device)
    full_model = spec["factory"]()

    # -- Remove classification head -------------------------------------------
    if model_name.startswith("efficientnet"):
        # EfficientNet: features are in model.features + model.avgpool
        # classifier is model.classifier
        full_model.classifier = torch.nn.Identity()
    elif model_name == "resnet50":
        full_model.fc = torch.nn.Identity()
    else:
        raise ValueError(f"Head removal not configured for: {model_name}")

    full_model = full_model.to(device)
    full_model.eval()

    # -- Preprocessing transform -----------------------------------------------
    transform = T.Compose(
        [
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    logger.info(
        "Model loaded: %s | input=%dx%d | feature_dim=%d | device=%s",
        model_name,
        input_size,
        input_size,
        feature_dim,
        device,
    )
    return full_model, transform, feature_dim


# =====================================================================
#  Image Preprocessing
# =====================================================================


def preprocess_image(
    image_path: Path | str,
    transform: Any,
) -> tuple[Any | None, bool]:
    """Load a single image and apply the preprocessing pipeline.

    Parameters
    ----------
    image_path : Path or str
        Path to the JPEG image file.
    transform : torchvision.transforms.Compose
        Preprocessing transform pipeline.

    Returns
    -------
    tuple of (Tensor or None, bool)
        - Preprocessed image tensor in CHW format, or ``None`` on failure.
        - Success flag (``True`` if loaded and preprocessed successfully).
    """
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGB")
        tensor = transform(img)
        return tensor, True
    except Exception as exc:
        logger.warning("Failed to load/preprocess image %s: %s", image_path, exc)
        return None, False


# =====================================================================
#  Batched Embedding Extraction
# =====================================================================


def extract_image_embeddings_batch(
    image_paths: list[Path | str],
    model: Any,
    transform: Any,
    batch_size: int = 32,
    device: str = "cpu",
    feature_dim: int = 1280,
) -> tuple[np.ndarray, list[bool]]:
    """Extract embeddings for a list of image paths in batches.

    Handles I/O errors gracefully -- failed images produce zero-vector
    embeddings with a ``False`` status flag.

    Parameters
    ----------
    image_paths : list of Path or str
        Paths to image files.
    model : nn.Module
        Pretrained backbone in eval mode.
    transform : transforms.Compose
        Image preprocessing pipeline.
    batch_size : int
        Number of images to process per forward pass.
    device : str
        Device for inference (``"cuda"``, ``"cpu"``).
    feature_dim : int
        Expected output dimension from the backbone.

    Returns
    -------
    tuple of (np.ndarray, list of bool)
        - Embedding array of shape ``(n_images, feature_dim)``.
        - Per-image success flags.
    """
    import torch

    n_images = len(image_paths)
    embeddings = np.zeros((n_images, feature_dim), dtype=np.float32)
    statuses: list[bool] = [False] * n_images

    for batch_start in range(0, n_images, batch_size):
        batch_end = min(batch_start + batch_size, n_images)
        batch_paths = image_paths[batch_start:batch_end]

        # Load and preprocess batch
        tensors: list[Any] = []
        valid_indices: list[int] = []

        for i, path in enumerate(batch_paths):
            global_idx = batch_start + i
            tensor, success = preprocess_image(path, transform)
            if success and tensor is not None:
                tensors.append(tensor)
                valid_indices.append(global_idx)
                statuses[global_idx] = True

        if not tensors:
            continue

        # Stack and forward pass
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            features = model(batch_tensor)

        # Flatten if needed (e.g., (B, C, 1, 1) -> (B, C))
        if features.dim() > 2:
            features = features.flatten(start_dim=1)

        features_np = features.cpu().numpy().astype(np.float32)

        for local_idx, global_idx in enumerate(valid_indices):
            embeddings[global_idx] = features_np[local_idx]

        # Free GPU memory
        del batch_tensor, features
        if device == "cuda":
            torch.cuda.empty_cache()

    return embeddings, statuses


# =====================================================================
#  Image Path Resolution
# =====================================================================


def get_image_paths_for_split(
    pet_ids: list[str] | np.ndarray | pd.Series,
    split: str = "train",
) -> dict[str, list[Path]]:
    """Enumerate all image files for each PetID in a split.

    Parameters
    ----------
    pet_ids : array-like of str
        Unique PetIDs to resolve images for.
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    dict[str, list[Path]]
        Mapping from PetID to sorted list of image file paths.
    """
    base = cfg.RAW_TRAIN_IMAGES if split == "train" else cfg.RAW_TEST_IMAGES
    result: dict[str, list[Path]] = {}

    for pid in pet_ids:
        paths = sorted(base.glob(f"{pid}-*.jpg"))
        result[pid] = paths

    total_images = sum(len(v) for v in result.values())
    n_with_images = sum(1 for v in result.values() if len(v) > 0)
    logger.info(
        "Image path resolution (split=%s): %d PetIDs, %d with images, %d total files",
        split,
        len(result),
        n_with_images,
        total_images,
    )
    return result


# =====================================================================
#  Per-PetID Embedding Aggregation
# =====================================================================


def aggregate_embeddings_per_pet(
    embeddings_dict: dict[str, np.ndarray],
    pet_ids: list[str] | np.ndarray | pd.Series,
    embedding_dim: int,
    strategy: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-image embeddings into per-PetID vectors.

    Parameters
    ----------
    embeddings_dict : dict[str, np.ndarray]
        Mapping from PetID to stacked embedding array of shape
        ``(n_images, embedding_dim)``. May be empty for PetIDs with no
        images.
    pet_ids : array-like of str
        Ordered PetIDs for output alignment.
    embedding_dim : int
        Feature dimension from the backbone.
    strategy : str
        Aggregation strategy: ``"mean"`` or ``"max"``.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        - Aggregated embedding array ``(n_pets, embedding_dim)``.
        - ``has_image_embedding`` binary array ``(n_pets,)``.
        - ``actual_photo_count`` integer array ``(n_pets,)``.
    """
    if isinstance(pet_ids, pd.Series):
        pet_ids = pet_ids.tolist()

    n_pets = len(pet_ids)
    aggregated = np.zeros((n_pets, embedding_dim), dtype=np.float32)
    has_image = np.zeros(n_pets, dtype=np.int32)
    photo_count = np.zeros(n_pets, dtype=np.int32)

    for i, pid in enumerate(pet_ids):
        embs = embeddings_dict.get(pid)
        if embs is not None and len(embs) > 0:
            if strategy == "mean":
                aggregated[i] = embs.mean(axis=0)
            elif strategy == "max":
                aggregated[i] = embs.max(axis=0)
            else:
                raise ValueError(f"Unknown aggregation strategy: {strategy}")
            has_image[i] = 1
            photo_count[i] = len(embs)
        # else: zero-vector default, has_image=0, photo_count=0

    n_with = int(has_image.sum())
    logger.info(
        "Aggregated embeddings (%s): %d PetIDs, %d with images, %d without",
        strategy,
        n_pets,
        n_with,
        n_pets - n_with,
    )
    return aggregated, has_image, photo_count


# =====================================================================
#  Image Quality Features
# =====================================================================


def compute_image_quality_features(
    pet_image_paths: dict[str, list[Path]],
    pet_ids: list[str] | np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Compute brightness, blur score, and size statistics per PetID.

    Parameters
    ----------
    pet_image_paths : dict[str, list[Path]]
        Mapping from PetID to image file paths.
    pet_ids : array-like of str
        Ordered PetIDs for output alignment.

    Returns
    -------
    pd.DataFrame
        One row per PetID with columns ``mean_image_brightness``,
        ``mean_blur_score``, ``image_size_std``.
    """
    from PIL import Image
    from scipy.ndimage import laplace

    if isinstance(pet_ids, pd.Series):
        pet_ids = pet_ids.tolist()

    rows: list[dict[str, Any]] = []

    for pid in pet_ids:
        paths = pet_image_paths.get(pid, [])
        brightnesses: list[float] = []
        blur_scores: list[float] = []
        areas: list[float] = []

        for p in paths:
            try:
                img = Image.open(p).convert("L")
                w, h = img.size
                arr = np.array(img, dtype=np.float64)

                brightnesses.append(float(arr.mean()))
                lap = laplace(arr)
                blur_scores.append(float(lap.var()))
                areas.append(float(w * h))
            except Exception as exc:
                logger.warning("Quality feature failed for %s: %s", p, exc)

        rows.append(
            {
                "mean_image_brightness": round(float(np.mean(brightnesses)), 2)
                if brightnesses
                else 0.0,
                "mean_blur_score": round(float(np.mean(blur_scores)), 2)
                if blur_scores
                else 0.0,
                "image_size_std": round(float(np.std(areas)), 2)
                if len(areas) > 1
                else 0.0,
            }
        )

    df = pd.DataFrame(rows, index=range(len(pet_ids)))
    logger.info("Computed image quality features for %d PetIDs", len(df))
    return df


# =====================================================================
#  Dimensionality Reduction
# =====================================================================


def reduce_image_dimensions(
    embeddings_train: np.ndarray,
    embeddings_test: np.ndarray,
    n_components: int = 100,
) -> tuple[np.ndarray, np.ndarray, Any, float]:
    """Fit PCA on train image embeddings, transform both splits.

    Parameters
    ----------
    embeddings_train : np.ndarray
        Train embedding array ``(n_train, d)``.
    embeddings_test : np.ndarray
        Test embedding array ``(n_test, d)``.
    n_components : int
        Number of PCA components to keep.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, PCA, float)
        - Reduced train embeddings ``(n_train, n_components)``.
        - Reduced test embeddings ``(n_test, n_components)``.
        - Fitted PCA object.
        - Cumulative explained variance ratio.
    """
    from sklearn.decomposition import PCA

    logger.info(
        "Fitting PCA with n_components=%d on train image embeddings ...",
        n_components,
    )
    pca = PCA(n_components=n_components, random_state=42)
    reduced_train = pca.fit_transform(embeddings_train)
    reduced_test = pca.transform(embeddings_test)

    explained_var = float(pca.explained_variance_ratio_.sum())
    logger.info(
        "PCA complete: %d -> %d dimensions, explained variance=%.4f",
        embeddings_train.shape[1],
        n_components,
        explained_var,
    )
    return reduced_train, reduced_test, pca, explained_var


# =====================================================================
#  Full Pipeline: Per-Split Embedding Extraction
# =====================================================================


def _extract_split_embeddings(
    pet_ids: list[str],
    pet_image_paths: dict[str, list[Path]],
    model: Any,
    transform: Any,
    batch_size: int,
    device: str,
    feature_dim: int,
) -> dict[str, np.ndarray]:
    """Extract per-image embeddings for a full split.

    Flattens all images across all PetIDs into a single batch list,
    extracts embeddings, then re-groups by PetID.

    Parameters
    ----------
    pet_ids : list of str
        PetIDs to process.
    pet_image_paths : dict[str, list[Path]]
        Mapping from PetID to image paths.
    model : nn.Module
        Pretrained backbone.
    transform : transforms.Compose
        Image preprocessing.
    batch_size : int
        Batch size.
    device : str
        Compute device.
    feature_dim : int
        Backbone output dimension.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from PetID to embedding array ``(n_images, feature_dim)``.
    """
    # Flatten all image paths with their PetID mapping
    all_paths: list[Path] = []
    path_to_pet: list[str] = []

    for pid in pet_ids:
        paths = pet_image_paths.get(pid, [])
        for p in paths:
            all_paths.append(p)
            path_to_pet.append(pid)

    if not all_paths:
        logger.warning("No images found for extraction.")
        return {}

    logger.info("Extracting embeddings for %d images ...", len(all_paths))

    # Extract all embeddings in batches
    embeddings, statuses = extract_image_embeddings_batch(
        all_paths,
        model,
        transform,
        batch_size,
        device,
        feature_dim,
    )

    # Log failure rate
    n_failed = sum(1 for s in statuses if not s)
    if n_failed > 0:
        logger.warning(
            "Image extraction failures: %d / %d (%.2f%%)",
            n_failed,
            len(statuses),
            100.0 * n_failed / len(statuses),
        )

    # Re-group embeddings by PetID
    result: dict[str, list[np.ndarray]] = {}
    for idx, (pid, success) in enumerate(zip(path_to_pet, statuses)):
        if success:
            if pid not in result:
                result[pid] = []
            result[pid].append(embeddings[idx])

    # Stack per-PetID
    stacked: dict[str, np.ndarray] = {}
    for pid, emb_list in result.items():
        stacked[pid] = np.stack(emb_list, axis=0)

    return stacked


# =====================================================================
#  Orchestrator
# =====================================================================


def extract_image_features(
    pet_ids_train: list[str] | np.ndarray | pd.Series,
    pet_ids_test: list[str] | np.ndarray | pd.Series,
    config: dict[str, Any],
    metadata_train: pd.DataFrame | None = None,
    metadata_test: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Orchestrate all image feature extraction steps.

    Parameters
    ----------
    pet_ids_train : array-like of str
        Train PetIDs.
    pet_ids_test : array-like of str
        Test PetIDs.
    config : dict
        Configuration dict with keys:
        - ``model_name``: str
        - ``batch_size``: int
        - ``device``: str or None
        - ``apply_pca``: bool
        - ``pca_components``: int
        - ``aggregation``: str (``"mean"`` or ``"max"``)
        - ``feature_dim``: int
    metadata_train : pd.DataFrame, optional
        Pre-aggregated metadata features for train.
    metadata_test : pd.DataFrame, optional
        Pre-aggregated metadata features for test.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, dict, dict)
        - Train image feature DataFrame.
        - Test image feature DataFrame.
        - Fitted artifacts dict (e.g., PCA object).
        - Extraction log dict.
    """
    import torch

    if isinstance(pet_ids_train, (pd.Series, np.ndarray)):
        pet_ids_train = list(pet_ids_train)
    if isinstance(pet_ids_test, (pd.Series, np.ndarray)):
        pet_ids_test = list(pet_ids_test)

    fitted_artifacts: dict[str, Any] = {}
    log: dict[str, Any] = {"steps": []}

    model_name = config.get("model_name", "efficientnet_v2_s")
    batch_size = config.get("batch_size", 32)
    device = _resolve_device(config.get("device"))
    aggregation = config.get("aggregation", "mean")
    feature_dim = config.get("feature_dim", 1280)
    apply_pca = config.get("apply_pca", False)
    pca_components = config.get("pca_components", 100)

    # -- Reproducibility -------------------------------------------------------
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # -- Step 1: Load model ----------------------------------------------------
    model, transform, actual_feature_dim = load_pretrained_backbone(
        model_name,
        device,
    )
    if actual_feature_dim != feature_dim:
        logger.warning(
            "Config feature_dim=%d != actual=%d. Using actual.",
            feature_dim,
            actual_feature_dim,
        )
        feature_dim = actual_feature_dim

    log["model_name"] = model_name
    log["feature_dim"] = feature_dim
    log["device"] = device
    log["steps"].append("load_pretrained_backbone")

    # -- Step 2: Resolve image paths -------------------------------------------
    train_paths = get_image_paths_for_split(pet_ids_train, "train")
    test_paths = get_image_paths_for_split(pet_ids_test, "test")
    log["steps"].append("resolve_image_paths")

    # -- Step 3: Extract train embeddings --------------------------------------
    train_emb_dict = _extract_split_embeddings(
        pet_ids_train,
        train_paths,
        model,
        transform,
        batch_size,
        device,
        feature_dim,
    )
    log["train_images_processed"] = sum(len(v) for v in train_paths.values())
    log["steps"].append("extract_train_embeddings")

    # -- Step 4: Aggregate train -----------------------------------------------
    train_agg, train_has_img, train_photo_ct = aggregate_embeddings_per_pet(
        train_emb_dict,
        pet_ids_train,
        feature_dim,
        aggregation,
    )
    log["steps"].append("aggregate_train_embeddings")

    # Free memory before test extraction
    del train_emb_dict
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -- Step 5: Extract test embeddings ---------------------------------------
    test_emb_dict = _extract_split_embeddings(
        pet_ids_test,
        test_paths,
        model,
        transform,
        batch_size,
        device,
        feature_dim,
    )
    log["test_images_processed"] = sum(len(v) for v in test_paths.values())
    log["steps"].append("extract_test_embeddings")

    # -- Step 6: Aggregate test ------------------------------------------------
    test_agg, test_has_img, test_photo_ct = aggregate_embeddings_per_pet(
        test_emb_dict,
        pet_ids_test,
        feature_dim,
        aggregation,
    )
    log["steps"].append("aggregate_test_embeddings")

    del test_emb_dict
    gc.collect()

    # Free model
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # -- Step 7: PCA (conditional) ---------------------------------------------
    effective_dim = feature_dim
    if apply_pca:
        train_agg, test_agg, pca_obj, explained_var = reduce_image_dimensions(
            train_agg,
            test_agg,
            pca_components,
        )
        fitted_artifacts["pca"] = pca_obj
        effective_dim = pca_components
        log["pca_explained_variance"] = explained_var
        log["pca_components"] = pca_components
        log["steps"].append("reduce_image_dimensions")

    # -- Step 8: Build embedding DataFrames ------------------------------------
    emb_col_names = [f"img_emb_{i}" for i in range(effective_dim)]

    train_emb_df = pd.DataFrame(train_agg, columns=emb_col_names)
    test_emb_df = pd.DataFrame(test_agg, columns=emb_col_names)

    # -- Step 9: Build auxiliary columns ---------------------------------------
    train_features = pd.DataFrame(
        {
            "PetID": pet_ids_train,
            "has_image_embedding": train_has_img,
            "actual_photo_count": train_photo_ct,
        }
    )
    test_features = pd.DataFrame(
        {
            "PetID": pet_ids_test,
            "has_image_embedding": test_has_img,
            "actual_photo_count": test_photo_ct,
        }
    )

    # Concatenate embeddings
    train_features = pd.concat(
        [train_features, train_emb_df],
        axis=1,
    )
    test_features = pd.concat(
        [test_features, test_emb_df],
        axis=1,
    )

    # -- Step 10: Image quality features ---------------------------------------
    train_quality = compute_image_quality_features(train_paths, pet_ids_train)
    test_quality = compute_image_quality_features(test_paths, pet_ids_test)

    for col in IMAGE_QUALITY_COLUMNS:
        train_features[col] = train_quality[col].values
        test_features[col] = test_quality[col].values

    log["steps"].append("compute_image_quality_features")

    # -- Step 11: Metadata features (if provided) ------------------------------
    if metadata_train is not None and metadata_test is not None:
        for col in IMAGE_METADATA_COLUMNS:
            if col in metadata_train.columns:
                train_features = train_features.merge(
                    metadata_train[["PetID", col]],
                    on="PetID",
                    how="left",
                )
                test_features = test_features.merge(
                    metadata_test[["PetID", col]],
                    on="PetID",
                    how="left",
                )
                train_features[col] = train_features[col].fillna(0.0).astype(np.float32)
                test_features[col] = test_features[col].fillna(0.0).astype(np.float32)
        log["steps"].append("merge_metadata_features")

    # -- Final cleanup: ensure no NaN ------------------------------------------
    nan_train = int(train_features.isna().sum().sum())
    nan_test = int(test_features.isna().sum().sum())
    if nan_train > 0:
        logger.warning(
            "Found %d NaN values in train image features, filling with 0", nan_train
        )
        train_features = train_features.fillna(0.0)
    if nan_test > 0:
        logger.warning(
            "Found %d NaN values in test image features, filling with 0", nan_test
        )
        test_features = test_features.fillna(0.0)

    log["n_features_train"] = len(train_features.columns) - 1  # exclude PetID
    log["n_features_test"] = len(test_features.columns) - 1
    log["n_rows_train"] = len(train_features)
    log["n_rows_test"] = len(test_features)
    log["aggregation"] = aggregation
    log["effective_embedding_dim"] = effective_dim

    logger.info(
        "Image FE complete: train=%s, test=%s",
        train_features.shape,
        test_features.shape,
    )

    return train_features, test_features, fitted_artifacts, log
