"""
Text feature extraction functions for Adoption Accelerator.

Provides text preprocessing, embedding extraction via pretrained sentence
transformers, and handcrafted text statistics. All functions are designed
for train/test parity and inference reproducibility.

Functions that require fitted artifacts (PCA) accept those artifacts as
explicit arguments -- they **never** compute them internally from the
input data. This separation guarantees deterministic inference.

Functions
---------
preprocess_descriptions(descriptions)
    Apply the full text preprocessing pipeline to a Series of
    descriptions.
extract_embeddings(texts, model_name, batch_size, device)
    Load the pretrained model and extract embeddings for a list of
    texts.
reduce_embedding_dimensions(emb_train, emb_test, n_components)
    Fit PCA on train embeddings, transform both splits.
compute_text_statistics(descriptions)
    Compute handcrafted text features (length, word count, etc.).
detect_languages(descriptions, sample_size)
    Run language detection on a sample of descriptions.
extract_text_features(df, split, config, fitted_artifacts)
    Orchestrate all text feature extraction steps.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("adoption_accelerator")

# -- Default placeholder for missing descriptions ----------------------------
_EMPTY_PLACEHOLDER = "no description"


# =====================================================================
#  Text Preprocessing
# =====================================================================


def preprocess_descriptions(
    descriptions: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    """Apply the full text preprocessing pipeline to descriptions.

    Pipeline steps:
    1. Replace null / empty descriptions with a canonical placeholder.
    2. Strip HTML tags and URLs.
    3. Normalize whitespace.
    4. Apply NFKD Unicode normalization.

    Casing is **preserved** (transformer tokenizers handle casing natively).
    Stop words are **not** removed (subword tokenizers degrade with removal).

    Parameters
    ----------
    descriptions : pd.Series
        Series of raw description strings.

    Returns
    -------
    tuple of (pd.Series, dict)
        - Cleaned descriptions Series (same index as input).
        - Statistics dict with counts of empty/cleaned descriptions and
          character count distribution (mean, median, min, max).
    """
    out = descriptions.copy()
    stats: dict[str, Any] = {}

    # Count nulls and empties before processing
    n_null = int(out.isna().sum())
    n_empty = int((out.fillna("").str.strip() == "").sum())
    stats["n_null"] = n_null
    stats["n_empty_or_null"] = n_empty

    # Step 1: Replace null/empty with placeholder
    out = out.fillna(_EMPTY_PLACEHOLDER)
    out = out.where(out.str.strip() != "", _EMPTY_PLACEHOLDER)

    # Step 2: Strip HTML tags and URLs/emails
    out = out.str.replace(r"<[^>]+>", " ", regex=True)
    out = out.str.replace(
        r"https?://\S+|www\.\S+|[\w.+-]+@[\w-]+\.[\w.-]+",
        " ",
        regex=True,
    )

    # Step 3: Normalize whitespace
    out = out.str.replace(r"\s+", " ", regex=True).str.strip()

    # Step 4: NFKD Unicode normalization
    out = out.apply(
        lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x
    )

    # Final safety: ensure no blanks remain
    out = out.where(out.str.strip() != "", _EMPTY_PLACEHOLDER)

    # Statistics on cleaned text
    char_lengths = out.str.len()
    stats["n_cleaned"] = len(out)
    stats["char_length_mean"] = round(float(char_lengths.mean()), 1)
    stats["char_length_median"] = round(float(char_lengths.median()), 1)
    stats["char_length_min"] = int(char_lengths.min())
    stats["char_length_max"] = int(char_lengths.max())

    logger.info(
        "Text preprocessing complete: %d descriptions (%d null, %d empty/null replaced)",
        len(out),
        n_null,
        n_empty,
    )
    return out, stats


# =====================================================================
#  Text Statistics
# =====================================================================


def compute_text_statistics(descriptions: pd.Series) -> pd.DataFrame:
    """Compute handcrafted text features from descriptions.

    Features computed per description:
    - ``description_length``: character count
    - ``word_count``: word count
    - ``sentence_count``: approximate sentence count (split on `.!?`)
    - ``avg_word_length``: mean word length in characters
    - ``uppercase_ratio``: fraction of uppercase characters
    - ``punctuation_ratio``: fraction of punctuation characters
    - ``digit_ratio``: fraction of digit characters

    Parameters
    ----------
    descriptions : pd.Series
        Preprocessed description strings.

    Returns
    -------
    pd.DataFrame
        One row per description (same index as input) with statistic
        columns.
    """
    stats = pd.DataFrame(index=descriptions.index)

    stats["description_length"] = descriptions.str.len().astype(np.int32)

    # Word count
    words = descriptions.str.split()
    stats["word_count"] = words.str.len().fillna(0).astype(np.int32)

    # Sentence count (approximate split on sentence-ending punctuation)
    stats["sentence_count"] = (
        descriptions.str.split(r"[.!?]+").str.len().clip(lower=1).astype(np.int32)
    )

    # Average word length
    avg_wl = words.apply(
        lambda ws: np.mean([len(w) for w in ws]) if ws and len(ws) > 0 else 0.0
    )
    stats["avg_word_length"] = avg_wl.astype(np.float32)

    # Character-level ratios
    total_chars = descriptions.str.len().replace(0, 1)  # avoid /0

    upper_counts = descriptions.str.count(r"[A-Z]")
    stats["uppercase_ratio"] = (upper_counts / total_chars).astype(np.float32)

    punct_counts = descriptions.str.count(r"[^\w\s]")
    stats["punctuation_ratio"] = (punct_counts / total_chars).astype(np.float32)

    digit_counts = descriptions.str.count(r"\d")
    stats["digit_ratio"] = (digit_counts / total_chars).astype(np.float32)

    logger.info("Computed text statistics: %d rows x %d features", *stats.shape)
    return stats


# =====================================================================
#  Language Detection
# =====================================================================


def detect_languages(
    descriptions: pd.Series,
    sample_size: int = 1_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run language detection on a sample of descriptions.

    Parameters
    ----------
    descriptions : pd.Series
        Preprocessed description strings.
    sample_size : int
        Number of descriptions to sample for detection.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    dict
        Language distribution report with keys ``sample_size``,
        ``language_counts`` (dict of language code -> count),
        ``dominant_language``, and ``multilingual_ratio``.
    """
    from langdetect import detect, LangDetectException
    from langdetect import DetectorFactory

    DetectorFactory.seed = seed

    # Sample without replacement (cap at available size)
    n = min(sample_size, len(descriptions))
    sample = descriptions.sample(n=n, random_state=seed)

    lang_results: list[str] = []
    for text in sample:
        try:
            lang = detect(str(text))
            lang_results.append(lang)
        except LangDetectException:
            lang_results.append("unknown")

    lang_counts = pd.Series(lang_results).value_counts().to_dict()
    dominant = max(lang_counts, key=lang_counts.get) if lang_counts else "unknown"
    en_count = lang_counts.get("en", 0)
    multilingual_ratio = round(1.0 - (en_count / n), 4) if n > 0 else 0.0

    report = {
        "sample_size": n,
        "language_counts": {k: int(v) for k, v in lang_counts.items()},
        "dominant_language": dominant,
        "multilingual_ratio": multilingual_ratio,
    }

    logger.info(
        "Language detection: dominant=%s, multilingual_ratio=%.2f%% (n=%d)",
        dominant,
        multilingual_ratio * 100,
        n,
    )
    return report


# =====================================================================
#  Embedding Extraction
# =====================================================================


def extract_embeddings(
    texts: list[str] | pd.Series,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Extract sentence-level embeddings using a pretrained model.

    Parameters
    ----------
    texts : list of str or pd.Series
        Input texts to encode.
    model_name : str
        Hugging Face model identifier for the sentence transformer.
    batch_size : int
        Batch size for encoding (higher = faster, more memory).
    device : str, optional
        Device to use (``"cuda"``, ``"cpu"``). If ``None``, auto-detects.

    Returns
    -------
    np.ndarray
        Embedding array of shape ``(n_texts, embedding_dim)``.
    """
    from sentence_transformers import SentenceTransformer

    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    logger.info(
        "Loading sentence transformer model: %s (batch_size=%d)",
        model_name,
        batch_size,
    )
    model = SentenceTransformer(model_name, device=device)

    logger.info("Extracting embeddings for %d texts ...", len(texts))
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    logger.info(
        "Embeddings extracted: shape=%s, dtype=%s", embeddings.shape, embeddings.dtype
    )
    return embeddings


# =====================================================================
#  Dimensionality Reduction (Conditional)
# =====================================================================


def reduce_embedding_dimensions(
    embeddings_train: np.ndarray,
    embeddings_test: np.ndarray,
    n_components: int = 100,
) -> tuple[np.ndarray, np.ndarray, Any, float]:
    """Fit PCA on train embeddings, transform both splits.

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
        "Fitting PCA with n_components=%d on train embeddings ...", n_components
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
#  Column Definitions
# =====================================================================


# Text statistics feature columns
TEXT_STAT_COLUMNS: list[str] = [
    "description_length",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "uppercase_ratio",
    "punctuation_ratio",
    "digit_ratio",
]

# Sentiment feature columns (from metadata.py aggregation)
SENTIMENT_COLUMNS: list[str] = [
    "doc_sentiment_score",
    "doc_sentiment_magnitude",
    "sentence_count_sentiment",
    "mean_sentence_score",
    "min_sentence_score",
    "max_sentence_score",
    "sentiment_variance",
    "entity_count",
    "entity_type_count",
]


def _build_embedding_column_names(dim: int, prefix: str = "text_emb") -> list[str]:
    """Build embedding column names: text_emb_0, text_emb_1, ..."""
    return [f"{prefix}_{i}" for i in range(dim)]


# Feature descriptions for schema registration
FEATURE_DESCRIPTIONS: dict[str, str] = {
    # Text statistics
    "description_length": "Character length of preprocessed description",
    "word_count": "Word count of preprocessed description",
    "sentence_count": "Approximate sentence count (split on .!?)",
    "avg_word_length": "Mean word length in characters",
    "uppercase_ratio": "Fraction of uppercase characters in description",
    "punctuation_ratio": "Fraction of punctuation characters in description",
    "digit_ratio": "Fraction of digit characters in description",
    # Sentiment features
    "doc_sentiment_score": "Document-level sentiment polarity (-1.0 to +1.0)",
    "doc_sentiment_magnitude": "Document-level sentiment intensity",
    "sentence_count_sentiment": "Number of sentences in sentiment JSON",
    "mean_sentence_score": "Mean sentiment score across all sentences",
    "min_sentence_score": "Minimum sentence-level sentiment score",
    "max_sentence_score": "Maximum sentence-level sentiment score",
    "sentiment_variance": "Variance of sentence-level sentiment scores",
    "entity_count": "Total named entities detected by NLP API",
    "entity_type_count": "Number of distinct entity types detected",
}


# =====================================================================
#  Orchestrator
# =====================================================================


def extract_text_features(
    df: pd.DataFrame,
    split: str,
    config: dict[str, Any],
    fitted_artifacts: dict[str, Any] | None = None,
    sentiment_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Orchestrate all text feature extraction steps.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with at least ``PetID`` and ``Description``.
    split : str
        ``"train"`` or ``"test"``.
    config : dict
        Configuration dict with keys:
        - ``model_name``: str -- sentence transformer model identifier
        - ``batch_size``: int -- encoding batch size
        - ``device``: str or None -- device for encoding
        - ``apply_pca``: bool -- whether to apply PCA
        - ``pca_components``: int -- number of PCA components
    fitted_artifacts : dict, optional
        Dict with pre-fitted artifacts (e.g., ``"pca"`` object). Required
        when ``split="test"`` and PCA is enabled.
    sentiment_df : pd.DataFrame, optional
        Pre-aggregated sentiment features DataFrame with ``PetID``.
        If ``None``, sentiment features are skipped.

    Returns
    -------
    tuple of (pd.DataFrame, dict, dict)
        - Text feature DataFrame (one row per PetID).
        - ``fitted_artifacts`` dict (updated with PCA if fitted).
        - Extraction log dict with step metadata.
    """
    fitted_artifacts = fitted_artifacts or {}
    log: dict[str, Any] = {"split": split, "steps": []}

    logger.info("Starting text feature extraction for split=%s", split)

    pet_ids = df["PetID"].values
    descriptions = df["Description"].copy()

    # Step 1: Preprocess descriptions
    cleaned, preprocess_stats = preprocess_descriptions(descriptions)
    log["preprocess_stats"] = preprocess_stats
    log["steps"].append("preprocess_descriptions")

    # Step 2: Compute handcrafted text statistics
    text_stats = compute_text_statistics(cleaned)
    text_stats.index = df.index  # align indices
    log["steps"].append("compute_text_statistics")

    # Step 3: Extract embeddings
    model_name = config.get("model_name", "sentence-transformers/all-mpnet-base-v2")
    batch_size = config.get("batch_size", 64)
    device = config.get("device", None)

    embeddings = extract_embeddings(
        cleaned,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
    )
    log["embedding_dim"] = int(embeddings.shape[1])
    log["steps"].append("extract_embeddings")

    # Step 4: Dimensionality reduction (conditional)
    apply_pca = config.get("apply_pca", False)
    if apply_pca:
        n_components = config.get("pca_components", 100)
        if split == "train":
            # We need a placeholder for test embeddings during train fitting;
            # PCA is fitted on train only. Test application happens in test call.
            embeddings, _, pca_obj, explained_var = reduce_embedding_dimensions(
                embeddings,
                embeddings[:1],  # dummy test for API
                n_components=n_components,
            )
            fitted_artifacts["pca"] = pca_obj
            log["pca_explained_variance"] = explained_var
        else:
            pca_obj = fitted_artifacts.get("pca")
            if pca_obj is not None:
                embeddings = pca_obj.transform(embeddings)
                log["pca_applied"] = True
        log["steps"].append("reduce_embedding_dimensions")

    # Step 5: Build embedding DataFrame
    emb_dim = embeddings.shape[1]
    emb_col_names = _build_embedding_column_names(emb_dim)
    emb_df = pd.DataFrame(
        embeddings.astype(np.float32),
        columns=emb_col_names,
        index=df.index,
    )
    log["steps"].append("build_embedding_dataframe")

    # Step 6: Merge all text features
    result = pd.DataFrame({"PetID": pet_ids}, index=df.index)

    # Add text statistics
    for col in TEXT_STAT_COLUMNS:
        result[col] = text_stats[col].values

    # Add embeddings
    for col in emb_col_names:
        result[col] = emb_df[col].values

    # Add sentiment features (if provided)
    if sentiment_df is not None:
        sent_merge = sentiment_df.copy()
        # Ensure we have all expected sentiment columns with defaults
        for col in SENTIMENT_COLUMNS:
            if col not in sent_merge.columns:
                sent_merge[col] = 0.0

        result = result.merge(
            sent_merge[["PetID"] + SENTIMENT_COLUMNS],
            on="PetID",
            how="left",
        )
        # Fill NaN sentiment values with defaults
        for col in SENTIMENT_COLUMNS:
            result[col] = result[col].fillna(0.0).astype(np.float32)

        log["sentiment_coverage"] = round(
            float(sentiment_df["PetID"].isin(pet_ids).sum() / len(pet_ids)), 4
        )
        log["steps"].append("merge_sentiment_features")
    else:
        log["sentiment_coverage"] = 0.0
        logger.info("Sentiment features not provided -- skipping.")

    # Ensure no NaN values
    nan_count = int(result.isna().sum().sum())
    if nan_count > 0:
        logger.warning(
            "Found %d NaN values in text features, filling with 0", nan_count
        )
        result = result.fillna(0.0)

    log["n_features"] = len(result.columns) - 1  # exclude PetID
    log["n_rows"] = len(result)

    logger.info(
        "Text FE complete for split=%s: %d rows x %d features",
        split,
        len(result),
        log["n_features"],
    )

    return result, fitted_artifacts, log
