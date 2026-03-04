"""
Tabular feature engineering functions for Adoption Accelerator.

All transformations are implemented as pure, vectorized functions that
operate on pandas DataFrames. Functions that require statistics derived
from training data (frequency maps, rescuer aggregates) accept those
statistics as explicit arguments -- they **never** compute them internally
from the input DataFrame. This separation guarantees train/test parity
and inference reproducibility.

Functions
---------
encode_pet_type(df)
    Create ``is_dog`` binary feature from ``Type``.
create_health_care_score(df)
    Combine Vaccinated, Dewormed, Sterilized into composite score.
recode_care_features(df)
    Apply tristate recoding to care features.
transform_numeric_features(df)
    Apply log1p, binning, and derived ratio transformations.
create_name_features(df)
    Compute name_length, name_word_count from the Name column.
create_breed_features(df, breed_freq_map)
    Create is_mixed_breed, breed_count, breed1_frequency.
create_color_features(df)
    Compute color_count and color missing indicators.
create_interaction_features(df)
    Construct interaction features from EDA hypotheses.
compute_rescuer_aggregates(train_df)
    Compute per-rescuer statistics from training data.
apply_rescuer_aggregates(df, rescuer_stats, default_values)
    Map rescuer aggregates onto a DataFrame.
frequency_encode(df, col, freq_map)
    Apply frequency encoding using a precomputed frequency map.
engineer_tabular_features(df, split, fitted_maps)
    Orchestrate all tabular feature engineering steps.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("adoption_accelerator")


# =====================================================================
#  Individual Feature Engineering Functions
# =====================================================================


def encode_pet_type(df: pd.DataFrame) -> pd.DataFrame:
    """Create ``is_dog`` binary feature from ``Type``.

    Type == 1 -> Dog, Type == 2 -> Cat.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``Type`` column.

    Returns
    -------
    pd.DataFrame
        Copy with ``is_dog`` added.
    """
    out = df.copy()
    out["is_dog"] = (out["Type"] == 1).astype(np.int8)
    return out


def create_health_care_score(df: pd.DataFrame) -> pd.DataFrame:
    """Combine Vaccinated, Dewormed, Sterilized into ``health_care_score``.

    Each feature uses PetFinder encoding: 1 = Yes, 2 = No, 3 = Not Sure.
    The score counts how many of the three are *Yes* (== 1), range 0--3.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Vaccinated``, ``Dewormed``, ``Sterilized`` columns.

    Returns
    -------
    pd.DataFrame
        Copy with ``health_care_score`` added.
    """
    out = df.copy()
    care_cols = ["Vaccinated", "Dewormed", "Sterilized"]
    out["health_care_score"] = sum((out[c] == 1).astype(np.int8) for c in care_cols)
    return out


def recode_care_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recode Vaccinated, Dewormed, Sterilized to tristate {1, 0, -1}.

    Original encoding: 1 = Yes, 2 = No, 3 = Not Sure.
    Target encoding:   1 = Yes, 0 = No, -1 = Not Sure.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Vaccinated``, ``Dewormed``, ``Sterilized``.

    Returns
    -------
    pd.DataFrame
        Copy with recoded columns.
    """
    out = df.copy()
    mapping = {1: 1, 2: 0, 3: -1}
    for col in ["Vaccinated", "Dewormed", "Sterilized"]:
        out[col] = out[col].map(mapping).astype(np.int8)
    return out


def transform_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p, binning, and ratio transformations to numeric columns.

    Creates:
    - ``log_age``, ``age_bin`` (ordinal-encoded)
    - ``is_free``, ``log_fee``, ``fee_per_pet``
    - ``is_single_pet``, ``log_quantity``
    - ``has_photos``, ``has_video``, ``log_photo_amt``

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Age``, ``Fee``, ``Quantity``, ``PhotoAmt``,
        ``VideoAmt`` columns.

    Returns
    -------
    pd.DataFrame
        Copy with new numeric features added.
    """
    out = df.copy()

    # -- Age --
    out["log_age"] = np.log1p(out["Age"]).astype(np.float32)

    age_bins = [0, 1, 3, 6, 12, 24, 60, 255]
    age_labels = list(range(len(age_bins) - 1))  # 0..6 ordinal
    out["age_bin"] = (
        pd.cut(
            out["Age"],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True,
            right=True,
        )
        .astype(float)
        .fillna(len(age_labels) - 1)
        .astype(np.int8)
    )

    # -- Fee --
    out["is_free"] = (out["Fee"] == 0).astype(np.int8)
    out["log_fee"] = np.log1p(out["Fee"]).astype(np.float32)
    out["fee_per_pet"] = (out["Fee"] / out["Quantity"].clip(lower=1)).astype(np.float32)

    # -- Quantity --
    out["is_single_pet"] = (out["Quantity"] == 1).astype(np.int8)
    out["log_quantity"] = np.log1p(out["Quantity"]).astype(np.float32)

    # -- Photo / Video --
    out["has_photos"] = (out["PhotoAmt"] > 0).astype(np.int8)
    out["has_video"] = (out["VideoAmt"] > 0).astype(np.int8)
    out["log_photo_amt"] = np.log1p(out["PhotoAmt"]).astype(np.float32)

    return out


def create_name_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute name-derived features.

    Creates ``name_length`` and ``name_word_count``. Uses the ``has_name``
    flag produced during cleaning (notebook 03). Names without a valid
    value get length and word-count of 0.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Name`` and ``has_name`` columns.

    Returns
    -------
    pd.DataFrame
        Copy with name features added.
    """
    out = df.copy()
    name_series = out["Name"].fillna("")
    out["name_length"] = (
        name_series.str.len().where(out["has_name"] == 1, 0).astype(np.int16)
    )
    out["name_word_count"] = (
        name_series.str.split()
        .str.len()
        .fillna(0)
        .where(out["has_name"] == 1, 0)
        .astype(np.int8)
    )
    return out


def create_breed_features(
    df: pd.DataFrame,
    breed_freq_map: pd.Series,
) -> pd.DataFrame:
    """Create breed-related features.

    Creates ``is_mixed_breed``, ``breed_count``, and ``breed1_frequency``.
    The frequency map must be pre-fitted from training data only.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Breed1`` and ``Breed2`` columns.
    breed_freq_map : pd.Series
        Index = Breed1 code, value = frequency proportion (fitted
        from train).

    Returns
    -------
    pd.DataFrame
        Copy with breed features added.
    """
    out = df.copy()
    out["is_mixed_breed"] = (out["Breed2"] != 0).astype(np.int8)
    out["breed_count"] = (1 + (out["Breed2"] != 0)).astype(np.int8)
    out["breed1_frequency"] = (
        out["Breed1"]
        .map(breed_freq_map)
        .fillna(breed_freq_map.min())
        .astype(np.float32)
    )
    return out


def create_color_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create color-related features.

    Creates ``color_count`` (number of non-zero Color columns, range
    1--3) and missing-indicator flags for Color2 and Color3.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Color1``, ``Color2``, ``Color3`` columns.

    Returns
    -------
    pd.DataFrame
        Copy with color features added.
    """
    out = df.copy()
    color_cols = ["Color1", "Color2", "Color3"]
    out["color_count"] = sum((out[c] != 0).astype(np.int8) for c in color_cols)
    out["breed2_missing"] = (out["Breed2"] == 0).astype(np.int8)
    out["color2_missing"] = (out["Color2"] == 0).astype(np.int8)
    out["color3_missing"] = (out["Color3"] == 0).astype(np.int8)
    return out


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct interaction features from EDA hypotheses.

    Creates:
    - ``age_x_type``: Age * is_dog (differential age dynamics)
    - ``health_x_vaccinated``: Health * Vaccinated (ordinal cross)

    Note: ``fee_per_pet`` is already created in :func:`transform_numeric_features`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``Age``, ``is_dog``, ``Health``, ``Vaccinated``.

    Returns
    -------
    pd.DataFrame
        Copy with interaction features added.
    """
    out = df.copy()
    out["age_x_type"] = (out["Age"] * out["is_dog"]).astype(np.float32)
    out["health_x_vaccinated"] = (out["Health"] * out["Vaccinated"]).astype(np.float32)
    return out


def compute_rescuer_aggregates(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-rescuer statistics from training data.

    Aggregations:
    - ``rescuer_pet_count`` : number of listings
    - ``rescuer_mean_photo_amt`` : mean PhotoAmt
    - ``rescuer_mean_fee`` : mean Fee

    Parameters
    ----------
    train_df : pd.DataFrame
        **Training** DataFrame only. Must contain ``RescuerID``,
        ``PhotoAmt``, ``Fee`` columns.

    Returns
    -------
    pd.DataFrame
        One row per unique ``RescuerID``, indexed by ``RescuerID``.
    """
    agg = train_df.groupby("RescuerID").agg(
        rescuer_pet_count=("RescuerID", "size"),
        rescuer_mean_photo_amt=("PhotoAmt", "mean"),
        rescuer_mean_fee=("Fee", "mean"),
    )
    agg["rescuer_pet_count"] = agg["rescuer_pet_count"].astype(np.int16)
    agg["rescuer_mean_photo_amt"] = agg["rescuer_mean_photo_amt"].astype(np.float32)
    agg["rescuer_mean_fee"] = agg["rescuer_mean_fee"].astype(np.float32)
    return agg


def apply_rescuer_aggregates(
    df: pd.DataFrame,
    rescuer_stats: pd.DataFrame,
    default_values: dict[str, float],
) -> pd.DataFrame:
    """Map rescuer aggregates onto a DataFrame.

    For rescuers that appear only in test (cold-start), provided
    ``default_values`` are used as fill values.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``RescuerID``.
    rescuer_stats : pd.DataFrame
        Per-rescuer aggregates (indexed by ``RescuerID``).
    default_values : dict
        Mapping of feature name to default value for cold-start
        rescuers (typically training-set global means).

    Returns
    -------
    pd.DataFrame
        Copy with rescuer aggregate features merged.
    """
    out = df.copy()
    merged = out[["RescuerID"]].merge(
        rescuer_stats,
        how="left",
        left_on="RescuerID",
        right_index=True,
    )
    for col, default in default_values.items():
        out[col] = merged[col].fillna(default).values
    return out


def frequency_encode(
    df: pd.DataFrame,
    col: str,
    freq_map: pd.Series,
    new_col: str | None = None,
) -> pd.DataFrame:
    """Apply frequency encoding using a precomputed frequency map.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    col : str
        Column name to encode.
    freq_map : pd.Series
        Index = category value, value = frequency proportion (from train).
    new_col : str, optional
        Output column name. Defaults to ``{col}_freq``.

    Returns
    -------
    pd.DataFrame
        Copy with the frequency-encoded column.
    """
    out = df.copy()
    target = new_col or f"{col}_freq"
    out[target] = out[col].map(freq_map).fillna(freq_map.min()).astype(np.float32)
    return out


# =====================================================================
#  Feature Column Selection
# =====================================================================

# Columns to keep in the final output feature matrix
FEATURE_COLUMNS: list[str] = [
    # Identifiers (kept for downstream merging, dropped before modelling)
    "PetID",
    # Binary / encoded
    "is_dog",
    "Gender",
    # Ordinal
    "MaturitySize",
    "FurLength",
    "Health",
    # Care features (recoded)
    "Vaccinated",
    "Dewormed",
    "Sterilized",
    "health_care_score",
    # Numeric transforms
    "Age",
    "log_age",
    "age_bin",
    "Fee",
    "log_fee",
    "is_free",
    "fee_per_pet",
    "Quantity",
    "log_quantity",
    "is_single_pet",
    "PhotoAmt",
    "log_photo_amt",
    "has_photos",
    "VideoAmt",
    "has_video",
    # Name features
    "has_name",
    "name_length",
    "name_word_count",
    # Breed features
    "Breed1",
    "Breed2",
    "is_mixed_breed",
    "breed_count",
    "breed1_frequency",
    # Color features
    "Color1",
    "Color2",
    "Color3",
    "color_count",
    "breed2_missing",
    "color2_missing",
    "color3_missing",
    # Interaction features
    "age_x_type",
    "health_x_vaccinated",
    # Rescuer aggregate features
    "rescuer_pet_count",
    "rescuer_mean_photo_amt",
    "rescuer_mean_fee",
    # Frequency-encoded
    "state_freq",
]

# Feature descriptions for schema registration
FEATURE_DESCRIPTIONS: dict[str, str] = {
    "PetID": "Unique pet listing identifier",
    "is_dog": "Binary: 1 = Dog, 0 = Cat",
    "Gender": "Original gender code (1=Male, 2=Female, 3=Mixed)",
    "MaturitySize": "Ordinal maturity size (1=Small to 4=Extra Large)",
    "FurLength": "Ordinal fur length (1=Short, 2=Medium, 3=Long)",
    "Health": "Ordinal health condition (1=Healthy to 3=Serious Injury)",
    "Vaccinated": "Tristate recoded: 1=Yes, 0=No, -1=Not Sure",
    "Dewormed": "Tristate recoded: 1=Yes, 0=No, -1=Not Sure",
    "Sterilized": "Tristate recoded: 1=Yes, 0=No, -1=Not Sure",
    "health_care_score": "Count of Yes (=1) among Vaccinated/Dewormed/Sterilized (0-3)",
    "Age": "Original age in months",
    "log_age": "log1p(Age) to reduce right skew",
    "age_bin": "Domain-informed age bin (0=neonate to 6=senior)",
    "Fee": "Original adoption fee",
    "log_fee": "log1p(Fee) for non-zero fee gradient",
    "is_free": "Binary: 1 = Fee is zero",
    "fee_per_pet": "Fee / Quantity ratio",
    "Quantity": "Original quantity of pets in listing",
    "log_quantity": "log1p(Quantity)",
    "is_single_pet": "Binary: 1 = Quantity is 1",
    "PhotoAmt": "Original photo count",
    "log_photo_amt": "log1p(PhotoAmt)",
    "has_photos": "Binary: 1 = PhotoAmt > 0",
    "VideoAmt": "Original video count",
    "has_video": "Binary: 1 = VideoAmt > 0",
    "has_name": "Binary: 1 = valid name present (from cleaning)",
    "name_length": "Character length of name (0 if no valid name)",
    "name_word_count": "Word count of name (0 if no valid name)",
    "Breed1": "Primary breed code",
    "Breed2": "Secondary breed code (0 = none)",
    "is_mixed_breed": "Binary: 1 = Breed2 is non-zero",
    "breed_count": "Number of breeds (1 or 2)",
    "breed1_frequency": "Frequency encoding of Breed1 from train distribution",
    "Color1": "Primary color code",
    "Color2": "Secondary color code (0 = none)",
    "Color3": "Tertiary color code (0 = none)",
    "color_count": "Number of non-zero color codes (1-3)",
    "breed2_missing": "Binary: 1 = Breed2 is 0 (no secondary breed)",
    "color2_missing": "Binary: 1 = Color2 is 0",
    "color3_missing": "Binary: 1 = Color3 is 0",
    "age_x_type": "Interaction: Age * is_dog",
    "health_x_vaccinated": "Interaction: Health * Vaccinated (ordinal cross)",
    "rescuer_pet_count": "Number of listings by this rescuer (train-fitted)",
    "rescuer_mean_photo_amt": "Mean PhotoAmt for this rescuer (train-fitted)",
    "rescuer_mean_fee": "Mean Fee for this rescuer (train-fitted)",
    "state_freq": "Frequency encoding of State from train distribution",
}


# =====================================================================
#  Orchestrator
# =====================================================================


def engineer_tabular_features(
    df: pd.DataFrame,
    split: str,
    fitted_maps: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Orchestrate all tabular feature engineering steps.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame (train or test).
    split : str
        ``"train"`` or ``"test"``. When ``"train"``, frequency maps and
        rescuer aggregates are computed and returned in ``fitted_maps``.
        When ``"test"``, pre-fitted maps must be supplied.
    fitted_maps : dict, optional
        Dictionary containing pre-fitted statistics:
        - ``breed_freq_map`` : pd.Series
        - ``state_freq_map`` : pd.Series
        - ``rescuer_stats`` : pd.DataFrame
        - ``rescuer_defaults`` : dict
        Required when ``split="test"``.

    Returns
    -------
    tuple of (pd.DataFrame, dict, dict)
        - Feature DataFrame with columns from :data:`FEATURE_COLUMNS`.
        - ``fitted_maps`` dict (computed if split is train, echoed if test).
        - Engineering log dict with step counts and diagnostics.
    """
    fitted_maps = fitted_maps or {}
    log: dict[str, Any] = {"split": split, "steps": []}

    logger.info("Starting tabular feature engineering for split=%s", split)

    # Step 3: Pet type binary encoding
    df = encode_pet_type(df)
    log["steps"].append("encode_pet_type")

    # Step 5: Health-care composite score (before recoding)
    df = create_health_care_score(df)
    log["steps"].append("create_health_care_score")

    # Step 6: Binary recoding of care features
    df = recode_care_features(df)
    log["steps"].append("recode_care_features")

    # Step 7-9: Numeric transformations
    df = transform_numeric_features(df)
    log["steps"].append("transform_numeric_features")

    # Step 11: Name features
    df = create_name_features(df)
    log["steps"].append("create_name_features")

    # Step 12: Breed features
    if split == "train":
        breed_freq_map = df["Breed1"].value_counts(normalize=True)
        fitted_maps["breed_freq_map"] = breed_freq_map
    else:
        breed_freq_map = fitted_maps["breed_freq_map"]
    df = create_breed_features(df, breed_freq_map)
    log["steps"].append("create_breed_features")

    # Step 13-14: Color features and missing indicators
    df = create_color_features(df)
    log["steps"].append("create_color_features")

    # Step 15: Interaction features
    df = create_interaction_features(df)
    log["steps"].append("create_interaction_features")

    # Step 16: Rescuer aggregation
    if split == "train":
        rescuer_stats = compute_rescuer_aggregates(df)
        rescuer_defaults = {
            "rescuer_pet_count": float(rescuer_stats["rescuer_pet_count"].mean()),
            "rescuer_mean_photo_amt": float(
                rescuer_stats["rescuer_mean_photo_amt"].mean()
            ),
            "rescuer_mean_fee": float(rescuer_stats["rescuer_mean_fee"].mean()),
        }
        fitted_maps["rescuer_stats"] = rescuer_stats
        fitted_maps["rescuer_defaults"] = rescuer_defaults
    else:
        rescuer_stats = fitted_maps["rescuer_stats"]
        rescuer_defaults = fitted_maps["rescuer_defaults"]

    df = apply_rescuer_aggregates(df, rescuer_stats, rescuer_defaults)
    log["steps"].append("rescuer_aggregates")

    # Step 17: State frequency encoding
    if split == "train":
        state_freq_map = df["State"].value_counts(normalize=True)
        fitted_maps["state_freq_map"] = state_freq_map
    else:
        state_freq_map = fitted_maps["state_freq_map"]

    df = frequency_encode(df, "State", state_freq_map, new_col="state_freq")
    log["steps"].append("state_frequency_encode")

    # Step 19: Feature selection and column ordering
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing_features = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_features:
        logger.warning("Missing expected feature columns: %s", missing_features)
        log["missing_features"] = list(missing_features)

    result = df[feature_cols].copy()

    log["n_features"] = len(feature_cols)
    log["n_rows"] = len(result)
    log["null_counts"] = result.isna().sum().to_dict()

    logger.info(
        "Tabular FE complete for split=%s: %d rows x %d features",
        split,
        len(result),
        len(feature_cols),
    )

    return result, fitted_maps, log
