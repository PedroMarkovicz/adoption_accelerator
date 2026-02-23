"""
Data cleaning utilities for the Adoption Accelerator project.

Deterministic, replayable cleaning transformations driven by
validation findings.  Every function returns the transformed
DataFrame and a log entry describing what changed.

Functions
---------
clean_tabular(df, split, ref_breeds, ref_colors, ref_states)
    Orchestrate all cleaning steps.  Return cleaned df + cleaning log.
handle_missing_names(df)
    Standardize null/empty Name values.  Add ``has_name`` flag.
normalize_text_fields(df, columns)
    Strip whitespace and normalize encoding on text columns.
fix_invalid_codes(df, col, valid_ids, fallback)
    Replace FK values not present in *valid_ids* with *fallback*.
enforce_dtypes(df, schema)
    Cast columns to canonical schema dtypes.
"""

from __future__ import annotations

import logging
import unicodedata
from typing import Any

import pandas as pd

from adoption_accelerator.data.schemas import TabularSchema, get_tabular_schema

logger = logging.getLogger("adoption_accelerator")


# ────────────────────────────────────────────────────────────────────
# Missing names
# ────────────────────────────────────────────────────────────────────


def handle_missing_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Standardize null / empty / placeholder ``Name`` values and add ``has_name`` flag.

    Rules applied in order:

    1. Empty strings ``""`` and whitespace-only strings → ``NaN``
    2. Placeholder patterns (e.g. "No Name", "Unnamed", "None",
       "N/A", "Unknown", dashes, dots, question marks) → ``NaN``
    3. Numeric-only strings (e.g. "12345") → ``NaN``
    4. Symbol-only strings (no alphanumeric characters) → ``NaN``
    5. ``has_name`` = 1 if Name is still non-null after cleaning, else 0

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``Name`` column.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (modified DataFrame, cleaning log entry)
    """
    df = df.copy()
    before_nulls = int(df["Name"].isna().sum())

    # 1. Normalize empties / whitespace-only → NaN
    df["Name"] = df["Name"].replace(r"^\s*$", pd.NA, regex=True)
    df.loc[df["Name"].isna(), "Name"] = pd.NA

    after_empty_fix = int(df["Name"].isna().sum())
    empty_converted = after_empty_fix - before_nulls

    # 2-4. Detect placeholder / numeric-only / symbol-only patterns
    _PLACEHOLDER_RE = r"(?i)^(no\s*name|unnamed|none|n/?a|unknown|\-+|\.+|\?+)$"
    _NUMERIC_ONLY_RE = r"^\d+$"
    _SYMBOL_ONLY_RE = r"^[^a-zA-Z0-9\s]+$"

    name_str = df["Name"].fillna("")
    is_placeholder = name_str.str.match(_PLACEHOLDER_RE)
    is_numeric = name_str.str.match(_NUMERIC_ONLY_RE)
    is_symbol = name_str.str.match(_SYMBOL_ONLY_RE)
    noisy_mask = is_placeholder | is_numeric | is_symbol

    n_placeholder = int(is_placeholder.sum())
    n_numeric = int(is_numeric.sum())
    n_symbol = int(is_symbol.sum())
    n_noisy = int(noisy_mask.sum())

    # Set noisy names to NaN
    df.loc[noisy_mask, "Name"] = pd.NA

    after_nulls = int(df["Name"].isna().sum())

    # 5. Create binary flag
    df["has_name"] = (~df["Name"].isna()).astype(int)

    log_entry = {
        "step": "handle_missing_names",
        "before_nulls": before_nulls,
        "after_nulls": after_nulls,
        "empty_strings_converted": empty_converted,
        "placeholders_converted": n_placeholder,
        "numeric_only_converted": n_numeric,
        "symbol_only_converted": n_symbol,
        "total_noisy_converted": n_noisy,
        "has_name_added": True,
    }
    logger.info(
        "handle_missing_names: %d empty→NaN, %d noisy→NaN "
        "(placeholder=%d, numeric=%d, symbol=%d), %d total nulls, has_name flag added",
        empty_converted, n_noisy, n_placeholder, n_numeric, n_symbol, after_nulls,
    )
    return df, log_entry


# ────────────────────────────────────────────────────────────────────
# Text normalization
# ────────────────────────────────────────────────────────────────────


def normalize_text_fields(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Strip whitespace and normalize encoding on text columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list[str]
        Column names to normalize.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (modified DataFrame, cleaning log entry)
    """
    df = df.copy()
    details: dict[str, int] = {}

    for col in columns:
        if col not in df.columns:
            continue
        original = df[col].copy()

        # Strip leading/trailing whitespace
        df[col] = df[col].astype(str).str.strip()

        # Normalize Unicode to NFC
        df[col] = df[col].apply(
            lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) and x != "<NA>" else x
        )

        # Restore NaN where appropriate
        df.loc[df[col].isin(["nan", "<NA>", "None", ""]), col] = pd.NA

        changed = int((original.fillna("") != df[col].fillna("")).sum())
        details[col] = changed

    log_entry = {
        "step": "normalize_text_fields",
        "columns": columns,
        "changes_per_column": details,
    }
    logger.info("normalize_text_fields: %s", details)
    return df, log_entry


# ────────────────────────────────────────────────────────────────────
# Invalid code remediation
# ────────────────────────────────────────────────────────────────────


def fix_invalid_codes(
    df: pd.DataFrame,
    col: str,
    valid_ids: set[int],
    fallback: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace values in *col* not present in *valid_ids* with *fallback*.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column to fix.
    valid_ids : set[int]
        Set of valid values (including 0 if 0 is semantically valid).
    fallback : int
        Replacement value for invalid entries.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (modified DataFrame, log entry with fix count)
    """
    df = df.copy()
    mask = ~df[col].isin(valid_ids)
    fix_count = int(mask.sum())
    invalid_values = sorted(df.loc[mask, col].unique().tolist()) if fix_count else []

    df.loc[mask, col] = fallback

    log_entry = {
        "step": "fix_invalid_codes",
        "column": col,
        "fixes": fix_count,
        "invalid_values_found": invalid_values,
        "fallback": fallback,
    }
    logger.info("fix_invalid_codes [%s]: %d fixes (fallback=%d)", col, fix_count, fallback)
    return df, log_entry


def fix_breed_swap(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fix rows where ``Breed1=0`` but ``Breed2≠0`` by swapping.

    Moves the valid secondary breed into the primary slot and zeros
    the secondary slot.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (modified DataFrame, log entry)
    """
    df = df.copy()
    mask = (df["Breed1"] == 0) & (df["Breed2"] != 0)
    swap_count = int(mask.sum())

    if swap_count > 0:
        df.loc[mask, "Breed1"] = df.loc[mask, "Breed2"]
        df.loc[mask, "Breed2"] = 0

    log_entry = {
        "step": "fix_breed_swap",
        "swapped_rows": swap_count,
        "details": "Moved Breed2 → Breed1 where Breed1=0 and Breed2≠0",
    }
    logger.info("fix_breed_swap: %d rows swapped", swap_count)
    return df, log_entry


# ────────────────────────────────────────────────────────────────────
# Dtype enforcement
# ────────────────────────────────────────────────────────────────────


def enforce_dtypes(
    df: pd.DataFrame,
    schema: TabularSchema,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Cast DataFrame columns to match the canonical schema dtypes.

    Parameters
    ----------
    df : pd.DataFrame
    schema : TabularSchema

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (modified DataFrame, log entry with cast details)
    """
    df = df.copy()
    casts: list[dict[str, str]] = []

    for spec in schema.columns:
        if spec.name not in df.columns:
            continue
        current_dtype = str(df[spec.name].dtype)
        target = spec.dtype

        if current_dtype == target:
            continue

        try:
            if target == "int64":
                df[spec.name] = df[spec.name].fillna(0).astype("int64")
            elif target == "float64":
                df[spec.name] = df[spec.name].astype("float64")
            elif target == "object":
                df[spec.name] = df[spec.name].astype("object")
            else:
                df[spec.name] = df[spec.name].astype(target)

            casts.append({"column": spec.name, "from": current_dtype, "to": target})
        except (ValueError, TypeError) as exc:
            logger.warning("Could not cast %s from %s to %s: %s", spec.name, current_dtype, target, exc)

    log_entry = {
        "step": "enforce_dtypes",
        "casts": casts,
        "total_casts": len(casts),
    }
    logger.info("enforce_dtypes: %d column(s) cast", len(casts))
    return df, log_entry


# ────────────────────────────────────────────────────────────────────
# Orchestrator
# ────────────────────────────────────────────────────────────────────


def clean_tabular(
    df: pd.DataFrame,
    split: str,
    ref_breeds: pd.DataFrame,
    ref_colors: pd.DataFrame,
    ref_states: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Orchestrate all cleaning steps on a tabular DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (train or test).
    split : str
        ``"train"`` or ``"test"``.
    ref_breeds : pd.DataFrame
        Breed reference table.
    ref_colors : pd.DataFrame
        Color reference table.
    ref_states : pd.DataFrame
        State reference table.

    Returns
    -------
    tuple[pd.DataFrame, list[dict]]
        (cleaned DataFrame, list of cleaning log entries)
    """
    schema = get_tabular_schema(split)
    cleaning_log: list[dict[str, Any]] = []

    logger.info("── Cleaning [%s] (%d rows) ──", split, len(df))

    # Step 1: Handle missing names
    df, log = handle_missing_names(df)
    cleaning_log.append(log)

    # Step 2: Normalize text fields
    df, log = normalize_text_fields(df, columns=["Name", "Description"])
    cleaning_log.append(log)

    # Step 3: Fix Breed1/Breed2 swap (Breed2→Breed1 where Breed1=0)
    df, log = fix_breed_swap(df)
    cleaning_log.append(log)

    # Step 4: Fix remaining invalid Breed1 values
    valid_breed_ids = set(ref_breeds["BreedID"].unique()) | {0}
    df, log = fix_invalid_codes(df, "Breed1", valid_breed_ids, fallback=307)
    cleaning_log.append(log)

    # Step 5: Fix invalid Breed2 values
    df, log = fix_invalid_codes(df, "Breed2", valid_breed_ids, fallback=0)
    cleaning_log.append(log)

    # Step 6: Fix invalid Color codes (defensive — no issues found in validation)
    valid_color_ids = set(ref_colors["ColorID"].unique()) | {0}
    for col in ["Color1", "Color2", "Color3"]:
        df, log = fix_invalid_codes(df, col, valid_color_ids, fallback=0)
        cleaning_log.append(log)

    # Step 7: Fix invalid State codes (defensive)
    valid_state_ids = set(ref_states["StateID"].unique())
    df, log = fix_invalid_codes(df, "State", valid_state_ids, fallback=int(ref_states["StateID"].mode().iloc[0]))
    cleaning_log.append(log)

    # Step 8: Enforce canonical dtypes
    df, log = enforce_dtypes(df, schema)
    cleaning_log.append(log)

    logger.info("── Cleaning [%s] complete — %d steps applied ──", split, len(cleaning_log))
    return df, cleaning_log
