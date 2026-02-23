"""
Data validation utilities for the Adoption Accelerator project.

Provides schema validation, domain checks, referential integrity
verification, null/duplicate analysis, file-coverage audits, and
report generation.  All functions return structured result dicts so
that the notebook can aggregate them into a final validation report.

Functions
---------
validate_schema(df, schema_name)
    Validate columns, dtypes and shape against a predefined schema.
validate_domain(df, rules)
    Check column values fall within specified valid domains.
validate_referential_integrity(df, ref_df, fk_col, pk_col)
    Verify FK→PK mapping.  Return orphan records.
check_nulls(df)
    Compute null counts / fractions per column.
check_duplicates(df, key_col)
    Detect duplicates on a key column.
check_file_coverage(pet_ids, directory, extension)
    Verify which PetIDs have matching files in a directory.
generate_validation_report(results, report_dir)
    Persist aggregated results as JSON + Markdown.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from adoption_accelerator.data.schemas import TabularSchema, get_tabular_schema

logger = logging.getLogger("adoption_accelerator")


# ── Internal helpers ────────────────────────────────────────────────


def _dtype_compat(actual: str, expected: str) -> bool:
    """Check if *actual* pandas dtype is compatible with *expected*.

    Allows, e.g., ``int32`` to match ``int64``, and ``float32`` to
    match ``float64``.
    """
    _family = {
        "int": {"int8", "int16", "int32", "int64", "Int8", "Int16", "Int32", "Int64"},
        "float": {"float16", "float32", "float64", "Float32", "Float64"},
        "object": {"object", "string", "str"},
    }
    for family, members in _family.items():
        if expected in members and actual in members:
            return True
    return actual == expected


# ────────────────────────────────────────────────────────────────────
# Schema validation
# ────────────────────────────────────────────────────────────────────


def validate_schema(
    df: pd.DataFrame,
    schema_name: str = "train",
) -> dict[str, Any]:
    """Validate a DataFrame's columns and dtypes against the canonical schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    schema_name : str
        ``"train"`` or ``"test"`` — selects the corresponding schema.

    Returns
    -------
    dict
        Keys: ``passed`` (bool), ``missing_columns``, ``extra_columns``,
        ``dtype_mismatches`` (list of dicts), ``details`` (str).
    """
    schema: TabularSchema = get_tabular_schema(schema_name)
    expected = set(schema.column_names)
    actual = set(df.columns.tolist())

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    dtype_mismatches: list[dict[str, str]] = []
    for spec in schema.columns:
        if spec.name in df.columns:
            actual_dtype = str(df[spec.name].dtype)
            # Accept compatible families (e.g. int32 vs int64)
            if not _dtype_compat(actual_dtype, spec.dtype):
                dtype_mismatches.append(
                    {
                        "column": spec.name,
                        "expected": spec.dtype,
                        "actual": actual_dtype,
                    }
                )

    passed = len(missing) == 0  # extra cols are OK; dtypes are warnings
    details = "Schema validation passed." if passed else (
        f"Missing columns: {missing}"
    )

    result = {
        "check": "schema_validation",
        "split": schema_name,
        "passed": passed,
        "missing_columns": missing,
        "extra_columns": extra,
        "dtype_mismatches": dtype_mismatches,
        "details": details,
    }
    logger.info(
        "Schema validation [%s]: %s", schema_name, "PASS" if passed else "FAIL"
    )
    return result


# ────────────────────────────────────────────────────────────────────
# Domain validation
# ────────────────────────────────────────────────────────────────────


def validate_domain(
    df: pd.DataFrame,
    schema_name: str = "train",
) -> dict[str, Any]:
    """Check every column's values against the domain rules in the schema.

    Supports set-based domains (``valid_values``) and range-based domains
    (``min_value``).  FK columns are *not* checked here — use
    :func:`validate_referential_integrity` for those.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    schema_name : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    dict
        Keys: ``passed`` (bool), ``column_results`` (list of per-column
        dicts with violation details).
    """
    schema = get_tabular_schema(schema_name)
    column_results: list[dict[str, Any]] = []
    all_passed = True

    for spec in schema.columns:
        if spec.name not in df.columns:
            continue

        col = df[spec.name]
        violations: int = 0
        invalid_values: list[Any] = []

        # Set-based domain check
        if spec.valid_values is not None:
            mask = ~col.isin(spec.valid_values)
            violations = int(mask.sum())
            if violations > 0:
                invalid_values = sorted(col[mask].unique().tolist())

        # Range-based domain check
        elif spec.min_value is not None and spec.fk_reference is None:
            mask = col < spec.min_value
            violations = int(mask.sum())
            if violations > 0:
                invalid_values = sorted(col[mask].unique().tolist())

        col_passed = violations == 0
        if not col_passed:
            all_passed = False

        column_results.append(
            {
                "column": spec.name,
                "passed": col_passed,
                "violations": violations,
                "invalid_values": invalid_values[:20],  # cap for report size
            }
        )

    result = {
        "check": "domain_validation",
        "split": schema_name,
        "passed": all_passed,
        "column_results": column_results,
    }
    logger.info(
        "Domain validation [%s]: %s", schema_name, "PASS" if all_passed else "FAIL"
    )
    return result


# ────────────────────────────────────────────────────────────────────
# Referential integrity
# ────────────────────────────────────────────────────────────────────


def validate_referential_integrity(
    df: pd.DataFrame,
    ref_df: pd.DataFrame,
    fk_col: str,
    pk_col: str,
    allow_zero: bool = False,
) -> dict[str, Any]:
    """Verify all values in *fk_col* exist in the reference table's *pk_col*.

    Parameters
    ----------
    df : pd.DataFrame
        Data table containing the foreign key column.
    ref_df : pd.DataFrame
        Reference table containing the primary key column.
    fk_col : str
        Column name in *df*.
    pk_col : str
        Column name in *ref_df*.
    allow_zero : bool
        If ``True``, value ``0`` is exempt from the FK check (used for
        optional secondary breeds/colours).

    Returns
    -------
    dict
        Keys: ``passed`` (bool), ``total``, ``valid``, ``orphans``,
        ``orphan_values``, ``coverage_pct``.
    """
    fk_values = df[fk_col]
    valid_ids = set(ref_df[pk_col].unique())

    if allow_zero:
        # Exclude zero from the check
        check_mask = fk_values != 0
        fk_to_check = fk_values[check_mask]
    else:
        fk_to_check = fk_values

    total = len(fk_to_check)
    orphan_mask = ~fk_to_check.isin(valid_ids)
    orphan_count = int(orphan_mask.sum())
    orphan_values = sorted(fk_to_check[orphan_mask].unique().tolist()) if orphan_count else []
    valid_count = total - orphan_count
    coverage = round(valid_count / total * 100, 2) if total > 0 else 100.0

    passed = orphan_count == 0
    result = {
        "check": "referential_integrity",
        "fk_column": fk_col,
        "pk_column": pk_col,
        "passed": passed,
        "total": total,
        "valid": valid_count,
        "orphans": orphan_count,
        "orphan_values": orphan_values[:50],
        "coverage_pct": coverage,
    }
    logger.info(
        "Referential integrity [%s → %s]: %.1f%% coverage (%d orphans)",
        fk_col, pk_col, coverage, orphan_count,
    )
    return result


# ────────────────────────────────────────────────────────────────────
# Null analysis
# ────────────────────────────────────────────────────────────────────


def check_nulls(
    df: pd.DataFrame,
    schema_name: str = "train",
) -> dict[str, Any]:
    """Compute null counts and fractions per column.

    Classifies each column as *critical* (nulls not allowed by schema)
    or *expected* (nullable column).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    schema_name : str
        ``"train"`` or ``"test"`` — determines nullable expectations.

    Returns
    -------
    dict
        Keys: ``passed`` (bool — no critical nulls), ``columns`` (list).
    """
    schema = get_tabular_schema(schema_name)
    total_rows = len(df)
    columns: list[dict[str, Any]] = []
    critical_violations = False

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        null_fraction = round(null_count / total_rows, 4) if total_rows else 0.0

        spec = schema.get_column(col)
        nullable = spec.nullable if spec else True
        severity = "expected" if nullable else "critical"

        if null_count > 0 and not nullable:
            critical_violations = True

        columns.append(
            {
                "column": col,
                "null_count": null_count,
                "null_fraction": null_fraction,
                "nullable": nullable,
                "severity": severity if null_count > 0 else "ok",
            }
        )

    result = {
        "check": "null_analysis",
        "split": schema_name,
        "passed": not critical_violations,
        "total_rows": total_rows,
        "columns": columns,
    }
    logger.info(
        "Null analysis [%s]: %s", schema_name, "PASS" if result["passed"] else "FAIL"
    )
    return result


# ────────────────────────────────────────────────────────────────────
# Duplicate detection
# ────────────────────────────────────────────────────────────────────


def check_duplicates(
    df: pd.DataFrame,
    key_col: str,
) -> dict[str, Any]:
    """Detect duplicates on *key_col*.

    Parameters
    ----------
    df : pd.DataFrame
    key_col : str
        Column to check for uniqueness.

    Returns
    -------
    dict
        Keys: ``passed`` (bool), ``total``, ``unique``, ``duplicates``,
        ``duplicate_values``.
    """
    total = len(df)
    unique = df[key_col].nunique()
    dup_count = total - unique
    dup_values: list[Any] = []
    if dup_count > 0:
        dup_values = (
            df[key_col][df[key_col].duplicated(keep=False)]
            .unique()
            .tolist()[:50]
        )

    passed = dup_count == 0
    result = {
        "check": "duplicate_detection",
        "key_column": key_col,
        "passed": passed,
        "total": total,
        "unique": unique,
        "duplicates": dup_count,
        "duplicate_values": dup_values,
    }
    logger.info(
        "Duplicate check [%s]: %s (%d dupes)", key_col, "PASS" if passed else "FAIL", dup_count
    )
    return result


# ────────────────────────────────────────────────────────────────────
# File coverage
# ────────────────────────────────────────────────────────────────────


def check_file_coverage(
    pet_ids: pd.Series,
    directory: Path | str,
    extension: str = ".json",
) -> dict[str, Any]:
    """Verify which PetIDs have at least one matching file in *directory*.

    Files are expected to be named ``{PetID}{ext}`` or
    ``{PetID}-{index}{ext}``.

    Parameters
    ----------
    pet_ids : pd.Series
        Series of PetID values to look up.
    directory : Path or str
        Directory to scan.
    extension : str
        File extension (e.g. ``".json"``).

    Returns
    -------
    dict
        Keys: ``passed`` (bool — coverage ≥ 95%), ``total_pets``,
        ``covered``, ``missing``, ``coverage_pct``,
        ``missing_pet_ids`` (truncated list).
    """
    directory = Path(directory)
    if not directory.exists():
        return {
            "check": "file_coverage",
            "directory": str(directory),
            "passed": False,
            "total_pets": len(pet_ids),
            "covered": 0,
            "missing": len(pet_ids),
            "coverage_pct": 0.0,
            "missing_pet_ids": [],
            "details": f"Directory not found: {directory}",
        }

    # Build a set of PetID prefixes from filenames
    file_pet_ids: set[str] = set()
    for f in directory.iterdir():
        if f.suffix.lower() == extension.lower() and f.is_file():
            stem = f.stem  # e.g. "abc123" or "abc123-1"
            pet_prefix = stem.split("-")[0]
            file_pet_ids.add(pet_prefix)

    unique_ids = pet_ids.unique()
    total = len(unique_ids)
    covered = sum(1 for pid in unique_ids if pid in file_pet_ids)
    missing_count = total - covered
    coverage_pct = round(covered / total * 100, 2) if total > 0 else 100.0

    missing_ids = [pid for pid in unique_ids if pid not in file_pet_ids]

    passed = coverage_pct >= 95.0
    result = {
        "check": "file_coverage",
        "directory": str(directory.name),
        "passed": passed,
        "total_pets": total,
        "covered": covered,
        "missing": missing_count,
        "coverage_pct": coverage_pct,
        "missing_pet_ids": missing_ids[:30],
    }
    logger.info(
        "File coverage [%s]: %.1f%% (%d/%d)",
        directory.name, coverage_pct, covered, total,
    )
    return result


# ────────────────────────────────────────────────────────────────────
# Cross-column consistency
# ────────────────────────────────────────────────────────────────────


def check_cross_column_consistency(
    df: pd.DataFrame,
    image_dir: Path | str | None = None,
    sample_size: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Run cross-column logical consistency checks.

    Checks:
    1. ``Breed2 ≠ 0`` ⟹ ``Breed1 ≠ 0``
    2. ``Color2/3`` populated only when ``Color1`` is populated
    3. ``PhotoAmt`` vs actual image file count (sampled)

    Parameters
    ----------
    df : pd.DataFrame
    image_dir : Path or str, optional
        Directory with pet images. If provided, a sampled PhotoAmt
        check is performed.
    sample_size : int
        Number of PetIDs to sample for the PhotoAmt vs file-count check.
    seed : int
        Random seed for the sample.

    Returns
    -------
    dict
        Aggregated consistency results.
    """
    checks: list[dict[str, Any]] = []

    # 1. Breed2 ≠ 0 ⟹ Breed1 ≠ 0
    breed_issue = int(((df["Breed2"] != 0) & (df["Breed1"] == 0)).sum())
    checks.append({
        "rule": "Breed2 populated implies Breed1 populated",
        "violations": breed_issue,
        "passed": breed_issue == 0,
    })

    # 2. Color2/3 only populated if Color1 populated
    color2_issue = int(((df["Color2"] != 0) & (df["Color1"] == 0)).sum())
    color3_issue = int(((df["Color3"] != 0) & (df["Color1"] == 0)).sum())
    checks.append({
        "rule": "Color2 populated implies Color1 populated",
        "violations": color2_issue,
        "passed": color2_issue == 0,
    })
    checks.append({
        "rule": "Color3 populated implies Color1 populated",
        "violations": color3_issue,
        "passed": color3_issue == 0,
    })

    # 3. PhotoAmt vs actual image file count (sampled)
    if image_dir is not None:
        image_dir = Path(image_dir)
        if image_dir.exists():
            sample_ids = df["PetID"].sample(
                n=min(sample_size, len(df)), random_state=seed
            )
            mismatches = 0
            mismatch_examples: list[dict[str, Any]] = []
            for pid in sample_ids:
                expected_photos = int(
                    df.loc[df["PetID"] == pid, "PhotoAmt"].iloc[0]
                )
                actual_files = list(image_dir.glob(f"{pid}-*.jpg"))
                actual_count = len(actual_files)
                if expected_photos != actual_count:
                    mismatches += 1
                    if len(mismatch_examples) < 10:
                        mismatch_examples.append({
                            "PetID": pid,
                            "PhotoAmt": expected_photos,
                            "actual_files": actual_count,
                        })
            checks.append({
                "rule": "PhotoAmt matches actual image file count (sampled)",
                "sample_size": len(sample_ids),
                "violations": mismatches,
                "passed": mismatches == 0,
                "examples": mismatch_examples,
            })

    all_passed = all(c["passed"] for c in checks)
    return {
        "check": "cross_column_consistency",
        "passed": all_passed,
        "rules": checks,
    }


# ────────────────────────────────────────────────────────────────────
# Report generation
# ────────────────────────────────────────────────────────────────────


def generate_validation_report(
    results: list[dict[str, Any]],
    report_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Aggregate validation results into a structured report.

    Persists:
    - ``reports/validation_report.json``
    - ``reports/validation_summary.md``

    Parameters
    ----------
    results : list[dict]
        List of individual validation result dicts.
    report_dir : Path or str, optional
        Directory to write reports.  Defaults to ``<project>/reports/``.

    Returns
    -------
    dict
        The complete report object (also written to disk).
    """
    from adoption_accelerator import config as cfg  # local to avoid circular

    if report_dir is None:
        report_dir = cfg.REPORTS_DIR
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    overall_passed = all(r.get("passed", True) for r in results)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_passed": overall_passed,
        "total_checks": len(results),
        "passed_checks": sum(1 for r in results if r.get("passed", True)),
        "failed_checks": sum(1 for r in results if not r.get("passed", True)),
        "results": results,
    }

    # ── JSON report ─────────────────────────────────────────────────
    json_path = report_dir / "validation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Validation report saved to %s", json_path)

    # ── Markdown summary ────────────────────────────────────────────
    md_lines = [
        "# Data Validation Summary",
        "",
        f"**Generated:** {report['generated_at']}  ",
        f"**Overall:** {'PASS ✅' if overall_passed else 'FAIL ❌'}  ",
        f"**Checks:** {report['passed_checks']}/{report['total_checks']} passed  ",
        "",
        "---",
        "",
        "| # | Check | Split / Scope | Passed |",
        "|---|-------|---------------|--------|",
    ]
    for i, r in enumerate(results, 1):
        check_name = r.get("check", "unknown")
        scope = r.get("split", r.get("fk_column", r.get("directory", "")))
        status = "✅" if r.get("passed", True) else "❌"
        md_lines.append(f"| {i} | {check_name} | {scope} | {status} |")

    md_lines.extend(["", "---", ""])

    # Append detail sections for failures
    for r in results:
        if not r.get("passed", True):
            md_lines.append(f"### ❌ {r.get('check', 'unknown')}")
            md_lines.append(f"```json\n{json.dumps(r, indent=2, default=str)}\n```")
            md_lines.append("")

    md_path = report_dir / "validation_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Validation summary saved to %s", md_path)

    return report
