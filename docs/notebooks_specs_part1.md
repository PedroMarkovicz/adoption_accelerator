# Adoption Accelerator — Notebook Specifications (Part 1)

> **Version:** 1.0  
> **Status:** Draft  
> **Last Updated:** 2026-02-21  
> **Author:** Pedro Markovicz  
> **Scope:** Notebooks `00` through `06` — Project Setup, Data Pipeline & Exploratory Data Analysis  
> **Parent Document:** `project_specs.md` v1.0

---

## Table of Contents

1. [Overview & Governance](#1-overview--governance)
2. [Notebook 00 — Project Setup](#2-notebook-00--project-setup)
3. [Notebook 01 — Data Ingestion](#3-notebook-01--data-ingestion)
4. [Notebook 02 — Data Validation](#4-notebook-02--data-validation)
5. [Notebook 03 — Data Cleaning](#5-notebook-03--data-cleaning)
6. [Notebook 04 — EDA: Tabular](#6-notebook-04--eda-tabular)
7. [Notebook 05 — EDA: Text & Sentiment](#7-notebook-05--eda-text--sentiment)
8. [Notebook 06 — EDA: Images & Metadata](#8-notebook-06--eda-images--metadata)
9. [Cross-Notebook `src/` Architecture Map](#9-cross-notebook-src-architecture-map)
10. [Artifact Registry (Part 1)](#10-artifact-registry-part-1)
11. [Conventions & Standards](#11-conventions--standards)

---

## 1. Overview & Governance

### 1.1. Purpose

This document defines the execution specification for the first seven notebooks of the Adoption Accelerator project. It serves as an **experimentation governance manual** — prescribing what each notebook must accomplish, what it must produce, what it must validate, and what reusable logic must be extracted into the `src/adoption_accelerator/` core library.

### 1.2. Guiding Principles

| Principle | Enforcement |
|-----------|-------------|
| **Notebooks are thin orchestrators** | All reusable logic lives in `src/`. Notebooks call functions, display results, and document findings — they never define reusable functions inline. |
| **Immutable raw data** | `data/raw/` is never modified after initial download. Every transformation produces new artifacts in downstream directories. |
| **Parquet for all processed data** | CSVs are only read; all outputs are Apache Parquet with enforced schemas. |
| **Deterministic reproducibility** | Fixed seed (`SEED = 42`), pinned dependencies, and config-driven parameters ensure any notebook can be re-run from scratch with identical results. |
| **Unidirectional dependencies** | Each notebook depends only on artifacts from previous notebooks. No backward or circular dependencies. |
| **Validation gates** | Every notebook includes explicit pass/fail checkpoints. The notebook must not proceed past a failed gate. |

### 1.3. Notebook Naming Convention

```
{NN}_{descriptive_slug}.ipynb
```

- `NN` — Two-digit sequential number (execution order).
- `descriptive_slug` — Lowercase, underscored, concise label.

### 1.4. Standard Notebook Structure

Every notebook follows a consistent internal layout:

| Section | Purpose |
|---------|---------|
| **Header Cell** | Title, objective, dependencies, expected runtime |
| **Imports & Config** | Import `src/` modules, load config, set seeds |
| **Execution Cells** | Step-by-step orchestration calling `src/` functions |
| **Validation Gate(s)** | Assert-based checkpoints with clear pass/fail messages |
| **Artifact Output** | Persist outputs (Parquet, JSON, figures) to designated directories |
| **Summary Cell** | Key findings, metrics, and pointer to the next notebook |

### 1.5. Dependency Chain (Part 1)

```
00_project_setup
      │
      ▼
01_data_ingestion
      │
      ▼
02_data_validation
      │
      ▼
03_data_cleaning
      │
      ├──────────────────────────────────┐
      ▼                                  ▼
04_eda_tabular        05_eda_text_sentiment
                                         │
                      ┌──────────────────┘
                      ▼
              06_eda_images_metadata
```

> Notebooks `04`, `05`, and `06` depend on `03` (cleaned data) but are **mutually independent** in their EDA scope. They can be executed in any order after `03` completes.

---

## 2. Notebook 00 — Project Setup

### 2.1. Objective

Bootstrap the development environment: install dependencies, configure paths, create the directory structure, authenticate with Kaggle, download the raw dataset, and verify file inventory integrity.

### 2.2. Scope

| In Scope | Out of Scope |
|----------|-------------|
| Dependency installation (`pyproject.toml`) | Data loading or parsing |
| Directory scaffold creation | Any data transformation |
| Kaggle authentication verification | Exploratory analysis |
| Dataset download and extraction | Schema validation |
| File inventory count verification | — |

### 2.3. Step-by-Step Process

| # | Step | Description |
|---|------|-------------|
| 1 | Install package | Install the project as editable package (`pip install -e .`) to make `src/adoption_accelerator` importable. |
| 2 | Import & version check | Import core libraries and print versions for reproducibility logging. |
| 3 | Load configuration | Import path constants from `src/adoption_accelerator/config.py`. All paths must originate from config — no hardcoded paths in the notebook. |
| 4 | Create directory structure | Ensure all project directories exist (`data/raw/`, `data/cleaned/`, `data/features/`, `data/submissions/`, `reports/figures/`, `reports/metrics/`, `artifacts/models/`). Idempotent operation. |
| 5 | Set global defaults | Set random seed (`SEED = 42`), pandas display options, matplotlib/seaborn theme, logging level. |
| 6 | Verify Kaggle credentials | Assert that `~/.kaggle/kaggle.json` exists and is readable. Fail with clear instructions if missing. |
| 7 | Download dataset | Download `petfinder-adoption-prediction` from Kaggle API into `data/raw/`. Skip if already downloaded (idempotent). |
| 8 | Extract dataset | Unzip into `data/raw/`. Organize files into the expected subdirectory layout. |
| 9 | File inventory | List all files in `data/raw/` with counts per directory and file sizes. |
| 10 | **Validation gate** | Assert expected file counts (see §2.4). |

### 2.4. Validation Checkpoints

| Gate ID | Assertion | Expected Value |
|---------|-----------|---------------|
| `G00-1` | `data/raw/train/train.csv` exists | `True` |
| `G00-2` | `data/raw/test/test.csv` exists | `True` |
| `G00-3` | File count in `train_images/` | 58,311 |
| `G00-4` | File count in `test_images/` | 14,465 |
| `G00-5` | File count in `train_metadata/` | 58,311 |
| `G00-6` | File count in `test_metadata/` | 14,465 |
| `G00-7` | File count in `train_sentiment/` | 14,442 |
| `G00-8` | File count in `test_sentiment/` | 3,865 |
| `G00-9` | Reference CSVs exist (`breed_labels.csv`, `color_labels.csv`, `state_labels.csv`) | `True` |
| `G00-10` | `src/adoption_accelerator` is importable | `True` |

### 2.5. Expected Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Raw dataset (complete) | `data/raw/` | CSV, JPEG, JSON |
| Directory scaffold | All project directories | Filesystem |

### 2.6. Dependencies

- None (this is the entry-point notebook).

### 2.7. Required `src/` Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `get_project_root()` | `src/adoption_accelerator/utils/paths.py` | Return the absolute path to the project root directory. |
| `get_data_path(stage, subset)` | `src/adoption_accelerator/utils/paths.py` | Resolve data paths by stage (`raw`, `cleaned`, `features`) and subset (`train`, `test`). |
| `ensure_directories()` | `src/adoption_accelerator/utils/paths.py` | Create the full directory scaffold idempotently. |
| `setup_logging(level)` | `src/adoption_accelerator/utils/logging.py` | Configure project-wide logging with consistent formatting. |

**Rationale:** Path resolution and directory management are utility concerns shared by every downstream module. They must be centralized to prevent path-string duplication across notebooks.

---

## 3. Notebook 01 — Data Ingestion

### 3.1. Objective

Load all raw data sources into memory in their native formats, perform initial structural inspection (shape, dtypes, head/tail), and persist a **data inventory report**. This notebook answers: *"What do we have?"*

### 3.2. Scope

| In Scope | Out of Scope |
|----------|-------------|
| Loading `train.csv` and `test.csv` as DataFrames | Schema validation (→ notebook 02) |
| Loading reference tables (`breed_labels`, `color_labels`, `state_labels`) | Data cleaning (→ notebook 03) |
| Loading sample sentiment and metadata JSONs | Statistical analysis or EDA |
| Structural inspection: shape, dtypes, memory usage, null counts | Any data transformation |
| Documenting column-level summary for each data source | Feature engineering |
| Merging reference labels for display (not for processing) | — |

### 3.3. Step-by-Step Process

| # | Step | Description |
|---|------|-------------|
| 1 | Import & config | Import `src/` ingestion module. Load path config. |
| 2 | Load tabular data | Read `train.csv` and `test.csv` using the ingestion module. Store as DataFrames. |
| 3 | Inspect tabular data | For each DataFrame: print shape, dtypes, memory usage, first/last 5 rows, null count per column, unique count per column. |
| 4 | Load reference tables | Read `breed_labels.csv`, `color_labels.csv`, `state_labels.csv`. Inspect each. |
| 5 | Map reference labels | Join breed, color, and state names onto train/test for human-readable inspection (display only — not persisted as processed data). |
| 6 | Inspect sentiment JSONs | Load 3–5 sample sentiment JSONs from `train_sentiment/`. Display their structure and fields. Document the schema. |
| 7 | Inspect metadata JSONs | Load 3–5 sample metadata JSONs from `train_metadata/`. Display their structure (`labelAnnotations`, `dominantColors`, `cropHints`). Document the schema. |
| 8 | Inspect image files | Load 2–3 sample images from `train_images/`. Display with `matplotlib`. Report image dimensions and file sizes. |
| 9 | Data inventory summary | Produce a summary table of all data sources: source, format, row/file count, column count, memory footprint. |
| 10 | **Validation gate** | Assert expected shapes and column names (see §3.4). |

### 3.4. Validation Checkpoints

| Gate ID | Assertion | Expected Value |
|---------|-----------|---------------|
| `G01-1` | `train.csv` row count | 14,993 |
| `G01-2` | `test.csv` row count | 3,972 |
| `G01-3` | `train.csv` column count | 24 |
| `G01-4` | `test.csv` column count | 23 (no `AdoptionSpeed`) |
| `G01-5` | `AdoptionSpeed` present in train, absent in test | `True` |
| `G01-6` | `breed_labels.csv` row count | 307 |
| `G01-7` | `color_labels.csv` row count | 7 |
| `G01-8` | `state_labels.csv` row count | 15 |
| `G01-9` | All expected columns present in `train.csv` | Match schema from `project_specs.md` §2.5 |
| `G01-10` | `PetID` uniqueness in train and test | All unique |

### 3.5. Expected Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Data inventory report | `reports/data_inventory.md` or `reports/data_inventory.json` | Markdown or JSON |

> **No processed data files are produced.** This notebook is read-only with respect to data — it only inspects.

### 3.6. Dependencies

| Dependency | Source |
|------------|--------|
| Raw dataset in `data/raw/` | Notebook `00` |
| `src/adoption_accelerator` importable | Notebook `00` |

### 3.7. Required `src/` Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `load_tabular(split)` | `src/adoption_accelerator/data/ingestion.py` | Load `train.csv` or `test.csv` as a pandas DataFrame with appropriate dtypes. `split` ∈ {`"train"`, `"test"`}. |
| `load_reference_table(name)` | `src/adoption_accelerator/data/ingestion.py` | Load a reference CSV (`"breed"`, `"color"`, `"state"`) as a DataFrame. |
| `load_sentiment_json(pet_id, index)` | `src/adoption_accelerator/data/ingestion.py` | Load a single sentiment JSON file for a given `PetID` and image index. |
| `load_metadata_json(pet_id, index)` | `src/adoption_accelerator/data/ingestion.py` | Load a single metadata JSON file for a given `PetID` and image index. |
| `load_image(pet_id, index)` | `src/adoption_accelerator/data/ingestion.py` | Load a single image file as a NumPy array or PIL Image. |
| `get_file_inventory(directory)` | `src/adoption_accelerator/data/ingestion.py` | Recursively list files in a directory with counts and sizes. Return a summary DataFrame. |

**Rationale:** All I/O operations must be centralized in `data/ingestion.py` so that path resolution, dtype enforcement, and error handling are consistent across notebooks and pipelines.

---

## 4. Notebook 02 — Data Validation

### 4.1. Objective

Apply systematic schema validation, referential integrity checks, and domain-specific constraint validation to all raw tabular data. Produce a **validation report** documenting every check and its result. This notebook answers: *"Is the data structurally correct and internally consistent?"*

### 4.2. Scope

| In Scope | Out of Scope |
|----------|-------------|
| Column type validation against expected schema | Fixing or cleaning data (→ notebook 03) |
| Domain value-range checks (e.g., `Type` ∈ {1, 2}) | Statistical analysis or EDA |
| Referential integrity (Breed1/2 → breed_labels, Color1/2/3 → color_labels, State → state_labels) | Feature engineering |
| Null pattern analysis (which columns, what fraction) | Text or image analysis |
| Duplicate detection (PetID uniqueness) | — |
| Cross-column logical consistency checks | — |
| Sentiment/metadata file-to-PetID coverage analysis | — |

### 4.3. Step-by-Step Process

| # | Step | Description |
|---|------|-------------|
| 1 | Import & config | Import `src/` validation and ingestion modules. |
| 2 | Load data | Load train and test DataFrames via ingestion module. |
| 3 | Schema validation | Validate all columns exist, have expected dtypes, and have expected names. Log deviations. |
| 4 | Domain value checks | For each categorical/ordinal column, verify all values fall within the expected domain (see §4.4 validation rules). Flag out-of-range values. |
| 5 | Null analysis | For each column: count nulls, compute null fraction, classify as critical vs. expected. |
| 6 | Duplicate check | Verify `PetID` uniqueness in both train and test. |
| 7 | Referential integrity | Verify every `Breed1`, `Breed2`, `Color1`, `Color2`, `Color3`, and `State` value maps to a valid entry in the corresponding reference table. Log orphan IDs. |
| 8 | Cross-column consistency | Verify logical constraints: (a) if `Breed2 ≠ 0`, then `Breed2` must exist in breed_labels; (b) `Color2` and `Color3` should only be populated if `Color1` is populated; (c) `PhotoAmt` should correlate with actual image file count for a sample. |
| 9 | File coverage analysis | Verify that every `PetID` in `train.csv` has at least one matching file in `train_sentiment/` and `train_metadata/`. Report coverage percentages. |
| 10 | Target distribution (train only) | Compute `AdoptionSpeed` class distribution. Report counts and proportions. Flag severe imbalance if present. |
| 11 | **Validation gate** | Assert all critical checks pass (see §4.5). |
| 12 | Validation report | Persist the validation report as a structured artifact. |

### 4.4. Domain Validation Rules

| Column | Valid Values | Null Allowed |
|--------|-------------|-------------|
| `Type` | {1, 2} | No |
| `Gender` | {1, 2, 3} | No |
| `MaturitySize` | {1, 2, 3, 4, 0} | No |
| `FurLength` | {1, 2, 3, 0} | No |
| `Vaccinated` | {1, 2, 3} | No |
| `Dewormed` | {1, 2, 3} | No |
| `Sterilized` | {1, 2, 3} | No |
| `Health` | {1, 2, 3} | No |
| `Age` | ≥ 0 (integer) | No |
| `Quantity` | ≥ 1 (integer) | No |
| `Fee` | ≥ 0 (numeric) | No |
| `VideoAmt` | ≥ 0 (integer) | No |
| `PhotoAmt` | ≥ 0 (numeric) | No |
| `Breed1` | FK → breed_labels.BreedID | No |
| `Breed2` | 0 or FK → breed_labels.BreedID | No |
| `Color1` | FK → color_labels.ColorID | No |
| `Color2` | 0 or FK → color_labels.ColorID | No |
| `Color3` | 0 or FK → color_labels.ColorID | No |
| `State` | FK → state_labels.StateID | No |
| `Name` | Free text | Yes (may be empty string or NaN) |
| `Description` | Free text | Yes (may be empty) |
| `AdoptionSpeed` (train) | {0, 1, 2, 3, 4} | No |

### 4.5. Validation Checkpoints

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| `G02-1` | All expected columns present in train and test | **Critical** — halt |
| `G02-2` | `PetID` is unique in train; unique in test | **Critical** — halt |
| `G02-3` | `Type` values all in {1, 2} | **Critical** — halt |
| `G02-4` | `AdoptionSpeed` values all in {0, 1, 2, 3, 4} | **Critical** — halt |
| `G02-5` | All categorical/ordinal columns within valid domain | **Warning** — log and continue |
| `G02-6` | Referential integrity for `Breed1` ≥ 99% | **Warning** — log orphans |
| `G02-7` | Sentiment file coverage ≥ 95% of PetIDs | **Warning** — log missing |
| `G02-8` | Metadata file coverage consistent with `PhotoAmt` | **Warning** — log mismatches |
| `G02-9` | No unexpected null columns in required fields | **Critical** — halt |
| `G02-10` | Target class distribution — no class < 5% of total | **Informational** — log |

### 4.6. Expected Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Validation report | `reports/validation_report.json` | JSON |
| Validation summary | `reports/validation_summary.md` | Markdown |

> **No data modifications.** This notebook is strictly diagnostic. All findings feed into notebook `03` for remediation.

### 4.7. Dependencies

| Dependency | Source |
|------------|--------|
| Raw data loaded and inspected | Notebook `01` |

### 4.8. Required `src/` Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `validate_schema(df, schema_name)` | `src/adoption_accelerator/data/validation.py` | Validate that a DataFrame's columns, dtypes, and shape conform to a predefined schema. Return a validation result object. |
| `validate_domain(df, rules)` | `src/adoption_accelerator/data/validation.py` | Check that each column's values fall within specified valid domains. Return per-column pass/fail with violation counts. |
| `validate_referential_integrity(df, ref_df, fk_col, pk_col)` | `src/adoption_accelerator/data/validation.py` | Verify all values in `fk_col` exist in the reference table's `pk_col`. Return orphan records. |
| `check_nulls(df)` | `src/adoption_accelerator/data/validation.py` | Compute null counts and fractions per column. Classify as critical vs. expected. |
| `check_duplicates(df, key_col)` | `src/adoption_accelerator/data/validation.py` | Detect duplicates on `key_col`. Return duplicate rows if any. |
| `check_file_coverage(pet_ids, directory, extension)` | `src/adoption_accelerator/data/validation.py` | Verify which PetIDs have matching files in a directory. Return coverage statistics. |
| `generate_validation_report(results)` | `src/adoption_accelerator/data/validation.py` | Aggregate all validation results into a structured report (JSON + markdown). |
| `get_tabular_schema()` | `src/adoption_accelerator/data/schemas.py` | Return the expected schema definition for the tabular data (column names, types, constraints). |

**Rationale:** Validation logic is required at multiple stages (initial validation, post-cleaning re-validation, feature pipeline pre-checks). Centralizing in `data/validation.py` with a schema-driven approach enables reuse and ensures consistent quality gates.

---

## 5. Notebook 03 — Data Cleaning

### 5.1. Objective

Apply targeted data transformations to remediate all issues identified in the validation report (notebook 02). Produce clean, schema-conformant Parquet files ready for exploratory analysis and downstream feature engineering. This notebook answers: *"Is the data now ready for analysis and modeling?"*

### 5.2. Scope

| In Scope | Out of Scope |
|----------|-------------|
| Null handling (imputation or flagging) | Feature engineering (→ notebooks 07–10) |
| Dtype enforcement and casting | Exploratory visualizations (→ notebooks 04–06) |
| Free-text field normalization (`Name`, `Description`) | Model training |
| Handling of invalid/orphan categorical codes | Sentiment or metadata feature extraction |
| Encoding-safe column name standardization | Image preprocessing |
| Persisting cleaned data as Parquet | — |
| Post-cleaning re-validation | — |

### 5.3. Step-by-Step Process

| # | Step | Description |
|---|------|-------------|
| 1 | Import & config | Import `src/` cleaning and validation modules. |
| 2 | Load raw data | Load train and test DataFrames via ingestion module. |
| 3 | Load validation report | Load the validation report from notebook 02 to identify issues requiring remediation. |
| 4 | Handle missing names | Apply the naming cleaning strategy: empty strings and NaN in `Name` → standardized null representation. Create `has_name` flag (future feature, but the flag is created here for completeness). |
| 5 | Normalize text fields | Strip leading/trailing whitespace from `Name` and `Description`. Normalize encoding (UTF-8). |
| 6 | Handle invalid breed codes | Address any `Breed1`/`Breed2` values not found in breed_labels: map to a designated "Unknown" category or 0. |
| 7 | Handle invalid color codes | Address any `Color1`/`Color2`/`Color3` values not found in color_labels: map invalid values to 0. |
| 8 | Handle invalid state codes | Address any `State` values not found in state_labels. |
| 9 | Enforce dtypes | Cast all columns to their canonical dtypes as defined in the schema (integers for codes, floats where appropriate). |
| 10 | Standardize column names | Ensure all column names follow a consistent convention (lowercase_snake_case or preserve original — choose one and enforce). |
| 11 | Post-cleaning re-validation | Run the full validation suite (notebook 02's functions) against the cleaned DataFrames. All critical gates must now pass. |
| 12 | **Validation gate** | Assert all critical validation checks pass on cleaned data (see §5.4). |
| 13 | Persist cleaned data | Save train and test DataFrames as Parquet to `data/cleaned/train.parquet` and `data/cleaned/test.parquet`. |
| 14 | Cleaning summary | Log a summary of all transformations applied, with before/after counts. |

### 5.4. Validation Checkpoints

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| `G03-1` | Cleaned train/test pass full schema validation | **Critical** — halt |
| `G03-2` | No critical null violations remain | **Critical** — halt |
| `G03-3` | All referential integrity checks pass (Breed, Color, State) | **Critical** — halt |
| `G03-4` | Cleaned train row count == raw train row count (no accidental row drops) | **Critical** — halt |
| `G03-5` | Cleaned test row count == raw test row count | **Critical** — halt |
| `G03-6` | `data/cleaned/train.parquet` file exists and is readable | **Critical** — halt |
| `G03-7` | `data/cleaned/test.parquet` file exists and is readable | **Critical** — halt |
| `G03-8` | Parquet round-trip integrity: reload and compare shape/dtypes | **Critical** — halt |
| `G03-9` | `AdoptionSpeed` distribution unchanged vs. raw | **Critical** — halt |

### 5.5. Expected Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Cleaned train data | `data/cleaned/train.parquet` | Parquet |
| Cleaned test data | `data/cleaned/test.parquet` | Parquet |
| Cleaning log | `reports/cleaning_log.json` | JSON |

### 5.6. Dependencies

| Dependency | Source |
|------------|--------|
| Raw data files in `data/raw/` | Notebook `00` |
| Validation report | Notebook `02` |
| Validation functions | `src/adoption_accelerator/data/validation.py` |

### 5.7. Required `src/` Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `clean_tabular(df, split)` | `src/adoption_accelerator/data/cleaning.py` | Orchestrate all cleaning steps on a tabular DataFrame. Return the cleaned DataFrame and a cleaning log. |
| `handle_missing_names(df)` | `src/adoption_accelerator/data/cleaning.py` | Standardize null/empty `Name` values. Add `has_name` binary flag. |
| `normalize_text_fields(df, columns)` | `src/adoption_accelerator/data/cleaning.py` | Strip whitespace, normalize encoding on specified text columns. |
| `fix_invalid_codes(df, col, valid_ids, fallback)` | `src/adoption_accelerator/data/cleaning.py` | Replace values in `col` not present in `valid_ids` with `fallback`. Return count of fixes. |
| `enforce_dtypes(df, schema)` | `src/adoption_accelerator/data/cleaning.py` | Cast DataFrame columns to match the canonical schema dtypes. |
| `save_parquet(df, path)` | `src/adoption_accelerator/data/ingestion.py` | Save a DataFrame as Parquet with consistent settings (compression, engine). |
| `load_parquet(path)` | `src/adoption_accelerator/data/ingestion.py` | Load a Parquet file as a DataFrame. |

**Rationale:** Cleaning operations must be deterministic and replayable. Encapsulating them in `data/cleaning.py` ensures the same transformations can be applied by the notebook interactively and by the pipeline script non-interactively, producing identical results.

---

## 6. Notebook 04 — EDA: Tabular

### 6.1. Objective

Conduct comprehensive exploratory data analysis on the **tabular (structured) features** of the cleaned dataset. Generate statistical summaries, distribution profiles, correlation analyses, and bivariate analyses against the target variable (`AdoptionSpeed`). Produce publication-quality visualizations and a structured findings report. This notebook answers: *"What patterns exist in the structured data that may predict adoption speed?"*

### 6.2. Scope

| In Scope | Out of Scope |
|----------|-------------|
| Univariate distribution analysis for all tabular columns | Text analysis (`Description`) — → notebook 05 |
| Bivariate analysis: each feature vs. `AdoptionSpeed` | Image analysis — → notebook 06 |
| Correlation analysis (numeric features) | Sentiment or metadata JSON analysis — → notebooks 05/06 |
| Categorical frequency analysis | Feature engineering |
| Target variable deep-dive (class distribution, balance) | Model training |
| Rescuer-level aggregate statistics | — |
| Cross-tabulation of key categorical pairs | — |
| Outlier identification in numeric features | — |
| Interaction hypothesis documentation | — |

### 6.3. Step-by-Step Process

| # | Step | Description |
|---|------|-------------|
| 1 | Import & config | Import `src/` modules. Load cleaned data from Parquet. |
| 2 | Target analysis | Deep-dive into `AdoptionSpeed`: class counts, proportions, bar chart. Quantify class imbalance ratio. |
| 3 | Univariate — numeric features | For `Age`, `Fee`, `Quantity`, `PhotoAmt`, `VideoAmt`: descriptive statistics (mean, median, std, skewness, kurtosis), histograms, box plots. Identify outliers. |
| 4 | Univariate — categorical features | For `Type`, `Gender`, `MaturitySize`, `FurLength`, `Vaccinated`, `Dewormed`, `Sterilized`, `Health`: value counts and bar charts. |
| 5 | Univariate — high-cardinality features | For `Breed1`, `Breed2`, `State`, `Color1`: top-N frequency analysis. Long-tail identification. |
| 6 | Bivariate — numeric vs. target | For each numeric feature: grouped statistics by `AdoptionSpeed` class, violin/box plots per class. Assess discriminative power. |
| 7 | Bivariate — categorical vs. target | For each categorical feature: stacked bar charts or heatmaps showing adoption speed distribution per category. |
| 8 | Correlation analysis | Compute correlation matrix for all numeric features (Pearson). Compute Cramér's V for categorical–categorical associations. Visualize as heatmaps. |
| 9 | Breed analysis | Top breeds by volume and by mean adoption speed. Dog vs. Cat breed distribution. Mixed breed prevalence. |
| 10 | Geographic analysis | Adoption speed distribution by `State`. Volume by state. |
| 11 | Fee analysis | Fee distribution by adoption speed class. Free vs. paid adoption correlation. |
| 12 | Rescuer analysis | Compute rescuer-level statistics: listing count, mean adoption speed, median fee, mean `PhotoAmt`. Identify high-volume rescuers. Assess whether rescuer identity is predictive. |
| 13 | Interaction hypotheses | Document observed interactions (e.g., Age × Type, Fee × State, Vaccinated × Health) as hypotheses for future feature engineering. |
| 14 | Outlier summary | Summarize outliers detected in numeric features. Document thresholds and counts. |
| 15 | Save figures | Persist all key visualizations to `reports/figures/eda_tabular/`. |
| 16 | **Validation gate** | Assert that all expected analyses were completed and figures were saved (see §6.4). |
| 17 | Findings report | Persist a structured findings summary. |

### 6.4. Validation Checkpoints

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| `G04-1` | Cleaned train data loaded with expected shape and columns | **Critical** — halt |
| `G04-2` | All 5 `AdoptionSpeed` classes present in train data | **Critical** — halt |
| `G04-3` | At least one figure saved per analysis section | **Informational** — warn |
| `G04-4` | No NaN values introduced during analysis (data integrity preserved) | **Critical** — halt |
| `G04-5` | Findings report persisted | **Warning** — log |

### 6.5. Expected Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Tabular EDA figures | `reports/figures/eda_tabular/` | PNG |
| Tabular EDA findings | `reports/eda_tabular_findings.json` | JSON |

### 6.6. Dependencies

| Dependency | Source |
|------------|--------|
| `data/cleaned/train.parquet` | Notebook `03` |
| Reference tables (for label mapping) | `data/raw/*.csv` |

### 6.7. Required `src/` Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `plot_target_distribution(y)` | `src/adoption_accelerator/utils/visualization.py` | Plot and return the AdoptionSpeed class distribution chart. |
| `plot_numeric_distribution(df, col, target_col)` | `src/adoption_accelerator/utils/visualization.py` | Generate histogram + box plot for a numeric column, optionally grouped by target. |
| `plot_categorical_distribution(df, col, target_col)` | `src/adoption_accelerator/utils/visualization.py` | Generate bar chart for a categorical column, optionally stacked/grouped by target. |
| `plot_correlation_matrix(df, columns, method)` | `src/adoption_accelerator/utils/visualization.py` | Compute and visualize a correlation heatmap. |
| `compute_cramers_v(df, col1, col2)` | `src/adoption_accelerator/utils/visualization.py` | Compute Cramér's V statistic for two categorical columns. |
| `compute_descriptive_stats(df, columns)` | `src/adoption_accelerator/utils/visualization.py` | Return a DataFrame of descriptive statistics (mean, median, std, skew, kurtosis) for specified columns. |
| `save_figure(fig, name, subdir)` | `src/adoption_accelerator/utils/visualization.py` | Save a matplotlib/seaborn figure to `reports/figures/{subdir}/{name}.png`. |

**Rationale:** Visualization and statistical summary functions are reused across all three EDA notebooks (04, 05, 06) and will also be needed for report generation and the Streamlit exploration page. Centralizing in `utils/visualization.py` prevents duplication and ensures visual consistency.

---

## 7. Notebook 05 — EDA: Text & Sentiment

### 7.1. Objective

Conduct exploratory data analysis on the **text modality**: the `Description` column (free-text listing descriptions) and the pre-computed Google NLP sentiment analysis JSONs. Document text characteristics, language patterns, and sentiment correlations with `AdoptionSpeed`. This notebook answers: *"What textual and sentiment features may be predictive of adoption speed?"*

### 7.2. Scope

| In Scope | Out of Scope |
|----------|-------------|
| `Description` length, word count, and character analysis | Text embedding extraction (→ notebook 08) |
| Language detection and multilingual assessment | NLP model fine-tuning |
| Missing description analysis | Feature engineering |
| Sentiment JSON parsing and aggregation at PetID level | Image analysis (→ notebook 06) |
| Sentiment score/magnitude distributions | Model training |
| Sentiment vs. `AdoptionSpeed` correlation | — |
| Entity analysis from sentiment JSONs | — |
| Word frequency and keyword analysis | — |

### 7.3. Step-by-Step Process

| # | Step | Description |
|---|------|-------------|
| 1 | Import & config | Import `src/` modules. Load cleaned train data from Parquet. |
| 2 | Description availability | Count and report: listings with non-empty descriptions vs. empty/null. Compute coverage percentage. |
| 3 | Text length analysis | Compute `description_length` (chars), `word_count`, `sentence_count` for each listing. Plot distributions. Analyze distributions per `AdoptionSpeed` class. |
| 4 | Language analysis | Sample descriptions and identify predominant language(s). Assess whether multilingual content is present. Document implications for text embedding strategy. |
| 5 | Missing description profiling | Profile listings without descriptions: how does their adoption speed distribution compare to listings with descriptions? |
| 6 | Word frequency analysis | Compute top-N most frequent words (after basic tokenization and stop-word removal). Generate word frequency charts. Optionally: word clouds per adoption speed class. |
| 7 | Load sentiment JSONs | Use `src/` ingestion functions to batch-load all train sentiment JSONs. Parse into a structured DataFrame at PetID-level with: `doc_sentiment_score`, `doc_sentiment_magnitude`, `sentence_count`, `mean_sentence_score`, `entity_count`. |
| 8 | Sentiment coverage | Report what fraction of train PetIDs have corresponding sentiment JSONs. Profile listings without sentiment data. |
| 9 | Sentiment distributions | Plot distributions of `doc_sentiment_score` and `doc_sentiment_magnitude`. Box plots per `AdoptionSpeed` class. |
| 10 | Sentiment vs. target | Compute correlation between sentiment features and `AdoptionSpeed`. Visualize with scatter plots or grouped statistics. |
| 11 | Entity analysis | Extract entity types and top entities from sentiment JSONs. Assess entity diversity and salience. |
| 12 | Key findings | Document observed patterns and hypotheses for text feature engineering (e.g., "longer descriptions correlate with faster adoption"). |
| 13 | Save figures | Persist all key visualizations to `reports/figures/eda_text/`. |
| 14 | **Validation gate** | Assert analysis completeness (see §7.4). |

### 7.4. Validation Checkpoints

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| `G05-1` | Cleaned train data loaded with `Description` column | **Critical** — halt |
| `G05-2` | Sentiment JSONs loaded for ≥ 95% of PetIDs | **Warning** — log |
| `G05-3` | No exceptions during batch sentiment JSON parsing | **Critical** — halt |
| `G05-4` | Sentiment DataFrame has expected columns: `PetID`, `doc_sentiment_score`, `doc_sentiment_magnitude` | **Critical** — halt |
| `G05-5` | At least one figure saved per analysis section | **Informational** — warn |
| `G05-6` | Findings report persisted | **Warning** — log |

### 7.5. Expected Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Text/Sentiment EDA figures | `reports/figures/eda_text/` | PNG |
| Text EDA findings | `reports/eda_text_findings.json` | JSON |
| Aggregated sentiment DataFrame (optional interim) | `data/processed/sentiment_aggregated.parquet` | Parquet |

### 7.6. Dependencies

| Dependency | Source |
|------------|--------|
| `data/cleaned/train.parquet` | Notebook `03` |
| `data/raw/train_sentiment/*.json` | Notebook `00` (download) |

### 7.7. Required `src/` Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `load_all_sentiment_jsons(split)` | `src/adoption_accelerator/data/ingestion.py` | Batch-load all sentiment JSONs for a given split. Return a list of parsed dicts keyed by `PetID`. |
| `parse_sentiment_to_dataframe(sentiment_records)` | `src/adoption_accelerator/data/ingestion.py` | Convert raw sentiment JSON records into a structured DataFrame with document-level and sentence-level aggregates. |
| `compute_text_statistics(descriptions)` | `src/adoption_accelerator/utils/visualization.py` | Compute character length, word count, and sentence count for a Series of text. Return a DataFrame of statistics. |
| `plot_text_length_distributions(stats_df, target_col)` | `src/adoption_accelerator/utils/visualization.py` | Plot text length distributions, optionally grouped by target class. |
| `plot_sentiment_distributions(sentiment_df, target_col)` | `src/adoption_accelerator/utils/visualization.py` | Plot sentiment score and magnitude distributions grouped by target. |

**Rationale:** Sentiment JSON parsing produces a structured DataFrame that will be reused in the metadata/sentiment feature pipeline (notebook 10, `src/features/metadata.py`). The parsing logic belongs in `data/ingestion.py` because it is a raw-data-to-DataFrame transformation — not a feature engineering step. Text statistical functions are shared with `utils/visualization.py` for consistency.

---

## 8. Notebook 06 — EDA: Images & Metadata

### 8.1. Objective

Conduct exploratory data analysis on the **image modality** and the pre-computed **Google Vision API metadata JSONs**. Profile image characteristics (count, dimensions, quality proxies), analyze Vision API labels and color information, and correlate image/metadata features with `AdoptionSpeed`. This notebook answers: *"What visual and metadata features may be predictive of adoption speed?"*

### 8.2. Scope

| In Scope | Out of Scope |
|----------|-------------|
| Image count per listing analysis | Image embedding extraction (→ notebook 09) |
| Image dimension and file-size profiling (sampled) | CNN/ViT feature extraction |
| Vision API label analysis (from metadata JSONs) | Feature engineering |
| Vision API dominant color analysis | Model training |
| Vision API crop hint analysis | Sentiment analysis (→ notebook 05) |
| Metadata coverage vs. `PhotoAmt` consistency | — |
| Image quality proxies (brightness, blur) — sampled | — |
| Metadata features vs. `AdoptionSpeed` correlation | — |

### 8.3. Step-by-Step Process

| # | Step | Description |
|---|------|-------------|
| 1 | Import & config | Import `src/` modules. Load cleaned train data from Parquet. |
| 2 | Photo count analysis | Analyze `PhotoAmt` distribution. Plot distribution vs. `AdoptionSpeed`. |
| 3 | Image file audit | For a sample of PetIDs, verify that the number of actual image files matches `PhotoAmt`. Report consistency. |
| 4 | Image dimension profiling | Load a sample of images (e.g., 500–1000). Compute width, height, aspect ratio, file size. Plot distributions. Identify standard dimensions vs. outliers. |
| 5 | Image quality proxies (sampled) | For the sampled images: compute average brightness and blur score (e.g., Laplacian variance). Profile distributions. |
| 6 | Load metadata JSONs | Batch-load a representative sample (or all) of metadata JSONs from `train_metadata/`. Parse label annotations, dominant colors, and crop hints. |
| 7 | Metadata coverage | Report what fraction of PetIDs have metadata files. Verify consistency with `PhotoAmt`. |
| 8 | Vision label analysis | Extract unique vision labels across all metadata. Compute label frequency. Identify top-N labels. Analyze label count per image and per listing. |
| 9 | Vision label vs. target | For top labels (e.g., "cute", "puppy", "cat"): compute adoption speed distribution for listings where label is present vs. absent. |
| 10 | Dominant color analysis | Extract dominant colors from metadata. Compute average brightness from dominant colors. Analyze color diversity (number of dominant colors). |
| 11 | Crop hint analysis | Extract crop confidence scores. Plot distribution. Correlate with `AdoptionSpeed`. |
| 12 | Image gallery | Display a grid of sample images for each `AdoptionSpeed` class (e.g., 4 per class) for qualitative inspection. |
| 13 | Key findings | Document observed patterns and hypotheses for image/metadata feature engineering. |
| 14 | Save figures | Persist all key visualizations to `reports/figures/eda_images/`. |
| 15 | **Validation gate** | Assert analysis completeness (see §8.4). |

### 8.4. Validation Checkpoints

| Gate ID | Assertion | Severity |
|---------|-----------|----------|
| `G06-1` | Cleaned train data loaded with `PhotoAmt` and `PetID` columns | **Critical** — halt |
| `G06-2` | Image sample loaded successfully (no I/O errors) | **Critical** — halt |
| `G06-3` | Metadata JSONs parsed without critical exceptions | **Critical** — halt |
| `G06-4` | Metadata coverage ≥ 95% of PetIDs with `PhotoAmt > 0` | **Warning** — log |
| `G06-5` | At least one figure saved per analysis section | **Informational** — warn |
| `G06-6` | Findings report persisted | **Warning** — log |

### 8.5. Expected Artifacts

| Artifact | Location | Format |
|----------|----------|--------|
| Image/Metadata EDA figures | `reports/figures/eda_images/` | PNG |
| Image EDA findings | `reports/eda_images_findings.json` | JSON |
| Aggregated metadata DataFrame (optional interim) | `data/processed/metadata_aggregated.parquet` | Parquet |

### 8.6. Dependencies

| Dependency | Source |
|------------|--------|
| `data/cleaned/train.parquet` | Notebook `03` |
| `data/raw/train_images/*.jpg` | Notebook `00` (download) |
| `data/raw/train_metadata/*.json` | Notebook `00` (download) |

### 8.7. Required `src/` Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `load_all_metadata_jsons(split)` | `src/adoption_accelerator/data/ingestion.py` | Batch-load all metadata JSONs for a given split. Return a list of parsed dicts keyed by `PetID` and image index. |
| `parse_metadata_to_dataframe(metadata_records)` | `src/adoption_accelerator/data/ingestion.py` | Convert raw metadata JSON records into a structured DataFrame with label annotations, dominant colors, and crop confidence. |
| `get_image_paths_for_pet(pet_id, split)` | `src/adoption_accelerator/data/ingestion.py` | Return a list of image file paths for a given `PetID`. |
| `compute_image_stats(image_paths)` | `src/adoption_accelerator/data/ingestion.py` | Compute width, height, aspect ratio, file size, brightness, and blur score for a list of image paths. Return a DataFrame. |
| `plot_image_grid(images, titles, nrows, ncols)` | `src/adoption_accelerator/utils/visualization.py` | Display a grid of images with titles. |
| `plot_label_frequency(label_counts, top_n)` | `src/adoption_accelerator/utils/visualization.py` | Plot a bar chart of top-N Vision API label frequencies. |

**Rationale:** Metadata JSON parsing follows the same pattern as sentiment parsing (notebook 05): raw file → structured DataFrame. Both belong in `data/ingestion.py`. Image statistics computation (brightness, blur) is also an I/O-bound operation on raw files. The reusable plotting functions in `utils/visualization.py` ensure consistent visual output.

---

## 9. Cross-Notebook `src/` Architecture Map

### 9.1. Module Inventory (Part 1)

The following `src/` modules are required by notebooks 00–06. Each module is listed with its functions and the notebooks that consume them.

```
src/adoption_accelerator/
│
├── __init__.py
│
├── config.py                          # Consumed by: ALL notebooks
│   └── (Configuration loading, path constants, seed, global settings)
│
├── data/
│   ├── __init__.py
│   ├── schemas.py                     # Consumed by: 02, 03
│   │   ├── get_tabular_schema()
│   │   └── (Schema definitions for validation)
│   │
│   ├── ingestion.py                   # Consumed by: 01, 02, 03, 04, 05, 06
│   │   ├── load_tabular(split)
│   │   ├── load_reference_table(name)
│   │   ├── load_sentiment_json(pet_id, index)
│   │   ├── load_metadata_json(pet_id, index)
│   │   ├── load_image(pet_id, index)
│   │   ├── load_all_sentiment_jsons(split)
│   │   ├── load_all_metadata_jsons(split)
│   │   ├── parse_sentiment_to_dataframe(records)
│   │   ├── parse_metadata_to_dataframe(records)
│   │   ├── get_image_paths_for_pet(pet_id, split)
│   │   ├── compute_image_stats(image_paths)
│   │   ├── get_file_inventory(directory)
│   │   ├── save_parquet(df, path)
│   │   └── load_parquet(path)
│   │
│   ├── validation.py                  # Consumed by: 02, 03
│   │   ├── validate_schema(df, schema_name)
│   │   ├── validate_domain(df, rules)
│   │   ├── validate_referential_integrity(df, ref_df, fk_col, pk_col)
│   │   ├── check_nulls(df)
│   │   ├── check_duplicates(df, key_col)
│   │   ├── check_file_coverage(pet_ids, directory, extension)
│   │   └── generate_validation_report(results)
│   │
│   └── cleaning.py                    # Consumed by: 03
│       ├── clean_tabular(df, split)
│       ├── handle_missing_names(df)
│       ├── normalize_text_fields(df, columns)
│       ├── fix_invalid_codes(df, col, valid_ids, fallback)
│       └── enforce_dtypes(df, schema)
│
└── utils/
    ├── __init__.py
    ├── paths.py                       # Consumed by: 00, and transitively by ALL
    │   ├── get_project_root()
    │   ├── get_data_path(stage, subset)
    │   └── ensure_directories()
    │
    ├── logging.py                     # Consumed by: 00, and transitively by ALL
    │   └── setup_logging(level)
    │
    └── visualization.py              # Consumed by: 04, 05, 06
        ├── plot_target_distribution(y)
        ├── plot_numeric_distribution(df, col, target_col)
        ├── plot_categorical_distribution(df, col, target_col)
        ├── plot_correlation_matrix(df, columns, method)
        ├── compute_cramers_v(df, col1, col2)
        ├── compute_descriptive_stats(df, columns)
        ├── compute_text_statistics(descriptions)
        ├── plot_text_length_distributions(stats_df, target_col)
        ├── plot_sentiment_distributions(sentiment_df, target_col)
        ├── plot_image_grid(images, titles, nrows, ncols)
        ├── plot_label_frequency(label_counts, top_n)
        └── save_figure(fig, name, subdir)
```

### 9.2. Design Rationale

| Module | Placement Rationale |
|--------|-------------------|
| `config.py` | Single source of truth for all settings. Every module imports paths and constants from here rather than hardcoding. Lives at package root for universal access. |
| `data/schemas.py` | Schema definitions are data contracts — they describe the expected structure of incoming data. Separating from validation logic enables schema reuse in other contexts (e.g., inference input validation). |
| `data/ingestion.py` | All file I/O operations (CSV, Parquet, JSON, image loading) centralized here. This ensures consistent path resolution, dtype handling, and error management regardless of caller. |
| `data/validation.py` | Validation rules are reusable across notebooks (initial validation, post-cleaning re-validation, feature pipeline pre-checks). Schema-driven design enables config-based rule definition. |
| `data/cleaning.py` | Cleaning transformations must be identical whether run from a notebook or a pipeline script. Centralizing ensures deterministic reproducibility. |
| `utils/paths.py` | Path management is cross-cutting. Every module needs it. Placed in `utils/` to avoid circular dependencies. |
| `utils/logging.py` | Logging configuration is cross-cutting infrastructure. Standard formatting and level management. |
| `utils/visualization.py` | Plotting functions are shared across three EDA notebooks and the future Streamlit app. Prevents code duplication and ensures visual consistency. |

### 9.3. Dependency Graph Within `src/`

```
utils/paths.py ◄─── config.py ◄─── data/schemas.py
                         ▲               ▲
                         │               │
                    data/ingestion.py    data/validation.py
                         ▲               ▲
                         │               │
                    data/cleaning.py ────┘
                    
utils/visualization.py ◄─── (standalone, imports only config for paths)
utils/logging.py       ◄─── (standalone)
```

No circular dependencies. All arrows point from consumers to providers.

---

## 10. Artifact Registry (Part 1)

### 10.1. Complete Artifact Inventory

All artifacts produced by notebooks 00–06, with their producers and downstream consumers.

| Artifact | Location | Format | Producer | Consumer(s) |
|----------|----------|--------|----------|-------------|
| Raw dataset | `data/raw/` | CSV, JPEG, JSON | NB 00 | NB 01–06, all downstream |
| Directory scaffold | Project directories | Filesystem | NB 00 | All notebooks |
| Data inventory report | `reports/data_inventory.json` | JSON | NB 01 | Documentation |
| Validation report | `reports/validation_report.json` | JSON | NB 02 | NB 03 |
| Validation summary | `reports/validation_summary.md` | Markdown | NB 02 | Documentation |
| Cleaned train data | `data/cleaned/train.parquet` | Parquet | NB 03 | NB 04, 05, 06, NB 07+ |
| Cleaned test data | `data/cleaned/test.parquet` | Parquet | NB 03 | NB 07+ |
| Cleaning log | `reports/cleaning_log.json` | JSON | NB 03 | Documentation |
| Tabular EDA figures | `reports/figures/eda_tabular/` | PNG | NB 04 | Reports, documentation |
| Tabular EDA findings | `reports/eda_tabular_findings.json` | JSON | NB 04 | Feature engineering decisions |
| Text EDA figures | `reports/figures/eda_text/` | PNG | NB 05 | Reports, documentation |
| Text EDA findings | `reports/eda_text_findings.json` | JSON | NB 05 | Feature engineering decisions |
| Sentiment aggregated (interim) | `data/processed/sentiment_aggregated.parquet` | Parquet | NB 05 | NB 06, feature pipeline |
| Image EDA figures | `reports/figures/eda_images/` | PNG | NB 06 | Reports, documentation |
| Image EDA findings | `reports/eda_images_findings.json` | JSON | NB 06 | Feature engineering decisions |
| Metadata aggregated (interim) | `data/processed/metadata_aggregated.parquet` | Parquet | NB 06 | Feature pipeline |

### 10.2. Critical Path Artifacts

The following artifacts are **required** for downstream notebook execution (Part 2 and beyond):

| Artifact | Required By |
|----------|------------|
| `data/cleaned/train.parquet` | All feature engineering notebooks (07–10) |
| `data/cleaned/test.parquet` | All feature engineering notebooks (07–10) |
| `data/raw/train_images/` | Image feature extraction (NB 09) |
| `data/raw/train_metadata/` | Metadata feature extraction (NB 10) |
| `data/raw/train_sentiment/` | Metadata/sentiment feature extraction (NB 10) |

---

## 11. Conventions & Standards

### 11.1. Code Style

| Convention | Standard |
|-----------|----------|
| Python version | ≥ 3.11 |
| Formatting | Consistent style enforced by formatter (e.g., `ruff format`) |
| Linting | Static analysis via linter (e.g., `ruff check`) |
| Type hints | Required for all `src/` function signatures |
| Docstrings | Required for all `src/` public functions (Google style) |
| Import order | Standard library → third-party → local (`src/`) |

### 11.2. Function Naming

| Pattern | Usage |
|---------|-------|
| `load_*` | Functions that read data from disk into memory. |
| `save_*` | Functions that write data from memory to disk. |
| `parse_*` | Functions that transform raw structures into structured DataFrames. |
| `validate_*` | Functions that check data integrity and return validation results. |
| `check_*` | Functions that perform a single validation check (building blocks for `validate_*`). |
| `clean_*` | Functions that apply transformations to fix data issues. |
| `compute_*` | Functions that calculate statistics or derived values. |
| `plot_*` | Functions that create and return visualization objects. |
| `get_*` | Functions that retrieve configuration values or resolve paths. |

### 11.3. Notebook Cell Discipline

| Rule | Rationale |
|------|-----------|
| No function definitions in notebooks | All reusable logic in `src/`. |
| No magic numbers | All constants come from config or are named variables. |
| Each code cell has a single responsibility | Readability and debugging. |
| Markdown cell before each code section | Documents intent, not implementation. |
| Output cells display results, not raw DataFrames | Use `.head()`, summary tables, or plots — never dump full DataFrames. |

### 11.4. Data Persistence Rules

| Rule | Details |
|------|---------|
| Always Parquet for DataFrames | `engine='pyarrow'`, `compression='snappy'`. |
| Always JSON for reports | Structured, human-readable, version-controllable. |
| Always PNG for figures | `dpi=150`, `bbox_inches='tight'`. |
| Filenames include context | E.g., `train.parquet`, not `data.parquet`. |
| No overwriting without versioning | If rerunning changes output, use versioned directories. |

### 11.5. Seed and Reproducibility

| Setting | Value |
|---------|-------|
| Global random seed | `42` (set in `config.py`) |
| NumPy seed | `np.random.seed(SEED)` |
| Python random seed | `random.seed(SEED)` |
| Sampling operations | Always pass `random_state=SEED` |

---

*End of Notebook Specifications — Part 1.*
