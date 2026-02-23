# Adoption Accelerator — Technical Architecture Specification

> **Version:** 1.0  
> **Status:** Draft  
> **Last Updated:** 2026-02-21  
> **Author:** Pedro Markovicz  
> **Project:** Adoption Accelerator — PetFinder Adoption Speed Prediction

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Definition](#2-problem-definition)
3. [High-Level System Architecture](#3-high-level-system-architecture)
4. [Data Architecture](#4-data-architecture)
5. [Multimodal Feature Engineering Strategy](#5-multimodal-feature-engineering-strategy)
6. [Model Training Architecture](#6-model-training-architecture)
7. [Interpretability Layer](#7-interpretability-layer)
8. [Inference Pipeline Design](#8-inference-pipeline-design)
9. [Frontend Integration Contract](#9-frontend-integration-contract)
10. [Generative AI Multi-Agent Integration](#10-generative-ai-multi-agent-integration)
11. [Directory Structure](#11-directory-structure)
12. [Scalability & MLOps Considerations](#12-scalability--mlops-considerations)
13. [Future Extension Possibilities](#13-future-extension-possibilities)

---

## 1. Executive Summary

Adoption Accelerator is a production-grade multimodal machine learning system designed to predict the adoption speed of pets listed on the PetFinder platform. The system ingests structured (tabular), unstructured text (listing descriptions), and unstructured image data to produce a multiclass adoption speed prediction — along with interpretable explanations and actionable recommendations.

The platform comprises three integrated layers:

- **ML Core** — A modular library for data processing, feature engineering, model training, inference, and interpretability across all data modalities.
- **Frontend Application** — A Streamlit-based interface allowing users to submit pet profiles and receive predictions with confidence scores and explanations.
- **Generative AI Layer** — A LangGraph multi-agent system that consumes structured interpretability outputs to generate human-readable insights, improvement suggestions, and optimized listing descriptions.

The architecture prioritizes separation of concerns, reproducibility, artifact versioning, and clean integration contracts between layers.

---

## 2. Problem Definition

### 2.1. Prediction Target

**AdoptionSpeed** — an ordinal multiclass variable representing how quickly a pet was adopted after listing:

| Class | Meaning |
|-------|---------|
| 0 | Adopted on the same day as listing |
| 1 | Adopted between 1–7 days |
| 2 | Adopted between 8–30 days |
| 3 | Adopted between 31–90 days |
| 4 | Not adopted after 100 days |

### 2.2. Ordinal Nature

Classes have a natural order (0 < 1 < 2 < 3 < 4). This property can be exploited via:

- Ordinal regression formulations
- Ordinal-aware loss functions (e.g., weighted cross-entropy with distance penalties)
- Threshold-based cumulative probability models
- Evaluation with order-sensitive metrics

### 2.3. Evaluation Metric

**Quadratic Weighted Kappa (QWK)** — the official Kaggle competition metric. QWK penalizes predictions more heavily the further they are from the true class, which aligns with the ordinal structure of the target.

### 2.4. Data Modalities

| Modality | Source | Count (Train / Test) |
|----------|--------|---------------------|
| Tabular | `train.csv` / `test.csv` | 14,993 / 3,972 rows |
| Images | `train_images/` / `test_images/` | 58,311 / 14,465 files |
| Image Metadata | `train_metadata/` / `test_metadata/` | 58,311 / 14,465 JSON files |
| Text Sentiment | `train_sentiment/` / `test_sentiment/` | 14,442 / 3,865 JSON files |
| Reference Tables | `breed_labels.csv`, `color_labels.csv`, `state_labels.csv` | Static lookups |

### 2.5. Tabular Feature Schema (Raw)

| Column | Type | Description |
|--------|------|-------------|
| Type | Categorical | 1 = Dog, 2 = Cat |
| Name | Text | Pet name (may be empty) |
| Age | Numeric | Age in months |
| Breed1, Breed2 | Categorical ID | Primary/secondary breed (FK → breed_labels) |
| Gender | Categorical | 1 = Male, 2 = Female, 3 = Mixed (group) |
| Color1, Color2, Color3 | Categorical ID | Up to 3 colors (FK → color_labels) |
| MaturitySize | Ordinal | 1 = Small, 2 = Medium, 3 = Large, 4 = Extra Large |
| FurLength | Ordinal | 1 = Short, 2 = Medium, 3 = Long |
| Vaccinated | Categorical | 1 = Yes, 2 = No, 3 = Not Sure |
| Dewormed | Categorical | 1 = Yes, 2 = No, 3 = Not Sure |
| Sterilized | Categorical | 1 = Yes, 2 = No, 3 = Not Sure |
| Health | Ordinal | 1 = Healthy, 2 = Minor Injury, 3 = Serious Injury |
| Quantity | Numeric | Number of pets in the listing |
| Fee | Numeric | Adoption fee (0 = free) |
| State | Categorical ID | Malaysian state (FK → state_labels) |
| RescuerID | Identifier | Rescuer unique ID |
| VideoAmt | Numeric | Number of videos uploaded |
| Description | Text | Free-text listing description |
| PetID | Identifier | Unique pet listing ID |
| PhotoAmt | Numeric | Number of photos uploaded |
| AdoptionSpeed | Target | 0–4 ordinal class (train only) |

### 2.6. Auxiliary Data Schemas

**Image Metadata (Google Vision API output):**

- `labelAnnotations` — object/scene labels with confidence scores
- `imagePropertiesAnnotation.dominantColors` — RGB color palette with pixel fractions
- `cropHintsAnnotation` — suggested crop bounding boxes with confidence

**Text Sentiment (Google NLP API output):**

- `sentences[].sentiment.score` — sentiment polarity per sentence (−1.0 to +1.0)
- `sentences[].sentiment.magnitude` — sentiment intensity per sentence
- `entities[].name` / `entities[].type` / `entities[].salience` — named entities
- Document-level `sentiment.score` and `sentiment.magnitude`

---

## 3. High-Level System Architecture

```
                            ┌─────────────────────────────────────────┐
                            │          CONFIGURATION LAYER            │
                            │     configs/ (YAML-based settings)      │
                            └────────────────────┬────────────────────┘
                                                 │
┌──────────────────────┐    ┌────────────────────▼────────────────────┐
│                      │    │                                         │
│   NOTEBOOKS          │    │          CORE ML LIBRARY                │
│   (Research &        │───▶│        src/adoption_accelerator/        │
│    Experimentation)  │    │                                         │
│                      │    │  ┌─────────┐ ┌──────────┐ ┌──────────┐ │
└──────────────────────┘    │  │  data/   │ │features/ │ │training/ │ │
                            │  └─────────┘ └──────────┘ └──────────┘ │
┌──────────────────────┐    │  ┌───────────┐ ┌────────────────────┐  │
│                      │    │  │inference/ │ │ interpretability/  │  │
│   PIPELINES          │───▶│  └───────────┘ └────────────────────┘  │
│   (Orchestration)    │    │                                         │
│                      │    └──────┬─────────────────┬────────────────┘
└──────────────────────┘           │                 │
                            ┌──────▼──────┐   ┌──────▼──────────────┐
                            │  STREAMLIT  │   │   LANGGRAPH         │
                            │  FRONTEND   │   │   MULTI-AGENT       │
                            │  app/       │   │   agents/           │
                            └─────────────┘   └─────────────────────┘
                                   │                    │
                                   └────────┬───────────┘
                                            ▼
                                    ┌──────────────┐
                                    │   END USER   │
                                    └──────────────┘
```

### 3.1. Layer Responsibilities

| Layer | Responsibility | Consumers |
|-------|---------------|-----------|
| **Configuration** | Centralized settings for paths, seeds, model hyperparameters, feature definitions. Single source of truth. | All layers |
| **Core ML Library** | All reusable logic: data I/O, validation, feature engineering, training, inference, and interpretability. Installed as an editable Python package. | Notebooks, Pipelines, App, Agents |
| **Notebooks** | Interactive research, EDA, experimentation. Thin orchestration over the core library. Never contain reusable logic. | Data scientists (development only) |
| **Pipelines** | Deterministic, scriptable execution of end-to-end workflows (feature extraction, training, evaluation). Suitable for CI/CD and automation. | CI/CD, Makefile, cron jobs |
| **Frontend** | User-facing interface for pet profile submission and prediction display. Communicates with core only through the inference contract. | End users |
| **Agent System** | Multi-agent reasoning layer consuming interpretability outputs to generate explanations and recommendations. Communicates with core only through defined tools. | End users (via frontend) |

### 3.2. Dependency Flow

All dependencies are **unidirectional**. No circular imports exist between layers:

```
configs ← core ← {notebooks, pipelines, app, agents}

Within core:
  data → features → training → inference
                                   ↑
                           interpretability
```

The frontend and agent layers depend on `inference` and `interpretability` respectively, but never on `training` or `features` directly.

---

## 4. Data Architecture

### 4.1. Data Lifecycle

```
raw/ ──► cleaned/ ──► features/ ──► artifacts/
 │        (stage 1)    (stage 2)     (stage 3)
 │
 └── IMMUTABLE: never modified after initial download
```

### 4.2. Directory Layout

```
data/
├── raw/                          # Immutable — Kaggle competition data
│   ├── train/
│   │   └── train.csv
│   ├── test/
│   │   ├── test.csv
│   │   └── sample_submission.csv
│   ├── train_images/             # 58,311 JPEG files
│   ├── test_images/              # 14,465 JPEG files
│   ├── train_metadata/           # 58,311 Google Vision API JSONs
│   ├── test_metadata/            # 14,465 Google Vision API JSONs
│   ├── train_sentiment/          # 14,442 Google NLP API JSONs
│   ├── test_sentiment/           # 3,865 Google NLP API JSONs
│   ├── breed_labels.csv
│   ├── color_labels.csv
│   └── state_labels.csv
│
├── cleaned/                      # Cleaned tabular data (post-validation)
│   ├── train.parquet
│   └── test.parquet
│
├── features/                     # Materialized feature store (versioned)
│   ├── tabular/
│   │   └── v1/
│   │       ├── train.parquet
│   │       ├── test.parquet
│   │       └── schema.json
│   ├── text/
│   │   └── v1/
│   ├── image/
│   │   └── v1/
│   ├── metadata/
│   │   └── v1/
│   └── integrated/
│       └── v1/                   # Final merged feature matrix
│           ├── train.parquet
│           ├── test.parquet
│           └── schema.json
│
└── submissions/                  # Generated Kaggle submission files
    └── submission_v1_20260221.csv
```

### 4.3. Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Immutability** | `data/raw/` is never modified after download. All transformations produce new files in downstream directories. |
| **Parquet over CSV** | All intermediate and processed data uses Apache Parquet for type preservation, compression, and read performance. |
| **Versioned features** | Each feature set is stored under a version directory (e.g., `v1/`). Changing a text embedding model produces `v2/` without overwriting `v1/`. |
| **Schema registration** | Every versioned feature set includes a `schema.json` defining column names, types, source modality, and generation config hash. |
| **Deterministic derivation** | Any file in `cleaned/`, `features/`, or `submissions/` can be fully reproduced from `raw/` using the core library and a fixed config. |

### 4.4. Feature Version Tracking

Each `schema.json` records:

```
{
  "version": "v1",
  "modality": "text",
  "created_at": "2026-02-21T14:30:00Z",
  "config_hash": "sha256:abc123...",
  "model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "columns": ["text_emb_0", "text_emb_1", ..., "sentiment_score", "sentiment_magnitude"],
  "n_rows_train": 14993,
  "n_rows_test": 3972
}
```

This allows the training pipeline to verify which feature versions were used to produce a given model, ensuring full lineage traceability.

---

## 5. Multimodal Feature Engineering Strategy

### 5.1. Overview

The system extracts features from four data sources independently, then fuses them into a single feature matrix for model training.

```
                  ┌──────────────────────────────────────────────────────────────┐
                  │                   FEATURE ENGINEERING                        │
                  │                                                              │
  train.csv ──────┤   ┌───────────────┐                                         │
                  │   │  Tabular       │──► tabular/v1/train.parquet             │
                  │   │  Pipeline      │                                         │
                  │   └───────────────┘                                         │
                  │                                                              │
  Description ────┤   ┌───────────────┐                                         │
  column          │   │  Text          │──► text/v1/train.parquet                │
                  │   │  Pipeline      │                                         │
                  │   └───────────────┘                                         │
                  │                                                              │
  train_images/ ──┤   ┌───────────────┐                                         │
                  │   │  Image         │──► image/v1/train.parquet               │
                  │   │  Pipeline      │                                         │
                  │   └───────────────┘                                         │
                  │                                                              │
  train_metadata/ ┤   ┌───────────────┐                                         │
  train_sentiment/│   │  Metadata &    │──► metadata/v1/train.parquet            │
                  │   │  Sentiment     │                                         │
                  │   └───────────────┘                                         │
                  │                                             ┌──────────────┐│
                  │           ALL PARQUETS ─── merge ──────────►│ integrated/  ││
                  │                                             │ v1/train.pq  ││
                  │                                             └──────────────┘│
                  └──────────────────────────────────────────────────────────────┘
```

### 5.2. Tabular Pipeline

**Input:** Cleaned `train.csv` / `test.csv` columns (excluding `Description`, `PetID`, `RescuerID`).

**Operations:**

| Category | Techniques |
|----------|-----------|
| Encoding | Label encoding for tree models; target encoding for high-cardinality (Breed1, State) |
| Interaction features | Breed1 × Type, Age × MaturitySize, Fee ÷ Quantity (per-pet fee) |
| Aggregation features | Rescuer-level stats (mean adoption speed, listing count, avg photos) |
| Missing indicators | Binary flags for Breed2 = 0, Color2 = 0, Color3 = 0 |
| Ordinal exploitation | Health, MaturitySize, FurLength treated as ordinal integers |
| Binary recoding | Vaccinated/Dewormed/Sterilized mapped to: {Yes: 1, No: 0, Not Sure: −1} |
| Name features | has_name (binary), name_length, name_word_count |

**Output:** `features/tabular/v1/train.parquet` — one row per PetID with all engineered tabular features.

### 5.3. Text Embedding Pipeline

**Input:** `Description` column from cleaned tabular data.

**Strategy:** Use a pretrained sentence transformer model to generate dense embedding vectors. The model choice is externalized to `configs/features/text.yaml`.

**Operations:**

| Step | Detail |
|------|--------|
| Preprocessing | Lowercase, strip HTML/URLs, handle empty/null descriptions |
| Embedding extraction | Sentence-level embeddings via pretrained model (e.g., `all-MiniLM-L6-v2`) |
| Dimensionality | Native embedding size (e.g., 384-dim) or PCA-reduced if needed |
| Auxiliary features | `description_length`, `word_count`, `sentence_count`, `language_detected` |
| Sentiment integration | Merge pre-computed Google NLP sentiment scores (`score`, `magnitude`) from sentiment JSONs |

**Output:** `features/text/v1/train.parquet` — one row per PetID with embedding dimensions + text statistics + sentiment scores.

**Inference parity:** The same preprocessing and model must be applied identically during training and inference. The pretrained model artifact is included in the model bundle.

### 5.4. Image Embedding Pipeline

**Input:** JPEG images from `train_images/` and `test_images/`.

**Strategy:** Use a pretrained CNN or Vision Transformer backbone to extract feature vectors from pet images. The backbone choice is externalized to `configs/features/image.yaml`.

**Operations:**

| Step | Detail |
|------|--------|
| Preprocessing | Resize, center crop, normalize to model-expected input |
| Embedding extraction | Forward pass through pretrained backbone up to the penultimate layer |
| Aggregation | Multiple images per pet → mean/max pooling of embeddings across all photos |
| Auxiliary features | `photo_count`, `avg_image_brightness`, `avg_image_blur_score` |

**Output:** `features/image/v1/train.parquet` — one row per PetID with aggregated image embeddings + image statistics.

**Computational note:** Image feature extraction is the most expensive pipeline (hours on CPU, minutes on GPU). Features must be materialized and cached aggressively. Re-extraction should only occur when the backbone model changes.

### 5.5. Metadata & Sentiment Pipeline

**Input:** Google Vision API JSONs (`train_metadata/`) and Google NLP API JSONs (`train_sentiment/`).

**Operations:**

| Source | Extracted Features |
|--------|--------------------|
| Vision — labels | Top-N label scores, count of labels above threshold, presence of specific labels (e.g., "cute", "puppy") |
| Vision — colors | Dominant color RGB values, color diversity score, brightness proxy |
| Vision — crop hints | Crop confidence score (proxy for image composition quality) |
| Sentiment | Document-level score and magnitude, sentence-level aggregations (mean, min, max), entity count and types |

**Output:** `features/metadata/v1/train.parquet` — one row per PetID with structured metadata features.

### 5.6. Feature Fusion Strategy

**Method:** Late fusion via horizontal concatenation.

All per-PetID feature parquets are joined on `PetID` to produce a single wide feature matrix:

```
integrated/v1/train.parquet = JOIN(
    tabular/v1/train.parquet,
    text/v1/train.parquet,
    image/v1/train.parquet,
    metadata/v1/train.parquet
) ON PetID
```

**Rationale for late fusion over early/mid fusion:**

- Tree-based models (LightGBM, XGBoost) handle heterogeneous feature types natively
- Late fusion preserves modality-independent feature versioning
- Simpler debugging — each modality can be validated in isolation
- Feature importance remains attributable to specific modalities
- Modal ablation studies are trivial (train with/without a modality)

**The integrated schema must record which columns originate from which modality.** This metadata is critical for the interpretability layer to group SHAP values by modality.

---

## 6. Model Training Architecture

### 6.1. Training Strategy

| Aspect | Decision |
|--------|----------|
| **Primary model family** | Gradient-boosted trees (LightGBM / XGBoost / CatBoost) |
| **Justification** | State-of-the-art for tabular data with heterogeneous features; fast training; native SHAP support; handles mixed feature types without extensive preprocessing |
| **Secondary exploration** | Neural network approaches (tabular transformers, multi-modal end-to-end) as experimental alternatives |
| **Validation** | Stratified K-Fold cross-validation (K=5 or K=10), preserving target distribution |
| **Metric** | Quadratic Weighted Kappa (QWK) as primary; accuracy and per-class F1 as secondary |
| **Ordinal handling** | Post-processing threshold optimization (optimize class boundaries on QWK) |

### 6.2. Experiment Tracking

Each training run produces a structured record:

```
{
  "run_id": "run_20260221_143000",
  "model_type": "lightgbm",
  "feature_versions": {
    "tabular": "v1",
    "text": "v1",
    "image": "v1",
    "metadata": "v1",
    "integrated": "v1"
  },
  "config": { ... },            // Full hyperparameter config
  "metrics": {
    "cv_qwk_mean": 0.412,
    "cv_qwk_std": 0.015,
    "cv_qwk_folds": [0.405, 0.42, 0.398, 0.415, 0.422],
    "cv_accuracy": 0.38,
    "per_class_f1": [0.25, 0.30, 0.42, 0.35, 0.28]
  },
  "training_time_seconds": 45,
  "timestamp": "2026-02-21T14:30:00Z"
}
```

Tracking is file-based (JSON/YAML) during early development. Migration to MLflow or Weights & Biases is supported by the architecture but not required initially.

### 6.3. Model Registry

The model registry is a filesystem-based artifact store:

```
artifacts/
└── models/
    ├── lgbm_baseline_v1/
    │   ├── model.joblib
    │   ├── preprocessors/
    │   │   ├── tabular_pipeline.joblib
    │   │   ├── text_encoder_config.json
    │   │   └── image_encoder_config.json
    │   ├── explainer.joblib
    │   ├── feature_schema.json
    │   ├── metrics.json
    │   └── config.yaml
    │
    └── lgbm_tuned_v2/
        └── ...
```

### 6.4. Artifact Bundle Specification

A model artifact bundle is the **atomic deployable unit**. It contains everything needed to produce a prediction from raw inputs, with no external dependencies on the training environment.

| Artifact | Purpose | Format |
|----------|---------|--------|
| `model.joblib` | Trained classifier | joblib-serialized model object |
| `preprocessors/tabular_pipeline.joblib` | Tabular feature transformations (encoders, scalers) | joblib |
| `preprocessors/text_encoder_config.json` | Reference to pretrained text model (name, version) | JSON |
| `preprocessors/image_encoder_config.json` | Reference to pretrained image backbone (name, version) | JSON |
| `explainer.joblib` | Pre-fitted SHAP explainer (e.g., `TreeExplainer`) | joblib |
| `feature_schema.json` | Ordered list of expected feature names, types, and modality tags | JSON |
| `metrics.json` | Cross-validation and evaluation metrics | JSON |
| `config.yaml` | Full training configuration snapshot | YAML |

**Atomicity guarantee:** Loading a bundle directory is sufficient to reconstruct the entire inference pipeline. No implicit state or external lookups are required.

### 6.5. Threshold Optimization

Since AdoptionSpeed is ordinal and QWK is the evaluation metric, raw multiclass probabilities are post-processed:

1. Train model to output class probabilities `P(y=0), ..., P(y=4)`
2. Compute expected value: `E = Σ(i × P(y=i))` for each sample
3. Optimize 4 threshold boundaries on the validation set to maximize QWK
4. Store optimized thresholds in the model bundle

This approach often outperforms direct argmax classification for ordinal targets evaluated with QWK.

---

## 7. Interpretability Layer

### 7.1. Purpose

The interpretability layer is a **first-class component** — not an afterthought. It serves two distinct consumers with different requirements:

| Consumer | Needs |
|----------|-------|
| **Frontend (Streamlit)** | Per-prediction feature contribution summary for visual display |
| **Agent System (LangGraph)** | Structured, machine-parseable attribution data for reasoning and recommendation generation |

### 7.2. Architecture

```
src/adoption_accelerator/interpretability/
├── __init__.py
├── explainer.py              # SHAP wrapper for local + global explanations
├── feature_importance.py     # Global feature importance (pre-computed at train time)
├── counterfactual.py         # "What-if" analysis for recommendation generation
└── contracts.py              # ExplanationResult dataclass
```

### 7.3. Global Interpretability

Computed once during training and stored as part of the model bundle:

- **Global feature importance** — mean absolute SHAP values across the training set
- **Per-modality importance** — aggregated SHAP values grouped by modality (tabular, text, image, metadata)
- **Top-K features per class** — which features most influence each AdoptionSpeed class

Stored in `artifacts/models/<name>/global_importance.json`.

### 7.4. Local Interpretability (Per-Prediction)

Computed at inference time for each individual prediction:

- **SHAP values** — per-feature contribution to the predicted class
- **Modality attribution** — percentage contribution of each modality to the prediction
- **Top positive/negative factors** — ranked list of features pushing the prediction toward/away from the predicted class
- **Counterfactual suggestions** — "Changing feature X from A to B would shift the prediction from class 3 to class 2"

### 7.5. ExplanationResult Contract

```
ExplanationResult:
├── prediction_id: str
├── predicted_class: int (0–4)
├── global_importance:
│   ├── top_features: [{name, importance, modality}, ...]
│   └── modality_weights: {tabular: 0.45, text: 0.25, image: 0.20, metadata: 0.10}
├── local_contributions:
│   ├── tabular: [{feature, value, shap_value}, ...]
│   ├── text: [{feature, value, shap_value}, ...]
│   ├── image: [{feature, value, shap_value}, ...]
│   └── metadata: [{feature, value, shap_value}, ...]
├── top_positive_factors: [{feature, shap_value, modality}, ...]
├── top_negative_factors: [{feature, shap_value, modality}, ...]
├── counterfactuals: [
│   {feature, current_value, suggested_value, predicted_class_change, qwk_impact},
│   ...
│ ]
└── modality_attribution:
    ├── tabular: float       # % contribution of tabular modality
    ├── text: float
    ├── image: float
    └── metadata: float
```

### 7.6. SHAP Compatibility

| Model Type | SHAP Explainer | Performance |
|------------|---------------|-------------|
| LightGBM | `TreeExplainer` | Fast (native tree path computation) |
| XGBoost | `TreeExplainer` | Fast |
| CatBoost | `TreeExplainer` | Fast |
| Neural Networks | `DeepExplainer` or `GradientExplainer` | Slow (requires batched computation) |

The `TreeExplainer` is pre-fitted during training and serialized into the model bundle (`explainer.joblib`). This avoids refitting at inference time and guarantees consistent explanations.

### 7.7. Counterfactual Engine

The counterfactual module identifies actionable feature changes that would improve adoption speed:

1. For each modifiable feature (e.g., `Vaccinated`, `PhotoAmt`, `Description`), simulate alternative values
2. Re-run the preprocessor and predictor with the modified input
3. Record which changes produce a class improvement
4. Rank by feasibility and impact

**Non-modifiable features** (e.g., `Age`, `Breed1`) are excluded from counterfactual generation to ensure actionable recommendations.

---

## 8. Inference Pipeline Design

### 8.1. Pipeline Architecture

```
Raw Input ──► Preprocessor ──► Feature Vector ──► Predictor ──► PredictionResult
                  │                                    │
                  │                              ┌─────▼──────┐
                  │                              │  Explainer  │
                  │                              └─────┬──────┘
                  │                                    │
                  └────────────────────────────────────►ExplanationResult
```

The inference pipeline is a single callable that accepts raw user inputs and returns a structured prediction result. It is the **sole integration point** between the ML core and external consumers (frontend, agents).

### 8.2. Pipeline Components

```
src/adoption_accelerator/inference/
├── __init__.py
├── contracts.py       # PredictionRequest, PredictionResult (Pydantic models)
├── preprocessor.py    # Raw input → feature vector (uses same transforms as training)
├── predictor.py       # Feature vector → class probabilities → predicted class
└── pipeline.py        # Orchestrates preprocessor → predictor → (optional) explainer
```

### 8.3. PredictionRequest Contract

```
PredictionRequest:
├── tabular:
│   ├── type: int (1 or 2)
│   ├── name: str | None
│   ├── age: int
│   ├── breed1: int
│   ├── breed2: int | None
│   ├── gender: int
│   ├── color1: int
│   ├── color2: int | None
│   ├── color3: int | None
│   ├── maturity_size: int
│   ├── fur_length: int
│   ├── vaccinated: int
│   ├── dewormed: int
│   ├── sterilized: int
│   ├── health: int
│   ├── quantity: int
│   ├── fee: float
│   ├── state: int
│   └── video_amt: int
├── description: str                  # Free-text listing description
├── images: list[bytes | Path]        # One or more images (file bytes or paths)
└── options:
    ├── include_explanation: bool      # Whether to compute SHAP explanations
    └── include_counterfactuals: bool  # Whether to compute what-if analysis
```

### 8.4. PredictionResult Contract

```
PredictionResult:
├── prediction: int                    # Predicted AdoptionSpeed (0–4)
├── prediction_label: str              # Human-readable label
├── probabilities: {0: float, 1: float, 2: float, 3: float, 4: float}
├── confidence: float                  # max(probabilities)
├── explanation: ExplanationResult | None    # If requested
├── metadata:
│   ├── model_version: str
│   ├── model_type: str
│   ├── feature_versions: {tabular: str, text: str, image: str, metadata: str}
│   ├── inference_time_ms: float
│   └── timestamp: str
```

### 8.5. Prediction Labels

| Class | Label |
|-------|-------|
| 0 | "Same-day adoption" |
| 1 | "Adopted within 1 week" |
| 2 | "Adopted within 1 month" |
| 3 | "Adopted within 1–3 months" |
| 4 | "Not adopted (100+ days)" |

### 8.6. Design Constraints

| Constraint | Rationale |
|-----------|-----------|
| **Stateless** | Each prediction call is independent. No session state between calls. |
| **Deterministic** | Same input + same model bundle → same output. No randomness in inference. |
| **Fail-safe** | Missing images or empty descriptions produce valid predictions using available features with degraded confidence. |
| **Latency target** | < 2 seconds per prediction (excluding image upload time), suitable for interactive use. |

---

## 9. Frontend Integration Contract

### 9.1. Architecture

```
┌──────────────────────────────────────┐
│          STREAMLIT APP               │
│                                      │
│  ┌────────────┐    ┌──────────────┐  │
│  │ Input Form │    │ Result Panel │  │
│  │            │    │              │  │
│  │ - Tabular  │    │ - Prediction │  │
│  │ - Text     │    │ - Confidence │  │
│  │ - Images   │    │ - Explain.   │  │
│  └──────┬─────┘    └──────▲──────┘  │
│         │                  │         │
│  ┌──────▼──────────────────┴──────┐  │
│  │        ML Service Layer        │  │
│  │    app/services/ml_service.py  │  │
│  └──────────────┬─────────────────┘  │
└─────────────────│─────────────────────┘
                  │
                  ▼
    src/adoption_accelerator/inference/pipeline.py
```

### 9.2. Service Layer Responsibilities

The `ml_service.py` wrapper exists to isolate frontend concerns from ML concerns:

| Responsibility | Location |
|---------------|----------|
| Model loading with `@st.cache_resource` | `ml_service.py` |
| Input validation and user error messages | `ml_service.py` |
| Conversion of uploaded files to `PredictionRequest` | `ml_service.py` |
| Call to `inference.pipeline` | `ml_service.py` |
| Formatting of `PredictionResult` for display | `ml_service.py` |
| SHAP visualization rendering | `app/components/explanation_panel.py` |

The frontend **never** directly imports from `training/`, `features/`, or `data/`.

### 9.3. Frontend Expected Inputs

| Input | UI Component | Required |
|-------|-------------|----------|
| Pet type (Dog/Cat) | Radio button | Yes |
| Age (months) | Number input | Yes |
| Primary breed | Dropdown (from breed_labels) | Yes |
| Secondary breed | Dropdown | No |
| Gender | Radio button | Yes |
| Colors (1–3) | Dropdown(s) | Color1 required |
| Maturity size | Dropdown | Yes |
| Fur length | Dropdown | Yes |
| Vaccinated / Dewormed / Sterilized | Radio buttons | Yes |
| Health condition | Dropdown | Yes |
| Quantity | Number input | Yes |
| Adoption fee | Number input | Yes |
| State | Dropdown (from state_labels) | Yes |
| Number of videos | Number input | Yes |
| Description | Text area | No (empty handled gracefully) |
| Photos | File uploader (multi) | No (zero-photo handled gracefully) |

### 9.4. Frontend Expected Outputs

| Output | Display Component | Source Field |
|--------|------------------|-------------|
| Predicted class | Large badge / header | `PredictionResult.prediction_label` |
| Confidence gauge | Progress bar or gauge chart | `PredictionResult.confidence` |
| Class probabilities | Bar chart (5 bars) | `PredictionResult.probabilities` |
| Top factors | Ranked list with icons | `ExplanationResult.top_positive_factors`, `top_negative_factors` |
| Modality contribution | Pie/donut chart | `ExplanationResult.modality_attribution` |
| Recommendations | Card list | `ExplanationResult.counterfactuals` (filtered to actionable) |
| AI-generated insights | Text block | Agent system output (if connected) |

---

## 10. Generative AI Multi-Agent Integration

### 10.1. Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    LANGGRAPH SYSTEM                          │
│                                                              │
│  ┌──────────┐    ┌─────────────┐    ┌───────────────────┐   │
│  │ Explainer│    │ Recommender │    │ Description Writer│   │
│  │ Agent    │    │ Agent       │    │ Agent             │   │
│  └────┬─────┘    └──────┬──────┘    └────────┬──────────┘   │
│       │                 │                     │              │
│  ┌────▼─────────────────▼─────────────────────▼──────────┐  │
│  │                     TOOLS                              │  │
│  │  ┌─────────────┐ ┌──────────────┐ ┌────────────────┐  │  │
│  │  │ prediction  │ │ explanation  │ │ counterfactual │  │  │
│  │  │ _tool       │ │ _tool        │ │ _tool          │  │  │
│  │  └──────┬──────┘ └──────┬───────┘ └───────┬────────┘  │  │
│  └─────────│───────────────│─────────────────│───────────┘  │
└────────────│───────────────│─────────────────│──────────────┘
             │               │                 │
             ▼               ▼                 ▼
   inference/pipeline   interpretability/   interpretability/
                        explainer           counterfactual
```

### 10.2. Agent Descriptions

| Agent | Role | Consumes | Produces |
|-------|------|----------|----------|
| **Explainer** | Translates structured SHAP data into natural language explanations | `ExplanationResult.local_contributions`, `modality_attribution` | Human-readable text explaining why the pet received its predicted adoption speed |
| **Recommender** | Suggests concrete actions to improve adoption speed | `ExplanationResult.counterfactuals`, `top_negative_factors` | Ranked list of actionable recommendations with expected impact |
| **Description Writer** | Generates or improves listing descriptions optimized for faster adoption | `ExplanationResult.local_contributions.text`, original `Description` | Rewritten description incorporating language patterns correlated with faster adoption |
| **Analyst** | Identifies patterns and segment-level insights | `global_importance`, multiple `PredictionResult` instances | Statistical summaries and trend analyses |

### 10.3. Tool Specifications

Each tool is a thin adapter between the LangGraph framework and the core ML library.

**prediction_tool:**
```
Input:  PredictionRequest (structured pet profile)
Output: PredictionResult (prediction + probabilities + confidence)
Core:   inference.pipeline.predict()
```

**explanation_tool:**
```
Input:  PredictionResult + PredictionRequest
Output: ExplanationResult (SHAP values + modality attribution)
Core:   interpretability.explainer.explain()
```

**counterfactual_tool:**
```
Input:  PredictionRequest + target_class (desired AdoptionSpeed)
Output: List of {feature, current_value, suggested_value, impact}
Core:   interpretability.counterfactual.generate()
```

**feature_tool:**
```
Input:  feature_name or modality
Output: Feature metadata (description, distribution stats, importance rank)
Core:   features.registry.lookup()
```

### 10.4. Data Contract Requirements

The agent system requires all interpretability outputs to be:

| Requirement | Reason |
|-------------|--------|
| **JSON-serializable** | LLM tool outputs must be parseable text |
| **Self-descriptive** | Each field includes enough context for an LLM to reason about it without external documentation |
| **Modality-tagged** | Every feature contribution must indicate whether it comes from tabular, text, image, or metadata |
| **Actionability-tagged** | Counterfactuals must indicate whether a suggestion is user-actionable (e.g., "add more photos" = actionable, "change breed" = not actionable) |
| **Bounded** | Top-K truncation on SHAP values (e.g., top 10 per modality) to avoid overwhelming LLM context windows |

### 10.5. LangGraph State

The shared state flowing through the agent graph:

```
AgentState:
├── pet_profile: PredictionRequest
├── prediction: PredictionResult
├── explanation: ExplanationResult
├── recommendations: list[Recommendation]
├── improved_description: str | None
├── conversation_history: list[Message]
└── metadata:
    ├── session_id: str
    └── timestamp: str
```

---

## 11. Directory Structure

```
adoption_accelerator/
│
├── pyproject.toml
├── Makefile
├── README.md
├── project_specs.md                          # This document
├── .env
├── .gitignore
│
│
│ ── CONFIGURATION ────────────────────────────────────────────
│
├── configs/
│   ├── base.yaml                             # Global: paths, seeds, constants
│   ├── features/
│   │   ├── tabular.yaml                      # Tabular feature definitions
│   │   ├── text.yaml                         # Text model, max_length, etc.
│   │   └── image.yaml                        # Image backbone, size, batch_size
│   ├── training/
│   │   ├── baseline.yaml                     # Baseline training config
│   │   └── tuned.yaml                        # Optimized training config
│   └── inference/
│       └── serving.yaml                      # Inference: model path, thresholds
│
│
│ ── CORE ML LIBRARY ──────────────────────────────────────────
│
├── src/
│   └── adoption_accelerator/
│       ├── __init__.py
│       ├── config.py                         # YAML loader, path resolver, PARAMS
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── schemas.py                    # Pandera / Pydantic data schemas
│       │   ├── ingestion.py                  # CSV, JSON, image file readers
│       │   ├── validation.py                 # Schema enforcement, integrity checks
│       │   └── cleaning.py                   # Data transformations and fixes
│       │
│       ├── features/
│       │   ├── __init__.py
│       │   ├── tabular.py                    # Tabular feature engineering
│       │   ├── text.py                       # Text embedding extraction
│       │   ├── image.py                      # Image embedding extraction
│       │   ├── metadata.py                   # Vision API + Sentiment API features
│       │   ├── integration.py                # Cross-modality feature fusion
│       │   └── registry.py                   # Feature catalog with metadata
│       │
│       ├── training/
│       │   ├── __init__.py
│       │   ├── trainer.py                    # Training loop with CV and logging
│       │   ├── evaluation.py                 # QWK, confusion matrix, per-class F1
│       │   ├── selection.py                  # Model comparison and selection
│       │   └── artifacts.py                  # Bundle save/load operations
│       │
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── contracts.py                  # PredictionRequest, PredictionResult
│       │   ├── preprocessor.py               # Raw input → feature vector
│       │   ├── predictor.py                  # Feature vector → probabilities
│       │   └── pipeline.py                   # End-to-end orchestration
│       │
│       ├── interpretability/
│       │   ├── __init__.py
│       │   ├── contracts.py                  # ExplanationResult dataclass
│       │   ├── explainer.py                  # SHAP: global + local explanations
│       │   ├── feature_importance.py         # Global importance computation
│       │   └── counterfactual.py             # What-if analysis engine
│       │
│       └── utils/
│           ├── __init__.py
│           ├── paths.py                      # Centralized path resolution
│           ├── logging.py                    # Logging configuration
│           └── visualization.py              # Reusable plotting functions
│
│
│ ── NOTEBOOKS ────────────────────────────────────────────────
│
├── notebooks/
│   ├── 00_project_setup.ipynb
│   ├── 01_data_ingestion.ipynb
│   ├── 02_data_validation.ipynb
│   ├── 03_data_cleaning.ipynb
│   ├── 04_eda_tabular.ipynb
│   ├── 05_eda_text_sentiment.ipynb
│   ├── 06_eda_images_metadata.ipynb
│   ├── 07_feature_engineering_tabular.ipynb
│   ├── 08_feature_extraction_text.ipynb
│   ├── 09_feature_extraction_images.ipynb
│   ├── 10_feature_integration.ipynb
│   ├── 11_modeling_baseline.ipynb
│   ├── 12_modeling_tuning.ipynb
│   ├── 13_interpretability.ipynb
│   ├── 14_inference_pipeline_test.ipynb
│   └── 15_submission.ipynb
│
│
│ ── PIPELINES ────────────────────────────────────────────────
│
├── pipelines/
│   ├── run_features.py                       # End-to-end feature extraction
│   ├── run_training.py                       # Full training with config
│   ├── run_evaluation.py                     # Evaluate existing model
│   └── run_inference.py                      # Batch inference
│
│
│ ── FRONTEND ─────────────────────────────────────────────────
│
├── app/
│   ├── __init__.py
│   ├── main.py                               # Streamlit entry point
│   ├── pages/
│   │   ├── prediction.py                     # Prediction form + result display
│   │   ├── exploration.py                    # Dataset exploration / insights
│   │   └── recommendations.py                # Agent-generated recommendations
│   ├── components/
│   │   ├── input_form.py                     # Tabular + text + image input
│   │   ├── result_display.py                 # Prediction + confidence display
│   │   └── explanation_panel.py              # SHAP visualization panel
│   └── services/
│       └── ml_service.py                     # Adapter: frontend → inference
│
│
│ ── AGENT SYSTEM ─────────────────────────────────────────────
│
├── agents/
│   ├── __init__.py
│   ├── graph.py                              # LangGraph graph definition
│   ├── state.py                              # Shared agent state schema
│   ├── nodes/
│   │   ├── explainer.py                      # Explanation generation agent
│   │   ├── recommender.py                    # Improvement suggestion agent
│   │   ├── description_writer.py             # Listing description optimizer
│   │   └── analyst.py                        # Pattern analysis agent
│   └── tools/
│       ├── prediction_tool.py                # Tool: invoke inference pipeline
│       ├── explanation_tool.py               # Tool: invoke interpretability
│       ├── feature_tool.py                   # Tool: query feature registry
│       └── counterfactual_tool.py            # Tool: run what-if analysis
│
│
│ ── DATA ─────────────────────────────────────────────────────
│
├── data/
│   ├── raw/                                  # Immutable Kaggle data
│   ├── cleaned/                              # Post-cleaning tabular data
│   ├── features/                             # Versioned feature store
│   │   ├── tabular/v1/
│   │   ├── text/v1/
│   │   ├── image/v1/
│   │   ├── metadata/v1/
│   │   └── integrated/v1/
│   └── submissions/                          # Kaggle submission files
│
│
│ ── ARTIFACTS ────────────────────────────────────────────────
│
├── artifacts/
│   └── models/
│       └── <model_name>_<version>/           # Atomic model bundles
│
│
│ ── QUALITY ──────────────────────────────────────────────────
│
├── tests/
│   ├── unit/
│   │   ├── test_schemas.py
│   │   ├── test_features_tabular.py
│   │   ├── test_features_text.py
│   │   ├── test_inference_pipeline.py
│   │   └── test_contracts.py
│   └── integration/
│       ├── test_feature_pipeline.py
│       └── test_inference_end_to_end.py
│
├── reports/
│   ├── figures/
│   └── metrics/
│
└── docker/
    ├── Dockerfile.app                        # Streamlit container
    └── Dockerfile.training                   # Training container (GPU)
```

---

## 12. Scalability & MLOps Considerations

### 12.1. Current Stage: Local Development

| Concern | Current Approach |
|---------|-----------------|
| Package management | `uv` + `pyproject.toml` |
| Experiment tracking | File-based JSON/YAML in `artifacts/` |
| Feature store | Filesystem Parquet in `data/features/` |
| Model registry | Filesystem bundles in `artifacts/models/` |
| CI/CD | `Makefile` targets |
| Containerization | Docker files for app and training |

### 12.2. Growth Path

| When | Upgrade | Trigger |
|------|---------|---------|
| Multiple collaborators | MLflow for experiment tracking | JSON tracking becomes unmanageable |
| Frequent retraining | DVC for data versioning | Need to version large binary data in Git |
| Production deployment | FastAPI wrapping inference pipeline | Streamlit latency becomes a bottleneck |
| Team scaling | Pre-commit hooks + CI pipeline (GitHub Actions) | Code quality becomes inconsistent |
| Large-scale serving | Model registry (MLflow, BentoML) | Multiple models serving simultaneously |
| Real-time data | Feature store (Feast) | Shift from static dataset to live listings |

### 12.3. Reproducibility Guarantees

| Mechanism | Ensures |
|-----------|---------|
| `pyproject.toml` + `uv.lock` | Exact dependency versions |
| `configs/*.yaml` | All hyperparameters externalized and version-controlled |
| `data/raw/` immutability | Input data never mutated |
| Feature version directories | Feature transformation reproducibility |
| Model bundle config snapshots | Full training config stored alongside model |
| Random seed in `configs/base.yaml` | Deterministic splits and model initialization |

### 12.4. Testing Strategy

| Level | Scope | Frequency |
|-------|-------|-----------|
| **Unit tests** | Individual functions in `src/` (schemas, transformers, contracts) | Every commit |
| **Integration tests** | End-to-end pipeline on a small data sample | Every PR |
| **Data validation** | Schema checks on new data files | Before any pipeline run |
| **Model validation** | QWK threshold gates (new model must beat previous) | Before model promotion |

---

## 13. Future Extension Possibilities

### 13.1. Near-Term (Within Project Scope)

| Extension | Design Support |
|-----------|---------------|
| Ensemble models | `artifacts/` supports multiple bundles; `predictor.py` can average probabilities |
| Neural multimodal model | `training/` is model-agnostic; add a new trainer, same feature/inference pipeline |
| A/B testing between models | `inference/pipeline.py` can accept model version as parameter |
| Batch prediction | `pipelines/run_inference.py` already supports batch mode |

### 13.2. Medium-Term (Post-Competition)

| Extension | Design Support |
|-----------|---------------|
| REST API | Wrap `inference/pipeline.py` in FastAPI; contracts are already Pydantic-compatible |
| Real-time retraining | `pipelines/run_training.py` is scriptable and config-driven |
| Multi-language descriptions | `features/text.py` abstraction supports swapping multilingual models |
| Additional data sources | Add new feature module in `features/`, register in `registry.py`, update `integration.py` |

### 13.3. Long-Term (Platform Evolution)

| Extension | Design Support |
|-----------|---------------|
| Multi-tenant SaaS | Service layer abstraction in `app/services/` supports auth and tenant isolation |
| Streaming predictions | Stateless inference pipeline is compatible with message queue consumption |
| Automated ML (AutoML) | `training/trainer.py` can be extended with automated search frameworks |
| Fine-tuned LLMs | Agent nodes in `agents/nodes/` are independently replaceable |

---

## Appendix A: Key Design Decisions Log

| # | Decision | Rationale | Alternatives Considered |
|---|----------|-----------|------------------------|
| 1 | Monorepo structure | Single repository avoids cross-repo versioning overhead at this project scale | Multi-repo (core, app, agents) |
| 2 | Late fusion for features | Preserves modality independence; enables ablation; tree models handle heterogeneous input natively | Early/mid fusion; end-to-end neural multimodal |
| 3 | SHAP for interpretability | Native TreeExplainer for gradient-boosted models; structured output; well-documented library | LIME (less stable), Captum (PyTorch-only) |
| 4 | Parquet for intermediate data | Type-safe, compressed, columnar, fast I/O | CSV (no types, slow), HDF5 (less ecosystem support) |
| 5 | YAML configs over Hydra | Simpler, sufficient for current scale, no framework lock-in | Hydra (powerful but opinionated), dotenv (too flat) |
| 6 | Filesystem model registry over MLflow | Lower setup cost; migration path exists when needed | MLflow (premature complexity at current scale) |
| 7 | Pydantic contracts over plain dicts | Type safety, validation, serialization to JSON for agent consumption | Dataclasses (less validation), plain dicts (no safety) |
| 8 | Notebooks for research only | Prevents "notebook jungle"; keeps reusable code in testable, diffable `.py` modules | All-notebook approach (untestable, poor diffs) |

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **Artifact bundle** | Atomic package containing model, preprocessors, explainer, schemas, and config |
| **Feature store** | Versioned directory of materialized feature matrices in Parquet format |
| **Inference contract** | Formal specification of input/output types for the prediction pipeline |
| **Late fusion** | Combining independently extracted features at the model input level |
| **Model bundle** | See Artifact bundle |
| **Modality** | A distinct data type: tabular, text, image, or metadata |
| **QWK** | Quadratic Weighted Kappa — the primary evaluation metric |
| **SHAP** | SHapley Additive exPlanations — method for computing feature contributions |
| **Counterfactual** | A hypothetical input modification that would change the prediction |

---

*End of specification.*
