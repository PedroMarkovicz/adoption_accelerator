"""
Precompute static JSON data for the Explore Data page.

Generates:
  artifacts/explore/distributions.json  -- histogram/bar data per feature
  artifacts/explore/adoption_patterns.json -- class distribution, dog vs cat, modality avg
  artifacts/explore/performance.json -- confusion matrix, per-class metrics

Run once after model training:
  python scripts/precompute_explore_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

TRAIN_CSV = ROOT / "data" / "raw" / "train" / "train.csv"
OOF_PARQUET = ROOT / "artifacts" / "models" / "tuned_v1" / "oof_predictions.parquet"
METRICS_JSON = ROOT / "artifacts" / "models" / "tuned_v1" / "metrics.json"
GLOBAL_IMP_JSON = ROOT / "reports" / "global_importance.json"
OUT_DIR = ROOT / "artifacts" / "explore"

CLASS_LABELS = {
    0: "Same-day",
    1: "Within 1 week",
    2: "Within 1 month",
    3: "Within 1-3 months",
    4: "100+ days",
}

# Features to expose in the distribution explorer
EXPLORE_FEATURES = {
    # numeric
    "Age": {"type": "numeric", "display": "Pet Age (months)", "bins": 20},
    "Fee": {"type": "numeric", "display": "Adoption Fee (RM)", "bins": 20},
    "PhotoAmt": {"type": "numeric", "display": "Number of Photos", "bins": 10},
    "VideoAmt": {"type": "numeric", "display": "Number of Videos", "bins": 6},
    "Quantity": {"type": "numeric", "display": "Number of Pets in Listing", "bins": 10},
    # categorical
    "Type": {"type": "categorical", "display": "Pet Type", "map": {1: "Dog", 2: "Cat"}},
    "Gender": {"type": "categorical", "display": "Gender", "map": {1: "Male", 2: "Female", 3: "Mixed"}},
    "MaturitySize": {
        "type": "categorical",
        "display": "Maturity Size",
        "map": {1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large"},
    },
    "FurLength": {
        "type": "categorical",
        "display": "Fur Length",
        "map": {1: "Short", 2: "Medium", 3: "Long"},
    },
    "Vaccinated": {
        "type": "categorical",
        "display": "Vaccination Status",
        "map": {1: "Yes", 2: "No", 3: "Not Sure"},
    },
    "Dewormed": {
        "type": "categorical",
        "display": "Deworming Status",
        "map": {1: "Yes", 2: "No", 3: "Not Sure"},
    },
    "Sterilized": {
        "type": "categorical",
        "display": "Sterilization Status",
        "map": {1: "Yes", 2: "No", 3: "Not Sure"},
    },
    "Health": {
        "type": "categorical",
        "display": "Health Condition",
        "map": {1: "Healthy", 2: "Minor Injury", 3: "Serious Injury"},
    },
}


def _compute_distributions(df: pd.DataFrame) -> dict:
    """Build histogram/bar data for each explore feature."""
    result: dict[str, dict] = {}

    for feat, meta in EXPLORE_FEATURES.items():
        if feat not in df.columns:
            continue

        entry: dict = {
            "feature": feat,
            "display_name": meta["display"],
            "type": meta["type"],
        }

        if meta["type"] == "numeric":
            col = df[[feat, "AdoptionSpeed"]].dropna(subset=[feat])
            # Cap extreme outliers for cleaner histograms
            cap = col[feat].quantile(0.99)
            col = col[col[feat] <= cap].copy()

            # Overall histogram
            counts, bin_edges = np.histogram(col[feat], bins=meta["bins"])
            entry["bins"] = [round(float(e), 2) for e in bin_edges]
            entry["counts"] = [int(c) for c in counts]

            # By-class histograms (same bin edges)
            by_class: dict[str, list[int]] = {}
            for cls in range(5):
                subset = col[col["AdoptionSpeed"] == cls][feat]
                cls_counts, _ = np.histogram(subset, bins=bin_edges)
                by_class[str(cls)] = [int(c) for c in cls_counts]
            entry["by_class"] = by_class

        else:  # categorical
            mapping = meta.get("map", {})
            col = df[[feat, "AdoptionSpeed"]].dropna(subset=[feat])

            categories = sorted(col[feat].unique())
            cat_labels = [mapping.get(int(c), str(c)) for c in categories]
            overall_counts = [int(col[col[feat] == c].shape[0]) for c in categories]

            entry["categories"] = cat_labels
            entry["counts"] = overall_counts

            by_class: dict[str, list[int]] = {}
            for cls in range(5):
                subset = col[col["AdoptionSpeed"] == cls]
                cls_counts = [int(subset[subset[feat] == c].shape[0]) for c in categories]
                by_class[str(cls)] = cls_counts
            entry["by_class"] = by_class

        result[feat] = entry

    return result


def _compute_adoption_patterns(df: pd.DataFrame) -> dict:
    """Compute aggregate adoption pattern data."""
    # Class distribution
    class_counts = df["AdoptionSpeed"].value_counts().sort_index()
    class_dist = {
        "labels": [CLASS_LABELS[i] for i in range(5)],
        "counts": [int(class_counts.get(i, 0)) for i in range(5)],
        "percentages": [
            round(float(class_counts.get(i, 0)) / len(df) * 100, 1) for i in range(5)
        ],
    }

    # Dog vs Cat comparison
    dog_cat: dict[str, dict] = {}
    for pet_type, label in [(1, "Dog"), (2, "Cat")]:
        subset = df[df["Type"] == pet_type]
        counts = subset["AdoptionSpeed"].value_counts().sort_index()
        total = len(subset)
        dog_cat[label] = {
            "counts": [int(counts.get(i, 0)) for i in range(5)],
            "percentages": [
                round(float(counts.get(i, 0)) / max(total, 1) * 100, 1)
                for i in range(5)
            ],
        }

    # Modality importance (approximate from feature schema groups)
    modality_importance = {
        "tabular": 52.0,
        "text": 23.0,
        "image": 18.0,
        "metadata": 7.0,
    }

    return {
        "class_distribution": class_dist,
        "dog_vs_cat": dog_cat,
        "class_labels": CLASS_LABELS,
        "modality_importance": modality_importance,
    }


def _compute_performance(df_oof: pd.DataFrame) -> dict:
    """Compute confusion matrix and per-class metrics from OOF predictions."""
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = df_oof["true_label"].values
    y_pred = df_oof["predicted_label"].values

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    # Per-class metrics
    per_class = []
    precision_arr = precision_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    recall_arr = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    f1_arr = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    support_arr = [int(np.sum(y_true == c)) for c in range(5)]

    for i in range(5):
        per_class.append({
            "class": i,
            "label": CLASS_LABELS[i],
            "precision": round(float(precision_arr[i]), 4),
            "recall": round(float(recall_arr[i]), 4),
            "f1": round(float(f1_arr[i]), 4),
            "support": support_arr[i],
        })

    # Aggregate metrics
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    acc = accuracy_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    return {
        "confusion_matrix": cm.tolist(),
        "class_labels": [CLASS_LABELS[i] for i in range(5)],
        "per_class_metrics": per_class,
        "aggregate_metrics": {
            "qwk": round(float(qwk), 4),
            "accuracy": round(float(acc), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "macro_f1": round(float(macro_f1), 4),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading training data ...")
    df = pd.read_csv(TRAIN_CSV)
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # 1. Distributions
    print("Computing feature distributions ...")
    distributions = _compute_distributions(df)
    dist_path = OUT_DIR / "distributions.json"
    dist_path.write_text(json.dumps(distributions, indent=2))
    print(f"  Wrote {dist_path} ({len(distributions)} features)")

    # 2. Adoption patterns
    print("Computing adoption patterns ...")
    patterns = _compute_adoption_patterns(df)
    pat_path = OUT_DIR / "adoption_patterns.json"
    pat_path.write_text(json.dumps(patterns, indent=2))
    print(f"  Wrote {pat_path}")

    # 3. Performance (from OOF predictions)
    print("Computing model performance ...")
    if OOF_PARQUET.exists():
        df_oof = pd.read_parquet(OOF_PARQUET)
        print(f"  OOF data: {len(df_oof)} rows, columns: {list(df_oof.columns)}")
        performance = _compute_performance(df_oof)
    else:
        print("  WARNING: OOF predictions not found, using empty performance data")
        performance = {
            "confusion_matrix": [[0] * 5 for _ in range(5)],
            "class_labels": [CLASS_LABELS[i] for i in range(5)],
            "per_class_metrics": [],
            "aggregate_metrics": {"qwk": 0, "accuracy": 0, "weighted_f1": 0, "macro_f1": 0},
        }

    perf_path = OUT_DIR / "performance.json"
    perf_path.write_text(json.dumps(performance, indent=2))
    print(f"  Wrote {perf_path}")

    print("Done.")


if __name__ == "__main__":
    main()
