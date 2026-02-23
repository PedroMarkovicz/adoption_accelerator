"""
Centralized project configuration.

All path constants, seeds, and global settings are defined here.
Notebooks and modules import from this module — no hardcoded paths.
"""

from __future__ import annotations

from adoption_accelerator.utils.paths import get_project_root

# ── Project root ────────────────────────────────────────────────────
PROJECT_ROOT = get_project_root()

# ── Random seed ─────────────────────────────────────────────────────
SEED = 42

# ── Data paths ──────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_CLEANED = DATA_DIR / "cleaned"
DATA_FEATURES = DATA_DIR / "features"
DATA_SUBMISSIONS = DATA_DIR / "submissions"

# ── Raw data subdirectories ─────────────────────────────────────────
RAW_TRAIN_CSV = DATA_RAW / "train" / "train.csv"
RAW_TEST_CSV = DATA_RAW / "test" / "test.csv"
RAW_TRAIN_IMAGES = DATA_RAW / "train_images"
RAW_TEST_IMAGES = DATA_RAW / "test_images"
RAW_TRAIN_METADATA = DATA_RAW / "train_metadata"
RAW_TEST_METADATA = DATA_RAW / "test_metadata"
RAW_TRAIN_SENTIMENT = DATA_RAW / "train_sentiment"
RAW_TEST_SENTIMENT = DATA_RAW / "test_sentiment"

# ── Reference tables ────────────────────────────────────────────────
REF_BREED_LABELS = DATA_RAW / "breed_labels.csv"
REF_COLOR_LABELS = DATA_RAW / "color_labels.csv"
REF_STATE_LABELS = DATA_RAW / "state_labels.csv"

# ── Reports ─────────────────────────────────────────────────────────
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES = REPORTS_DIR / "figures"
REPORTS_METRICS = REPORTS_DIR / "metrics"

# ── Artifacts ───────────────────────────────────────────────────────
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_MODELS = ARTIFACTS_DIR / "models"

# ── Kaggle ──────────────────────────────────────────────────────────
KAGGLE_COMPETITION = "petfinder-adoption-prediction"

# ── Expected file counts (validation gates) ────────────────────────
EXPECTED_FILE_COUNTS = {
    "train_images": 58_311,
    "test_images": 14_465,
    "train_metadata": 58_311,
    "test_metadata": 14_465,
    "train_sentiment": 14_442,
    "test_sentiment": 3_865,
}
