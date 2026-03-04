"""
Output formatting for inference results.

Produces Kaggle submission CSVs and detailed prediction Parquets.
Separated from prediction logic to allow consumer-specific formatting
without modifying the prediction path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CLASS_LABELS = {
    0: "Same-day adoption",
    1: "Adopted within 1 week",
    2: "Adopted within 1 month",
    3: "Adopted within 1-3 months",
    4: "Not adopted (100+ days)",
}


def format_predictions(
    pet_ids: pd.Index | np.ndarray,
    predicted_classes: np.ndarray,
    probabilities: np.ndarray,
    expected_values: np.ndarray,
    argmax_classes: np.ndarray | None = None,
) -> pd.DataFrame:
    """Assemble the prediction detail DataFrame.

    Parameters
    ----------
    pet_ids : Index or array
        Pet identifiers.
    predicted_classes : ndarray
        Threshold-based class predictions (0-4).
    probabilities : ndarray of shape (n, 5)
        Class probabilities.
    expected_values : ndarray
        Weighted expected values.
    argmax_classes : ndarray or None
        Argmax predictions for comparison.

    Returns
    -------
    pd.DataFrame
        Detailed prediction table.
    """
    n_classes = probabilities.shape[1]
    confidence = probabilities.max(axis=1)

    df = pd.DataFrame(
        {
            "PetID": pet_ids,
            "predicted_class": predicted_classes.astype(np.int8),
            "predicted_label": [
                CLASS_LABELS.get(int(c), str(c)) for c in predicted_classes
            ],
            "confidence": confidence,
            "expected_value": expected_values,
        }
    )

    for i in range(n_classes):
        df[f"prob_class_{i}"] = probabilities[:, i]

    if argmax_classes is not None:
        df["argmax_class"] = argmax_classes.astype(np.int8)

    df["prediction_method"] = "threshold"
    df = df.set_index("PetID")

    logger.info(
        "Formatted prediction detail: %d rows, %d columns", df.shape[0], df.shape[1]
    )
    return df


def generate_submission(
    pet_ids: pd.Index | np.ndarray,
    predicted_classes: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Produce and save the Kaggle-format submission CSV.

    Parameters
    ----------
    pet_ids : Index or array
        Test PetIDs.
    predicted_classes : ndarray
        Integer predictions (0-4).
    output_path : str or Path
        Destination CSV path.

    Returns
    -------
    Path
        The written file path.

    Raises
    ------
    ValueError
        If format constraints are violated.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {"PetID": pet_ids, "AdoptionSpeed": predicted_classes.astype(int)}
    )

    # Validate
    valid_classes = {0, 1, 2, 3, 4}
    actual_classes = set(submission["AdoptionSpeed"].unique())
    if not actual_classes.issubset(valid_classes):
        raise ValueError(f"Invalid prediction values: {actual_classes - valid_classes}")

    if submission["PetID"].duplicated().any():
        raise ValueError("Duplicate PetIDs in submission")

    submission.to_csv(output_path, index=False)
    logger.info(
        "Submission saved: %s (%d rows, %d columns)",
        output_path,
        submission.shape[0],
        submission.shape[1],
    )
    return output_path
