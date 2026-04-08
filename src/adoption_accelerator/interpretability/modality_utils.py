"""
Modality availability detection utilities.

Provides the canonical function for determining which data modalities
are actually present in a prediction request.  This is used by the
interpretability layer to distinguish "meaningful SHAP signal" from
"zero-vector baseline offset" produced by absent modalities.

Background (Issue #2 — Ghost Modality Attribution):
----------------------------------------------------
When a modality is absent (e.g. no description, no images), the feature
builder inserts zero-vectors.  SHAP computes contributions relative to
the background distribution's expected value, NOT relative to "no data".
A zero-vector therefore has a non-zero SHAP contribution because it
differs from the mean embedding in the training set.

For text embeddings (~384 dims) and image embeddings (~484 dims), even
small per-dimension baseline offsets accumulate into dominant modality
totals.  Without modality awareness, a pet with no description can appear
to have 65% of its prediction driven by "description content".

This module provides the ground truth of which modalities are present,
allowing the aggregation layer to exclude absent-modality contributions
from the modality breakdown shown to users.

Reference: ``docs/diagnosis/planning/correction_planning.md`` Issue #2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from adoption_accelerator.inference.contracts import PredictionRequest


def detect_modality_availability(request: "PredictionRequest") -> dict[str, bool]:
    """Determine which modalities are actually provided in the request.

    Parameters
    ----------
    request : PredictionRequest
        The raw prediction request from the user.

    Returns
    -------
    dict[str, bool]
        Keys: ``"tabular"``, ``"text"``, ``"image"``, ``"metadata"``.
        Value is ``True`` if the modality has real user-supplied data;
        ``False`` if it is absent (SHAP contributions = baseline offset).

    Notes
    -----
    - **tabular** is always ``True``: structured pet attributes are
      required fields in ``PredictionRequest.tabular``.
    - **metadata** is always ``True``: metadata features are derived
      from tabular data (e.g. document-level sentiment from text when
      present, otherwise imputed), so they are always computable.
    - **text** is ``True`` only when ``request.description`` contains
      non-whitespace content.
    - **image** is ``True`` only when at least one image path/byte
      object is provided.
    """
    has_text = bool(
        getattr(request, "description", None)
        and str(request.description).strip()
    )
    has_image = bool(
        getattr(request, "images", None)
        and len(request.images) > 0
    )

    return {
        "tabular": True,    # always present — required input
        "text": has_text,   # True only when description is non-empty
        "image": has_image, # True only when ≥1 image is supplied
        "metadata": True,   # derived from tabular; always computable
    }
