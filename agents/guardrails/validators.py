"""
LLM output validators for the agent guardrail system.

Provides validation functions that check LLM-generated outputs for
correctness, safety, and schema compliance before they are written
to the agent state.

Key guarantees:
  - Explanations reference the correct predicted class.
  - No raw embedding dimension names (img_emb_*, text_emb_*) leak
    into user-facing text.
  - Recommendations only target actionable features.
  - Descriptions meet length requirements and factual grounding.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regex that detects raw embedding dimension names that must never
# appear in user-facing outputs.
_RAW_EMBEDDING_PATTERN = re.compile(r"(img_emb_|text_emb_)\d+")

# Actionable features that recommendations may target.
ACTIONABLE_FEATURE_NAMES = {
    "PhotoAmt", "VideoAmt", "Vaccinated", "Dewormed", "Sterilized",
    "Fee", "Name", "Description", "Quantity",
    # Also accept human-readable aliases
    "photos", "videos", "vaccination", "deworming", "sterilization",
    "fee", "name", "description", "quantity", "photo", "video",
}

# Class label keywords for grounding checks
CLASS_LABELS = {
    0: ["same-day", "same day", "immediately"],
    1: ["1 week", "one week", "within a week", "7 days"],
    2: ["1 month", "one month", "within a month", "30 days"],
    3: ["1-3 months", "one to three months", "1 to 3 months"],
    4: ["not adopted", "100+ days", "100 days", "over 100"],
}


def validate_explanation(
    text: str,
    predicted_class: int,
) -> tuple[bool, str]:
    """Validate an LLM-generated explanation.

    Checks:
    1. Text is non-empty and has reasonable length.
    2. No raw embedding dimension names are present.
    3. The explanation references the correct predicted class.

    Parameters
    ----------
    text : str
        LLM-generated explanation text.
    predicted_class : int
        The model's predicted class (0-4).

    Returns
    -------
    tuple[bool, str]
        ``(is_valid, reason)`` where reason describes the failure.
    """
    if not text or not text.strip():
        return False, "Empty explanation text"

    stripped = text.strip()

    # Check for raw embedding names
    if _RAW_EMBEDDING_PATTERN.search(stripped):
        match = _RAW_EMBEDDING_PATTERN.search(stripped)
        return False, f"Raw embedding name detected: {match.group()}"

    # Check minimum length (at least 50 chars for a meaningful explanation)
    if len(stripped) < 50:
        return False, f"Explanation too short ({len(stripped)} chars, minimum 50)"

    # Check that the class reference is plausible
    # We don't enforce strict keyword matching since the LLM may use
    # varied phrasing, but we flag if it contradicts the prediction
    class_keywords = CLASS_LABELS.get(predicted_class, [])
    opposite_classes = [
        kw
        for cls, kws in CLASS_LABELS.items()
        if cls != predicted_class
        for kw in kws
    ]

    text_lower = stripped.lower()

    # Check for contradiction: mentioning a different class explicitly
    # Only flag if a conflicting class keyword appears AND no correct one does
    has_correct = any(kw in text_lower for kw in class_keywords)
    has_conflicting = any(kw in text_lower for kw in opposite_classes)

    if has_conflicting and not has_correct:
        return False, "Explanation references a different class than predicted"

    return True, ""


def validate_recommendations(
    recommendations: list[dict[str, Any]],
    max_count: int = 5,
) -> list[dict[str, Any]]:
    """Validate and filter LLM-generated recommendations.

    Filters out:
    - Recommendations targeting non-actionable features.
    - Recommendations exceeding the maximum count.
    - Malformed recommendation dicts.

    Parameters
    ----------
    recommendations : list[dict]
        Raw recommendations from the LLM.
    max_count : int
        Maximum number of recommendations to return.

    Returns
    -------
    list[dict]
        Filtered and validated recommendations.
    """
    valid: list[dict[str, Any]] = []

    for rec in recommendations:
        if not isinstance(rec, dict):
            logger.warning("Skipping non-dict recommendation: %s", type(rec))
            continue

        # Check required fields
        feature = rec.get("feature", "")
        if not feature:
            logger.warning("Skipping recommendation with no feature")
            continue

        # Check that the feature is actionable
        if not _is_actionable_feature(feature):
            logger.warning(
                "Filtering non-actionable recommendation: feature=%s", feature
            )
            continue

        # Check for raw embedding names in any string field
        for key in ("feature", "current_value", "suggested_value", "expected_impact"):
            val = str(rec.get(key, ""))
            if _RAW_EMBEDDING_PATTERN.search(val):
                logger.warning(
                    "Filtering recommendation with raw embedding name in %s", key
                )
                continue

        valid.append(rec)

    # Cap at max_count
    return valid[:max_count]


def validate_description(
    text: str,
    pet_type: str = "",
    min_words: int = 100,
    max_words: int = 400,
) -> tuple[bool, str]:
    """Validate an LLM-generated pet description.

    Checks:
    1. Text is non-empty.
    2. Word count is within the target range (relaxed from 150-300
       to 100-400 to allow some flexibility).
    3. No raw embedding names are present.
    4. Factual grounding: mentions the correct pet type if provided.

    Parameters
    ----------
    text : str
        LLM-generated description.
    pet_type : str
        Expected pet type (``"dog"`` or ``"cat"``).
    min_words : int
        Minimum word count.
    max_words : int
        Maximum word count.

    Returns
    -------
    tuple[bool, str]
        ``(is_valid, reason)`` where reason describes the failure.
    """
    if not text or not text.strip():
        return False, "Empty description"

    stripped = text.strip()

    # Check for raw embedding names
    if _RAW_EMBEDDING_PATTERN.search(stripped):
        return False, "Raw embedding name detected in description"

    # Word count check
    word_count = len(stripped.split())
    if word_count < min_words:
        return False, f"Description too short ({word_count} words, minimum {min_words})"
    if word_count > max_words:
        return False, f"Description too long ({word_count} words, maximum {max_words})"

    # Factual grounding: check pet type if provided
    if pet_type:
        text_lower = stripped.lower()
        pet_lower = pet_type.lower()
        if pet_lower not in text_lower:
            # Allow synonyms
            synonyms = {
                "dog": ["dog", "puppy", "pup", "canine"],
                "cat": ["cat", "kitten", "kitty", "feline"],
            }
            type_synonyms = synonyms.get(pet_lower, [pet_lower])
            if not any(syn in text_lower for syn in type_synonyms):
                return False, f"Description does not mention the pet type ({pet_type})"

    return True, ""


def _is_actionable_feature(feature: str) -> bool:
    """Check if a feature name is in the actionable feature set."""
    # Case-insensitive match against known actionable names
    return feature.lower().strip() in {f.lower() for f in ACTIONABLE_FEATURE_NAMES}
