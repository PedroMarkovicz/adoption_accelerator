"""
Deterministic fallback generators for the agent guardrail system.

Provides template-based fallbacks that produce valid output when
LLM calls fail, time out, or return invalid responses.  These
fallbacks use ``FeatureEntry.description`` from the built feature
registry and never reference raw embedding dimension names.

All fallbacks are deterministic --- they require no LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from agents.state import Recommendation

from adoption_accelerator.features.display_names import USER_DISPLAY_NAMES

logger = logging.getLogger(__name__)

# Class labels for template generation
CLASS_LABELS = {
    0: "Same-day adoption",
    1: "Adopted within 1 week",
    2: "Adopted within 1 month",
    3: "Adopted within 1-3 months",
    4: "Not adopted (100+ days)",
}

# Actionable features with recommendation templates
_ACTIONABLE_RECOMMENDATION_TEMPLATES: dict[str, dict[str, str]] = {
    "PhotoAmt": {
        "category": "photo",
        "template": "Adding more high-quality photos can significantly improve listing visibility",
        "suggested": "5-7 photos",
    },
    "VideoAmt": {
        "category": "photo",
        "template": "Adding a video showing the pet's personality can boost adoption interest",
        "suggested": "1-2 videos",
    },
    "Vaccinated": {
        "category": "health",
        "template": "Completing vaccinations and documenting them increases adopter confidence",
        "suggested": "Yes (vaccinated)",
    },
    "Dewormed": {
        "category": "health",
        "template": "Deworming the pet and updating the listing shows responsible care",
        "suggested": "Yes (dewormed)",
    },
    "Sterilized": {
        "category": "health",
        "template": "Sterilized pets are preferred by many adopters and shelters",
        "suggested": "Yes (sterilized)",
    },
    "Fee": {
        "category": "listing_details",
        "template": "Adjusting the adoption fee can make the listing more accessible",
        "suggested": "Lower or free",
    },
    "Name": {
        "category": "listing_details",
        "template": "Giving the pet a name creates emotional connection with potential adopters",
        "suggested": "A friendly, memorable name",
    },
    "Description": {
        "category": "description",
        "template": "A detailed, engaging description highlighting the pet's personality improves adoption rates",
        "suggested": "150-300 word engaging description",
    },
}


def generate_fallback_explanation(
    predicted_class: int,
    prediction_label: str,
    confidence: float,
    modality_contributions: dict[str, float],
    top_factors: list[dict[str, Any]],
) -> str:
    """Generate a deterministic explanation from structured data.

    Uses context-aware templates and interpreted factor descriptions
    to build a readable explanation without any LLM calls.

    Parameters
    ----------
    predicted_class : int
        Predicted class (0-4).
    prediction_label : str
        Human-readable class label.
    confidence : float
        Prediction confidence.
    modality_contributions : dict[str, float]
        Per-modality SHAP contribution proportions.
    top_factors : list[dict]
        Top interpreted factors with ``name``, ``description``,
        ``shap_magnitude``, ``direction``, ``modality`` keys.

    Returns
    -------
    str
        Template-based explanation.
    """
    # Determine prediction sentiment
    is_fast = predicted_class <= 2
    speed_phrase = (
        "a relatively fast adoption"
        if is_fast
        else "a slower adoption timeline"
    )

    # Confidence qualifier
    if confidence >= 0.5:
        conf_phrase = "with moderate confidence"
    elif confidence >= 0.3:
        conf_phrase = "though with some uncertainty"
    else:
        conf_phrase = "though the model shows notable uncertainty across classes"

    # Build modality narrative
    modality_labels = {
        "tabular": "listing attributes",
        "text": "description content",
        "image": "photo characteristics",
        "metadata": "image metadata",
    }
    sorted_mods = sorted(
        modality_contributions.items(), key=lambda x: x[1], reverse=True
    )
    present_mods = [(m, p) for m, p in sorted_mods if p > 0.01]

    if len(present_mods) == 1:
        mod, pct = present_mods[0]
        modality_sentence = (
            f"The prediction is driven entirely by {modality_labels.get(mod, mod)}."
        )
    elif len(present_mods) >= 2:
        primary_mod, primary_pct = present_mods[0]
        secondary_parts = [
            modality_labels.get(m, m) for m, _ in present_mods[1:]
        ]
        secondary_str = " and ".join(secondary_parts)
        modality_sentence = (
            f"The prediction is primarily driven by "
            f"{modality_labels.get(primary_mod, primary_mod)} "
            f"({primary_pct:.0%}), with additional influence from {secondary_str}."
        )
    else:
        modality_sentence = "The prediction is based on multiple data signals."

    # Build top factors narrative (up to 3, more natural phrasing)
    positive_factors = [
        f for f in top_factors[:5] if f.get("direction") == "positive"
    ][:2]
    negative_factors = [
        f for f in top_factors[:5] if f.get("direction") == "negative"
    ][:2]

    def _factor_label(f: dict[str, Any]) -> str:
        name = f.get("name", "")
        return USER_DISPLAY_NAMES.get(name, f.get("description", name))

    def _unique_labels(factors: list[dict[str, Any]]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for f in factors:
            label = _factor_label(f)
            if label not in seen:
                seen.add(label)
                result.append(label)
        return result

    factor_sentences = []
    if positive_factors:
        pos_descs = _unique_labels(positive_factors)
        pos_str = " and ".join(pos_descs)
        factor_sentences.append(
            f"Factors supporting faster adoption include {pos_str}."
        )
    if negative_factors:
        neg_descs = _unique_labels(negative_factors)
        neg_str = " and ".join(neg_descs)
        factor_sentences.append(
            f"Working against faster adoption: {neg_str}."
        )

    factors_text = " ".join(factor_sentences) if factor_sentences else ""

    # Assemble
    explanation = (
        f"The model predicts {speed_phrase} ({prediction_label}), "
        f"{conf_phrase}. "
        f"{modality_sentence}"
    )
    if factors_text:
        explanation += f" {factors_text}"

    return explanation


def generate_fallback_recommendations(
    predicted_class: int,
    top_negative_factors: list[dict[str, Any]],
    current_values: dict[str, Any] | None = None,
) -> list[Recommendation]:
    """Generate rule-based recommendations from negative SHAP factors.

    Maps top negative factors to actionable features and generates
    recommendations ranked by SHAP magnitude.

    Parameters
    ----------
    predicted_class : int
        Current predicted class (0-4).
    top_negative_factors : list[dict]
        Negative factors from the interpreted explanation.
    current_values : dict | None
        Current values of actionable features from the request.

    Returns
    -------
    list[Recommendation]
        Up to 5 rule-based recommendations.
    """
    if current_values is None:
        current_values = {}

    recommendations: list[Recommendation] = []
    used_features: set[str] = set()
    priority = 1

    # First, try to map negative factors to actionable features
    for factor in top_negative_factors:
        if priority > 5:
            break

        name = factor.get("name", "")
        # Try to match factor to an actionable feature
        matched_feature = _match_to_actionable(name)
        if matched_feature and matched_feature not in used_features:
            template = _ACTIONABLE_RECOMMENDATION_TEMPLATES[matched_feature]
            current = str(current_values.get(matched_feature, ""))

            recommendations.append(
                Recommendation(
                    feature=matched_feature,
                    current_value=current,
                    suggested_value=template["suggested"],
                    expected_impact=template["template"],
                    priority=priority,
                    category=template["category"],
                    actionable=True,
                )
            )
            used_features.add(matched_feature)
            priority += 1

    # Fill remaining slots with generic recommendations
    for feat, template in _ACTIONABLE_RECOMMENDATION_TEMPLATES.items():
        if priority > 5:
            break
        if feat in used_features:
            continue

        current = str(current_values.get(feat, ""))
        recommendations.append(
            Recommendation(
                feature=feat,
                current_value=current,
                suggested_value=template["suggested"],
                expected_impact=template["template"],
                priority=priority,
                category=template["category"],
                actionable=True,
            )
        )
        used_features.add(feat)
        priority += 1

    return recommendations


def _match_to_actionable(factor_name: str) -> str | None:
    """Try to match an interpreted factor name to an actionable feature."""
    name_lower = factor_name.lower()

    mapping = {
        "photo": "PhotoAmt",
        "image": "PhotoAmt",
        "video": "VideoAmt",
        "vaccin": "Vaccinated",
        "deworm": "Dewormed",
        "steriliz": "Sterilized",
        "fee": "Fee",
        "name": "Name",
        "description": "Description",
        "text_semantic": "Description",
    }

    for keyword, feature in mapping.items():
        if keyword in name_lower:
            return feature

    return None


def with_timeout(
    timeout_seconds: float,
    fallback_fn: Callable[..., Any],
    fallback_args: tuple = (),
    fallback_kwargs: dict[str, Any] | None = None,
):
    """Decorator factory that wraps an async function with a timeout.

    If the wrapped function exceeds ``timeout_seconds``, it is cancelled
    and the ``fallback_fn`` is called instead.

    Parameters
    ----------
    timeout_seconds : float
        Maximum allowed execution time.
    fallback_fn : callable
        Synchronous fallback function to call on timeout.
    fallback_args : tuple
        Positional args for the fallback function.
    fallback_kwargs : dict | None
        Keyword args for the fallback function.

    Returns
    -------
    callable
        Decorated async function.
    """
    if fallback_kwargs is None:
        fallback_kwargs = {}

    def decorator(async_fn):
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    async_fn(*args, **kwargs),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout (%.1fs) for %s, using fallback",
                    timeout_seconds,
                    async_fn.__name__,
                )
                return fallback_fn(*fallback_args, **fallback_kwargs)

        wrapper.__name__ = async_fn.__name__
        wrapper.__doc__ = async_fn.__doc__
        return wrapper

    return decorator
