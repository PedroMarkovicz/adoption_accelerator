"""synthesizer node — fuses the evidence board into narrative, headline,
and a grounded optimized description."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from adoption_accelerator.agents.contracts import NodeError, TraceEntry
from adoption_accelerator.agents.listing_facts import (
    ListingFacts,
    build_listing_facts,
)
from adoption_accelerator.agents.llm.client import extract_usage, get_chat_model
from adoption_accelerator.agents.llm.registry import resolve_role
from adoption_accelerator.agents.state import AgentState

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_TIMEOUT_SECONDS = 20.0

# Visual-trait phrases that must be grounded in observed_traits when a
# description mentions them. Deliberately narrow: every added pattern is
# another chance to reject honest copy.
_VISUAL_TRAIT_PATTERNS = [
    r"blue eyes", r"green eyes", r"amber eyes",
    r"black and white", r"brown coat", r"white coat", r"black coat",
    r"golden coat", r"spotted", r"striped", r"fluffy coat",
    r"curly coat", r"wavy coat", r"tabby", r"brindle",
]

# Words the analyst and the writer use interchangeably. Without this, a
# truthful "fluffy coat" is rejected because the analyst wrote "fluffy fur".
_EQUIVALENT_WORDS: dict[str, set[str]] = {
    "coat": {"coat", "fur", "hair"},
    "fur": {"coat", "fur", "hair"},
    "hair": {"coat", "fur", "hair"},
    "eyes": {"eyes", "eye"},
}

_DASH_PATTERN = re.compile(r"\s*[—–]+\s*")
_MALE_PRONOUNS = re.compile(r"\b(?:he|him|his|himself)\b", re.IGNORECASE)
_FEMALE_PRONOUNS = re.compile(r"\b(?:she|her|hers|herself)\b", re.IGNORECASE)


class SynthesisOutput(BaseModel):
    narrative: str
    headline: str
    optimized_description: Optional[str] = None


def _word_present(word: str, observed: str) -> bool:
    """Is this trait word (or an accepted equivalent) in observed_traits?"""
    for variant in _EQUIVALENT_WORDS.get(word, {word}):
        if re.search(rf"\b{re.escape(variant)}\b", observed):
            return True
    return False


def _violates_grounding(description: str, observed_traits: list[str]) -> str | None:
    """Return the offending trait phrase when the description claims a visual
    trait whose significant words are not all present in observed_traits."""
    text = description.lower()
    observed = " ".join(observed_traits).lower()
    for pattern in _VISUAL_TRAIT_PATTERNS:
        # Does the description actually claim this trait? (word-boundary, so
        # "spotted" does not match inside "unspotted")
        if not re.search(rf"\b{re.escape(pattern)}\b", text):
            continue
        # Grounded if every significant word of the trait appears (word-boundary)
        # in observed_traits. "and" is not significant.
        words = [w for w in pattern.split() if w != "and"]
        if all(_word_present(w, observed) for w in words):
            continue  # grounded -> not a violation
        return pattern
    return None


def _strip_dashes(text: str) -> str:
    """Remove em and en dashes, the most reliable tell of machine-written
    copy. A run of one or more dashes between digits is a range, so it
    becomes ' to '. Elsewhere it becomes a comma, unless punctuation
    already precedes it (then a single space suffices) or there is no
    text on one side of it (then nothing, rather than stray punctuation
    at a string boundary)."""

    def replace(match: re.Match[str]) -> str:
        previous = match.string[: match.start()].rstrip()
        following = match.string[match.end() :].lstrip()
        prev_char = previous[-1:]
        next_char = following[:1]
        if prev_char.isdigit() and next_char.isdigit():
            return " to "
        if not prev_char or not next_char:
            return ""
        if prev_char in ",;:.!?":
            return " "
        return ", "

    cleaned = _DASH_PATTERN.sub(replace, text)
    # Collapse only runs of spaces/tabs; a paragraph break (newlines) must
    # survive, since narrative/headline copy can be multi-paragraph.
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _strip_accents(value: str) -> str:
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(ch)
    )


def _normalize_name(text: str, name: Optional[str]) -> str:
    """Rewrite accent variants of the pet's name to the declared spelling.

    The model has decorated names before ("Bebe" became "Bebé"), which
    contradicts the listing heading. Tokens that differ from the name only
    in capitalization are left alone, so a pet called "Happy" does not
    turn every "happy" into a name. When the offending token was
    capitalized (e.g. it opened the sentence), the replacement keeps that
    capitalization even if the declared name is stored lowercase.

    Silently inert for a declared name containing a space or hyphen (e.g.
    "Miss Daisy"): the matching is done token-by-token (``\\w+``), so it
    can never match a multi-token name as a whole and no replacement
    happens. Confirmed harmless -- the text is left as the model wrote it,
    never corrupted -- but such names simply do not benefit from this
    repair.
    """
    if not name:
        return text
    target = _strip_accents(name).casefold()
    if not target:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.casefold() == name.casefold():
            return token
        if _strip_accents(token).casefold() == target:
            if token[:1].isupper():
                return name[:1].upper() + name[1:]
            return name
        return token

    return re.sub(r"\w+", replace, text)


def _pronoun_mismatch(description: str, facts: ListingFacts) -> bool:
    """True when the copy contradicts the shelter's own record on sex.

    Only meaningful for one animal of known sex: a group listing may
    legitimately use both sets while describing individual animals.
    """
    if facts.is_group or facts.sex not in ("male", "female"):
        return False
    wrong = _FEMALE_PRONOUNS if facts.sex == "male" else _MALE_PRONOUNS
    return bool(wrong.search(description))


def _build_user_prompt(state: AgentState, facts: ListingFacts) -> str:
    ev = state["prediction_evidence"]
    lines = [
        "PREDICTION EVIDENCE:",
        f"- class {ev.predicted_class} ({ev.prediction_label}), "
        f"confidence {ev.class_confidence:.1%}",
        f"- uncertainty: {ev.uncertainty_reading}",
    ]
    for d in ev.key_drivers:
        lines.append(f"- [{d.direction}] {d.reading}")

    visual = state.get("visual_evidence")
    if visual is not None:
        lines.append("VISUAL EVIDENCE:")
        lines.append(f"- overall appeal: {visual.overall_visual_appeal}/10")
        lines.append(
            "- confirmed physical traits (raw material for the ad, not a "
            f"checklist to recite): {', '.join(visual.observed_traits) or 'none'}"
        )
        if visual.appeal_hooks:
            lines.append(
                "- adopter impression from the photos (let it shape tone and "
                "word choice; never restate it as confirmed behavior): "
                f"{', '.join(visual.appeal_hooks)}"
            )
        lines.append(f"- strategy: {visual.photo_strategy_summary}")
        for flag in visual.consistency_flags:
            lines.append(f"- consistency flag: {flag}")
    else:
        lines.append("VISUAL EVIDENCE: none (no photos were provided/analyzed)")

    recs = state.get("recommendation_evidence")
    if recs is not None and recs.recommendations:
        lines.append("VALIDATED RECOMMENDATIONS (measured by the real model):")
        for r in recs.recommendations:
            lines.append(
                f"- P{r.priority} {r.action}: {r.measured_impact.expected_speedup}"
            )
    else:
        lines.append("VALIDATED RECOMMENDATIONS: none")

    lines.append(facts.as_prompt_block())
    request = state.get("request")
    lines.append(f"ORIGINAL DESCRIPTION: {request.description or '(none)'}")
    return "\n".join(lines)


def _fallback_narrative(state: AgentState) -> str:
    ev = state["prediction_evidence"]
    parts = [
        f"The model predicts: {ev.prediction_label} "
        f"(confidence {ev.class_confidence:.0%})."
    ]
    recs = state.get("recommendation_evidence")
    if recs is not None and recs.recommendations:
        top = recs.recommendations[0]
        parts.append(f"Top validated action: {top.action} "
                     f"({top.measured_impact.expected_speedup}).")
    return " ".join(parts)


async def synthesizer_node(state: AgentState) -> dict:
    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    timestamp = state.get("timestamp", "")

    if state.get("prediction_evidence") is None:
        return {
            "narrative": "Prediction unavailable.", "headline": "",
            "optimized_description": None,
            "errors": [NodeError(node="synthesizer", error_type="missing_input",
                                 message="prediction_evidence missing",
                                 timestamp=timestamp, recoverable=True)],
            "trace": [_trace(started_at, t0, "error", {})],
        }

    errors: list[NodeError] = []
    meta: dict = {}
    try:
        system = (_PROMPTS_DIR / "synthesizer_system.txt").read_text(
            encoding="utf-8"
        )
        model = get_chat_model("synthesizer").with_structured_output(
            SynthesisOutput, include_raw=True
        )
        request = state.get("request")
        facts = build_listing_facts(request.tabular, request.labels)
        result = await asyncio.wait_for(
            model.ainvoke([("system", system),
                           ("user", _build_user_prompt(state, facts))]),
            timeout=_TIMEOUT_SECONDS,
        )
        output: SynthesisOutput = result["parsed"]
        raw = result["raw"]

        narrative = _normalize_name(_strip_dashes(output.narrative), facts.name)
        headline = _normalize_name(_strip_dashes(output.headline), facts.name)
        description = output.optimized_description

        if description:
            # Repair first, so the rejection checks see the corrected text.
            description = _strip_dashes(description)
            description = _normalize_name(description, facts.name)

            if _pronoun_mismatch(description, facts):
                logger.warning(
                    "Dropping description: pronouns contradict declared sex "
                    "'%s'", facts.sex,
                )
                meta["description_dropped"] = "pronoun_mismatch"
                description = None

        if description:
            visual = state.get("visual_evidence")
            observed = visual.observed_traits if visual is not None else []
            # The declared record is authoritative too, not just the photo
            # analyst's wording: a truthful "<declared color> coat" must not
            # be dropped just because the analyst never used that phrasing
            # (or there were no photos at all). The fur_length entry exists
            # only to supply the "coat" token, so a claim like "fluffy coat"
            # still requires the analyst to have actually observed fluff.
            grounding_corpus = observed + [
                facts.colors or "", facts.breed or "", f"{facts.fur_length} coat",
            ]
            offending = _violates_grounding(description, grounding_corpus)
            if offending is not None:
                logger.warning(
                    "Dropping ungrounded description (claims '%s')", offending
                )
                meta["description_dropped"] = offending
                description = None

        resolved = resolve_role("synthesizer")
        meta["model"] = resolved.api_model
        meta["llm_usage"] = extract_usage(raw, resolved.model_key)
        return {
            "narrative": narrative, "headline": headline,
            "optimized_description": description,
            "errors": errors,
            "trace": [_trace(started_at, t0, "success", meta)],
        }

    except Exception as exc:
        logger.warning("synthesizer failed: %s. Template fallback.", exc)
        errors.append(NodeError(node="synthesizer", error_type="llm_failure",
                                message=str(exc), timestamp=timestamp,
                                recoverable=True))
        return {
            "narrative": _fallback_narrative(state),
            "headline": state["prediction_evidence"].prediction_label,
            "optimized_description": None,
            "errors": errors,
            "trace": [_trace(started_at, t0, "success", {"used_fallback": True})],
        }


def _trace(started_at: str, t0: float, status: str, metadata: dict) -> TraceEntry:
    return TraceEntry(
        node="synthesizer",
        started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
        status=status,
        metadata=metadata,
    )
