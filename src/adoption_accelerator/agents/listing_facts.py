"""Decodes a ``TabularInput`` into the human-readable facts the writing
nodes need.

The synthesizer used to receive raw integer codes (``gender code 1``),
which the LLM discarded in favor of guessing from the photos. This module
is the single place that turns codes into words, so prompts never carry a
code the model has to interpret.

Breed, color and state names cannot be resolved here: their CSVs live
under ``app/api/assets`` and ``data/`` is excluded from the container
image. The API layer injects them as ``ListingLabels``.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from adoption_accelerator.inference.contracts import ListingLabels, TabularInput

_SPECIES: dict[int, str] = {1: "dog", 2: "cat"}
_SIZE: dict[int, str] = {1: "small", 2: "medium", 3: "large", 4: "extra large"}
_FUR: dict[int, str] = {1: "short", 2: "medium", 3: "long"}
_TRISTATE: dict[int, str] = {1: "yes", 2: "no", 3: "not sure"}
_HEALTH: dict[int, str] = {
    1: "healthy", 2: "minor injury", 3: "serious injury"
}

# gender code -> (sex, subject, object, possessive)
_SEX_AND_PRONOUNS: dict[int, tuple[str, str, str, str]] = {
    1: ("male", "he", "him", "his"),
    2: ("female", "she", "her", "her"),
}
_UNKNOWN_SEX = ("unknown", "they", "them", "their")

# Life-stage thresholds mirror ageBand() in
# frontend/lib/listing/labels.ts so the two layers never disagree about
# whether an animal counts as a senior.
_ADULT_FROM_MONTHS = 12
_SENIOR_FROM_MONTHS = 84


class ListingFacts(BaseModel):
    """Everything about a listing that a writer can state as fact."""

    species: str
    name: Optional[str] = None
    quantity: int = 1
    is_group: bool = False
    sex: str
    pronoun_subject: str
    pronoun_object: str
    pronoun_possessive: str
    age_phrase: str
    life_stage: str
    breed: Optional[str] = None
    colors: Optional[str] = None
    size: str
    fur_length: str
    health: str
    vaccinated: str
    dewormed: str
    sterilized: str
    fee_phrase: str
    location: Optional[str] = None

    def as_prompt_block(self) -> str:
        """Render the facts as an authoritative block for an LLM prompt."""
        lines = [
            "PET FACTS (the shelter's own record; authoritative, and they "
            "override any impression from the photos):"
        ]
        if self.is_group:
            lines.append(
                f"- this listing covers {self.quantity} {self.species}s together"
            )
        lines.append(f"- name: {self.name or '(unnamed)'}")
        lines.append(f"- species: {self.species}")
        lines.append(f"- sex: {self.sex}")
        lines.append(
            f"- pronouns to use: {self.pronoun_subject}/"
            f"{self.pronoun_object}/{self.pronoun_possessive}"
        )
        lines.append(f"- age: {self.age_phrase} ({self.life_stage})")
        if self.breed:
            lines.append(f"- breed: {self.breed}")
        if self.colors:
            lines.append(f"- colors: {self.colors}")
        lines.append(f"- size when grown: {self.size}")
        lines.append(f"- coat length: {self.fur_length}")
        lines.append(f"- health: {self.health}")
        lines.append(
            f"- vaccinated: {self.vaccinated}, dewormed: {self.dewormed}, "
            f"sterilized: {self.sterilized}"
        )
        lines.append(f"- fee: {self.fee_phrase}")
        if self.location:
            lines.append(f"- location: {self.location}")
        return "\n".join(lines)


def _age_phrase(months: int) -> str:
    """Grammatical age wording. The old prompt passed a bare integer and
    produced copy that read 'Listed as 1 months old'."""
    if months <= 0:
        return "newborn"
    if months < 12:
        return f"{months} month{'' if months == 1 else 's'} old"
    years, rest = divmod(months, 12)
    year_part = f"{years} year{'' if years == 1 else 's'}"
    if rest == 0:
        return f"{year_part} old"
    return f"{year_part} {rest} month{'' if rest == 1 else 's'} old"


def _life_stage(species: str, months: int) -> str:
    if months < _ADULT_FROM_MONTHS:
        return "puppy" if species == "dog" else "kitten"
    if months < _SENIOR_FROM_MONTHS:
        return "adult"
    return "senior"


def _fee_phrase(fee: float) -> str:
    if fee <= 0:
        return "no adoption fee"
    amount = int(fee) if float(fee).is_integer() else fee
    return f"RM {amount} adoption fee"


def build_listing_facts(
    tabular: TabularInput,
    labels: Optional[ListingLabels] = None,
) -> ListingFacts:
    """Decode a ``TabularInput`` into writer-ready facts.

    ``labels`` carries the breed/color/state names the API layer resolved;
    when it is absent those fields are simply left unset rather than
    falling back to an ID no reader could interpret.
    """
    species = _SPECIES.get(tabular.type, "pet")
    quantity = max(1, tabular.quantity)
    is_group = quantity > 1

    sex, subject, obj, possessive = _SEX_AND_PRONOUNS.get(
        tabular.gender, _UNKNOWN_SEX
    )
    if tabular.gender not in _SEX_AND_PRONOUNS and is_group:
        # "Mixed" across several animals is real information; on a single
        # animal it is contradictory input, so it stays "unknown".
        sex = "mixed"
    if is_group:
        subject, obj, possessive = "they", "them", "their"

    colors = ", ".join(labels.colors) if labels and labels.colors else None
    name = (tabular.name or "").strip() or None

    return ListingFacts(
        species=species,
        name=name,
        quantity=quantity,
        is_group=is_group,
        sex=sex,
        pronoun_subject=subject,
        pronoun_object=obj,
        pronoun_possessive=possessive,
        age_phrase=_age_phrase(tabular.age),
        life_stage=_life_stage(species, tabular.age),
        breed=labels.breed if labels else None,
        colors=colors,
        size=_SIZE.get(tabular.maturity_size, "unknown"),
        fur_length=_FUR.get(tabular.fur_length, "unknown"),
        health=_HEALTH.get(tabular.health, "unknown"),
        vaccinated=_TRISTATE.get(tabular.vaccinated, "not sure"),
        dewormed=_TRISTATE.get(tabular.dewormed, "not sure"),
        sterilized=_TRISTATE.get(tabular.sterilized, "not sure"),
        fee_phrase=_fee_phrase(tabular.fee),
        location=labels.state if labels else None,
    )
