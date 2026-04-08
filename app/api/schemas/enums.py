"""
Enumerations for the Adoption Accelerator API.

AdoptionSpeedClass maps the integer class (0-4) to the human-readable
label used by the ML pipeline, mirroring CLASS_LABELS in
inference/pipeline.py but expressed as a standalone IntEnum so that
the API layer has no direct dependency on the ML package.
"""

from __future__ import annotations

from enum import IntEnum

_CLASS_LABELS: dict[int, str] = {
    0: "Same-day adoption",
    1: "Adopted within 1 week",
    2: "Adopted within 1 month",
    3: "Adopted within 1-3 months",
    4: "Not adopted (100+ days)",
}


class AdoptionSpeedClass(IntEnum):
    SAME_DAY = 0
    WITHIN_1_WEEK = 1
    WITHIN_1_MONTH = 2
    WITHIN_1_TO_3_MONTHS = 3
    NOT_ADOPTED = 4

    @property
    def label(self) -> str:
        return _CLASS_LABELS[self.value]
