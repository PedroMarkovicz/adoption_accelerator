"""
Enumerations for the Adoption Accelerator API.

AdoptionSpeedClass maps the integer class (0-4) to the human-readable
label used by the ML pipeline, reading its labels from
adoption_accelerator.target_labels, the single owner.
"""

from __future__ import annotations

from enum import IntEnum

from adoption_accelerator.target_labels import labels

_CLASS_LABELS: dict[int, str] = labels("display")


class AdoptionSpeedClass(IntEnum):
    SAME_DAY = 0
    WITHIN_1_WEEK = 1
    WITHIN_1_MONTH = 2
    WITHIN_1_TO_3_MONTHS = 3
    NOT_ADOPTED = 4

    @property
    def label(self) -> str:
        return _CLASS_LABELS[self.value]
