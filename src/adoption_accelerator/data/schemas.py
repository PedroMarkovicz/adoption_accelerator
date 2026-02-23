"""
Tabular data schema definitions for the Adoption Accelerator project.

Provides the canonical schema contract used by validation and cleaning
modules.  Each schema specifies expected column names, data types,
nullable constraints, and valid domain values.

Functions
---------
get_tabular_schema(split)
    Return the expected schema definition for train or test tabular data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Schema dataclass ────────────────────────────────────────────────


@dataclass(frozen=True)
class ColumnSpec:
    """Specification for a single DataFrame column."""

    name: str
    dtype: str  # pandas dtype family: "int64", "object", "float64", …
    nullable: bool = False
    valid_values: set[Any] | None = None  # finite domain (set-based)
    min_value: Any | None = None  # lower bound (range-based)
    fk_reference: str | None = None  # reference table name for FK cols
    allow_zero: bool = False  # whether 0 is an acceptable "no value"


@dataclass(frozen=True)
class TabularSchema:
    """Complete schema definition for a tabular split."""

    split: str
    columns: list[ColumnSpec] = field(default_factory=list)

    @property
    def column_names(self) -> list[str]:
        """Return ordered list of expected column names."""
        return [c.name for c in self.columns]

    @property
    def required_non_null_columns(self) -> list[str]:
        """Return column names that must not contain nulls."""
        return [c.name for c in self.columns if not c.nullable]

    def get_column(self, name: str) -> ColumnSpec | None:
        """Retrieve a ColumnSpec by name, or None if not found."""
        for c in self.columns:
            if c.name == name:
                return c
        return None


# ── Column definitions (shared) ────────────────────────────────────

_SHARED_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("Type", "int64", nullable=False, valid_values={1, 2}),
    ColumnSpec("Name", "object", nullable=True),
    ColumnSpec("Age", "int64", nullable=False, min_value=0),
    ColumnSpec("Breed1", "int64", nullable=False, fk_reference="breed"),
    ColumnSpec(
        "Breed2", "int64", nullable=False, fk_reference="breed", allow_zero=True
    ),
    ColumnSpec("Gender", "int64", nullable=False, valid_values={1, 2, 3}),
    ColumnSpec("Color1", "int64", nullable=False, fk_reference="color"),
    ColumnSpec(
        "Color2", "int64", nullable=False, fk_reference="color", allow_zero=True
    ),
    ColumnSpec(
        "Color3", "int64", nullable=False, fk_reference="color", allow_zero=True
    ),
    ColumnSpec("MaturitySize", "int64", nullable=False, valid_values={0, 1, 2, 3, 4}),
    ColumnSpec("FurLength", "int64", nullable=False, valid_values={0, 1, 2, 3}),
    ColumnSpec("Vaccinated", "int64", nullable=False, valid_values={1, 2, 3}),
    ColumnSpec("Dewormed", "int64", nullable=False, valid_values={1, 2, 3}),
    ColumnSpec("Sterilized", "int64", nullable=False, valid_values={1, 2, 3}),
    ColumnSpec("Health", "int64", nullable=False, valid_values={1, 2, 3}),
    ColumnSpec("Quantity", "int64", nullable=False, min_value=1),
    ColumnSpec("Fee", "int64", nullable=False, min_value=0),
    ColumnSpec("State", "int64", nullable=False, fk_reference="state"),
    ColumnSpec("RescuerID", "object", nullable=False),
    ColumnSpec("VideoAmt", "int64", nullable=False, min_value=0),
    ColumnSpec("Description", "object", nullable=True),
    ColumnSpec("PetID", "object", nullable=False),
    ColumnSpec("PhotoAmt", "float64", nullable=False, min_value=0),
]

_ADOPTION_SPEED = ColumnSpec(
    "AdoptionSpeed", "int64", nullable=False, valid_values={0, 1, 2, 3, 4}
)


# ── Public API ──────────────────────────────────────────────────────


def get_tabular_schema(split: str = "train") -> TabularSchema:
    """Return the expected schema for the tabular data.

    Parameters
    ----------
    split : str
        ``"train"`` or ``"test"``.

    Returns
    -------
    TabularSchema
        Schema object with column specs, domain rules, and FK references.

    Raises
    ------
    ValueError
        If *split* is not ``"train"`` or ``"test"``.
    """
    split = split.lower().strip()
    if split not in ("train", "test"):
        raise ValueError(f"Invalid split '{split}'. Must be 'train' or 'test'.")

    columns = list(_SHARED_COLUMNS)
    if split == "train":
        columns.append(_ADOPTION_SPEED)

    return TabularSchema(split=split, columns=columns)
