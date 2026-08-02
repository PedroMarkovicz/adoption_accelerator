# app/api/services/meta_service.py
"""Reference data for the /meta endpoint: canonical class labels and the
categorical option lists (breeds/colors/states) the frontend wizard needs.

CSVs are read from app/api/assets, which ships in the API container image.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from app.api.schemas.enums import AdoptionSpeedClass

_ASSETS = Path(__file__).resolve().parent.parent / "assets"

_MATURITY = {1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large"}
_FUR = {1: "Short", 2: "Medium", 3: "Long"}


def adoption_speed_classes() -> list[dict]:
    return [{"index": c.value, "label": c.label} for c in AdoptionSpeedClass]


def maturity_sizes() -> list[dict]:
    return [{"id": k, "label": v} for k, v in _MATURITY.items()]


def fur_lengths() -> list[dict]:
    return [{"id": k, "label": v} for k, v in _FUR.items()]


def _read_csv(name: str) -> list[dict[str, str]]:
    with open(_ASSETS / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def load_breeds() -> list[dict]:
    return [
        {"id": int(r["BreedID"]), "type": int(r["Type"]), "name": r["BreedName"]}
        for r in _read_csv("breed_labels.csv")
    ]


@lru_cache(maxsize=1)
def load_colors() -> list[dict]:
    return [
        {"id": int(r["ColorID"]), "name": r["ColorName"]}
        for r in _read_csv("color_labels.csv")
    ]


@lru_cache(maxsize=1)
def load_states() -> list[dict]:
    return [
        {"id": int(r["StateID"]), "name": r["StateName"]}
        for r in _read_csv("state_labels.csv")
    ]


def build_meta(model_version: str, modality_breakdown: dict[str, int]) -> dict:
    return {
        "model_version": model_version,
        "modality_breakdown": modality_breakdown,
        "adoption_speed_classes": adoption_speed_classes(),
        "breeds": load_breeds(),
        "colors": load_colors(),
        "states": load_states(),
        "maturity_sizes": maturity_sizes(),
        "fur_lengths": fur_lengths(),
    }
