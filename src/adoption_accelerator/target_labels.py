"""The single owner of adoption-speed class labels.

Reads ``configs/inference/serving.yaml`` and validates that every
rendering covers exactly ``0..n_classes-1``. Both ``src/`` and
``configs/inference/`` ship in the API container, so API modules import
from here too.

This deliberately reverses the note in ``app/api/schemas/enums.py`` about
keeping the API layer free of any ML-package dependency. One validated
owner is worth more than that separation, which in practice produced
eleven copies in four renderings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator

from adoption_accelerator import config

Rendering = Literal["display", "short", "inline", "axis"]

_DEFAULT_SERVING_PATH = (
    config.PROJECT_ROOT / "configs" / "inference" / "serving.yaml"
)

_target_cache: dict[Path, "TargetConfig"] = {}


class TargetConfig(BaseModel):
    n_classes: int
    class_labels: dict[str, dict[int, str]]

    @model_validator(mode="after")
    def _validate_rendering_coverage(self) -> "TargetConfig":
        if self.n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {self.n_classes}")
        expected = set(range(self.n_classes))
        for rendering, mapping in self.class_labels.items():
            if set(mapping) != expected:
                raise ValueError(
                    f"rendering '{rendering}' must cover classes "
                    f"{sorted(expected)}, got {sorted(mapping)}"
                )
            for index, label in mapping.items():
                if not label.strip():
                    raise ValueError(
                        f"rendering '{rendering}' has an empty label at {index}"
                    )
        return self


def load_target_config(path: Path | None = None) -> TargetConfig:
    """Load and validate the target section of the serving config."""
    resolved = (path or _DEFAULT_SERVING_PATH).resolve()
    if resolved not in _target_cache:
        with open(resolved, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _target_cache[resolved] = TargetConfig.model_validate(raw["target"])
    return _target_cache[resolved]


def labels(rendering: Rendering = "display") -> dict[int, str]:
    """Class index to label for one rendering.

    Raises KeyError for an unknown rendering so a typo fails loudly.
    """
    return dict(load_target_config().class_labels[rendering])


def ordered_labels(rendering: Rendering = "display") -> list[str]:
    """Labels in class-index order, for plot axes and other list consumers."""
    mapping = load_target_config().class_labels[rendering]
    return [mapping[i] for i in sorted(mapping)]


def clear_target_cache() -> None:
    """Clear the config cache (tests)."""
    _target_cache.clear()
