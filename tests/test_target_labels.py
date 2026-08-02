"""Tests for the single owner of adoption-speed class labels.

Before this module the labels were declared in eight places across four
renderings, and one consumer keyed them by string while the rest used
ints.
"""

from __future__ import annotations

import pytest
import yaml

from adoption_accelerator import target_labels


@pytest.fixture(autouse=True)
def _clear_cache():
    target_labels.clear_target_cache()
    yield
    target_labels.clear_target_cache()


def test_every_rendering_covers_every_class():
    cfg = target_labels.load_target_config()
    for rendering, mapping in cfg.class_labels.items():
        assert set(mapping) == set(range(cfg.n_classes)), rendering


def test_display_labels_match_the_shipped_values():
    assert target_labels.labels("display") == {
        0: "Same-day adoption",
        1: "Adopted within 1 week",
        2: "Adopted within 1 month",
        3: "Adopted within 1-3 months",
        4: "Not adopted (100+ days)",
    }


def test_short_labels_match_the_shipped_values():
    assert target_labels.labels("short") == {
        0: "Same-day",
        1: "Within 1 week",
        2: "Within 1 month",
        3: "Within 1-3 months",
        4: "100+ days",
    }


def test_inline_labels_match_the_shipped_values():
    assert target_labels.labels("inline") == {
        0: "adopted same day",
        1: "adopted within 1 week",
        2: "adopted within 1 month",
        3: "adopted within 1-3 months",
        4: "not adopted after 100 days",
    }


def test_axis_labels_match_the_shipped_values():
    assert target_labels.ordered_labels("axis") == [
        "Same day (0)",
        "1-7 days (1)",
        "8-30 days (2)",
        "31-90 days (3)",
        "100+ days (4)",
    ]


def test_ordered_labels_are_index_ordered():
    ordered = target_labels.ordered_labels("display")
    mapping = target_labels.labels("display")
    assert ordered == [mapping[i] for i in range(len(mapping))]


def test_unknown_rendering_is_rejected():
    with pytest.raises(KeyError):
        target_labels.labels("nonexistent")


def test_incomplete_rendering_is_rejected(tmp_path):
    payload = {
        "target": {
            "n_classes": 5,
            "class_labels": {
                "display": {0: "a", 1: "b", 2: "c", 3: "d", 4: "e"},
                "short": {0: "a"},
            },
        }
    }
    path = tmp_path / "serving.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="short"):
        target_labels.load_target_config(path)


def test_every_consumer_reads_the_same_owner():
    """Contract test: no module may hold its own copy."""
    from adoption_accelerator.inference import formatter, pipeline
    from app.api.schemas.enums import AdoptionSpeedClass

    display = target_labels.labels("display")
    assert pipeline.CLASS_LABELS == display
    assert formatter.CLASS_LABELS == display
    for member in AdoptionSpeedClass:
        assert member.label == display[member.value]
