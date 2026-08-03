"""Tests for the actionable-feature spec table.

The regression these guard: the same eight feature names were listed in
five separate structures, so adding a ninth required five coordinated
edits and missing one failed silently.
"""

from __future__ import annotations

import pytest

from adoption_accelerator.agents.tools import actionable_features as af
from adoption_accelerator.contracts_test_helpers import make_request
from adoption_accelerator.inference.contracts import PredictionRequest


def test_table_is_not_empty():
    assert af.ACTIONABLE_FEATURES


def test_every_feature_is_coercible():
    """One table entry is enough to make a feature fully functional."""
    samples = {"int": "3", "float": "12.5", "str": "Bebe"}
    for name, spec in af.ACTIONABLE_FEATURES.items():
        value = af.coerce(name, samples[spec.kind])
        expected_type = {"int": int, "float": float, "str": str}[spec.kind]
        assert isinstance(value, expected_type), name


def test_every_feature_is_applicable_and_readable():
    """A real write/read round trip: the value must actually land on the
    field ``apply_change`` claims to write, not merely produce *some*
    non-None reading. A typo'd target key must fail this test.

    The sample is derived from each feature's own current value rather
    than hardcoded, so it is guaranteed to differ from whatever
    make_request happens to default to -- now or after a future edit.
    A hardcoded sample can silently collide with a default and mask a
    broken apply_change: this happened twice with fixed literals, once
    for "Rex" (the default tabular name) and once for the int literal
    2 (the default sterilized value). Deriving from current_value rules
    out that whole class of false pass.
    """
    request = make_request(description="Friendly young dog.")
    for name, spec in af.ACTIONABLE_FEATURES.items():
        before = af.current_value(request, name)
        if spec.kind == "int":
            sample = int(before) + 7
        elif spec.kind == "float":
            sample = float(before) + 7.0
        else:
            sample = f"{before}_changed" if before else "Bebe"
        data = request.model_dump()
        af.apply_change(data, name, sample)
        mutated = PredictionRequest(**data)
        assert af.current_value(mutated, name) == str(sample), name


def test_current_value_is_permissive_for_a_non_actionable_feature():
    """Unlike coerce/apply_change, current_value must degrade gracefully:
    lookup_feature reports on non-actionable features by name, and
    _finalize_items reads back LLM-supplied feature names that are not
    constrained to the whitelist. Both rely on "" rather than a raise."""
    request = make_request(description="Friendly young dog.")
    assert af.current_value(request, "Breed1") == ""


def test_coerce_rejects_an_unknown_feature():
    with pytest.raises(KeyError):
        af.coerce("Breed1", "1")


def test_apply_change_rejects_an_unknown_feature():
    request = make_request(description="Friendly young dog.")
    with pytest.raises(KeyError):
        af.apply_change(request.model_dump(), "Age", 3)


def test_photo_amt_rewrites_the_image_list():
    request = make_request(description="Friendly young dog.")
    data = request.model_dump()
    af.apply_change(data, "PhotoAmt", 4)
    assert len(data["images"]) == 4


def test_sweep_candidates_come_from_the_table():
    candidates = af.sweep_candidates()
    assert candidates
    for name, value in candidates:
        assert name in af.ACTIONABLE_FEATURES
        assert af.ACTIONABLE_FEATURES[name].sweep_default == value
