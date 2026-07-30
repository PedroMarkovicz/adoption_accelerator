"""Tests for the measured-impact summary text.

The summary must be derived from the measured probability shift, never
assumed. A change that moves probability mass toward slower adoption, or
moves nothing at all, must not be described as an improvement.

Shift fixtures are the real values recorded in the Bebe and Yuki runs
(docs/article/assets/{bebe,yuki}-report.json).
"""

from adoption_accelerator.agents.subgraphs.recommendation_agent import _speedup_text


def test_class_improvement_is_reported_as_a_class_move():
    text = _speedup_text(2, 1, {0: 0.0, 1: 0.30, 2: -0.30, 3: 0.0, 4: 0.0})
    assert "moves the prediction" in text
    assert "adopted within 1 week" in text.lower()


def test_class_regression_is_not_called_an_improvement():
    """A change that pushes the predicted class to a slower bucket."""
    text = _speedup_text(2, 3, {0: 0.0, 1: -0.30, 2: -0.10, 3: 0.40, 4: 0.0})
    assert "improve" not in text.lower()
    assert "slower" in text.lower()


def test_same_class_with_mass_toward_faster_is_an_improvement():
    """Bebe: PhotoAmt 1 -> 4. Expected class value moves down."""
    shift = {0: -0.0006, 1: 0.0023, 2: 0.0006, 3: -0.0004, 4: -0.0019}
    text = _speedup_text(2, 2, shift)
    assert "improves class probabilities" in text


def test_same_class_with_mass_toward_slower_is_reported_as_worse():
    """Yuki: PhotoAmt 1 -> 6. Class 3 gains 0.0595; this is not an
    improvement and must never be labelled as one."""
    shift = {0: -0.0022, 1: -0.0264, 2: -0.0104, 3: 0.0595, 4: -0.0205}
    text = _speedup_text(2, 2, shift)
    assert "improve" not in text.lower()
    assert "slower" in text.lower()


def test_same_class_with_slight_regression_is_reported_as_worse():
    """Yuki: toggling Vaccinated. Small but real move toward slower."""
    shift = {0: -0.0002, 1: -0.0044, 2: -0.0016, 3: 0.0053, 4: 0.0009}
    text = _speedup_text(2, 2, shift)
    assert "improve" not in text.lower()


def test_all_zero_shift_is_reported_as_no_measurable_change():
    """Yuki: setting Fee to 0 when the fee is already 0."""
    shift = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    text = _speedup_text(2, 2, shift)
    assert "no measurable change" in text.lower()
    assert "improve" not in text.lower()


def test_negligible_shift_is_reported_as_no_measurable_change():
    """Below the noise floor, claim nothing."""
    shift = {0: 0.0, 1: 0.0001, 2: -0.0001, 3: 0.0, 4: 0.0}
    text = _speedup_text(2, 2, shift)
    assert "no measurable change" in text.lower()


def test_missing_shift_does_not_claim_an_improvement():
    """Defensive: an empty shift map must not produce a positive claim."""
    text = _speedup_text(2, 2, {})
    assert "improve" not in text.lower()
