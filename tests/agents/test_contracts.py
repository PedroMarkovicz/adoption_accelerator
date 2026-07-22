"""Contract validation tests for evidence types and AdoptionReport."""

import pytest
from pydantic import ValidationError

from adoption_accelerator.agents.contracts import (
    AdoptionReport,
    Evidence,
    FactorInsight,
    MeasuredImpact,
    PhotoAssessment,
    PhotoContent,
    PhotoQuality,
    PredictionEvidence,
    RecommendationEvidence,
    ReportMetadata,
    ValidatedRecommendation,
    VisualEvidence,
)


def make_prediction_evidence(**overrides):
    base = dict(
        source="data_analyst",
        confidence="high",
        generated_by="deterministic",
        predicted_class=2,
        prediction_label="Adopted within 1 month",
        probabilities={0: 0.1, 1: 0.2, 2: 0.4, 3: 0.2, 4: 0.1},
        class_confidence=0.4,
        modality_contributions={"tabular": 1.0},
        modality_available={"tabular": True, "text": False, "image": False},
        key_drivers=[],
        uncertainty_reading="",
    )
    base.update(overrides)
    return PredictionEvidence(**base)


def test_prediction_evidence_roundtrip():
    ev = make_prediction_evidence(
        key_drivers=[
            FactorInsight(
                feature="Age", display_name="Age (months)", value="3",
                direction="positive", shap_magnitude=0.08,
                modality="tabular", reading="Young pets adopt faster.",
            )
        ]
    )
    restored = PredictionEvidence.model_validate(ev.model_dump())
    assert restored.key_drivers[0].feature == "Age"
    assert restored.notes == []


def test_evidence_confidence_is_constrained():
    with pytest.raises(ValidationError):
        make_prediction_evidence(confidence="certain")


def test_visual_evidence_roundtrip():
    ev = VisualEvidence(
        source="visual_analyst", confidence="medium", generated_by="gpt-5-mini",
        photos=[
            PhotoAssessment(
                image_index=0,
                quality=PhotoQuality(sharpness=4, lighting=3, framing=4,
                                     background=2, issues=["cluttered background"]),
                content=PhotoContent(pet_visible=True, expression="alert",
                                     setting="indoor", distinctive_traits=["blue eyes"]),
                appeal_score=7,
                improvement_suggestions=["use a plain background"],
            )
        ],
        overall_visual_appeal=7,
        best_photo_index=0,
        observed_traits=["blue eyes"],
        consistency_flags=[],
        photo_strategy_summary="Lead with photo 0.",
    )
    restored = VisualEvidence.model_validate(ev.model_dump())
    assert restored.best_photo_index == 0


def test_recommendation_requires_measured_impact():
    with pytest.raises(ValidationError):
        ValidatedRecommendation(
            action="Lower fee", feature="Fee", current_value="200",
            suggested_value="0", priority=1, category="listing_details",
            rationale="Fee is the top negative driver.",
        )


def test_adoption_report_minimal_degraded():
    """Every failure path must still yield a valid report."""
    report = AdoptionReport(
        prediction=make_prediction_evidence(),
        visual=None,
        recommendations=None,
        narrative="Prediction: class 2.",
        optimized_description=None,
        headline="Likely adopted within 1 month.",
        metadata=ReportMetadata(
            session_id="s1", ml_model_version="tuned_v1",
            llm_models={}, timing_ms={}, estimated_cost_usd=0.0,
            errors=[], timestamp="2026-07-21T00:00:00Z",
        ),
    )
    assert report.visual is None
    assert report.recommendations is None
