"""Tests for the synthesizer node with a mocked LLM."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from adoption_accelerator.agents.contracts import (
    MeasuredImpact,
    PredictionEvidence,
    RecommendationEvidence,
    ValidatedRecommendation,
    VisualEvidence,
)
from adoption_accelerator.agents.nodes.synthesizer import (
    SynthesisOutput,
    synthesizer_node,
)
from adoption_accelerator.contracts_test_helpers import make_request


def make_state(with_visual=True, observed_traits=None):
    prediction_evidence = PredictionEvidence(
        source="data_analyst", confidence="high", generated_by="gpt-5-nano",
        predicted_class=3, prediction_label="Adopted within 1-3 months",
        probabilities={0: 0.05, 1: 0.1, 2: 0.25, 3: 0.4, 4: 0.2},
        class_confidence=0.4, modality_contributions={"tabular": 1.0},
        modality_available={"tabular": True, "text": True,
                            "image": with_visual},
    )
    visual = None
    if with_visual:
        visual = VisualEvidence(
            source="visual_analyst", confidence="medium",
            generated_by="gpt-5-mini", overall_visual_appeal=7,
            observed_traits=observed_traits or ["black and white coat"],
            photo_strategy_summary="Lead with photo 0.",
        )
    recs = RecommendationEvidence(
        source="recommendation_agent", confidence="high",
        generated_by="gpt-5-mini",
        recommendations=[ValidatedRecommendation(
            action="Waive the fee", feature="Fee", current_value="200",
            suggested_value="0",
            measured_impact=MeasuredImpact(
                class_before=3, class_after=2,
                probability_shift={0: 0.0, 1: 0.02, 2: 0.13, 3: -0.1, 4: -0.05},
                expected_speedup="moves the prediction to 'adopted within 1 month'",
            ),
        )],
    )
    return {
        "request": make_request(description="Nice dog."),
        "prediction_evidence": prediction_evidence,
        "visual_evidence": visual,
        "recommendation_evidence": recs,
        "timestamp": "t",
    }


def _model_returning(output):
    fake = AsyncMock()
    fake.ainvoke.return_value = {
        "parsed": output,
        "raw": AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 200,
                "output_tokens": 90,
                "total_tokens": 290,
            },
        ),
        "parsing_error": None,
    }
    return SimpleNamespace(with_structured_output=lambda schema, **kw: fake)


async def test_synthesizer_produces_all_outputs():
    output = SynthesisOutput(
        narrative="The model predicts adoption within 1-3 months. Waiving the "
                  "fee moves it to within 1 month.",
        headline="Likely adopted in 1-3 months; waive the fee to speed it up.",
        optimized_description="Meet Rex, a friendly dog with a black and white "
                              "coat looking for a home.",
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(make_state())
    assert updates["narrative"].startswith("The model predicts")
    assert updates["headline"]
    assert "black and white" in updates["optimized_description"]
    assert updates["trace"][0].metadata["llm_usage"]["model_key"] == "gpt-5-mini"


async def test_description_grounding_rejects_unobserved_traits():
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description="Meet Rex, with striking blue eyes.",  # not observed
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(make_state())
    # description dropped, narrative kept
    assert updates["optimized_description"] is None
    assert updates["narrative"] == "ok narrative for the report"


async def test_description_grounding_keeps_paraphrased_observed_trait():
    # observed_traits uses a hyphenated phrasing; description uses a
    # slightly different but truthful phrasing of the same trait.
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description="Meet Rex, a friendly dog with a golden coat.",
    )
    state = make_state(observed_traits=["golden-colored coat"])
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(state)
    # legitimately grounded (word-boundary match survives the hyphen) -> kept
    assert updates["optimized_description"] is not None
    assert "golden coat" in updates["optimized_description"]


async def test_description_grounding_word_boundary_rejects_substring_bleed():
    # observed_traits does not mention "spotted" at all. Naive substring
    # matching (e.g. against something like "unspotted") could previously
    # let an ungrounded "spotted" claim slip through; word-boundary
    # matching must still reject it here.
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description="Meet Rex, with a spotted coat.",
    )
    state = make_state(observed_traits=["black and white coat"])
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(state)
    # "spotted" was claimed but not observed -> dropped
    assert updates["optimized_description"] is None


async def test_llm_failure_falls_back_to_template():
    fake = AsyncMock()
    fake.ainvoke.side_effect = RuntimeError("down")
    model = SimpleNamespace(with_structured_output=lambda schema, **kw: fake)
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=model,
    ):
        updates = await synthesizer_node(make_state())
    assert "1-3 months" in updates["narrative"]
    assert updates["optimized_description"] is None
    assert any(e.error_type == "llm_failure" for e in updates["errors"])
