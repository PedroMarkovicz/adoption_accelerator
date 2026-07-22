"""End-to-end graph test with all LLM calls mocked."""

from unittest.mock import patch

import pytest

from adoption_accelerator.agents.contracts import (
    AdoptionReport,
    PredictionEvidence,
    RecommendationEvidence,
)
from adoption_accelerator.agents.graph import compile_report_graph
from adoption_accelerator.contracts_test_helpers import make_request


def _fake_data_analyst_updates(state):
    prediction = state["prediction"]
    interpreted = state["interpreted_explanation"]
    return {
        "prediction_evidence": PredictionEvidence(
            source="data_analyst", confidence="medium",
            generated_by="deterministic",
            predicted_class=prediction.prediction,
            prediction_label=prediction.prediction_label,
            probabilities=prediction.probabilities,
            class_confidence=prediction.confidence,
            modality_contributions=interpreted.modality_contributions,
            modality_available=interpreted.modality_available,
        ),
        "errors": [], "trace": [],
    }


async def test_graph_runs_end_to_end_without_llm():
    async def fake_data_analyst(state):
        return _fake_data_analyst_updates(state)

    async def fake_visual(state):
        return {"visual_evidence": None, "errors": [], "trace": []}

    async def fake_recommend(state):
        return {
            "recommendation_evidence": RecommendationEvidence(
                source="recommendation_agent", confidence="low",
                generated_by="deterministic",
            ),
            "errors": [], "trace": [],
        }

    async def fake_synth(state):
        return {"narrative": "n", "headline": "h",
                "optimized_description": None, "errors": [], "trace": []}

    with patch("adoption_accelerator.agents.graph.data_analyst_node",
               fake_data_analyst), \
         patch("adoption_accelerator.agents.graph.visual_analyst_node",
               fake_visual), \
         patch("adoption_accelerator.agents.graph.recommendation_agent_node",
               fake_recommend), \
         patch("adoption_accelerator.agents.graph.synthesizer_node",
               fake_synth):
        app = compile_report_graph()
        state = await app.ainvoke(
            {"request": make_request(), "errors": [], "trace": []}
        )

    report = state["report"]
    assert isinstance(report, AdoptionReport)
    assert report.prediction.predicted_class in range(5)
    assert report.visual is None
    assert report.narrative == "n"
    assert report.metadata.session_id
