"""Tests for the data_analyst node with a mocked LLM."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from adoption_accelerator.agents.contracts import PredictionEvidence
from adoption_accelerator.agents.nodes.data_analyst import (
    DataAnalystOutput,
    data_analyst_node,
)
from adoption_accelerator.inference.contracts import PredictionResult
from adoption_accelerator.interpretability.translator import (
    InterpretedExplanation,
    InterpretedFactor,
)


def make_state():
    prediction = PredictionResult(
        prediction=2, prediction_label="Adopted within 1 month",
        probabilities={0: 0.1, 1: 0.2, 2: 0.4, 3: 0.2, 4: 0.1},
        confidence=0.4,
    )
    interpreted = InterpretedExplanation(
        top_factors=[
            InterpretedFactor(
                name="Age", description="Pet age in months",
                shap_magnitude=0.08, direction="positive",
                modality="tabular", group="core", value="6",
            )
        ],
        modality_contributions={"tabular": 1.0},
        modality_available={"tabular": True, "text": False, "image": False},
    )
    return {
        "prediction": prediction,
        "interpreted_explanation": interpreted,
        "modality_available": interpreted.modality_available,
        "timestamp": "2026-07-21T00:00:00Z",
    }


async def test_data_analyst_builds_evidence_from_llm():
    fake_structured = AsyncMock()
    fake_structured.ainvoke.return_value = {
        "parsed": DataAnalystOutput(
            driver_readings=["Young age pushes adoption faster."],
            uncertainty_reading="Probabilities are moderately spread.",
        ),
        "raw": AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 120,
                "output_tokens": 40,
                "total_tokens": 160,
            },
        ),
        "parsing_error": None,
    }
    fake_model = SimpleNamespace(
        with_structured_output=lambda schema, **kw: fake_structured
    )
    with patch(
        "adoption_accelerator.agents.nodes.data_analyst.get_chat_model",
        return_value=fake_model,
    ):
        updates = await data_analyst_node(make_state())

    ev = updates["prediction_evidence"]
    assert isinstance(ev, PredictionEvidence)
    assert ev.predicted_class == 2
    assert ev.key_drivers[0].reading == "Young age pushes adoption faster."
    assert ev.generated_by != "deterministic"

    trace_meta = updates["trace"][0].metadata
    assert trace_meta["llm_usage"]["input_tokens"] == 120
    assert trace_meta["llm_usage"]["model_key"] == "gpt-5-nano"


async def test_data_analyst_falls_back_on_llm_failure():
    fake_structured = AsyncMock()
    fake_structured.ainvoke.side_effect = RuntimeError("api down")
    fake_model = SimpleNamespace(
        with_structured_output=lambda schema, **kw: fake_structured
    )
    with patch(
        "adoption_accelerator.agents.nodes.data_analyst.get_chat_model",
        return_value=fake_model,
    ):
        updates = await data_analyst_node(make_state())

    ev = updates["prediction_evidence"]
    assert ev is not None
    assert ev.generated_by == "deterministic"
    assert ev.predicted_class == 2
    assert any(e.error_type == "llm_failure" for e in updates["errors"])
