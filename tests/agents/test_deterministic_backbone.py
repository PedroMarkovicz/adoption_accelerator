"""Integration test: orchestrator + inference nodes against the real bundle."""

import pytest

from adoption_accelerator.agents.nodes.inference import inference_node
from adoption_accelerator.agents.nodes.orchestrator import orchestrator_node
from adoption_accelerator.inference.contracts import PredictionRequest, TabularInput


@pytest.fixture(scope="module")
def request_fixture() -> PredictionRequest:
    return PredictionRequest(
        tabular=TabularInput(
            type=1, name="Rex", age=6, breed1=307, gender=1, color1=1,
            maturity_size=2, fur_length=1, vaccinated=1, dewormed=1,
            sterilized=2, health=1, quantity=1, fee=0.0, state=41326,
        ),
        description="Friendly young dog, loves people.",
        images=[],
    )


def test_orchestrator_sets_session(request_fixture):
    updates = orchestrator_node({"request": request_fixture})
    assert updates["session_id"]
    assert updates["timestamp"]


def test_orchestrator_honors_incoming_session_id(request_fixture):
    updates = orchestrator_node({"request": request_fixture, "session_id": "fixed-id"})
    assert updates["session_id"] == "fixed-id"


def test_orchestrator_generates_session_id_when_absent(request_fixture):
    updates = orchestrator_node({"request": request_fixture})
    assert updates["session_id"]
    assert updates["session_id"] != "fixed-id"


def test_inference_produces_prediction_and_interpretation(request_fixture):
    state = {"request": request_fixture, "errors": [], "trace": []}
    state.update(orchestrator_node(state))
    updates = inference_node(state)
    assert updates["prediction"].prediction in range(5)
    assert updates["interpreted_explanation"] is not None
    assert updates["modality_available"]["tabular"] is True
    assert updates["modality_available"]["image"] is False
