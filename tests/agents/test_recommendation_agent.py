"""Tests for the bounded ReAct recommendation agent."""

from unittest.mock import patch

from langchain_core.messages import AIMessage

from adoption_accelerator.agents.contracts import (
    PredictionEvidence,
    RecommendationEvidence,
)
from adoption_accelerator.agents.subgraphs.recommendation_agent import (
    MAX_TOOL_CALLS,
    FinalRecommendationItem,
    FinalRecommendations,
    recommendation_agent_node,
)
from adoption_accelerator.contracts_test_helpers import make_request
from adoption_accelerator.inference.contracts import PredictionResult


class ScriptedToolModel:
    """Fake bindable chat model that emits a scripted sequence of
    AIMessages (tool calls, then plain stop), then a scripted structured
    output object."""

    def __init__(self, messages: list[AIMessage], final: FinalRecommendations):
        self._messages = list(messages)
        self._final = final
        self._overflow_calls = 0

    def bind_tools(self, tools):
        return self

    def with_structured_output(self, schema, **kw):
        outer = self

        class _Structured:
            async def ainvoke(self, _input):
                return {
                    "parsed": outer._final,
                    "raw": AIMessage(
                        content="",
                        usage_metadata={
                            "input_tokens": 50,
                            "output_tokens": 20,
                            "total_tokens": 70,
                        },
                    ),
                    "parsing_error": None,
                }

        return _Structured()

    async def ainvoke(self, _input):
        if self._messages:
            return self._messages.pop(0)
        # Scripted list exhausted: keep requesting a valid tool call
        # forever, so the caller's own bound (MAX_TOOL_CALLS) is the
        # only thing that can stop the loop.
        self._overflow_calls += 1
        return AIMessage(
            content="",
            tool_calls=[{
                "name": "run_counterfactual",
                "args": {"feature": "Fee", "value": "0"},
                "id": f"call_overflow_{self._overflow_calls}",
            }],
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


def make_state():
    prediction = PredictionResult(
        prediction=3, prediction_label="Adopted within 1-3 months",
        probabilities={0: 0.05, 1: 0.1, 2: 0.25, 3: 0.4, 4: 0.2},
        confidence=0.4,
    )
    evidence = PredictionEvidence(
        source="data_analyst", confidence="high", generated_by="deterministic",
        predicted_class=3, prediction_label=prediction.prediction_label,
        probabilities=prediction.probabilities, class_confidence=0.4,
        modality_contributions={"tabular": 1.0},
        modality_available={"tabular": True, "text": False, "image": False},
    )
    return {
        "request": make_request(),
        "prediction": prediction,
        "prediction_evidence": evidence,
        "visual_evidence": None,
        "timestamp": "t",
    }


async def test_agent_validates_recommendations_through_tools():
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "run_counterfactual",
            "args": {"feature": "Fee", "value": "0"},
            "id": "call_1",
        }],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    stop_msg = AIMessage(
        content="Done testing.",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    final = FinalRecommendations(
        items=[FinalRecommendationItem(
            measurement_id="m1", action="Waive the adoption fee",
            feature="Fee", suggested_value="0", priority=1,
            category="listing_details", rationale="Largest measured shift.",
        )],
        rejected_hypotheses=["Renaming alone showed no measured effect."],
    )
    model = ScriptedToolModel([tool_call_msg, stop_msg], final)
    with patch(
        "adoption_accelerator.agents.subgraphs.recommendation_agent.get_chat_model",
        return_value=model,
    ):
        updates = await recommendation_agent_node(make_state())

    ev = updates["recommendation_evidence"]
    assert isinstance(ev, RecommendationEvidence)
    assert ev.iterations_used == 1
    assert len(ev.recommendations) == 1
    rec = ev.recommendations[0]
    assert rec.feature == "Fee"
    # measured_impact must come from the real measurement log
    assert rec.measured_impact.class_before == 3
    assert set(rec.measured_impact.probability_shift.keys()) == {0, 1, 2, 3, 4}
    assert updates["trace"][0].metadata["llm_usage"]["model_key"] == "gpt-5-mini"


async def test_item_with_unknown_measurement_id_is_dropped():
    stop_msg = AIMessage(
        content="No tests needed.",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    final = FinalRecommendations(
        items=[FinalRecommendationItem(
            measurement_id="m99", action="x", feature="Fee",
            suggested_value="0", priority=1, category="listing_details",
            rationale="fabricated",
        )],
        rejected_hypotheses=[],
    )
    model = ScriptedToolModel([stop_msg], final)
    with patch(
        "adoption_accelerator.agents.subgraphs.recommendation_agent.get_chat_model",
        return_value=model,
    ):
        updates = await recommendation_agent_node(make_state())
    ev = updates["recommendation_evidence"]
    assert ev.recommendations == []
    assert any("m99" in n for n in ev.notes)


async def test_llm_failure_falls_back_to_deterministic_sweep():
    class FailingModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, _input):
            raise RuntimeError("api down")

    with patch(
        "adoption_accelerator.agents.subgraphs.recommendation_agent.get_chat_model",
        return_value=FailingModel(),
    ):
        updates = await recommendation_agent_node(make_state())
    ev = updates["recommendation_evidence"]
    assert ev is not None
    assert ev.generated_by == "deterministic"
    # deterministic sweep still measures through the real model
    for rec in ev.recommendations:
        assert rec.measured_impact.class_after <= rec.measured_impact.class_before
    assert "llm_usage" not in updates["trace"][0].metadata


async def test_tool_call_budget_never_exceeds_max_tool_calls():
    # The model ALWAYS requests another tool call (never a plain stop),
    # so an unbounded loop would run forever. If MAX_TOOL_CALLS truly
    # caps real tool execution, exactly MAX_TOOL_CALLS real predictions
    # run through the real pipeline and no more.
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "run_counterfactual",
            "args": {"feature": "Fee", "value": "0"},
            "id": "call_1",
        }],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    final = FinalRecommendations(items=[], rejected_hypotheses=[])
    model = ScriptedToolModel([tool_call_msg], final)
    with patch(
        "adoption_accelerator.agents.subgraphs.recommendation_agent.get_chat_model",
        return_value=model,
    ):
        updates = await recommendation_agent_node(make_state())

    ev = updates["recommendation_evidence"]
    assert ev.iterations_used == MAX_TOOL_CALLS
    assert any("budget" in n.lower() for n in ev.notes)


async def test_recommendation_agent_consults_the_timeout_loader():
    """Regression guard for the dead-config bug this task fixes: the node
    must ask runtime_config.node_timeout for its own timeout by name and
    pass that value through to asyncio.wait_for, not a hardcoded literal.
    Covers both call sites: the ReAct loop and the structured finalize
    call.
    """
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{
            "name": "run_counterfactual",
            "args": {"feature": "Fee", "value": "0"},
            "id": "call_1",
        }],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    stop_msg = AIMessage(
        content="Done testing.",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    final = FinalRecommendations(items=[], rejected_hypotheses=[])
    model = ScriptedToolModel([tool_call_msg, stop_msg], final)

    captured_timeouts = []

    async def fake_wait_for(coro, timeout):
        captured_timeouts.append(timeout)
        return await coro

    with patch(
        "adoption_accelerator.agents.subgraphs.recommendation_agent.get_chat_model",
        return_value=model,
    ), patch(
        "adoption_accelerator.agents.subgraphs.recommendation_agent.node_timeout",
        return_value=12345.0,
    ) as mock_node_timeout, patch(
        "adoption_accelerator.agents.subgraphs.recommendation_agent.asyncio.wait_for",
        side_effect=fake_wait_for,
    ):
        await recommendation_agent_node(make_state())

    assert mock_node_timeout.call_count == 2
    assert all(c.args == ("recommendation_agent",) for c in mock_node_timeout.call_args_list)
    assert captured_timeouts == [12345.0, 12345.0]
