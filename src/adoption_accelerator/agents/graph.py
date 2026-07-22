"""Evidence Board graph.

Topology:
    orchestrator -> inference -> [visual_analyst, data_analyst] (parallel)
                              -> recommendation_agent -> synthesizer
                              -> aggregator -> END

visual_analyst self-skips when the request has no images (in-node skip;
the topology stays linear so the fan-in never deadlocks)."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from adoption_accelerator.agents.nodes.aggregator import aggregator_node
from adoption_accelerator.agents.nodes.data_analyst import data_analyst_node
from adoption_accelerator.agents.nodes.inference import inference_node
from adoption_accelerator.agents.nodes.orchestrator import orchestrator_node
from adoption_accelerator.agents.nodes.synthesizer import synthesizer_node
from adoption_accelerator.agents.nodes.visual_analyst import visual_analyst_node
from adoption_accelerator.agents.state import AgentState
from adoption_accelerator.agents.subgraphs.recommendation_agent import (
    recommendation_agent_node,
)

logger = logging.getLogger(__name__)


def build_report_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("inference", inference_node)

    # Late-binding wrappers: the four LLM nodes are resolved by name at
    # call time (via the module object), not passed as function objects
    # captured at import time. This is what makes
    # ``patch("adoption_accelerator.agents.graph.<name>", fake)`` take
    # effect in tests -- a direct ``graph.add_node("data_analyst",
    # data_analyst_node)`` would bind the original function object and
    # ignore any later monkeypatch of the module attribute.
    #
    # NOTE: these must be ``async def`` wrappers, not plain lambdas that
    # return a coroutine. LangGraph decides whether to ``await`` a node
    # by checking ``iscoroutinefunction`` on the callable itself; a
    # lambda is a regular (sync) function even when its body returns a
    # coroutine object, so LangGraph would treat the returned coroutine
    # as the node's state update and fail with InvalidUpdateError.
    import adoption_accelerator.agents.graph as _self

    async def _visual_analyst(s: AgentState) -> dict:
        return await _self.visual_analyst_node(s)

    async def _data_analyst(s: AgentState) -> dict:
        return await _self.data_analyst_node(s)

    async def _recommendation_agent(s: AgentState) -> dict:
        return await _self.recommendation_agent_node(s)

    async def _synthesizer(s: AgentState) -> dict:
        return await _self.synthesizer_node(s)

    graph.add_node("visual_analyst", _visual_analyst)
    graph.add_node("data_analyst", _data_analyst)
    graph.add_node("recommendation_agent", _recommendation_agent)
    graph.add_node("synthesizer", _synthesizer)

    graph.add_node("aggregator", aggregator_node)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "inference")
    # Fan-out: both analysts run in parallel after inference
    graph.add_edge("inference", "visual_analyst")
    graph.add_edge("inference", "data_analyst")
    # Fan-in: recommendation agent waits for both
    graph.add_edge(["visual_analyst", "data_analyst"], "recommendation_agent")
    graph.add_edge("recommendation_agent", "synthesizer")
    graph.add_edge("synthesizer", "aggregator")
    graph.add_edge("aggregator", END)

    logger.info("Evidence Board graph built")
    return graph


def compile_report_graph() -> Any:
    """Build and compile. Invoke with
    ``await app.ainvoke({"request": req, "errors": [], "trace": []})``."""
    app = build_report_graph().compile()
    logger.info("Evidence Board graph compiled")
    return app


def get_graph_config(session_id: str) -> dict:
    """Runtime config for graph invocation (Langfuse callback when set).

    Usage: ``await app.ainvoke(state, config=get_graph_config(session_id))``.
    Returns ``{}`` when Langfuse is not configured, so it is always safe to
    pass through unconditionally.
    """
    from adoption_accelerator.agents.observability.langfuse import (
        get_langfuse_handler,
    )

    handler = get_langfuse_handler(session_id)
    if handler is None:
        return {}
    return {
        "callbacks": [handler],
        "metadata": {"langfuse_session_id": session_id},
    }
