"""
LangGraph graph definition and compilation.

Provides two graph builders:

1. ``build_deterministic_graph`` --- Phase 1 only (orchestrator -> inference
   -> aggregator -> END).  No LLM dependency.

2. ``build_full_graph`` --- Complete two-phase pipeline with fan-out /
   fan-in topology::

       orchestrator -> inference -> [explainer, recommender, description_writer*]
                                          |             |              |
                                          +--- fan-in --+--- aggregator -> END

   *description_writer only runs when the request has a non-empty description.

Conditional routing after the inference node determines which LLM nodes
are included in the fan-out.  All LLM nodes fan back into the aggregator.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langgraph.graph import END, StateGraph

from agents.nodes.aggregator import aggregator_node
from agents.nodes.description_writer import description_writer_node
from agents.nodes.explainer import explainer_node
from agents.nodes.inference import inference_node
from agents.nodes.orchestrator import orchestrator_node
from agents.nodes.recommender import recommender_node
from agents.state import AgentState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


def route_after_inference(state: AgentState) -> Sequence[str]:
    """Determine which LLM nodes to fan-out to after inference.

    Always includes ``explainer`` and ``recommender``.  Includes
    ``description_writer`` only when the request carries a non-empty
    description string.

    Parameters
    ----------
    state : AgentState
        Current graph state (post-inference).

    Returns
    -------
    list[str]
        Node names to execute in parallel.
    """
    targets = ["explainer", "recommender"]

    request = state.get("request")
    if request is not None:
        desc = getattr(request, "description", None) or ""
        if desc.strip():
            targets.append("description_writer")

    logger.info("Routing after inference -> %s", targets)
    return targets


# ---------------------------------------------------------------------------
# Deterministic-only graph (Phase 1)
# ---------------------------------------------------------------------------


def build_deterministic_graph() -> StateGraph:
    """Build the deterministic-only agent graph (Phase 1).

    Topology::

        orchestrator -> inference -> aggregator -> END

    Returns
    -------
    StateGraph
        Uncompiled graph.
    """
    graph = StateGraph(AgentState)

    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("inference", inference_node)
    graph.add_node("aggregator", aggregator_node)

    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "inference")
    graph.add_edge("inference", "aggregator")
    graph.add_edge("aggregator", END)

    logger.info("Deterministic graph built: orchestrator -> inference -> aggregator -> END")
    return graph


def compile_deterministic_graph() -> Any:
    """Build and compile the deterministic agent graph.

    Returns
    -------
    CompiledGraph
        Runnable graph app.
    """
    graph = build_deterministic_graph()
    app = graph.compile()
    logger.info("Deterministic graph compiled successfully")
    return app


# ---------------------------------------------------------------------------
# Full graph (Phase 1 + Phase 2)
# ---------------------------------------------------------------------------


def build_full_graph() -> StateGraph:
    """Build the complete agent graph with fan-out / fan-in topology.

    Topology::

        orchestrator -> inference --(conditional)--> [explainer,
                                                      recommender,
                                                      description_writer*]
                                                           |
                                                      aggregator -> END

    The conditional edge uses ``route_after_inference`` to decide which
    LLM nodes participate.  All selected LLM nodes execute concurrently
    (LangGraph fan-out), then converge on the aggregator (fan-in).

    Returns
    -------
    StateGraph
        Uncompiled graph.
    """
    graph = StateGraph(AgentState)

    # Register all nodes
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("inference", inference_node)
    graph.add_node("explainer", explainer_node)
    graph.add_node("recommender", recommender_node)
    graph.add_node("description_writer", description_writer_node)
    graph.add_node("aggregator", aggregator_node)

    # Sequential: orchestrator -> inference
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "inference")

    # Fan-out: inference -> LLM nodes via conditional routing
    graph.add_conditional_edges(
        "inference",
        route_after_inference,
        ["explainer", "recommender", "description_writer"],
    )

    # Fan-in: LLM nodes -> aggregator
    graph.add_edge("explainer", "aggregator")
    graph.add_edge("recommender", "aggregator")
    graph.add_edge("description_writer", "aggregator")

    # Terminal: aggregator -> END
    graph.add_edge("aggregator", END)

    logger.info(
        "Full graph built: orchestrator -> inference -> "
        "[explainer, recommender, description_writer] -> aggregator -> END"
    )
    return graph


def compile_full_graph() -> Any:
    """Build and compile the full agent graph.

    Returns
    -------
    CompiledGraph
        Runnable graph app.  Invoke with
        ``app.invoke({"request": prediction_request})``.
    """
    graph = build_full_graph()
    app = graph.compile()
    logger.info("Full graph compiled successfully")
    return app
