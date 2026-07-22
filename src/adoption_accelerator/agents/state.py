"""LangGraph shared state for the Evidence Board graph. State only —
response contracts live in ``contracts.py``."""

from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

from adoption_accelerator.agents.contracts import (
    NodeError,
    PredictionEvidence,
    RecommendationEvidence,
    AdoptionReport,
    TraceEntry,
    VisualEvidence,
)
from adoption_accelerator.inference.contracts import (
    PredictionRequest,
    PredictionResult,
)
from adoption_accelerator.interpretability.contracts import ExplanationResult
from adoption_accelerator.interpretability.translator import InterpretedExplanation


class AgentState(TypedDict, total=False):
    # Input (orchestrator)
    request: PredictionRequest
    session_id: str
    timestamp: str
    # Phase 1 (inference) — raw deterministic outputs consumed by data_analyst
    prediction: Optional[PredictionResult]
    explanation: Optional[ExplanationResult]
    interpreted_explanation: Optional[InterpretedExplanation]
    feature_vector: Optional[list[float]]
    feature_names: Optional[list[str]]
    modality_available: Optional[dict[str, bool]]
    # Phase 2a (parallel analysts)
    prediction_evidence: Optional[PredictionEvidence]
    visual_evidence: Optional[VisualEvidence]
    # Phase 2b (ReAct)
    recommendation_evidence: Optional[RecommendationEvidence]
    # Phase 3 (synthesizer)
    narrative: Optional[str]
    optimized_description: Optional[str]
    headline: Optional[str]
    # Final (aggregator)
    report: Optional[AdoptionReport]
    # Observability (accumulating)
    errors: Annotated[list[NodeError], operator.add]
    trace: Annotated[list[TraceEntry], operator.add]
