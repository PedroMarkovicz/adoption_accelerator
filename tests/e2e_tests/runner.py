"""
E2E test runner: executes scenarios, collects structured outputs,
and writes results to JSON files for manual inspection.

Produces one unified JSON file per scenario in the outputs/ directory
following Option B (preferred) from the requirements.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.graph import compile_full_graph
from agents.observability.metrics import MetricStore
from agents.observability.tracing import extract_trace_summary

from tests.e2e_tests.scenarios import TestScenario, build_all_scenarios
from tests.e2e_tests.validators import ValidationResult, validate_scenario_output

logger = logging.getLogger(__name__)

# Output directory for structured JSON results
_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run_scenario(
    scenario: TestScenario,
    app: Any,
    metric_store: MetricStore,
) -> dict[str, Any]:
    """Execute a single scenario and collect structured output.

    Parameters
    ----------
    scenario : TestScenario
        The scenario definition.
    app : CompiledGraph
        The compiled full graph.
    metric_store : MetricStore
        Metrics collector.

    Returns
    -------
    dict
        Structured output document for the scenario.
    """
    logger.info("Running scenario: %s", scenario.name)
    t0 = time.perf_counter()

    # Execute the graph
    result = app.invoke({"request": scenario.request})

    execution_time_ms = (time.perf_counter() - t0) * 1000

    # Record metrics
    metric_store.record_execution(result)

    # Extract structured data from the result
    output = _build_scenario_output(scenario, result, execution_time_ms)

    # Run validations
    validation = validate_scenario_output(
        scenario.name, result, scenario.expected_behavior
    )
    output["validation"] = validation.to_dict()

    logger.info(
        "Scenario '%s' completed: prediction=%s, time=%.0fms, "
        "checks=%d passed / %d failed",
        scenario.name,
        output["agents"]["inference"]["prediction"]["prediction"],
        execution_time_ms,
        len(validation.passed),
        len(validation.failed),
    )

    return output


def _build_scenario_output(
    scenario: TestScenario,
    result: dict[str, Any],
    execution_time_ms: float,
) -> dict[str, Any]:
    """Build the unified scenario output document."""
    response = result.get("response")
    trace_entries = result.get("trace", [])
    errors = result.get("errors", [])
    interpreted = result.get("interpreted_explanation")

    # Input summary
    t = scenario.request.tabular
    type_map = {1: "Dog", 2: "Cat"}
    gender_map = {1: "Male", 2: "Female", 3: "Mixed"}
    care_map = {1: "Yes", 2: "No", 3: "Not Sure"}
    health_map = {1: "Healthy", 2: "Minor Injury", 3: "Serious Injury"}

    input_summary = {
        "pet_type": type_map.get(t.type, str(t.type)),
        "name": t.name or "(none)",
        "age_months": t.age,
        "breed1": t.breed1,
        "breed2": t.breed2 or 0,
        "gender": gender_map.get(t.gender, str(t.gender)),
        "maturity_size": t.maturity_size,
        "fur_length": t.fur_length,
        "vaccinated": care_map.get(t.vaccinated, str(t.vaccinated)),
        "dewormed": care_map.get(t.dewormed, str(t.dewormed)),
        "sterilized": care_map.get(t.sterilized, str(t.sterilized)),
        "health": health_map.get(t.health, str(t.health)),
        "quantity": t.quantity,
        "fee": t.fee,
        "video_amt": t.video_amt,
        "has_description": bool(scenario.request.description and scenario.request.description.strip()),
        "description_length": len(scenario.request.description) if scenario.request.description else 0,
        "n_images": len(scenario.request.images),
    }

    # Orchestrator output
    orchestrator_output = {
        "session_id": result.get("session_id", ""),
        "timestamp": result.get("timestamp", ""),
    }

    # Inference output
    prediction = result.get("prediction")
    inference_output: dict[str, Any] = {"prediction": None, "explanation_summary": None}
    if prediction is not None:
        inference_output["prediction"] = {
            "prediction": prediction.prediction,
            "prediction_label": prediction.prediction_label,
            "confidence": round(prediction.confidence, 4),
            "probabilities": {
                str(k): round(v, 4) for k, v in prediction.probabilities.items()
            },
        }

    if interpreted is not None:
        inference_output["explanation_summary"] = {
            "modality_contributions": {
                k: round(v, 4) for k, v in interpreted.modality_contributions.items()
            },
            "top_factors": [
                {
                    "name": f.name,
                    "description": f.description,
                    "modality": f.modality,
                    "direction": f.direction,
                    "shap_magnitude": round(f.shap_magnitude, 4),
                    "group": f.group,
                }
                for f in interpreted.top_factors
            ],
            "aggregated_embeddings": {
                k: {
                    "n_dimensions": v.get("n_dimensions", 0),
                    "total_magnitude": round(v.get("total_magnitude", 0), 4),
                    "description": v.get("description", ""),
                }
                for k, v in interpreted.aggregated_embeddings.items()
            },
        }

    # Inference timing from trace
    inf_traces = [e for e in trace_entries if e.node == "inference"]
    if inf_traces:
        inference_output["timing_ms"] = inf_traces[0].duration_ms
        inference_output["feature_count"] = inf_traces[0].metadata.get("n_features")

    # Explainer output
    explainer_output: dict[str, Any] = {
        "narrative_explanation": "",
        "used_fallback": False,
    }
    if response is not None:
        explainer_output["narrative_explanation"] = response.narrative_explanation

    expl_traces = [e for e in trace_entries if e.node == "explainer"]
    if expl_traces:
        explainer_output["timing_ms"] = expl_traces[0].duration_ms
        explainer_output["status"] = expl_traces[0].status
        explainer_output["used_fallback"] = expl_traces[0].metadata.get("used_fallback", False)
        llm_usage = expl_traces[0].metadata.get("llm_usage")
        if llm_usage:
            explainer_output["llm_usage"] = llm_usage

    # Recommender output
    recommender_output: dict[str, Any] = {
        "recommendations": [],
        "used_fallback": False,
    }
    if response is not None:
        recommender_output["recommendations"] = [
            {
                "feature": r.feature,
                "current_value": r.current_value,
                "suggested_value": r.suggested_value,
                "expected_impact": r.expected_impact,
                "priority": r.priority,
                "category": r.category,
                "actionable": r.actionable,
            }
            for r in response.recommendations
        ]

    rec_traces = [e for e in trace_entries if e.node == "recommender"]
    if rec_traces:
        recommender_output["timing_ms"] = rec_traces[0].duration_ms
        recommender_output["status"] = rec_traces[0].status
        recommender_output["used_fallback"] = rec_traces[0].metadata.get("used_fallback", False)
        llm_usage = rec_traces[0].metadata.get("llm_usage")
        if llm_usage:
            recommender_output["llm_usage"] = llm_usage

    # Description writer output
    description_writer_output: dict[str, Any] = {
        "improved_description": None,
        "executed": False,
        "skipped": False,
    }
    dw_traces = [e for e in trace_entries if e.node == "description_writer"]
    if dw_traces:
        dw_trace = dw_traces[0]
        description_writer_output["executed"] = dw_trace.status != "skipped"
        description_writer_output["skipped"] = dw_trace.status == "skipped"
        description_writer_output["timing_ms"] = dw_trace.duration_ms
        description_writer_output["status"] = dw_trace.status

        if response is not None and response.improved_description:
            description_writer_output["improved_description"] = response.improved_description

        llm_usage = dw_trace.metadata.get("llm_usage")
        if llm_usage:
            description_writer_output["llm_usage"] = llm_usage

    # Aggregator output
    aggregator_output: dict[str, Any] = {}
    agg_traces = [e for e in trace_entries if e.node == "aggregator"]
    if agg_traces:
        aggregator_output["timing_ms"] = agg_traces[0].duration_ms
        aggregator_output["status"] = agg_traces[0].status
        aggregator_output["has_narrative"] = agg_traces[0].metadata.get("has_narrative", False)
        aggregator_output["n_recommendations"] = agg_traces[0].metadata.get("n_recommendations", 0)
        aggregator_output["has_improved_description"] = agg_traces[0].metadata.get(
            "has_improved_description", False
        )

    # Final response (the full AgentResponse as JSON)
    final_response: dict[str, Any] | None = None
    if response is not None:
        final_response = {
            "prediction": response.prediction,
            "prediction_label": response.prediction_label,
            "probabilities": {str(k): round(v, 4) for k, v in response.probabilities.items()},
            "confidence": round(response.confidence, 4),
            "narrative_explanation": response.narrative_explanation,
            "modality_contributions": {
                k: round(v, 4) for k, v in response.modality_contributions.items()
            },
            "top_positive_factors": [
                {
                    "feature": f.feature,
                    "display_name": f.display_name,
                    "value": f.value,
                    "shap_value": round(f.shap_value, 4),
                    "modality": f.modality,
                    "direction": f.direction,
                }
                for f in response.top_positive_factors
            ],
            "top_negative_factors": [
                {
                    "feature": f.feature,
                    "display_name": f.display_name,
                    "value": f.value,
                    "shap_value": round(f.shap_value, 4),
                    "modality": f.modality,
                    "direction": f.direction,
                }
                for f in response.top_negative_factors
            ],
            "recommendations": [
                {
                    "feature": r.feature,
                    "current_value": r.current_value,
                    "suggested_value": r.suggested_value,
                    "expected_impact": r.expected_impact,
                    "priority": r.priority,
                    "category": r.category,
                    "actionable": r.actionable,
                }
                for r in response.recommendations
            ],
            "improved_description": response.improved_description,
            "modality_available": getattr(response, "modality_available", {}),
            "metadata": {
                "session_id": response.metadata.session_id,
                "model_version": response.metadata.model_version,
                "model_type": response.metadata.model_type,
                "inference_time_ms": round(response.metadata.inference_time_ms, 2),
                "total_time_ms": round(response.metadata.total_time_ms, 2),
                "timestamp": response.metadata.timestamp,
                "nodes_executed": response.metadata.nodes_executed,
                "errors": [
                    {
                        "node": e.node,
                        "error_type": e.error_type,
                        "message": e.message,
                        "recoverable": e.recoverable,
                    }
                    for e in response.metadata.errors
                ],
            },
        }

    # Error summary
    error_summary = [
        {
            "node": e.node,
            "error_type": e.error_type,
            "message": e.message,
            "recoverable": e.recoverable,
        }
        for e in errors
    ]

    # Trace summary
    trace_summary = extract_trace_summary(result)

    return {
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        "tags": scenario.tags,
        "input": input_summary,
        "agents": {
            "orchestrator": orchestrator_output,
            "inference": inference_output,
            "explainer": explainer_output,
            "recommender": recommender_output,
            "description_writer": description_writer_output,
            "aggregator": aggregator_output,
        },
        "final_response": final_response,
        "errors": error_summary,
        "trace": trace_summary,
        "execution_metadata": {
            "execution_time_ms": round(execution_time_ms, 2),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def save_scenario_output(output: dict[str, Any], output_dir: Path | None = None) -> Path:
    """Save a scenario output to a JSON file.

    Parameters
    ----------
    output : dict
        Scenario output from ``run_scenario``.
    output_dir : Path | None
        Directory to write to. Defaults to ``tests/e2e_tests/outputs/``.

    Returns
    -------
    Path
        Path to the written file.
    """
    out_dir = output_dir or _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{output['scenario']}.json"
    filepath = out_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    logger.info("Scenario output saved: %s", filepath)
    return filepath


def run_all_scenarios(
    output_dir: Path | None = None,
    scenarios: list[TestScenario] | None = None,
) -> dict[str, Any]:
    """Execute all scenarios and produce a summary report.

    Parameters
    ----------
    output_dir : Path | None
        Output directory. Defaults to ``tests/e2e_tests/outputs/``.
    scenarios : list[TestScenario] | None
        Scenarios to run. Defaults to all built-in scenarios.

    Returns
    -------
    dict
        Summary report with per-scenario results and aggregate metrics.
    """
    out_dir = output_dir or _OUTPUT_DIR
    scenarios = scenarios or build_all_scenarios()

    # Compile graph once for all scenarios
    logger.info("Compiling full agent graph...")
    app = compile_full_graph()
    logger.info("Graph compiled successfully")

    metric_store = MetricStore()

    results: list[dict[str, Any]] = []
    total_passed = 0
    total_failed = 0
    total_warnings = 0

    t0_all = time.perf_counter()

    for i, scenario in enumerate(scenarios, 1):
        logger.info(
            "--- Scenario %d/%d: %s ---",
            i, len(scenarios), scenario.name,
        )

        try:
            output = run_scenario(scenario, app, metric_store)
            save_scenario_output(output, out_dir)

            v = output.get("validation", {})
            passed = v.get("passed_count", 0)
            failed = v.get("failed_count", 0)
            warns = v.get("warning_count", 0)
            total_passed += passed
            total_failed += failed
            total_warnings += warns

            results.append({
                "scenario": scenario.name,
                "status": "PASS" if failed == 0 else "FAIL",
                "checks_passed": passed,
                "checks_failed": failed,
                "warnings": warns,
                "execution_time_ms": output["execution_metadata"]["execution_time_ms"],
                "prediction": output["agents"]["inference"]["prediction"]["prediction"]
                if output["agents"]["inference"]["prediction"]
                else None,
            })

        except Exception as exc:
            logger.error("Scenario '%s' CRASHED: %s", scenario.name, exc, exc_info=True)
            total_failed += 1
            results.append({
                "scenario": scenario.name,
                "status": "CRASH",
                "error": str(exc),
            })

    total_time_ms = (time.perf_counter() - t0_all) * 1000

    # Aggregate metrics
    metrics_summary = metric_store.get_summary()

    # Build summary report
    summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_scenarios": len(scenarios),
        "scenarios_passed": sum(1 for r in results if r["status"] == "PASS"),
        "scenarios_failed": sum(1 for r in results if r["status"] in ("FAIL", "CRASH")),
        "total_checks_passed": total_passed,
        "total_checks_failed": total_failed,
        "total_warnings": total_warnings,
        "total_execution_time_ms": round(total_time_ms, 2),
        "per_scenario_results": results,
        "aggregate_metrics": metrics_summary,
    }

    # Save summary report
    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Summary report saved: %s", summary_path)

    return summary
