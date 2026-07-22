"""
E2E test runner: executes scenarios against the compiled Evidence Board
graph, collects structured outputs, and writes results to JSON files for
manual inspection.

Produces one unified JSON file per scenario in the outputs/ directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from adoption_accelerator.agents.graph import compile_report_graph
from adoption_accelerator.agents.observability.audit import build_audit_record

from tests.e2e_tests.scenarios import TestScenario, build_all_scenarios
from tests.e2e_tests.validators import ValidationResult, validate_scenario_output

logger = logging.getLogger(__name__)

# Output directory for structured JSON results
_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


async def run_scenario(scenario: TestScenario, app: Any) -> dict[str, Any]:
    """Execute a single scenario and collect structured output.

    Parameters
    ----------
    scenario : TestScenario
        The scenario definition.
    app : CompiledGraph
        The compiled Evidence Board graph.

    Returns
    -------
    dict
        Structured output document for the scenario.
    """
    logger.info("Running scenario: %s", scenario.name)
    t0 = time.perf_counter()

    # Execute the graph
    result = await app.ainvoke({"request": scenario.request, "errors": [], "trace": []})

    execution_time_ms = (time.perf_counter() - t0) * 1000

    # Extract structured data from the result
    output = _build_scenario_output(scenario, result, execution_time_ms)

    # Run validations
    validation = validate_scenario_output(
        scenario.name, result, scenario.expected_behavior
    )
    output["validation"] = validation.to_dict()

    report = result.get("report")
    logger.info(
        "Scenario '%s' completed: prediction=%s, time=%.0fms, "
        "checks=%d passed / %d failed",
        scenario.name,
        report.prediction.predicted_class if report is not None else None,
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
    report = result.get("report")
    trace_entries = result.get("trace", [])
    errors = result.get("errors", [])

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

    def _node_summary(node_name: str) -> dict[str, Any]:
        node_traces = [e for e in trace_entries if e.node == node_name]
        if not node_traces:
            return {"executed": False}
        entry = node_traces[0]
        summary: dict[str, Any] = {
            "executed": True,
            "status": entry.status,
            "timing_ms": round(entry.duration_ms, 2),
        }
        llm_usage = entry.metadata.get("llm_usage")
        if llm_usage:
            summary["llm_usage"] = llm_usage
        return summary

    agents_output = {
        "orchestrator": {
            "session_id": result.get("session_id", ""),
            "timestamp": result.get("timestamp", ""),
            **_node_summary("orchestrator"),
        },
        "inference": _node_summary("inference"),
        "data_analyst": _node_summary("data_analyst"),
        "visual_analyst": _node_summary("visual_analyst"),
        "recommendation_agent": _node_summary("recommendation_agent"),
        "synthesizer": _node_summary("synthesizer"),
        "aggregator": _node_summary("aggregator"),
    }

    # Final response: the full AdoptionReport, JSON-serialized
    final_response: dict[str, Any] | None = None
    if report is not None:
        final_response = report.model_dump(mode="json")

    error_summary = [
        {
            "node": e.node,
            "error_type": e.error_type,
            "message": e.message,
            "recoverable": e.recoverable,
        }
        for e in errors
    ]

    audit = build_audit_record(result)

    return {
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        "tags": scenario.tags,
        "input": input_summary,
        "agents": agents_output,
        "final_response": final_response,
        "errors": error_summary,
        "audit": audit,
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


async def _run_all_scenarios_async(
    out_dir: Path,
    scenarios: list[TestScenario],
) -> dict[str, Any]:
    logger.info("Compiling Evidence Board graph...")
    app = compile_report_graph()
    logger.info("Graph compiled successfully")

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
            output = await run_scenario(scenario, app)
            save_scenario_output(output, out_dir)

            v = output.get("validation", {})
            passed = v.get("passed_count", 0)
            failed = v.get("failed_count", 0)
            warns = v.get("warning_count", 0)
            total_passed += passed
            total_failed += failed
            total_warnings += warns

            report = output.get("final_response")
            results.append({
                "scenario": scenario.name,
                "status": "PASS" if failed == 0 else "FAIL",
                "checks_passed": passed,
                "checks_failed": failed,
                "warnings": warns,
                "execution_time_ms": output["execution_metadata"]["execution_time_ms"],
                "prediction": report["prediction"]["predicted_class"] if report else None,
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
    }

    # Save summary report
    summary_path = out_dir / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Summary report saved: %s", summary_path)

    return summary


def run_all_scenarios(
    output_dir: Path | None = None,
    scenarios: list[TestScenario] | None = None,
) -> dict[str, Any]:
    """Execute all scenarios and produce a summary report.

    Requires a real LLM API key: every scenario invokes the full
    Evidence Board graph, whose ``data_analyst``, ``visual_analyst``,
    ``recommendation_agent``, and ``synthesizer`` nodes call out to an
    LLM.

    Parameters
    ----------
    output_dir : Path | None
        Output directory. Defaults to ``tests/e2e_tests/outputs/``.
    scenarios : list[TestScenario] | None
        Scenarios to run. Defaults to all built-in scenarios.

    Returns
    -------
    dict
        Summary report with per-scenario results.
    """
    out_dir = output_dir or _OUTPUT_DIR
    scenarios = scenarios or build_all_scenarios()
    return asyncio.run(_run_all_scenarios_async(out_dir, scenarios))
