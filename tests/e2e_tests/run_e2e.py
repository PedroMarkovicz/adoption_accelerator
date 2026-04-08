"""
Main entry point for the E2E test suite.

Usage:
    python -m tests.e2e_tests.run_e2e           # run all scenarios
    python tests/e2e_tests/run_e2e.py            # run all scenarios

Outputs are written to tests/e2e_tests/outputs/ as structured JSON.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Configure logging before any imports that use logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger("e2e_tests")


def main() -> int:
    """Run the full E2E test suite."""
    from tests.e2e_tests.runner import run_all_scenarios

    output_dir = Path(__file__).resolve().parent / "outputs"

    print("=" * 70)
    print("  ADOPTION ACCELERATOR - END-TO-END TEST SUITE")
    print("=" * 70)
    print()

    summary = run_all_scenarios(output_dir=output_dir)

    # Print results
    print()
    print("=" * 70)
    print("  E2E TEST RESULTS")
    print("=" * 70)
    print()

    for result in summary["per_scenario_results"]:
        status = result["status"]
        name = result["scenario"]

        if status == "PASS":
            marker = "[PASS]"
        elif status == "FAIL":
            marker = "[FAIL]"
        else:
            marker = "[CRASH]"

        if "execution_time_ms" in result:
            time_str = f"{result['execution_time_ms']:.0f}ms"
        else:
            time_str = "N/A"

        checks = ""
        if "checks_passed" in result:
            checks = f" ({result['checks_passed']} passed"
            if result.get("checks_failed", 0) > 0:
                checks += f", {result['checks_failed']} FAILED"
            if result.get("warnings", 0) > 0:
                checks += f", {result['warnings']} warnings"
            checks += ")"

        pred = ""
        if result.get("prediction") is not None:
            pred = f" -> class {result['prediction']}"

        print(f"  {marker} {name}{pred} [{time_str}]{checks}")

        if status == "CRASH":
            print(f"         Error: {result.get('error', 'unknown')}")

    print()
    print("-" * 70)
    print(f"  Scenarios: {summary['scenarios_passed']}/{summary['total_scenarios']} passed")
    print(f"  Checks:    {summary['total_checks_passed']} passed, "
          f"{summary['total_checks_failed']} failed, "
          f"{summary['total_warnings']} warnings")
    print(f"  Total time: {summary['total_execution_time_ms']:.0f}ms")
    print(f"  Outputs:   {output_dir}/")
    print("-" * 70)

    # Print aggregate metrics
    metrics = summary.get("aggregate_metrics", {})
    if metrics.get("request_count", 0) > 0:
        print()
        print("  AGGREGATE METRICS")
        print()

        inf = metrics.get("inference_latency_ms", {})
        total = metrics.get("total_latency_ms", {})
        print(f"  Inference latency: mean={inf.get('mean', 0):.0f}ms, "
              f"p50={inf.get('p50', 0):.0f}ms, "
              f"max={inf.get('max', 0):.0f}ms")
        print(f"  Total latency:     mean={total.get('mean', 0):.0f}ms, "
              f"p50={total.get('p50', 0):.0f}ms, "
              f"max={total.get('max', 0):.0f}ms")

        pred_dist = metrics.get("prediction_distribution", {})
        if pred_dist:
            dist_str = ", ".join(
                f"class {k}: {v}" for k, v in sorted(pred_dist.items())
            )
            print(f"  Prediction dist:   {dist_str}")

        confidence = metrics.get("confidence", {})
        if confidence.get("count", 0) > 0:
            print(f"  Confidence:        mean={confidence.get('mean', 0):.4f}, "
                  f"min={confidence.get('min', 0):.4f}, "
                  f"max={confidence.get('max', 0):.4f}")

        fallback = metrics.get("fallback_rate", {})
        if fallback:
            fb_str = ", ".join(
                f"{node}: {rate:.0%}" for node, rate in fallback.items()
            )
            print(f"  Fallback rates:    {fb_str}")

        llm_tokens = metrics.get("llm_token_usage", {})
        if llm_tokens:
            total_in = sum(v.get("input_tokens", 0) for v in llm_tokens.values())
            total_out = sum(v.get("output_tokens", 0) for v in llm_tokens.values())
            print(f"  LLM tokens:        {total_in} input, {total_out} output "
                  f"({total_in + total_out} total)")

    print()
    print("=" * 70)

    if summary["scenarios_failed"] > 0:
        print("  SOME SCENARIOS FAILED - review output files for details")
        print("=" * 70)
        return 1
    else:
        print("  ALL SCENARIOS PASSED SUCCESSFULLY")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    sys.exit(main())
