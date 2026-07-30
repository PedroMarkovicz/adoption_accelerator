import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Recommendations } from "@/components/report/Recommendations";
import type { RecommendationEvidence } from "@/lib/types";
import type { SpeedClass } from "@/lib/spectrum";

const classes = [
  { label: "Same-day adoption" },
  { label: "Adopted within 1 week" },
  { label: "Adopted within 1 month" },
  { label: "Adopted within 1-3 months" },
  { label: "Not adopted (100+ days)" },
] as unknown as SpeedClass[];

function evidence(expected_speedup: string): RecommendationEvidence {
  return {
    recommendations: [
      {
        action: "Increase number of photos to 6",
        feature: "PhotoAmt",
        current_value: "1",
        suggested_value: "6",
        measured_impact: {
          class_before: 2,
          class_after: 2,
          probability_shift: { 3: 0.0595 },
          expected_speedup,
        },
        priority: 1,
        category: "Photos",
        rationale: "Counterfactual test moved mass toward slower adoption.",
      },
    ],
    rejected_hypotheses: [],
    iterations_used: 1,
    confidence: "high",
    source: "react-agent",
    generated_by: "gpt-5-mini",
    notes: [],
  } as unknown as RecommendationEvidence;
}

describe("Recommendations", () => {
  it("renders a measured regression without softening it into an improvement", () => {
    render(
      <Recommendations
        recs={evidence(
          "shifts probability toward slower adoption without changing the predicted class",
        )}
        classes={classes}
      />,
    );
    expect(screen.getByText(/shifts probability toward slower adoption/)).toBeInTheDocument();
    expect(screen.queryByText(/improves class probabilities/)).not.toBeInTheDocument();
  });

  it("renders a null result as no measurable change", () => {
    render(
      <Recommendations
        recs={evidence("no measurable change in the predicted probabilities")}
        classes={classes}
      />,
    );
    expect(screen.getByText(/no measurable change/)).toBeInTheDocument();
    expect(screen.queryByText(/improves/)).not.toBeInTheDocument();
  });

  it("still renders a genuine improvement as one", () => {
    render(
      <Recommendations
        recs={evidence(
          "improves class probabilities without changing the predicted class",
        )}
        classes={classes}
      />,
    );
    expect(screen.getByText(/improves class probabilities/)).toBeInTheDocument();
  });
});
