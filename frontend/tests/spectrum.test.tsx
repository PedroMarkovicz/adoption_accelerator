import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SpeedSpectrum } from "@/components/report/SpeedSpectrum";
import { buildSpeedClasses } from "@/lib/spectrum";

const classes = buildSpeedClasses([
  { index: 0, label: "Same-day adoption" },
  { index: 1, label: "Adopted within 1 week" },
  { index: 2, label: "Adopted within 1 month" },
  { index: 3, label: "Adopted within 1-3 months" },
  { index: 4, label: "Not adopted (100+ days)" },
]);

describe("SpeedSpectrum", () => {
  it("renders a marker labelled by the predicted class", () => {
    render(<SpeedSpectrum classes={classes} markerClass={1} confidence={0.78} />);
    expect(screen.getByTestId("spectrum-marker")).toHaveAttribute("data-class", "1");
    expect(screen.getByText(/78%/)).toBeInTheDocument();
  });
});
