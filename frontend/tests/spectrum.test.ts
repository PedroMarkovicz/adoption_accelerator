import { describe, it, expect } from "vitest";
import { buildSpeedClasses, expectedPosition, SPECTRUM_COLORS } from "@/lib/spectrum";

describe("spectrum math", () => {
  it("assigns fixed colors by index", () => {
    const classes = buildSpeedClasses([
      { index: 0, label: "Same-day adoption" },
      { index: 1, label: "Adopted within 1 week" },
    ]);
    expect(classes[0].color).toBe(SPECTRUM_COLORS[0]);
    expect(classes[1].label).toBe("Adopted within 1 week");
  });

  it("expectedPosition is the probability-weighted class index", () => {
    expect(expectedPosition({ "0": 1 })).toBeCloseTo(0);
    expect(expectedPosition({ "4": 1 })).toBeCloseTo(4);
    expect(expectedPosition({ "0": 0.5, "2": 0.5 })).toBeCloseTo(1);
  });
});
