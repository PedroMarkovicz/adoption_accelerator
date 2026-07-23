import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PeelBack } from "@/components/report/PeelBack";
import { buildSpeedClasses } from "@/lib/spectrum";
import { fullReport } from "./fixtures/report";

const classes = buildSpeedClasses([
  { index: 0, label: "Same-day adoption" }, { index: 1, label: "Adopted within 1 week" },
  { index: 2, label: "Adopted within 1 month" }, { index: 3, label: "Adopted within 1-3 months" },
  { index: 4, label: "Not adopted (100+ days)" },
]);

describe("PeelBack", () => {
  it("reveals key drivers on expand", async () => {
    render(<PeelBack report={fullReport} classes={classes} />);
    await userEvent.click(screen.getByRole("button", { name: /see how the AI decided/i }));
    expect(await screen.findByText(/Photo appeal/)).toBeInTheDocument();
  });
});
