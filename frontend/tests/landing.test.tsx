import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import Home from "@/app/page";

describe("Landing", () => {
  it("shows the thesis and a CTA to predict", () => {
    render(<Home />);
    expect(screen.getByRole("link", { name: /predict adoption speed/i })).toHaveAttribute("href", "/predict");
  });
});
