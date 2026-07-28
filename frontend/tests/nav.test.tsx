import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { SiteHeader } from "@/components/nav/SiteHeader";

describe("SiteHeader", () => {
  it("links the wordmark home and exposes Predict and Explore nav links", () => {
    render(<SiteHeader />);
    expect(screen.getByRole("link", { name: /adoption accelerator/i })).toHaveAttribute("href", "/");
    expect(screen.getByRole("link", { name: /predict/i })).toHaveAttribute("href", "/predict");
    expect(screen.getByRole("link", { name: /explore/i })).toHaveAttribute("href", "/explore");
  });
});
