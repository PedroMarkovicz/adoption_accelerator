import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { reportImageUrl } from "@/lib/images";
import { ReportPhoto } from "@/components/report/ReportPhoto";

describe("reportImageUrl", () => {
  it("builds the BFF image path", () => {
    expect(reportImageUrl("abc-123", 2)).toBe("/api/predict/abc-123/images/2");
  });
});

describe("ReportPhoto", () => {
  it("renders an image with a descriptive alt", () => {
    render(<ReportPhoto sessionId="s1" index={0} total={3} />);
    const img = screen.getByAltText("Uploaded photo 1 of 3");
    expect(img).toHaveAttribute("src", "/api/predict/s1/images/0");
  });

  it("falls back to a placeholder when the image fails to load", () => {
    render(<ReportPhoto sessionId="s1" index={0} total={1} />);
    fireEvent.error(screen.getByAltText("Uploaded photo 1 of 1"));
    expect(screen.getByText("Photo no longer available")).toBeInTheDocument();
  });
});
