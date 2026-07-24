import { describe, it, expect, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { Dossier } from "@/components/report/Dossier";
import { fullReport, degradedReport, degradedReportWithPhotos } from "./fixtures/report";

vi.mock("@/lib/useMeta", () => ({
  useMeta: () => ({ data: { adoption_speed_classes: [
    { index: 0, label: "Same-day adoption" }, { index: 1, label: "Adopted within 1 week" },
    { index: 2, label: "Adopted within 1 month" }, { index: 3, label: "Adopted within 1-3 months" },
    { index: 4, label: "Not adopted (100+ days)" },
  ] } }) }));

describe("Dossier surface", () => {
  it("renders the verdict headline and a recommendation", () => {
    render(<Dossier report={fullReport} listing={null} />);
    expect(screen.getByText("Likely adopted within a week")).toBeInTheDocument();
    expect(screen.getByText(/Add a brighter lead photo/)).toBeInTheDocument();
    expect(screen.getByText(/Meet a friendly, healthy companion/)).toBeInTheDocument();
  });

  it("degrades gracefully when generative layers are missing", () => {
    render(<Dossier report={degradedReport} listing={null} />);
    expect(screen.getByText(/generative layers were unavailable/i)).toBeInTheDocument();
  });

  it("still shows uploaded photos when generative layers fail, without contradicting itself", () => {
    render(<Dossier report={degradedReportWithPhotos} listing={null} />);
    // The fallback lead photo in the hero and the photo-feedback gallery both
    // legitimately render photo 1 (visual is null, so the hero falls back to
    // index 0) -- so this proves both spots render it, each by its own
    // (now distinct) accessible name.
    expect(screen.getByAltText("Lead photo: uploaded photo 1 of 2")).toBeInTheDocument();
    expect(screen.getByAltText("Uploaded photo 1 of 2")).toBeInTheDocument();
    const note = screen.getByText(/generative layers were unavailable/i);
    expect(note.textContent).not.toMatch(/photo feedback/i);
  });

  it("shows the best photo beside the verdict", () => {
    const withImages = {
      ...fullReport,
      metadata: { ...fullReport.metadata, session_id: "s1", image_count: 2 },
    } as typeof fullReport;
    const { container } = render(<Dossier report={withImages} listing={null} />);
    const hero = container.querySelector("header") as HTMLElement;
    expect(within(hero).getByAltText("Lead photo: uploaded photo 1 of 2")).toHaveAttribute(
      "src", "/api/predict/s1/images/0",
    );
  });

  it("does not render a lead photo when the model's best index is out of range", () => {
    const outOfRange = {
      ...fullReport,
      metadata: { ...fullReport.metadata, session_id: "s1", image_count: 2 },
      visual: { ...fullReport.visual, best_photo_index: 2 },
    } as typeof fullReport;
    const { container } = render(<Dossier report={outOfRange} listing={null} />);
    const header = container.querySelector("header");
    expect(within(header!).queryByRole("img")).toBeNull();
  });
});
