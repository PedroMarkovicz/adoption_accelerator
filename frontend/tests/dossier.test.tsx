import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
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
    render(<Dossier report={fullReport} />);
    expect(screen.getByText("Likely adopted within a week")).toBeInTheDocument();
    expect(screen.getByText(/Add a brighter lead photo/)).toBeInTheDocument();
    expect(screen.getByText(/Meet a friendly, healthy companion/)).toBeInTheDocument();
  });

  it("degrades gracefully when generative layers are missing", () => {
    render(<Dossier report={degradedReport} />);
    expect(screen.getByText(/generative layers were unavailable/i)).toBeInTheDocument();
  });

  it("still shows uploaded photos when generative layers fail, without contradicting itself", () => {
    render(<Dossier report={degradedReportWithPhotos} />);
    expect(screen.getByAltText(/Uploaded photo 1 of 2/)).toBeInTheDocument();
    const note = screen.getByText(/generative layers were unavailable/i);
    expect(note.textContent).not.toMatch(/photo feedback/i);
  });
});
