import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Wizard } from "@/components/wizard/Wizard";

vi.mock("@/lib/useMeta", () => ({
  useMeta: () => ({
    data: {
      breeds: [{ id: 307, type: 1, name: "Mixed Breed" }, { id: 1, type: 1, name: "Affenpinscher" }],
      colors: [{ id: 1, name: "Black" }],
      states: [{ id: 41336, name: "Johor" }],
      maturity_sizes: [{ id: 1, label: "Small" }, { id: 2, label: "Medium" }],
      fur_lengths: [{ id: 1, label: "Short" }],
      adoption_speed_classes: [],
      model_version: "tuned_v1", modality_breakdown: {},
    },
    isLoading: false,
  }),
}));

function renderWizard() {
  const client = new QueryClient();
  return render(<QueryClientProvider client={client}><Wizard /></QueryClientProvider>);
}

describe("Wizard", () => {
  it("starts on The basics and advances to Appearance", async () => {
    renderWizard();
    expect(screen.getByText("The basics")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /continue/i }));
    expect(await screen.findByText("Appearance")).toBeInTheDocument();
  });
});
