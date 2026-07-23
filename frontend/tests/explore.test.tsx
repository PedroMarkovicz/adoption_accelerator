import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { GlobalImportance } from "@/components/explore/GlobalImportance";

describe("GlobalImportance", () => {
  it("renders feature rows", () => {
    render(<GlobalImportance items={[
      { rank: 1, feature: "photo_appeal", display_name: "Photo appeal", mean_abs_shap: 0.3 },
      { rank: 2, feature: "age", display_name: "Age", mean_abs_shap: 0.2 },
    ]} />);
    expect(screen.getByText("Photo appeal")).toBeInTheDocument();
    expect(screen.getByText("Age")).toBeInTheDocument();
  });
});
