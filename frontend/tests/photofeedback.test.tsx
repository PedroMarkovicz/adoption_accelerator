import { describe, it, expect } from "vitest";
import { render, screen, within } from "@testing-library/react";
import { PhotoFeedback } from "@/components/report/PhotoFeedback";
import type { VisualEvidence } from "@/lib/types";

const visual = {
  photos: [
    {
      image_index: 0,
      quality: { sharpness: 4, lighting: 4, framing: 3, background: 5, issues: [] },
      content: { pet_visible: true, expression: "friendly", setting: "indoor", distinctive_traits: [] },
      appeal_score: 8,
      improvement_suggestions: ["Crop tighter on the face."],
    },
  ],
  overall_visual_appeal: 8,
  best_photo_index: 0,
  observed_traits: [],
  consistency_flags: [],
  photo_strategy_summary: "Lead with the close-up.",
  confidence: "high",
  source: "vlm",
  generated_by: "gpt-5-mini",
  notes: [],
} as unknown as VisualEvidence;

describe("PhotoFeedback", () => {
  it("pairs each photo with its assessment and badges the best one", () => {
    render(<PhotoFeedback sessionId="s1" imageCount={1} visual={visual} />);
    expect(screen.getByAltText("Uploaded photo 1 of 1")).toHaveAttribute(
      "src", "/api/predict/s1/images/0",
    );
    expect(screen.getByText("Appeal 8/10")).toBeInTheDocument();
    expect(screen.getByText("BEST")).toBeInTheDocument();
    expect(screen.getByText("Crop tighter on the face.")).toBeInTheDocument();
  });

  it("renders photos with an honest note when visual analysis is missing", () => {
    render(<PhotoFeedback sessionId="s1" imageCount={2} visual={null} />);
    expect(screen.getByAltText("Uploaded photo 1 of 2")).toBeInTheDocument();
    expect(screen.getByAltText("Uploaded photo 2 of 2")).toBeInTheDocument();
    expect(
      screen.getByText("Visual analysis was unavailable for this run."),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Appeal/)).not.toBeInTheDocument();
  });

  it("renders photos beyond the analyst cap without inventing an assessment", () => {
    render(<PhotoFeedback sessionId="s1" imageCount={3} visual={visual} />);
    expect(screen.getByAltText("Uploaded photo 3 of 3")).toBeInTheDocument();
    expect(screen.getAllByText("No visual assessment for this photo.")).toHaveLength(2);
  });

  it("renders nothing when there are no images", () => {
    const { container } = render(
      <PhotoFeedback sessionId="s1" imageCount={0} visual={null} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("pairs the assessment by image_index, not array position", () => {
    const shiftedVisual = {
      ...visual,
      photos: [
        {
          image_index: 2,
          quality: { sharpness: 3, lighting: 3, framing: 3, background: 3, issues: [] },
          content: { pet_visible: true, expression: "calm", setting: "outdoor", distinctive_traits: [] },
          appeal_score: 6,
          improvement_suggestions: [],
        },
      ],
    } as unknown as VisualEvidence;

    render(<PhotoFeedback sessionId="s1" imageCount={3} visual={shiftedVisual} />);

    const photoThreeCard = screen.getByTestId("photo-card-2");
    expect(within(photoThreeCard).getByText("Appeal 6/10")).toBeInTheDocument();

    const photoOneCard = screen.getByTestId("photo-card-0");
    expect(within(photoOneCard).getByText("No visual assessment for this photo.")).toBeInTheDocument();

    expect(screen.getAllByText("No visual assessment for this photo.")).toHaveLength(2);
  });
});
