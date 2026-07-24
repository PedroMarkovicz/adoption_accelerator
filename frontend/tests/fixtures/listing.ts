import type { ListingInput } from "@/lib/listing/buildListingHtml";
import type { AdoptionReport, PetProfileRequest } from "@/lib/types";

export const report = {
  prediction: {
    predicted_class: 1,
    prediction_label: "Adopted within a week",
    probabilities: { "0": 0.1, "1": 0.6, "2": 0.2, "3": 0.05, "4": 0.05 },
    class_confidence: 0.6,
  },
  visual: { best_photo_index: 1 },
  recommendations: null,
  optimized_description: "Milo greets every visitor at the gate.",
  headline: "Likely adopted within a week",
  metadata: { session_id: "s1", ml_model_version: "tuned_v1", timestamp: "2026-07-23T10:00:00Z", image_count: 2 },
} as unknown as AdoptionReport;

export const profile = {
  pet_type: "Dog",
  name: "Milo",
  age_months: 8,
  gender: "Male",
  breed1: 307,
  maturity_size: 2,
  fur_length: 1,
  vaccinated: "Yes",
  dewormed: "Yes",
  sterilized: "No",
  health: "Healthy",
  fee: 50,
  description: "dog for adoption pls contact",
} as unknown as PetProfileRequest;

export const labels = {
  title: "Meet Milo",
  species: "Dog",
  breed: "Mixed Breed",
  colors: "Black",
  age: "8 months",
  gender: "Male",
  size: "Medium",
  fur: "Short",
  state: "Selangor",
  fee: "RM 50",
  health: [{ label: "Vaccinated", value: "Yes" }],
};

export const speedClasses = [
  { index: 0, label: "Same day", color: "#1FA363" },
  { index: 1, label: "Within a week", color: "#7DB33A" },
  { index: 2, label: "Within a month", color: "#E8B23A" },
  { index: 3, label: "Within three months", color: "#E77A3C" },
  { index: 4, label: "Not adopted", color: "#D14D5A" },
];

export const recommendations = {
  recommendations: [
    { action: "Add a second clear photo", priority: 1,
      measured_impact: { expected_speedup: "one class faster" } },
    { action: "Sterilize before listing", priority: 2,
      measured_impact: { expected_speedup: "+12% same-week" } },
    { action: "Lower the fee to RM 20", priority: 3,
      measured_impact: { expected_speedup: "+7% same-week" } },
    { action: "Rewrite the title", priority: 4,
      measured_impact: { expected_speedup: "+2% same-week" } },
  ],
};

export function input(over: Partial<ListingInput> = {}): ListingInput {
  return {
    report,
    profile,
    labels,
    images: [
      { index: 0, dataUri: "data:image/jpeg;base64,AAA" },
      { index: 1, dataUri: "data:image/jpeg;base64,BBB" },
    ],
    fonts: null,
    classes: speedClasses,
    ...over,
  } as ListingInput;
}
