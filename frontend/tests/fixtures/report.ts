import type { AdoptionReport } from "@/lib/types";

export const fullReport = {
  prediction: {
    predicted_class: 1, prediction_label: "Adopted within 1 week",
    probabilities: { "0": 0.1, "1": 0.5, "2": 0.25, "3": 0.1, "4": 0.05 },
    class_confidence: 0.5,
    modality_contributions: { tabular: 0.5, text: 0.3, image: 0.2 },
    modality_available: { tabular: true, text: true, image: true },
    key_drivers: [
      { feature: "photo_appeal", display_name: "Photo appeal", value: "high", direction: "positive", shap_magnitude: 0.21, modality: "image", reading: "Strong lead photo." },
    ],
    uncertainty_reading: "Confident, but the 1-month class is plausible.",
    confidence: "high", source: "ensemble", generated_by: "deterministic", notes: [],
  },
  visual: {
    photos: [{ image_index: 0, quality: { sharpness: 4, lighting: 4, framing: 3, background: 4, issues: [] },
      content: { pet_visible: true, expression: "friendly", setting: "indoor", distinctive_traits: [] },
      appeal_score: 8, improvement_suggestions: ["Crop tighter on the face."] }],
    overall_visual_appeal: 8, best_photo_index: 0, observed_traits: ["fluffy"], consistency_flags: [],
    photo_strategy_summary: "Lead with the close-up.", confidence: "high", source: "vlm", generated_by: "gpt-5-mini", notes: [],
  },
  recommendations: {
    recommendations: [{ action: "Add a brighter lead photo", feature: "photo", current_value: "dim", suggested_value: "bright",
      measured_impact: { class_before: 2, class_after: 1, probability_shift: { "1": 0.12 }, expected_speedup: "About a week sooner" },
      priority: 1, category: "photo", rationale: "Brighter photos correlate with faster adoption." }],
    rejected_hypotheses: [], iterations_used: 3, confidence: "high", source: "react-agent", generated_by: "gpt-5-mini", notes: [],
  },
  narrative: "This pet is likely to be adopted within a week.",
  optimized_description: "Meet a friendly, healthy companion ready for a loving home.",
  headline: "Likely adopted within a week",
  metadata: { session_id: "abc", ml_model_version: "tuned_v1", timing_ms: { total: 4200 }, cost_usd: 0.004 },
} as unknown as AdoptionReport;

export const degradedReport = {
  prediction: fullReport.prediction,
  visual: null, recommendations: null, narrative: "", optimized_description: null, headline: "Prediction ready",
  metadata: { session_id: "def", ml_model_version: "tuned_v1", timing_ms: {}, cost_usd: 0 },
} as unknown as AdoptionReport;
