"use client";
import type { PredictionEvidence } from "@/lib/types";
export function Uncertainty({ prediction }: { prediction: PredictionEvidence }) {
  return <p className="text-sm text-muted">{prediction.uncertainty_reading || "No uncertainty note available."}</p>;
}
