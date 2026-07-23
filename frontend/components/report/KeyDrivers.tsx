"use client";
import type { PredictionEvidence } from "@/lib/types";

export function KeyDrivers({ prediction }: { prediction: PredictionEvidence }) {
  const drivers = [...(prediction.key_drivers ?? [])].sort((a, b) => b.shap_magnitude - a.shap_magnitude);
  const max = Math.max(...drivers.map((d) => d.shap_magnitude), 0.0001);
  return (
    <ul className="flex flex-col gap-3">
      {drivers.map((d, i) => (
        <li key={i} className="grid grid-cols-[1fr_auto] items-center gap-3">
          <div>
            <p className="text-sm font-medium">{d.display_name || d.feature}</p>
            {d.reading && <p className="text-xs text-muted">{d.reading}</p>}
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-32 rounded-full bg-ink/10">
              <div className="h-full rounded-full"
                style={{ width: `${(d.shap_magnitude / max) * 100}%`, background: d.direction === "positive" ? "var(--spectrum-0)" : "var(--spectrum-4)" }} />
            </div>
            <span className="w-16 text-right font-mono text-xs">{d.shap_magnitude.toFixed(3)}</span>
          </div>
        </li>
      ))}
    </ul>
  );
}
