"use client";
import type { PredictionEvidence } from "@/lib/types";
import type { SpeedClass } from "@/lib/spectrum";

export function Probabilities({ prediction, classes }: { prediction: PredictionEvidence; classes: SpeedClass[] }) {
  return (
    <ul className="flex flex-col gap-2">
      {classes.map((c) => {
        const p = prediction.probabilities[String(c.index)] ?? 0;
        return (
          <li key={c.index} className="grid grid-cols-[10rem_1fr_3rem] items-center gap-3 text-sm">
            <span className="truncate text-muted">{c.label}</span>
            <div className="h-2 rounded-full bg-ink/10">
              <div className="h-full rounded-full" style={{ width: `${p * 100}%`, background: c.color }} />
            </div>
            <span className="text-right font-mono text-xs">{Math.round(p * 100)}%</span>
          </li>
        );
      })}
    </ul>
  );
}
