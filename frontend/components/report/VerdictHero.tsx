"use client";
import type { AdoptionReport } from "@/lib/types";
import type { SpeedClass } from "@/lib/spectrum";
import { SpeedSpectrum } from "./SpeedSpectrum";

export function VerdictHero({ report, classes }: { report: AdoptionReport; classes: SpeedClass[] }) {
  const p = report.prediction;
  return (
    <header className="border-b border-ink/10 pb-10">
      <p className="font-mono text-xs uppercase tracking-widest text-muted">The verdict</p>
      <h1 className="mt-2 font-[family-name:var(--font-display)] text-4xl leading-tight md:text-5xl">
        {report.headline || p.prediction_label}
      </h1>
      {report.narrative && <p className="mt-4 max-w-2xl text-lg text-muted">{report.narrative}</p>}
      <div className="mt-8">
        <SpeedSpectrum classes={classes} markerClass={p.predicted_class} probabilities={p.probabilities} confidence={p.class_confidence} />
      </div>
    </header>
  );
}
