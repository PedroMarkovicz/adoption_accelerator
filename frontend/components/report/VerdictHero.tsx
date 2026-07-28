"use client";
import type { AdoptionReport } from "@/lib/types";
import type { SpeedClass } from "@/lib/spectrum";
import { SpeedSpectrum } from "./SpeedSpectrum";
import { ReportPhoto } from "./ReportPhoto";

export function VerdictHero({ report, classes }: { report: AdoptionReport; classes: SpeedClass[] }) {
  const p = report.prediction;
  const imageCount = report.metadata.image_count ?? 0;
  const leadIndex = report.visual?.best_photo_index ?? 0;
  const showPhoto = imageCount > 0 && leadIndex >= 0 && leadIndex < imageCount;

  return (
    <header className="border-b border-ink/10 pb-10">
      <div className="flex flex-col gap-6 sm:flex-row sm:items-start">
        {showPhoto && (
          <div className="h-44 w-44 shrink-0">
            <ReportPhoto
              sessionId={report.metadata.session_id}
              index={leadIndex}
              total={imageCount}
              label={`Lead photo: uploaded photo ${leadIndex + 1} of ${imageCount}`}
            />
          </div>
        )}
        <div className="min-w-0">
          <p className="font-mono text-xs uppercase tracking-widest text-muted">The verdict</p>
          <h1 className="mt-2 font-[family-name:var(--font-display)] text-4xl leading-tight md:text-5xl">
            {report.headline || p.prediction_label}
          </h1>
          {report.narrative && <p className="mt-4 max-w-2xl text-lg text-muted">{report.narrative}</p>}
        </div>
      </div>
      <div className="mt-8">
        <SpeedSpectrum classes={classes} markerClass={p.predicted_class} probabilities={p.probabilities} confidence={p.class_confidence} />
      </div>
    </header>
  );
}
