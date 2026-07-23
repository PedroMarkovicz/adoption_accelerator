"use client";
import type { VisualEvidence } from "@/lib/types";
import { Card } from "@/components/ui/Card";
import { ReportPhoto } from "./ReportPhoto";

const QUALITY_KEYS = [
  ["sharpness", "Sharpness"],
  ["lighting", "Lighting"],
  ["framing", "Framing"],
  ["background", "Background"],
] as const;

function QualityMeter({ label, value }: { label: string; value: number }) {
  return (
    <div className="grid grid-cols-[5.5rem_1fr_1.25rem] items-center gap-2 text-xs">
      <span className="text-muted">{label}</span>
      <div className="flex gap-1">
        {[1, 2, 3, 4, 5].map((n) => (
          <span
            key={n}
            className={`h-1.5 flex-1 rounded-full ${n <= value ? "bg-teal" : "bg-ink/10"}`}
          />
        ))}
      </div>
      <span className="text-right font-mono text-muted">{value}</span>
    </div>
  );
}

export function PhotoFeedback({ sessionId, imageCount, visual }: {
  sessionId: string;
  imageCount: number;
  visual: VisualEvidence | null;
}) {
  if (imageCount <= 0) return null;

  const photos = visual?.photos ?? [];
  const byIndex = new Map(photos.map((p) => [p.image_index, p]));

  return (
    <section>
      <h2 className="font-[family-name:var(--font-display)] text-2xl">Photo feedback</h2>
      {visual?.photo_strategy_summary && (
        <p className="mt-2 text-muted">{visual.photo_strategy_summary}</p>
      )}
      {!visual && (
        <p className="mt-2 text-muted">Visual analysis was unavailable for this run.</p>
      )}

      <div className="mt-6 flex flex-col gap-4">
        {Array.from({ length: imageCount }, (_, i) => {
          const photo = byIndex.get(i);
          const isBest = visual?.best_photo_index === i;
          return (
            <Card key={i} className="grid gap-4 sm:grid-cols-[10rem_minmax(0,1fr)]">
              <div className="h-40">
                <ReportPhoto sessionId={sessionId} index={i} total={imageCount} />
              </div>
              <div className="min-w-0">
                <div className="flex items-center justify-between gap-2">
                  <p className="font-medium">Photo {i + 1}</p>
                  {isBest && (
                    <span className="rounded-full bg-ink px-2.5 py-0.5 font-mono text-xs text-paper">
                      BEST
                    </span>
                  )}
                </div>
                {photo ? (
                  <>
                    <p className="mt-1 font-mono text-sm">Appeal {photo.appeal_score}/10</p>
                    <div className="mt-3 flex flex-col gap-1.5">
                      {QUALITY_KEYS.map(([key, label]) => (
                        <QualityMeter key={key} label={label} value={photo.quality[key]} />
                      ))}
                    </div>
                    {(photo.improvement_suggestions ?? []).length > 0 && (
                      <ul className="mt-3 list-disc pl-5 text-sm text-muted">
                        {(photo.improvement_suggestions ?? []).map((s, j) => (
                          <li key={j}>{s}</li>
                        ))}
                      </ul>
                    )}
                  </>
                ) : (
                  <p className="mt-1 text-sm text-muted">
                    No visual assessment for this photo.
                  </p>
                )}
              </div>
            </Card>
          );
        })}
      </div>
    </section>
  );
}
