"use client";
import type { VisualEvidence } from "@/lib/types";
import { Card } from "@/components/ui/Card";

export function PhotoFeedback({ visual }: { visual: VisualEvidence }) {
  return (
    <section>
      <h2 className="font-[family-name:var(--font-display)] text-2xl">Photo feedback</h2>
      {visual.photo_strategy_summary && <p className="mt-2 text-muted">{visual.photo_strategy_summary}</p>}
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {(visual.photos ?? []).map((ph) => (
          <Card key={ph.image_index}>
            <div className="flex items-center justify-between">
              <p className="font-medium">Photo {ph.image_index + 1}{visual.best_photo_index === ph.image_index ? " - best" : ""}</p>
              <span className="font-mono text-sm">appeal {ph.appeal_score}/10</span>
            </div>
            {(ph.improvement_suggestions ?? []).length > 0 && (
              <ul className="mt-3 list-disc pl-5 text-sm text-muted">
                {(ph.improvement_suggestions ?? []).map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            )}
          </Card>
        ))}
      </div>
    </section>
  );
}
