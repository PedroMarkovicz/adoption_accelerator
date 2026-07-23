"use client";
import type { RecommendationEvidence } from "@/lib/types";
import type { SpeedClass } from "@/lib/spectrum";
import { Card } from "@/components/ui/Card";

export function Recommendations({ recs, classes }: { recs: RecommendationEvidence; classes: SpeedClass[] }) {
  const items = [...(recs.recommendations ?? [])].sort((a, b) => a.priority - b.priority);
  return (
    <section>
      <h2 className="font-[family-name:var(--font-display)] text-2xl">Improve this listing</h2>
      <div className="mt-6 flex flex-col gap-4">
        {items.map((r, i) => {
          const before = classes[r.measured_impact.class_before]?.label ?? String(r.measured_impact.class_before);
          const after = classes[r.measured_impact.class_after]?.label ?? String(r.measured_impact.class_after);
          return (
            <Card key={i}>
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="font-medium">{r.action}</p>
                  {r.rationale && <p className="mt-1 text-sm text-muted">{r.rationale}</p>}
                </div>
                <span className="rounded-full bg-ink/5 px-3 py-1 font-mono text-xs">{r.category}</span>
              </div>
              <p className="mt-4 font-mono text-sm">
                <span className="text-muted">{before}</span>
                <span className="mx-2 text-teal">-&gt;</span>
                <span className="font-medium">{after}</span>
                <span className="ml-2 text-muted">({r.measured_impact.expected_speedup})</span>
              </p>
            </Card>
          );
        })}
      </div>
    </section>
  );
}
