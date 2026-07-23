"use client";
import type { AdoptionReport } from "@/lib/types";
import { useMeta } from "@/lib/useMeta";
import { buildSpeedClasses } from "@/lib/spectrum";
import { VerdictHero } from "./VerdictHero";
import { Recommendations } from "./Recommendations";
import { RewrittenCopy } from "./RewrittenCopy";
import { PhotoFeedback } from "./PhotoFeedback";
import { PeelBack } from "./PeelBack";

export function Dossier({ report }: { report: AdoptionReport }) {
  const { data: meta } = useMeta();
  const classes = buildSpeedClasses(meta?.adoption_speed_classes ?? []);
  const imageCount = report.metadata.image_count ?? 0;
  if (classes.length === 0) return <main className="mx-auto max-w-3xl px-6 py-24 text-center text-muted">Preparing the dossier...</main>;
  const noGenerative = !report.visual && !report.recommendations && !report.narrative;

  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <VerdictHero report={report} classes={classes} />
      <div className="mt-12 flex flex-col gap-14">
        {report.recommendations && <Recommendations recs={report.recommendations} classes={classes} />}
        {report.optimized_description && <RewrittenCopy text={report.optimized_description} />}
        {imageCount > 0 && (
          <PhotoFeedback
            sessionId={report.metadata.session_id}
            imageCount={imageCount}
            visual={report.visual ?? null}
          />
        )}
        {noGenerative && (
          <p className="rounded-xl border border-ink/10 bg-surface p-6 text-sm text-muted">
            The prediction is ready. The generative layers were unavailable for this run, so photo feedback and
            recommendations are not shown. Set OPENAI_API_KEY on the backend to enable them.
          </p>
        )}
        <PeelBack report={report} classes={classes} />
      </div>
    </main>
  );
}
