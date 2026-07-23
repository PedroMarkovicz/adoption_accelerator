"use client";
import type { AdoptionReport } from "@/lib/types";

export function AgentTrace({ report }: { report: AdoptionReport }) {
  const m = report.metadata as { timing_ms?: Record<string, number>; cost_usd?: number };
  const iters = report.recommendations?.iterations_used ?? 0;
  const totalMs = m.timing_ms ? Object.values(m.timing_ms).reduce((a, b) => a + b, 0) : 0;
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 font-mono text-sm">
      <dt className="text-muted">Pipeline</dt><dd>inference - visual - data - recommend - synthesize</dd>
      <dt className="text-muted">Agent iterations</dt><dd>{iters}</dd>
      <dt className="text-muted">Total time</dt><dd>{Math.round(totalMs)} ms</dd>
      <dt className="text-muted">Est. cost</dt><dd>${(m.cost_usd ?? 0).toFixed(4)}</dd>
    </dl>
  );
}
