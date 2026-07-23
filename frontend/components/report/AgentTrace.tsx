"use client";
import type { AdoptionReport } from "@/lib/types";

export function AgentTrace({ report }: { report: AdoptionReport }) {
  const timingMs = report.metadata.timing_ms ?? {};
  const cost = report.metadata.estimated_cost_usd ?? 0;
  const iters = report.recommendations?.iterations_used ?? 0;
  const totalMs = Object.values(timingMs).reduce((a, b) => a + b, 0);
  return (
    <dl className="grid grid-cols-2 gap-x-6 gap-y-2 font-mono text-sm">
      <dt className="text-muted">Pipeline</dt><dd>inference - visual - data - recommend - synthesize</dd>
      <dt className="text-muted">Agent iterations</dt><dd>{iters}</dd>
      <dt className="text-muted">Total time</dt><dd>{Math.round(totalMs)} ms</dd>
      <dt className="text-muted">Est. cost</dt><dd>${cost.toFixed(4)}</dd>
    </dl>
  );
}
