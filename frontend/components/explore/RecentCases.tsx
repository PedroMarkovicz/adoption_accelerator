"use client";
import Link from "next/link";
import type { RecentPredictionsResponse } from "@/lib/types-explore";

export function RecentCases({ data }: { data: RecentPredictionsResponse }) {
  if (data.predictions.length === 0) {
    return <p className="text-sm text-muted">No cases yet. Run a prediction to see it here.</p>;
  }
  return (
    <ul className="flex flex-col divide-y divide-ink/10">
      {data.predictions.map((p) => (
        <li key={p.session_id} className="flex items-center justify-between py-3 text-sm">
          <Link href={`/report/${p.session_id}`} className="font-mono text-teal hover:underline">{p.session_id.slice(0, 8)}</Link>
          <span>{p.prediction_label}</span>
          <span className="font-mono text-xs text-muted">{Math.round(p.confidence * 100)}%</span>
        </li>
      ))}
    </ul>
  );
}
