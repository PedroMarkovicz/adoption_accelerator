"use client";
import type { PerformanceResponse } from "@/lib/types-explore";
import { Card } from "@/components/ui/Card";

export function PerformancePanel({ data }: { data: PerformanceResponse }) {
  return (
    <div className="grid gap-4 md:grid-cols-2">
      {Object.entries(data.aggregate_metrics).map(([k, v]) => (
        <Card key={k}>
          <p className="font-mono text-xs uppercase tracking-widest text-muted">{k}</p>
          <p className="mt-1 font-[family-name:var(--font-display)] text-3xl">{typeof v === "number" ? v.toFixed(3) : v}</p>
        </Card>
      ))}
    </div>
  );
}
