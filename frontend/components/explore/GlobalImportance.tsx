"use client";
import type { GlobalFeatureImportance } from "@/lib/types-explore";

export function GlobalImportance({ items }: { items: GlobalFeatureImportance[] }) {
  const max = Math.max(...items.map((i) => i.mean_abs_shap), 0.0001);
  return (
    <ul className="flex flex-col gap-2">
      {items.map((it) => (
        <li key={it.rank} className="grid grid-cols-[12rem_1fr_3.5rem] items-center gap-3 text-sm">
          <span className="truncate">{it.display_name || it.feature}</span>
          <div className="h-2 rounded-full bg-ink/10">
            <div className="h-full rounded-full bg-teal" style={{ width: `${(it.mean_abs_shap / max) * 100}%` }} />
          </div>
          <span className="text-right font-mono text-xs">{it.mean_abs_shap.toFixed(3)}</span>
        </li>
      ))}
    </ul>
  );
}
