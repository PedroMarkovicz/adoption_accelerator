"use client";
import { usePerformance, useRecent } from "@/lib/useExplore";
import { GlobalImportance } from "@/components/explore/GlobalImportance";
import { PerformancePanel } from "@/components/explore/PerformancePanel";
import { RecentCases } from "@/components/explore/RecentCases";

export default function ExplorePage() {
  const perf = usePerformance();
  const recent = useRecent();
  return (
    <main className="mx-auto max-w-3xl px-6 py-12">
      <h1 className="font-[family-name:var(--font-display)] text-4xl">Inside the model</h1>
      <section className="mt-10">
        <h2 className="font-[family-name:var(--font-display)] text-2xl">Performance</h2>
        <div className="mt-6">{perf.data ? <PerformancePanel data={perf.data} /> : <p className="text-muted">Loading...</p>}</div>
      </section>
      <section className="mt-12">
        <h2 className="font-[family-name:var(--font-display)] text-2xl">What drives adoption speed</h2>
        <div className="mt-6">{perf.data ? <GlobalImportance items={perf.data.global_importance} /> : null}</div>
      </section>
      <section className="mt-12">
        <h2 className="font-[family-name:var(--font-display)] text-2xl">Recent cases</h2>
        <div className="mt-6">{recent.data ? <RecentCases data={recent.data} /> : null}</div>
      </section>
    </main>
  );
}
