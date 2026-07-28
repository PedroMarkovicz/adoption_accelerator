"use client";
import { useQuery } from "@tanstack/react-query";
import type { PerformanceResponse, RecentPredictionsResponse } from "./types-explore";

export function usePerformance() {
  return useQuery({
    queryKey: ["explore", "performance"],
    queryFn: () => fetch("/api/explore/performance", { cache: "no-store" }).then((r) => r.json() as Promise<PerformanceResponse>),
  });
}
export function useRecent() {
  return useQuery({
    queryKey: ["recent"],
    queryFn: () => fetch("/api/predictions/recent", { cache: "no-store" }).then((r) => r.json() as Promise<RecentPredictionsResponse>),
    refetchInterval: 10000,
  });
}
