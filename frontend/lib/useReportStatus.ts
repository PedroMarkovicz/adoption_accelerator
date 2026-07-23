"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

export function useReportStatus(id: string) {
  const q = useQuery({
    queryKey: ["report", id],
    queryFn: () => api.getStatus(id),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "done" || s === "error" ? false : 1500;
    },
  });
  return {
    status: q.data?.status ?? "running",
    report: q.data?.report ?? null,
    error: q.data?.error ?? (q.isError ? "Could not reach the server." : null),
    isPending: q.isPending,
  };
}
