"use client";
import { useQuery } from "@tanstack/react-query";
import { api } from "./api";

export function useReportStatus(id: string) {
  const q = useQuery({
    queryKey: ["report", id],
    queryFn: () => api.getStatus(id),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      if (s === "done" || s === "error") return false;
      if (query.state.status === "error") return false;
      return 1500;
    },
  });
  return {
    status: q.data?.status ?? (q.isError ? "error" : "running"),
    report: q.data?.report ?? null,
    error: q.data?.error ?? (q.isError ? (q.error instanceof Error ? q.error.message : "Could not reach the server.") : null),
    isPending: q.isPending,
  };
}
