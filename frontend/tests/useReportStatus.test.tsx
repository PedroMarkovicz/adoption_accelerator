import { describe, it, expect, vi } from "vitest";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useReportStatus } from "@/lib/useReportStatus";
import * as apiMod from "@/lib/api";

function wrap() {
  const client = new QueryClient();
  return ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

function wrapNoRetry() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return ({ children }: { children: React.ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useReportStatus", () => {
  it("returns done with the report once status resolves", async () => {
    vi.spyOn(apiMod.api, "getStatus").mockResolvedValue({
      session_id: "abc", status: "done",
      report: { prediction: { predicted_class: 1 } } as never,
    } as never);
    const { result } = renderHook(() => useReportStatus("abc"), { wrapper: wrap() });
    await waitFor(() => expect(result.current.status).toBe("done"));
    expect(result.current.report).toBeTruthy();
  });

  it("returns error status when the fetch itself fails", async () => {
    vi.spyOn(apiMod.api, "getStatus").mockRejectedValue(new Error("boom"));
    const { result } = renderHook(() => useReportStatus("x"), { wrapper: wrapNoRetry() });
    await waitFor(() => expect(result.current.status).toBe("error"));
  });
});
