import { describe, it, expect, vi, beforeEach } from "vitest";
import { api } from "@/lib/api";

beforeEach(() => { vi.restoreAllMocks(); });

describe("api client", () => {
  it("getStatus calls the BFF status route and returns JSON", async () => {
    const payload = { session_id: "abc", status: "done" };
    vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(payload), { status: 200 })));
    const res = await api.getStatus("abc");
    expect(res.status).toBe("done");
    expect(fetch).toHaveBeenCalledWith("/api/predict/abc/status", expect.objectContaining({ cache: "no-store" }));
  });

  it("createPrediction posts FormData to the BFF predict route", async () => {
    const payload = { session_id: "xyz", status: "running" };
    const fetchMock = vi.fn(async () => new Response(JSON.stringify(payload), { status: 202 }));
    vi.stubGlobal("fetch", fetchMock);
    const fd = new FormData();
    fd.append("profile", "{}");
    const res = await api.createPrediction(fd);
    expect(res.session_id).toBe("xyz");
    expect(fetchMock).toHaveBeenCalledWith("/api/predict", expect.objectContaining({ method: "POST", body: fd }));
  });
});
