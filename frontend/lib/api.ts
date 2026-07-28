import type { MetaResponse, ReportStatusResponse } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json())?.detail ?? detail; } catch {}
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getMeta: () => fetch("/api/meta", { cache: "no-store" }).then(json<MetaResponse>),
  getStatus: (id: string) =>
    fetch(`/api/predict/${id}/status`, { cache: "no-store" }).then(json<ReportStatusResponse>),
  createPrediction: (form: FormData) =>
    fetch("/api/predict", { method: "POST", body: form }).then(json<ReportStatusResponse>),
};
