const BACKEND = process.env.FASTAPI_URL ?? "http://localhost:8000";

export async function GET(_req: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  const res = await fetch(`${BACKEND}/predict/${id}/status`, { cache: "no-store" });
  const text = await res.text();
  return new Response(text, { status: res.status, headers: { "content-type": "application/json" } });
}
