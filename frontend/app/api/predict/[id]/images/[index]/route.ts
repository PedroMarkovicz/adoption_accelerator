const BACKEND = process.env.FASTAPI_URL ?? "http://localhost:8000";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string; index: string }> },
) {
  const { id, index } = await params;
  const res = await fetch(`${BACKEND}/predict/${id}/images/${index}`, {
    cache: "no-store",
  });

  if (!res.ok) {
    return new Response(null, { status: res.status });
  }

  const body = await res.arrayBuffer();
  return new Response(body, {
    status: 200,
    headers: {
      "content-type": res.headers.get("content-type") ?? "application/octet-stream",
      "cache-control": "private, max-age=3600",
    },
  });
}
