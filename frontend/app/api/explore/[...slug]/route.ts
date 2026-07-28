const BACKEND = process.env.FASTAPI_URL ?? "http://localhost:8000";

export async function GET(req: Request, { params }: { params: Promise<{ slug: string[] }> }) {
  const { slug } = await params;
  const search = new URL(req.url).search;
  const res = await fetch(`${BACKEND}/explore/${slug.join("/")}${search}`, { cache: "no-store" });
  const text = await res.text();
  return new Response(text, { status: res.status, headers: { "content-type": "application/json" } });
}
