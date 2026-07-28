const BACKEND = process.env.FASTAPI_URL ?? "http://localhost:8000";
export async function GET(req: Request) {
  const search = new URL(req.url).search;
  const res = await fetch(`${BACKEND}/predictions/recent${search}`, { cache: "no-store" });
  return new Response(await res.text(), { status: res.status, headers: { "content-type": "application/json" } });
}
