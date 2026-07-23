const BACKEND = process.env.FASTAPI_URL ?? "http://localhost:8000";

export async function GET() {
  const res = await fetch(`${BACKEND}/meta`, { next: { revalidate: 3600 } });
  const text = await res.text();
  return new Response(text, { status: res.status, headers: { "content-type": "application/json" } });
}
