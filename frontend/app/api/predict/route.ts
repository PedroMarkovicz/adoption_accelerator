const BACKEND = process.env.FASTAPI_URL ?? "http://localhost:8000";

export async function POST(req: Request) {
  const form = await req.formData();
  const res = await fetch(`${BACKEND}/predict`, { method: "POST", body: form });
  const text = await res.text();
  return new Response(text, { status: res.status, headers: { "content-type": "application/json" } });
}
