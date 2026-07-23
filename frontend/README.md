# Adoption Accelerator — Frontend

The web frontend for the Adoption Accelerator. It talks to the FastAPI backend through a Next.js BFF (Backend-for-Frontend), so the browser never calls the Python API directly and there is no CORS to configure.

## Prerequisites

- Node 20+ and npm
- The Python backend, runnable from the repo root (see the root `README.md` / `pyproject.toml` for its own setup)

## Running locally (two processes)

**1. Backend** — from the repo root:

```bash
uvicorn app.api.main:app --port 8000
```

This loads the ML model on startup; wait until `http://localhost:8000/health` returns `200` before starting the frontend.

**2. Frontend** — from `frontend/`:

```bash
npm install
cp .env.local.example .env.local   # sets FASTAPI_URL=http://localhost:8000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Notes

- `npm run gen:types` regenerates `lib/openapi.ts` from the backend's OpenAPI schema. It requires the backend to be running on `:8000`.
- Set `OPENAI_API_KEY` in the repo-root `.env` to enable the full multimodal/agentic path. Without it, the deterministic prediction (the verdict) still renders normally, and the generative sections show an "unavailable" note instead.

## Commands

```bash
npm run test        # unit tests (Vitest + Testing Library)
npm run build        # production build
npx playwright test  # end-to-end tests (needs the backend running on :8000)
```
