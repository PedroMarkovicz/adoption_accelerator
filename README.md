<div align="center">

# 🐾 Adoption Accelerator

### Multimodal ML + Agentic Generative AI for Pet Adoption Speed Prediction

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agents-green.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST-teal.svg)](https://fastapi.tiangolo.com/)
[![Next.js 16](https://img.shields.io/badge/Next.js-16-black.svg)](https://nextjs.org/)
[![LightGBM](https://img.shields.io/badge/ensemble-LightGBM·XGBoost·CatBoost-yellow.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

**How fast will this pet find a home — and what would make it faster?**

</div>

Adoption Accelerator predicts a pet's adoption speed from **tabular, text, and image** data, then runs a **multi-agent generative layer** that turns the raw model output into a plain-language verdict, *measured* recommendations, and a ready-to-use listing. It closes the loop from prediction to intervention — classical ML for the numbers, an agentic LLM system for the reasoning and the words.

Dataset: [PetFinder.my Adoption Prediction](https://www.kaggle.com/competitions/petfinder-adoption-prediction) (Kaggle).

---

## Why it's interesting

- **Multimodal by design.** A soft-voting ensemble (LightGBM · XGBoost · CatBoost) fuses tabular attributes, 768-dim text embeddings (`all-mpnet-base-v2`), and EfficientNet V2-S image embeddings into one ordinal adoption-speed prediction — with **per-modality SHAP** attribution.
- **Agentic reasoning over ML outputs.** A LangGraph *Evidence Board* of specialized agents reads the photos with a vision LLM, interprets the SHAP drivers, and proposes changes — and every recommendation's impact is **re-measured by the real ensemble, never estimated**.
- **Generative, but grounded.** The synthesizer weaves the modalities into a narrative and an AI-optimized listing, under strict grounding rules: it can only claim traits the vision model actually observed.
- **Config-driven MLOps.** Training, tuning, inference, and the whole agent stack are declared in **YAML**. Swapping an LLM, a timeout, or a model bundle is a config edit, not a code change.
- **Honest by contract.** Every layer degrades gracefully — if a model or the LLM fails, the deterministic prediction still ships and the UI states plainly what was unavailable. It never fabricates.

---

## Architecture

**1 · Multimodal ML pipeline**

```
raw data ─► feature engineering ─► soft-voting ensemble ─► ordinal probabilities ─► SHAP
            (tabular · text emb ·    (LightGBM/XGBoost/       (5 classes, QWK-       (per-modality
             image emb · vision       CatBoost, Optuna-        thresholded)           attribution)
             + NLP metadata)          tuned, 5-fold CV)
```

**2 · Evidence Board — LangGraph multi-agent system**

```
orchestrator
 └─ inference ................ deterministic ML core: prediction + SHAP (always present)
     ├─ visual_analyst ....... vision LLM reads the uploaded photos
     └─ data_analyst ......... interprets the SHAP drivers into plain language
         └─ recommendation_agent .. proposes changes, RE-MEASURES each on the real ensemble
             └─ synthesizer ....... grounded narrative + AI-optimized listing description
                 └─ aggregator .... assembles the final report
```

The deterministic prediction is always present; every generative layer is optional, so any failure path still returns a valid report. Runs are traced with Langfuse.

**3 · Interface — "The Dossier" (Next.js)**

A Next.js 16 App Router frontend (BFF pattern, OpenAPI-typed end to end) presents the verdict on a Speed Spectrum, pairs each uploaded photo with its visual assessment, and exports a self-contained listing artifact — a finished ad plus an evidence brief — as a downloadable HTML/PDF.

---

## Config-driven design

Model and agent organization lives in declarative YAML under `configs/` — the MLOps backbone:

| File | Governs |
|---|---|
| `configs/training/{baseline,tuned}.yaml` | Cross-validation, Optuna tuning budget, reproducibility seeds, validation gates |
| `configs/inference/serving.yaml` | Model bundle, latency targets, class metadata, diagnostics |
| `configs/agents/models.yaml` | LLM catalog (pricing, vision support), defaults, and **per-role** model assignment |
| `configs/agents/timeouts.yaml` | Per-node timeout budgets |

> `models.yaml` is a catalog + role map: assigning a vision-capable model to the visual analyst, or escalating the synthesizer to a stronger model, is one line of YAML — no code change.

---

## Tech stack

- **ML:** LightGBM · XGBoost · CatBoost · scikit-learn · SHAP · Optuna · sentence-transformers · EfficientNet V2-S
- **Agents:** LangGraph · LangChain · OpenAI (GPT-5 family) · Langfuse
- **Backend:** FastAPI · Pydantic
- **Frontend:** Next.js 16 · React 19 · TypeScript · TanStack Query · Zod
- **Tooling:** uv · pytest · Vitest · Playwright

---

## Quickstart

Two processes — full setup in [`frontend/README.md`](frontend/README.md):

```bash
# 1. Backend (repo root) — loads the ML model and compiles the agent graph
uvicorn app.api.main:app --port 8000

# 2. Frontend
cd frontend && npm install && npm run dev     # http://localhost:3000
```

An `OPENAI_API_KEY` in `.env` (repo root) enables the generative layers. Without it, the deterministic ML prediction still renders — the app degrades honestly.

---

## Project structure

```
src/         ML + agent packages (features, inference, agents/)
app/api/     FastAPI service — clean REST over the agent graph
frontend/    Next.js "Dossier" interface
configs/     YAML config for training, inference, and agents
artifacts/   Trained model bundles (tuned_v1) + precomputed explore data
notebooks/   Data → features → modeling → tuning → inference
tests/       pytest (backend/agents); frontend/ carries Vitest + Playwright
```

## License

MIT — see [LICENSE](LICENSE).
