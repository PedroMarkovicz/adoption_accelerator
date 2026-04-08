"""
FastAPI application factory for the Adoption Accelerator API.

Startup (lifespan):
  - Loads and warms the InferencePipeline singleton via get_inference_pipeline().
  - Compiles both the deterministic and full LangGraph agent graphs.
  - Starts the job store TTL cleanup background thread.
  - All are stored in app.state for use by routers.

CORS is configured to allow the Streamlit default origin (localhost:8501).

Usage:
  uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agents.graph import compile_deterministic_graph, compile_full_graph
from adoption_accelerator.inference.serving import get_inference_pipeline
from app.api.middleware.error_handler import register_error_handlers
from app.api.routers import explore, health, predict, predictions
from app.api.services.job_store import job_store

logger = logging.getLogger(__name__)

# Artifact paths
_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPLORE_DIR = _ROOT / "artifacts" / "explore"
_REPORTS_DIR = _ROOT / "reports"
_MODEL_DIR = _ROOT / "artifacts" / "models" / "tuned_v1"


def _load_json(path: Path) -> dict | list | None:
    """Load a JSON file, returning None if missing or malformed."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        logger.warning("JSON file not found: %s", path)
    except Exception as exc:
        logger.warning("Failed to load %s: %s", path, exc)
    return None


def _load_model_meta() -> dict:
    """Load model metadata from config.yaml."""
    config_path = _MODEL_DIR / "config.yaml"
    try:
        import yaml
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            return {
                "model_name": cfg.get("model_name", "SoftVotingEnsemble"),
                "model_version": cfg.get("feature_version", "tuned_v1"),
                "model_family": cfg.get("model_family", "ensemble"),
                "base_models": list(cfg.get("params", {}).keys()),
                "training_qwk": cfg.get("selected_qwk", 0.0),
            }
    except ImportError:
        logger.warning("PyYAML not installed; model metadata unavailable.")
    except Exception as exc:
        logger.warning("Failed to load model config: %s", exc)
    return {}


def _compute_modality_breakdown(pipeline) -> dict[str, int]:
    """Count features per modality from the feature schema."""
    if pipeline is None:
        return {}
    try:
        features = pipeline.feature_schema.get("features", [])
    except Exception:
        return {}

    counts = {"tabular": 0, "text": 0, "image": 0, "metadata": 0}
    for f in features:
        if f.startswith("text_emb_"):
            counts["text"] += 1
        elif f.startswith("img_emb_"):
            counts["image"] += 1
        elif f in (
            "doc_sentiment_score", "doc_sentiment_magnitude",
            "sentiment_variance", "sentence_count_sentiment",
            "entity_count", "entity_type_count",
            "description_length", "word_count", "n_sentences",
            "mean_word_length", "avg_word_length", "uppercase_ratio",
            "sentence_count",
        ):
            counts["text"] += 1
        elif f in (
            "mean_crop_confidence", "mean_blur_score", "mean_image_brightness",
        ):
            counts["image"] += 1
        else:
            counts["tabular"] += 1
    return counts


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML pipeline, compile graphs, and start cleanup on startup."""
    logger.info("Startup: loading InferencePipeline ...")
    app.state.pipeline = get_inference_pipeline()

    logger.info("Startup: compiling deterministic graph ...")
    app.state.deterministic_graph = compile_deterministic_graph()

    logger.info("Startup: compiling full agent graph ...")
    app.state.graph = compile_full_graph()

    logger.info("Startup: starting job store cleanup thread ...")
    job_store.start_cleanup_loop()

    # Load precomputed explore data (coerce None -> safe defaults so
    # getattr(..., default) in routers is never bypassed by a None attribute)
    logger.info("Startup: loading explore data ...")
    app.state.distributions = _load_json(_EXPLORE_DIR / "distributions.json") or {}
    app.state.adoption_patterns = _load_json(_EXPLORE_DIR / "adoption_patterns.json") or {}
    app.state.performance = _load_json(_EXPLORE_DIR / "performance.json") or {}

    # Global importance
    gi_data = _load_json(_REPORTS_DIR / "global_importance.json")
    app.state.global_importance = gi_data.get("global_top_k", []) if gi_data else []

    # Display names for features
    try:
        from adoption_accelerator.features.display_names import USER_DISPLAY_NAMES
        app.state.display_names = USER_DISPLAY_NAMES
    except ImportError:
        app.state.display_names = {}

    # Model metadata from config.yaml
    app.state.model_meta = _load_model_meta()
    app.state.modality_breakdown = _compute_modality_breakdown(app.state.pipeline)

    logger.info("Startup complete. API ready.")
    try:
        yield
    finally:
        logger.info("Shutdown: stopping job store cleanup ...")
        job_store.stop_cleanup_loop()
        logger.info("Shutdown: releasing resources.")
        app.state.pipeline = None
        app.state.deterministic_graph = None
        app.state.graph = None
        app.state.distributions = None
        app.state.adoption_patterns = None
        app.state.performance = None
        app.state.global_importance = None


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    application = FastAPI(
        title="Adoption Accelerator API",
        description=(
            "ML prediction and multi-agent explanation for pet adoption speed. "
            "Exposes the LangGraph pipeline behind a clean REST interface."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_error_handlers(application)

    application.include_router(health.router)
    application.include_router(predict.router)
    application.include_router(explore.router)
    application.include_router(predictions.router)

    return application


app = create_app()
