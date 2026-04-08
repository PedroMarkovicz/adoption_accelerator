"""Configuration constants for the Streamlit UI client."""

from __future__ import annotations

import os

API_BASE_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")
# Request timeouts (seconds)
PREDICT_TIMEOUT = 30
HEALTH_TIMEOUT = 5
STATUS_TIMEOUT = 10
EXPLORE_TIMEOUT = 10

LOW_CONFIDENCE_THRESHOLD = 0.30

# Phase 2 polling
PHASE2_POLL_INTERVAL = 1  # seconds between polls
PHASE2_TIMEOUT = 30  # hard timeout in seconds

# 5-class adoption speed colors — frontend_design.md Section 4.2
CLASS_COLORS: dict[int, dict[str, str]] = {
    0: {"bg": "#22c55e", "text": "white",   "label": "Same-day adoption"},
    1: {"bg": "#86efac", "text": "#1e293b", "label": "Adopted within 1 week"},
    2: {"bg": "#fbbf24", "text": "#1e293b", "label": "Adopted within 1 month"},
    3: {"bg": "#f97316", "text": "white",   "label": "Adopted within 1-3 months"},
    4: {"bg": "#ef4444", "text": "white",   "label": "Not adopted (100+ days)"},
}

# Modality colors — frontend_design.md Section 4.4
MODALITY_COLORS: dict[str, str] = {
    "tabular":  "#3b82f6",
    "text":     "#8b5cf6",
    "image":    "#14b8a6",
    "metadata": "#6b7280",
}

MODALITY_LABELS: dict[str, str] = {
    "tabular":  "Tabular",
    "text":     "Text",
    "image":    "Image",
    "metadata": "Metadata",
}

# Global color system — frontend_design.md Section 8.4
PALETTE: dict[str, str] = {
    "background":    "#f8fafc",
    "surface":       "#ffffff",
    "text_primary":  "#1e293b",
    "text_secondary": "#64748b",
    "border":        "#e2e8f0",
    "positive":      "#22c55e",
    "negative":      "#ef4444",
    "warning":       "#f59e0b",
    "info":          "#3b82f6",
}
