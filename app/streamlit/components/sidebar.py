"""Sidebar component: model version badge and connection status indicator.

Status indicator logic (Phase 4 -- Graceful Degradation):
  - Green:  Both model_status == "healthy" AND agent_status == "connected".
  - Yellow: Model is healthy but agent_status is "degraded" (LLM issues).
  - Red:    Model is offline, API is unreachable, or health returns an error.
"""

from __future__ import annotations

import streamlit as st

from api_client import AdoptionAPI
from config import API_BASE_URL


@st.cache_data(ttl=10)
def _fetch_health() -> dict | None:
    """Fetch the /health endpoint. Returns None if unreachable or error."""
    try:
        result = AdoptionAPI(API_BASE_URL).health()
        # The API client now returns an error dict on failure
        if isinstance(result, dict) and result.get("error"):
            return None
        return result
    except Exception:
        return None


# Status dot HTML fragments
_DOT_GREEN = '<span style="color:#22c55e;font-size:1.1em;">&#9679;</span>'
_DOT_YELLOW = '<span style="color:#f59e0b;font-size:1.1em;">&#9679;</span>'
_DOT_RED = '<span style="color:#ef4444;font-size:1.1em;">&#9679;</span>'


_FORM_DEFAULTS: dict = {
    "form_pet_type": "Dog",
    "form_name": "",
    "form_age_months": 6,
    "form_gender": "Male",
    "form_breed1": 0,
    "form_breed2": 0,
    "form_color1": 0,
    "form_color2": 0,
    "form_color3": 0,
    "form_maturity_size": 2,
    "form_fur_length": 1,
    "form_vaccinated": "Not Sure",
    "form_dewormed": "Not Sure",
    "form_sterilized": "Not Sure",
    "form_health": "Healthy",
    "form_fee": 0.0,
    "form_quantity": 1,
    "form_state": 0,
    "form_video_amt": 0,
    "form_description": "",
}

_RESULT_KEYS = [
    "prediction_result",
    "prediction_session_id",
    "phase2_poll_start",
    "phase2_status",
    "submitted_description",
    "connection_error",
    "_skeleton_css_injected",
    "_pending_improved_description",
    "_description_applied",
    "uploaded_images",
]


def _reset_app() -> None:
    """Reset all form inputs and prediction results to defaults."""
    for key, default in _FORM_DEFAULTS.items():
        st.session_state[key] = default
    for key in _RESULT_KEYS:
        st.session_state.pop(key, None)


def render_sidebar_info() -> None:
    """Append model info and connection status to the sidebar."""
    with st.sidebar:
        if st.button("Refresh Status", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    health = _fetch_health()

    with st.sidebar:
        st.markdown("---")

        if health is None:
            # API is completely unreachable
            dot = _DOT_RED
            status_label = "Offline"
            model_version = "N/A"
            model_type = ""
        else:
            model_status = health.get("model_status", "offline")
            agent_status = health.get("agent_status", "offline")
            model_version = health.get("model_version", "N/A")
            model_type = health.get("model_type", "")

            if model_status == "healthy" and agent_status == "connected":
                # Everything works
                dot = _DOT_GREEN
                status_label = "Healthy"
            elif model_status == "healthy" and agent_status in ("degraded", "offline"):
                # Model OK but LLM agents have issues
                dot = _DOT_YELLOW
                status_label = "Degraded (AI agents)"
            elif model_status == "degraded":
                # Model having issues
                dot = _DOT_YELLOW
                status_label = "Degraded (model)"
            else:
                # Model is offline
                dot = _DOT_RED
                status_label = "Offline"

        st.markdown(f"{dot} **{status_label}**", unsafe_allow_html=True)

        if model_version != "N/A":
            badge = f"Model: **{model_version}**"
            if model_type:
                badge += f" ({model_type})"
            st.caption(badge)
        else:
            st.caption("Model: N/A -- server unreachable")

        st.markdown("---")
        st.button(
            "Reset Application",
            use_container_width=True,
            on_click=_reset_app,
            help="Clear all inputs and results, returning to the initial state.",
        )
