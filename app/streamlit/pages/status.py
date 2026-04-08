"""System Status page -- health cards, recent predictions, model info."""

from __future__ import annotations

import streamlit as st

from api_client import AdoptionAPI
from config import API_BASE_URL, CLASS_COLORS

st.title("System Status")

api = AdoptionAPI(API_BASE_URL)


def _status_color(status: str) -> str:
    mapping = {
        "healthy":   "#22c55e",
        "connected": "#22c55e",
        "degraded":  "#f59e0b",
        "offline":   "#ef4444",
    }
    return mapping.get(status, "#6b7280")


def _status_dot(status: str) -> str:
    color = _status_color(status)
    return f'<span style="color:{color};font-size:1.1em;">&#9679;</span>'


# ── Fetch data ───────────────────────────────────────────────────────

health = api.health()
model_info = api.model_info()
predictions_resp = api.recent_predictions(limit=20)

has_health = not health.get("error")
has_model = not model_info.get("error")
has_predictions = not predictions_resp.get("error")

# ── Health Cards Row ─────────────────────────────────────────────────

if has_health:
    model_status = health.get("model_status", "offline")
    agent_status = health.get("agent_status", "offline")
else:
    model_status = "offline"
    agent_status = "offline"

# Compute avg response time and predictions today from recent predictions
avg_response_time = 0.0
total_today = 0
if has_predictions:
    total_today = predictions_resp.get("total_today", 0)
    preds = predictions_resp.get("predictions", [])
    times = [p.get("response_time_ms", 0) for p in preds if p.get("response_time_ms", 0) > 0]
    if times:
        avg_response_time = sum(times) / len(times)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"**Model Status**\n\n"
        f"{_status_dot(model_status)} {model_status.title()}",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"**AI Agent Status**\n\n"
        f"{_status_dot(agent_status)} {agent_status.title()}",
        unsafe_allow_html=True,
    )
with c3:
    display_time = f"{avg_response_time / 1000:.1f}s" if avg_response_time > 0 else "N/A"
    st.metric("Avg Response Time", display_time)
with c4:
    st.metric("Predictions Today", total_today)

st.markdown("---")

# ── Recent Predictions Table ─────────────────────────────────────────

st.subheader("Recent Predictions")

if not has_predictions:
    st.warning(predictions_resp.get("message", "Could not load recent predictions."))
else:
    preds = predictions_resp.get("predictions", [])
    if not preds:
        st.info("No predictions recorded yet. Make a prediction in the Predict tab to see entries here.")
    else:
        table_rows = []
        for p in preds:
            pred_cls = p.get("prediction", -1)
            pred_label = p.get("prediction_label", "N/A")
            confidence = p.get("confidence", 0)
            status = p.get("status", "N/A")

            # Color-coded prediction label
            cls_info = CLASS_COLORS.get(pred_cls, {"bg": "#6b7280", "text": "white"})

            table_rows.append({
                "Timestamp": p.get("timestamp", "N/A")[:19],
                "Pet Type": p.get("pet_type", "N/A"),
                "Prediction": pred_label,
                "Confidence": f"{confidence:.1%}" if confidence > 0 else "N/A",
                "Response Time": f"{p.get('response_time_ms', 0):.0f} ms",
                "Status": status,
            })

        st.dataframe(table_rows, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Model Information Panel ──────────────────────────────────────────

st.subheader("Model Information")

if not has_model:
    st.warning(model_info.get("message", "Could not load model information."))
elif not has_health:
    st.error(
        f"Cannot connect to the API server at **{API_BASE_URL}**. "
        "Please ensure the FastAPI backend is running."
    )
else:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**Name:** {model_info.get('model_name', 'N/A')}")
        st.markdown(f"**Version:** {model_info.get('model_version', 'N/A')}")
        st.markdown(f"**Family:** {model_info.get('model_family', 'N/A')}")

        base_models = model_info.get("base_models", [])
        if base_models:
            st.markdown(f"**Base Models:** {', '.join(base_models)}")

    with col_b:
        training_qwk = model_info.get("training_qwk", 0)
        st.markdown(f"**Training QWK:** {training_qwk:.4f}")
        st.markdown(f"**Feature Count:** {model_info.get('feature_count', 0)}")

        breakdown = model_info.get("modality_breakdown", {})
        if breakdown:
            st.markdown("**Modality Breakdown:**")
            for modality, count in breakdown.items():
                st.markdown(f"- {modality.title()}: {count} features")

    st.markdown("---")
    st.caption(f"API endpoint: {API_BASE_URL}")
