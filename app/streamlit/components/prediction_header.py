"""Prediction header: color-coded badge, confidence bar, low-confidence warning."""

from __future__ import annotations

import streamlit as st

from config import CLASS_COLORS, LOW_CONFIDENCE_THRESHOLD


def render_prediction_header(phase1: dict) -> None:
    """Render the prediction badge, confidence bar, and optional warning.

    Args:
        phase1: The phase1 dict from the PredictionStatusResponse.
    """
    prediction: int  = phase1["prediction"]
    label: str       = phase1["prediction_label"]
    confidence: float = phase1["confidence"]

    color_info = CLASS_COLORS.get(prediction, CLASS_COLORS[4])
    bg    = color_info["bg"]
    fg    = color_info["text"]

    st.markdown(
        f'<div style="'
        f'background:{bg};color:{fg};'
        f'padding:14px 24px;border-radius:10px;'
        f'font-size:1.4em;font-weight:600;'
        f'text-align:center;margin-bottom:12px;'
        f'">{label}</div>',
        unsafe_allow_html=True,
    )

    confidence_pct = int(round(confidence * 100))
    col_label, col_bar = st.columns([1, 4])
    with col_label:
        st.markdown(f"**{confidence_pct}%** confidence")
    with col_bar:
        st.progress(confidence)

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "Low-confidence prediction. Consider providing more details "
            "(description, breed, health status) to improve accuracy."
        )
