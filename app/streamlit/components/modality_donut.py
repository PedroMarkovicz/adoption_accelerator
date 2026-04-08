"""Modality contribution donut chart."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from config import MODALITY_COLORS, MODALITY_LABELS

_MODALITY_ORDER = ["tabular", "text", "image", "metadata"]


def render_modality_donut(phase1: dict) -> None:
    """Render a Plotly donut chart of modality contributions.

    Only segments whose modality_available entry is True are shown.

    Args:
        phase1: The phase1 dict from PredictionStatusResponse.
    """
    contributions: dict = phase1.get("modality_contributions", {})
    available: dict     = phase1.get("modality_available", {})

    labels  = []
    values  = []
    colors  = []

    for mod in _MODALITY_ORDER:
        if available.get(mod, False):
            contrib = contributions.get(mod, 0.0)
            if contrib > 0:
                labels.append(MODALITY_LABELS.get(mod, mod.title()))
                values.append(float(contrib))
                colors.append(MODALITY_COLORS[mod])

    if not labels:
        st.caption("No modality contribution data available.")
        return

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.52,
        marker=dict(colors=colors, line=dict(color="#ffffff", width=2)),
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.1%}<extra></extra>",
        sort=False,
    )])

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=8, b=8),
        height=260,
        paper_bgcolor="#ffffff",
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
