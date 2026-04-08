"""Probability distribution chart: horizontal Plotly bar chart for all 5 classes."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from config import CLASS_COLORS


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def render_probability_chart(phase1: dict) -> None:
    """Render a horizontal bar chart of adoption-speed class probabilities.

    The predicted class bar uses full-intensity color; all others use 30% opacity.

    Args:
        phase1: The phase1 dict from PredictionStatusResponse.
    """
    probs: dict      = phase1.get("probabilities", {})
    prediction: int  = phase1["prediction"]

    labels = []
    values = []
    colors = []

    for cls in range(5):
        info = CLASS_COLORS[cls]
        labels.append(info["label"])
        values.append(float(probs.get(str(cls), probs.get(cls, 0.0))))
        alpha = 1.0 if cls == prediction else 0.30
        colors.append(_hex_to_rgba(info["bg"], alpha))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))

    fig.update_layout(
        xaxis=dict(
            range=[0, 1.15],
            tickformat=".0%",
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=8, b=8),
        height=230,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
