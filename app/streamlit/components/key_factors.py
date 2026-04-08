"""Key contributing factors component: two-column positive/negative layout."""

from __future__ import annotations

import streamlit as st

from config import MODALITY_COLORS, MODALITY_LABELS

_MAX_FACTORS = 5


def _factor_row(factor: dict) -> str:
    modality   = factor.get("modality", "tabular")
    mod_color  = MODALITY_COLORS.get(modality, "#6b7280")
    mod_label  = MODALITY_LABELS.get(modality, modality.title())
    name       = factor.get("display_name", factor.get("feature", ""))
    value      = factor.get("value", "")

    return (
        f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:10px;">'
        f'{{arrow}}'
        f'<div style="line-height:1.4;">'
        f'<strong>{name}</strong>: {value}&nbsp;'
        f'<span style="'
        f'background:{mod_color};color:white;'
        f'padding:2px 7px;border-radius:4px;'
        f'font-size:0.72em;vertical-align:middle;">'
        f'{mod_label}'
        f'</span>'
        f'</div>'
        f'</div>'
    )


def render_key_factors(phase1: dict) -> None:
    """Render positive and negative SHAP factors in a two-column layout.

    Args:
        phase1: The phase1 dict from PredictionStatusResponse.
    """
    positives: list = phase1.get("top_positive_factors", [])[:_MAX_FACTORS]
    negatives: list = phase1.get("top_negative_factors", [])[:_MAX_FACTORS]

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown("**Helping Adoption**")
        if positives:
            arrow = '<span style="color:#22c55e;font-size:1.3em;flex-shrink:0;">&#8679;</span>'
            html = "".join(_factor_row(f).replace("{arrow}", arrow) for f in positives)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.caption("No positive factors identified.")

    with col_neg:
        st.markdown("**Slowing Adoption**")
        if negatives:
            arrow = '<span style="color:#ef4444;font-size:1.3em;flex-shrink:0;">&#8681;</span>'
            html = "".join(_factor_row(f).replace("{arrow}", arrow) for f in negatives)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.caption("No negative factors identified.")
