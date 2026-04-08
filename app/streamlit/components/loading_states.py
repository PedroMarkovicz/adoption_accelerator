"""Skeleton loader components for Phase 2 progressive loading."""

from __future__ import annotations

import streamlit as st

_SKELETON_BLOCK = (
    '<div style="'
    "background:linear-gradient(90deg,#e2e8f0 25%,#f1f5f9 50%,#e2e8f0 75%);"
    "background-size:200% 100%;"
    "animation:shimmer 1.5s infinite;"
    "border-radius:8px;height:{height}px;margin-bottom:12px;"
    '"></div>'
)

_SHIMMER_CSS = """
<style>
@keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
</style>
"""

_CSS_INJECTED_KEY = "_skeleton_css_injected"


def _inject_shimmer_css() -> None:
    """Inject the shimmer animation CSS once per page render."""
    if not st.session_state.get(_CSS_INJECTED_KEY):
        st.markdown(_SHIMMER_CSS, unsafe_allow_html=True)
        st.session_state[_CSS_INJECTED_KEY] = True


def _skeleton_block(height: int = 24) -> str:
    """Return HTML for a single gray shimmer block."""
    return _SKELETON_BLOCK.format(height=height)


def render_explanation_skeleton() -> None:
    """Skeleton loader for the AI Explanation section."""
    _inject_shimmer_css()
    st.subheader("AI Explanation")
    st.markdown(
        f'<div style="padding:8px 0;">'
        f"  {_skeleton_block(20)}"
        f"  {_skeleton_block(20)}"
        f"  {_skeleton_block(20)}"
        f"  {_skeleton_block(14)}"
        f"</div>"
        f'<p style="color:#94a3b8;font-size:0.85em;">Generating AI insights...</p>',
        unsafe_allow_html=True,
    )


def render_recommendations_skeleton() -> None:
    """Skeleton loader for the Recommendations section."""
    _inject_shimmer_css()
    st.subheader("Recommendations")
    st.markdown(
        f'<div style="padding:8px 0;">'
        f"  {_skeleton_block(40)}"
        f"  {_skeleton_block(40)}"
        f"  {_skeleton_block(40)}"
        f"</div>"
        f'<p style="color:#94a3b8;font-size:0.85em;">Analyzing improvement areas...</p>',
        unsafe_allow_html=True,
    )


def render_description_skeleton() -> None:
    """Skeleton loader for the Improved Description section."""
    _inject_shimmer_css()
    st.subheader("Improved Description")
    st.markdown(
        f'<div style="padding:8px 0;">'
        f"  {_skeleton_block(20)}"
        f"  {_skeleton_block(20)}"
        f"  {_skeleton_block(14)}"
        f"</div>"
        f'<p style="color:#94a3b8;font-size:0.85em;">Crafting improved description...</p>',
        unsafe_allow_html=True,
    )


def render_all_phase2_skeletons() -> None:
    """Render all Phase 2 skeleton loaders together."""
    st.divider()
    render_explanation_skeleton()
    render_recommendations_skeleton()
    render_description_skeleton()


def render_timeout_fallback() -> None:
    """Render a timeout message when Phase 2 exceeds the polling deadline."""
    st.divider()
    st.markdown(
        '<div style="'
        "background:#f8fafc;border:1px solid #e2e8f0;"
        "border-radius:8px;padding:20px;text-align:center;"
        "color:#64748b;"
        '">'
        "<strong>AI insights are taking longer than expected.</strong><br>"
        "The core prediction and factors above are still valid."
        "</div>",
        unsafe_allow_html=True,
    )
