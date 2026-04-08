"""Pet Card component: consolidated summary card for prediction results.

Renders a two-panel card (media + content) that ties the pet's identity
to its predicted adoption outcome.  Consumes Phase 1 (always), Phase 2
(when available), and form inputs from ``st.session_state``.
"""

from __future__ import annotations

import base64
from html import escape

import streamlit as st

from config import CLASS_COLORS, LOW_CONFIDENCE_THRESHOLD

_SIZE_LABELS = {1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large"}

_PAW_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" '
    'viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<ellipse cx="8" cy="6.5" rx="1.8" ry="2.2"/>'
    '<ellipse cx="16" cy="6.5" rx="1.8" ry="2.2"/>'
    '<ellipse cx="5" cy="11.5" rx="1.5" ry="2"/>'
    '<ellipse cx="19" cy="11.5" rx="1.5" ry="2"/>'
    '<path d="M12 17c-2.5 0-4.5-1.5-5-3.5 0 0-.5-2 1-3s3.5-1 4-1 '
    '2.5 0 4 1 1 3 1 3c-.5 2-2.5 3.5-5 3.5z"/>'
    "</svg>"
)


# -- Private helpers --------------------------------------------------------


def _format_age(months: int) -> str:
    """Convert months to a human-readable age string."""
    if months == 0:
        return "Less than 1 month old"
    if months == 1:
        return "1 month old"
    if months < 12:
        return f"{months} months old"
    years = months // 12
    remaining = months % 12
    if remaining == 0:
        return f"{years}y old"
    return f"{years}y {remaining}m old"


def _render_hero_image(images: list[tuple[str, bytes]], pet_name: str) -> str:
    """Return an ``<img>`` tag (base64) or a placeholder block."""
    alt = f"Photo of {escape(pet_name)}" if pet_name else "Pet photo"

    if images:
        _filename, raw = images[0]
        b64 = base64.b64encode(raw).decode()
        mime = "image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        return (
            '<div class="pet-card-hero">'
            f'<img src="data:{mime};base64,{b64}" alt="{alt}" '
            'style="width:100%;height:100%;object-fit:cover;display:block;"/>'
            "</div>"
        )

    return (
        '<div class="pet-card-hero pet-card-placeholder">'
        f"{_PAW_SVG}"
        '<span style="color:#94a3b8;font-size:12px;margin-top:8px;">'
        "No photo provided</span>"
        "</div>"
    )


def _health_badge(value: str) -> str:
    """Return a micro-badge ``<span>`` for a health field value."""
    cls_map = {"Yes": "pet-card-health-yes", "No": "pet-card-health-no"}
    cls = cls_map.get(value, "pet-card-health-unsure")
    return f'<span class="{cls}">{escape(value)}</span>'


def _render_health_badges() -> str:
    """Build the three-row health status grid from session state."""
    rows = [
        ("Vaccinated", st.session_state.get("form_vaccinated", "Not Sure")),
        ("Dewormed", st.session_state.get("form_dewormed", "Not Sure")),
        ("Sterilized", st.session_state.get("form_sterilized", "Not Sure")),
    ]
    inner = ""
    for label, val in rows:
        inner += (
            '<div class="pet-card-health-row">'
            f'<span class="pet-card-health-label">{label}</span>'
            f"{_health_badge(val)}"
            "</div>"
        )
    return f'<div class="pet-card-health-grid">{inner}</div>'


def _render_factors_summary(phase1: dict) -> str:
    """Return condensed HTML for top 3 positive + top 1 negative factors."""
    positive = phase1.get("top_positive_factors") or []
    negative = phase1.get("top_negative_factors") or []

    if not positive and not negative:
        return ""

    lines = ""
    for factor in positive[:3]:
        name = escape(factor.get("display_name", factor.get("feature", "")))
        shap = factor.get("shap_value", 0.0)
        lines += (
            '<div class="pet-card-factor">'
            '<span style="color:#22c55e;margin-right:6px;">&#9650;</span>'
            f'<span class="pet-card-factor-name">{name}</span>'
            f'<span class="pet-card-factor-value" style="color:#22c55e;">'
            f"+{shap:.2f}</span>"
            "</div>"
        )
    for factor in negative[:1]:
        name = escape(factor.get("display_name", factor.get("feature", "")))
        shap = factor.get("shap_value", 0.0)
        lines += (
            '<div class="pet-card-factor">'
            '<span style="color:#ef4444;margin-right:6px;">&#9660;</span>'
            f'<span class="pet-card-factor-name">{name}</span>'
            f'<span class="pet-card-factor-value" style="color:#ef4444;">'
            f"{shap:.2f}</span>"
            "</div>"
        )

    return (
        '<div class="pet-card-section-label" style="margin-top:16px;">'
        "TOP FACTORS</div>"
        f'<div class="pet-card-factors">{lines}</div>'
    )


def _build_card_html(  # noqa: PLR0913
    *,
    hero_html: str,
    health_html: str,
    factors_html: str,
    pet_name: str,
    pet_type: str,
    size_label: str,
    age_str: str,
    gender: str,
    description: str,
    description_is_ai: bool,
    prediction_label: str,
    confidence: float,
    accent_bg: str,
    accent_text: str,
    fee: float,
    recommendation_html: str,
    low_confidence: bool,
) -> str:
    """Assemble the complete card HTML + scoped CSS."""

    # -- Name --
    if pet_name:
        name_html = (
            '<span style="font-size:20px;font-weight:700;color:#1e293b;">'
            f"{escape(pet_name)}</span>"
        )
    else:
        name_html = (
            '<span style="font-size:20px;font-weight:700;color:#94a3b8;">'
            "Unnamed Pet</span>"
        )

    # -- Fee --
    fee_text = "Free" if fee == 0 else f"MYR {fee:.0f}"

    # -- Description --
    if description:
        ai_tag = ""
        if description_is_ai:
            ai_tag = (
                '<span style="display:inline-block;background:#3b82f6;color:white;'
                "font-size:10px;font-weight:600;padding:1px 6px;border-radius:4px;"
                'margin-left:8px;vertical-align:middle;">AI-enhanced</span>'
            )
        desc_block = (
            '<div class="pet-card-description">'
            f'<span class="pet-card-section-label">DESCRIPTION{ai_tag}</span>'
            '<p style="margin:6px 0 0;font-size:14px;color:#374151;'
            f'line-height:1.5;">{escape(description)}</p>'
            "</div>"
        )
    else:
        desc_block = (
            '<div class="pet-card-description">'
            '<p style="margin:0;font-size:14px;color:#94a3b8;font-style:italic;">'
            "No description provided</p></div>"
        )

    # -- Confidence --
    conf_pct = f"{int(round(confidence * 100))}%"
    warn_icon = ""
    if low_confidence:
        warn_icon = (
            '<span style="margin-left:6px;font-size:14px;" '
            'title="Low confidence prediction">&#9888;</span>'
        )

    # NOTE: Every line must start at column 0 (no leading spaces).
    # Streamlit's st.markdown parses Markdown *before* injecting HTML;
    # lines indented 4+ spaces are treated as Markdown code blocks and
    # rendered as raw text instead of being interpreted as HTML.
    html = (
        "<style>"
        f".pet-card{{"
        f"background:#ffffff;"
        f"border-radius:12px;"
        f"border:1px solid #e2e8f0;"
        f"border-top:4px solid {accent_bg};"
        f"box-shadow:0 1px 3px rgba(0,0,0,0.08);"
        f"overflow:hidden;"
        f"margin-bottom:24px;"
        f"}}"
        ".pet-card-body{"
        "display:flex;gap:24px;padding:24px;"
        "}"
        ".pet-card-media{"
        "flex:0 0 35%;min-width:0;"
        "}"
        ".pet-card-content{"
        "flex:1;min-width:0;"
        "}"
        ".pet-card-hero{"
        "width:100%;aspect-ratio:4/3;border-radius:8px;"
        "overflow:hidden;background:#f8fafc;"
        "}"
        ".pet-card-placeholder{"
        "display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;"
        "}"
        ".pet-card-health-grid{margin-top:16px;}"
        ".pet-card-health-row{"
        "display:flex;justify-content:space-between;"
        "align-items:center;padding:4px 0;"
        "}"
        ".pet-card-health-label{"
        "font-size:12px;font-weight:500;color:#475569;"
        "}"
        ".pet-card-health-yes{"
        "display:inline-block;padding:2px 8px;border-radius:4px;"
        "font-size:12px;font-weight:600;background:#dcfce7;color:#166534;"
        "}"
        ".pet-card-health-no{"
        "display:inline-block;padding:2px 8px;border-radius:4px;"
        "font-size:12px;font-weight:600;background:#fee2e2;color:#991b1b;"
        "}"
        ".pet-card-health-unsure{"
        "display:inline-block;padding:2px 8px;border-radius:4px;"
        "font-size:12px;font-weight:600;background:#f1f5f9;color:#64748b;"
        "}"
        ".pet-card-badge{"
        "display:inline-block;padding:2px 10px;border-radius:999px;"
        "font-size:12px;font-weight:600;background:#f1f5f9;color:#475569;"
        "margin-left:8px;vertical-align:middle;"
        "}"
        ".pet-card-subline{"
        "font-size:14px;color:#64748b;margin-top:4px;"
        "}"
        ".pet-card-description{"
        "background:#f8fafc;border-radius:8px;"
        "padding:12px 16px;margin-top:16px;"
        "}"
        ".pet-card-section-label{"
        "font-size:11px;font-weight:700;color:#94a3b8;"
        "letter-spacing:0.05em;text-transform:uppercase;"
        "}"
        f".pet-card-banner{{"
        f"display:flex;justify-content:space-between;align-items:center;"
        f"background:{accent_bg};color:{accent_text};"
        f"padding:12px 16px;border-radius:8px;margin-top:16px;"
        f"font-size:16px;font-weight:600;"
        f"}}"
        ".pet-card-factors{margin-top:8px;}"
        ".pet-card-factor{"
        "display:flex;align-items:center;"
        "font-size:13px;color:#374151;padding:3px 0;"
        "}"
        ".pet-card-factor-name{flex:1;}"
        ".pet-card-factor-value{"
        "font-weight:600;font-size:12px;margin-left:8px;"
        "}"
        ".pet-card-recommendation{"
        "background:#eff6ff;border-left:3px solid #3b82f6;"
        "border-radius:0 8px 8px 0;padding:10px 14px;margin-top:8px;"
        "font-size:13px;color:#1e40af;line-height:1.5;"
        "}"
        ".pet-card-fee{"
        "margin-top:12px;font-size:13px;color:#475569;"
        "}"
        ".pet-card-fee strong{color:#1e293b;}"
        "@media (max-width:768px){"
        ".pet-card-body{flex-direction:column;}"
        ".pet-card-media{flex:none;width:100%;}"
        ".pet-card-hero{aspect-ratio:16/9;}"
        "}"
        "</style>"
        '<div class="pet-card">'
        '<div class="pet-card-body">'
        '<div class="pet-card-media">'
        f"{hero_html}"
        f"{health_html}"
        f'<div class="pet-card-fee">Fee: <strong>{fee_text}</strong></div>'
        "</div>"
        '<div class="pet-card-content">'
        f"<div>{name_html}"
        f'<span class="pet-card-badge">{escape(pet_type)}</span>'
        f'<span class="pet-card-badge">{escape(size_label)}</span>'
        "</div>"
        f'<div class="pet-card-subline">'
        f"{escape(age_str)} &mdash; {escape(gender)}"
        "</div>"
        f"{desc_block}"
        '<div class="pet-card-banner">'
        f"<span>{escape(prediction_label)}</span>"
        f"<span>{conf_pct}{warn_icon}</span>"
        "</div>"
        f"{factors_html}"
        f"{recommendation_html}"
        "</div>"
        "</div>"
        "</div>"
    )

    return html


# -- Public API -------------------------------------------------------------


def render_pet_card(
    phase1: dict,
    phase2: dict | None,
    uploaded_images: list[tuple[str, bytes]],
) -> None:
    """Render the consolidated Pet Card summary.

    Parameters
    ----------
    phase1 : dict
        The phase1 dict from ``prediction_result``.  Must be non-None.
    phase2 : dict or None
        The phase2 dict, or ``None`` if still loading / failed.
    uploaded_images : list[tuple[str, bytes]]
        Uploaded pet photos.  Empty list if none were provided.
    """
    # -- Form inputs from session state --
    pet_name: str = st.session_state.get("form_name", "")
    pet_type: str = st.session_state.get("form_pet_type", "Dog")
    age_months: int = int(st.session_state.get("form_age_months", 0))
    gender: str = st.session_state.get("form_gender", "Male")
    maturity_size: int = int(st.session_state.get("form_maturity_size", 2))
    fee: float = float(st.session_state.get("form_fee", 0))
    raw_description: str = st.session_state.get("form_description", "")

    # -- Phase 1 data --
    prediction: int = phase1.get("prediction", 4)
    prediction_label: str = phase1.get("prediction_label", "Unknown")
    confidence: float = phase1.get("confidence", 0.0)
    color_info = CLASS_COLORS.get(prediction, CLASS_COLORS[4])

    # -- Phase 2 data (graceful degradation) --
    improved_description: str | None = None
    recommendation_html = ""
    if phase2 is not None:
        improved_description = phase2.get("improved_description")
        recommendations = phase2.get("recommendations") or []
        if recommendations:
            top_rec = min(recommendations, key=lambda r: r.get("priority", 999))
            impact = top_rec.get("expected_impact", "")
            if impact:
                recommendation_html = (
                    '<div class="pet-card-section-label" '
                    'style="margin-top:16px;">TOP RECOMMENDATION</div>'
                    '<div class="pet-card-recommendation">'
                    f"{escape(impact)}</div>"
                )

    # -- Choose best available description --
    if improved_description:
        description = improved_description
        description_is_ai = True
    elif raw_description.strip():
        description = raw_description
        description_is_ai = False
    else:
        description = ""
        description_is_ai = False

    # -- Assemble and render --
    card_html = _build_card_html(
        hero_html=_render_hero_image(uploaded_images, pet_name),
        health_html=_render_health_badges(),
        factors_html=_render_factors_summary(phase1),
        pet_name=pet_name,
        pet_type=pet_type,
        size_label=_SIZE_LABELS.get(maturity_size, "Medium"),
        age_str=_format_age(age_months),
        gender=gender,
        description=description,
        description_is_ai=description_is_ai,
        prediction_label=prediction_label,
        confidence=confidence,
        accent_bg=color_info["bg"],
        accent_text=color_info["text"],
        fee=fee,
        recommendation_html=recommendation_html,
        low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
    )

    st.markdown(card_html, unsafe_allow_html=True)
