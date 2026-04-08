"""Predict page: two-column layout with input form and progressive results.

Phase 1 results render immediately after POST /predict returns 202.
Phase 2 results load asynchronously via polling GET /predict/{id}/status.
"""

from __future__ import annotations

import time

import streamlit as st

from api_client import AdoptionAPI
from components.description_card import render_description_card
from components.explanation_card import render_explanation_card
from components.input_form import render_input_form
from components.key_factors import render_key_factors
from components.pet_card import render_pet_card
from components.loading_states import (
    render_all_phase2_skeletons,
    render_timeout_fallback,
)
from components.modality_donut import render_modality_donut
from components.prediction_header import render_prediction_header
from components.probability_chart import render_probability_chart
from components.recommendations import render_recommendations
from config import API_BASE_URL, PHASE2_POLL_INTERVAL, PHASE2_TIMEOUT

st.title("Predict Adoption Speed")

api = AdoptionAPI(API_BASE_URL)

col_input, col_results = st.columns([4, 6])

with col_input:
    profile, images, submitted = render_input_form()

with col_results:
    # -------------------------------------------------------------------
    # Handle new form submission
    # -------------------------------------------------------------------
    if submitted and profile is not None:
        # Clear previous results
        st.session_state.pop("prediction_result", None)
        st.session_state.pop("prediction_session_id", None)
        st.session_state.pop("phase2_poll_start", None)
        st.session_state.pop("phase2_status", None)
        st.session_state.pop("submitted_description", None)
        st.session_state.pop("connection_error", None)
        # Reset the shimmer CSS flag so it re-injects on fresh render
        st.session_state.pop("_skeleton_css_injected", None)

        result = api.predict(profile, images=images)

        # Check for error response from the API client
        if result.get("error"):
            status_code = result.get("status_code", 0)
            if status_code == 0:
                # Connection error
                st.error(
                    "Unable to connect to the prediction server. "
                    f"Make sure the API is running at {API_BASE_URL}."
                )
            elif status_code == 422:
                st.error("Invalid input -- please review your form values.")
                details = result.get("details")
                if details and isinstance(details, list):
                    for d in details:
                        if isinstance(d, dict):
                            field = d.get("field", "")
                            msg = d.get("message", d.get("msg", ""))
                            st.caption(f"  {field}: {msg}")
            else:
                st.error(result.get("message", "Server error. Please try again."))
            st.stop()

        st.session_state["prediction_result"] = result
        st.session_state["prediction_session_id"] = result.get("session_id")
        st.session_state["phase2_status"] = result.get("status", "phase1_ready")
        st.session_state["phase2_poll_start"] = time.time()
        st.session_state["submitted_description"] = profile.get("description", "")
        st.session_state["uploaded_images"] = images

    # -------------------------------------------------------------------
    # Connection error banner
    # -------------------------------------------------------------------
    if st.session_state.get("connection_error"):
        st.warning(
            "Unable to connect to the prediction server. Checking connection..."
        )

    # -------------------------------------------------------------------
    # Retrieve state
    # -------------------------------------------------------------------
    result: dict | None = st.session_state.get("prediction_result")
    session_id: str | None = st.session_state.get("prediction_session_id")
    phase2_status: str | None = st.session_state.get("phase2_status")
    poll_start: float | None = st.session_state.get("phase2_poll_start")
    submitted_desc: str = st.session_state.get("submitted_description", "")

    # -------------------------------------------------------------------
    # Empty state
    # -------------------------------------------------------------------
    if result is None:
        st.markdown(
            '<div style="'
            "display:flex;align-items:center;justify-content:center;"
            "height:300px;color:#64748b;font-size:1.1em;"
            '">'
            "Enter pet details to get started"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        phase1: dict | None = result.get("phase1")

        # ---------------------------------------------------------------
        # Error state
        # ---------------------------------------------------------------
        if result.get("status") == "error" and phase1 is None:
            st.error(result.get("error_message", "Prediction failed. Please try again."))
        elif phase1 is None:
            st.error("Prediction failed. No results available.")
        else:
            # -----------------------------------------------------------
            # Pet Card summary (always when Phase 1 exists)
            # -----------------------------------------------------------
            phase2_for_card = (
                result.get("phase2") if phase2_status == "complete" else None
            )
            render_pet_card(
                phase1,
                phase2_for_card,
                st.session_state.get("uploaded_images", []),
            )

            st.divider()

            # -----------------------------------------------------------
            # Render Phase 1 (always available)
            # -----------------------------------------------------------
            st.subheader("Prediction")
            render_prediction_header(phase1)

            st.subheader("Probability by Class")
            render_probability_chart(phase1)

            st.subheader("Key Contributing Factors")
            render_key_factors(phase1)

            st.subheader("What Drove This Prediction")
            render_modality_donut(phase1)

            # -----------------------------------------------------------
            # Phase 2: progressive loading via polling
            # -----------------------------------------------------------
            had_description = bool(submitted_desc.strip())

            if phase2_status == "complete":
                # Phase 2 is done -- render final components
                phase2 = result.get("phase2")
                st.divider()
                render_explanation_card(phase2)
                render_recommendations(phase2)
                render_description_card(phase2, had_description)

            elif phase2_status == "timeout":
                # Polling timed out
                render_timeout_fallback()

            elif phase2_status == "error":
                # Phase 2 errored but Phase 1 is valid
                st.divider()
                error_msg = result.get("error_message", "")
                if error_msg:
                    st.warning(f"AI insights encountered an error: {error_msg}")
                render_explanation_card(None)
                render_recommendations(None)
                render_description_card(None, had_description)

            else:
                # Phase 2 still in progress -- show skeletons and poll
                render_all_phase2_skeletons()

                # Check timeout
                if poll_start is not None and (time.time() - poll_start) >= PHASE2_TIMEOUT:
                    st.session_state["phase2_status"] = "timeout"
                    st.rerun()
                elif session_id is not None:
                    # Poll for Phase 2 completion
                    time.sleep(PHASE2_POLL_INTERVAL)
                    status_resp = api.get_status(session_id)

                    # If polling returned an error dict, treat as
                    # transient -- keep skeletons and retry on next rerun.
                    if status_resp.get("error"):
                        st.rerun()
                    else:
                        new_status = status_resp.get("status", "phase1_ready")

                        if new_status == "complete":
                            st.session_state["prediction_result"] = status_resp
                            st.session_state["phase2_status"] = "complete"
                            st.rerun()
                        elif new_status == "error":
                            st.session_state["prediction_result"] = status_resp
                            st.session_state["phase2_status"] = "error"
                            st.rerun()
                        else:
                            # Still processing -- rerun to show updated skeletons
                            st.rerun()

            # -----------------------------------------------------------
            # Metadata expander (always shown when Phase 1 is available)
            # -----------------------------------------------------------
            metadata: dict | None = result.get("metadata")
            if metadata:
                with st.expander("Prediction Details"):
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.metric(
                            "Inference Time",
                            f"{metadata.get('inference_time_ms', 0):.0f} ms",
                        )
                    with col_m2:
                        st.metric(
                            "Total Time",
                            f"{metadata.get('total_time_ms', 0):.0f} ms",
                        )
                    with col_m3:
                        sid = metadata.get("session_id", "N/A")
                        st.metric("Session", sid[:8] + "..." if len(sid) > 8 else sid)

                    nodes = metadata.get("nodes_executed", [])
                    if nodes:
                        st.caption("Nodes executed: " + ", ".join(nodes))

                    errors = metadata.get("errors", [])
                    if errors:
                        for err in errors:
                            st.warning(str(err))
