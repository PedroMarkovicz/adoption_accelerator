"""Explore Data page -- feature distributions, adoption patterns, model performance."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from api_client import AdoptionAPI
from config import API_BASE_URL, CLASS_COLORS, MODALITY_COLORS, MODALITY_LABELS

st.title("Explore Data")

api = AdoptionAPI(API_BASE_URL)

tab_dist, tab_patterns, tab_perf = st.tabs(
    ["Feature Distributions", "Adoption Patterns", "Model Performance"]
)

# ── Class color helpers ──────────────────────────────────────────────

CLASS_COLOR_LIST = [CLASS_COLORS[i]["bg"] for i in range(5)]


def _class_label(cls_id: int | str) -> str:
    idx = int(cls_id)
    return CLASS_COLORS.get(idx, {}).get("label", f"Class {idx}")


# =====================================================================
# Tab 1: Feature Distributions
# =====================================================================

with tab_dist:
    # Fetch available features
    feat_resp = api.explore_features()
    if feat_resp.get("error"):
        st.error(feat_resp.get("message", "Failed to load features."))
    else:
        features = feat_resp.get("features", [])
        if not features:
            st.info("No feature data available.")
        else:
            feature_names = [f["feature"] for f in features]
            feature_display = {f["feature"]: f["display_name"] for f in features}

            col_sel, col_toggle = st.columns([3, 1])
            with col_sel:
                selected = st.selectbox(
                    "Select Feature",
                    feature_names,
                    format_func=lambda x: feature_display.get(x, x),
                )
            with col_toggle:
                color_by_class = st.toggle("Color by adoption class", value=False)

            if selected:
                dist_resp = api.explore_distributions(selected, color_by_class)
                if dist_resp.get("error"):
                    st.error(dist_resp.get("message", "Failed to load distribution."))
                else:
                    data = dist_resp.get("data", {})
                    feat_type = data.get("type", "numeric")
                    display_name = data.get("display_name", selected)

                    fig = go.Figure()

                    if feat_type == "numeric":
                        bins = data.get("bins", [])
                        # Compute bin centers for bar positioning
                        bin_centers = [
                            round((bins[i] + bins[i + 1]) / 2, 2)
                            for i in range(len(bins) - 1)
                        ]
                        bin_width = bins[1] - bins[0] if len(bins) > 1 else 1

                        if color_by_class and data.get("by_class"):
                            for cls_id in range(5):
                                cls_counts = data["by_class"].get(str(cls_id), [])
                                fig.add_trace(go.Bar(
                                    x=bin_centers,
                                    y=cls_counts,
                                    name=_class_label(cls_id),
                                    marker_color=CLASS_COLOR_LIST[cls_id],
                                    width=bin_width * 0.9,
                                ))
                            fig.update_layout(barmode="stack")
                        else:
                            fig.add_trace(go.Bar(
                                x=bin_centers,
                                y=data.get("counts", []),
                                marker_color="#3b82f6",
                                width=bin_width * 0.9,
                            ))

                        fig.update_layout(
                            xaxis_title=display_name,
                            yaxis_title="Count",
                        )

                    else:  # categorical
                        categories = data.get("categories", [])
                        if color_by_class and data.get("by_class"):
                            for cls_id in range(5):
                                cls_counts = data["by_class"].get(str(cls_id), [])
                                fig.add_trace(go.Bar(
                                    x=categories,
                                    y=cls_counts,
                                    name=_class_label(cls_id),
                                    marker_color=CLASS_COLOR_LIST[cls_id],
                                ))
                            fig.update_layout(barmode="stack")
                        else:
                            fig.add_trace(go.Bar(
                                x=categories,
                                y=data.get("counts", []),
                                marker_color="#3b82f6",
                            ))

                        fig.update_layout(
                            xaxis_title=display_name,
                            yaxis_title="Count",
                        )

                    fig.update_layout(
                        title=f"Distribution: {display_name}",
                        height=450,
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                    )

                    st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# Tab 2: Adoption Patterns
# =====================================================================

with tab_patterns:
    patterns_resp = api.explore_patterns()
    if patterns_resp.get("error"):
        st.error(patterns_resp.get("message", "Failed to load adoption patterns."))
    else:
        # -- Class Distribution --
        st.subheader("Adoption Speed Distribution")
        class_dist = patterns_resp.get("class_distribution", {})
        labels = class_dist.get("labels", [])
        counts = class_dist.get("counts", [])
        percentages = class_dist.get("percentages", [])

        fig_class = go.Figure(go.Bar(
            x=labels,
            y=counts,
            marker_color=CLASS_COLOR_LIST[:len(labels)],
            text=[f"{p}%" for p in percentages],
            textposition="outside",
        ))
        fig_class.update_layout(
            height=380,
            template="plotly_white",
            yaxis_title="Number of Pets",
        )
        st.plotly_chart(fig_class, use_container_width=True)

        # -- Top 10 features & Modality side by side --
        col_feat, col_mod = st.columns(2)

        with col_feat:
            st.subheader("Top 10 Features by Importance")
            perf_resp = api.explore_performance()
            if perf_resp.get("error"):
                st.warning("Could not load feature importance data.")
            else:
                importance = perf_resp.get("global_importance", [])[:10]
                if importance:
                    feat_names = [f["display_name"] for f in reversed(importance)]
                    feat_values = [f["mean_abs_shap"] for f in reversed(importance)]
                    fig_imp = go.Figure(go.Bar(
                        x=feat_values,
                        y=feat_names,
                        orientation="h",
                        marker_color="#3b82f6",
                    ))
                    fig_imp.update_layout(
                        height=400,
                        template="plotly_white",
                        xaxis_title="Mean |SHAP|",
                        margin=dict(l=10),
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)

        with col_mod:
            st.subheader("Modality Importance")
            mod_data = patterns_resp.get("modality_importance", {})
            if mod_data:
                mod_labels = [MODALITY_LABELS.get(k, k) for k in mod_data]
                mod_values = list(mod_data.values())
                mod_colors = [MODALITY_COLORS.get(k, "#6b7280") for k in mod_data]

                fig_mod = go.Figure(go.Pie(
                    labels=mod_labels,
                    values=mod_values,
                    hole=0.45,
                    marker=dict(colors=mod_colors),
                    textinfo="label+percent",
                ))
                fig_mod.update_layout(
                    height=400,
                    template="plotly_white",
                    showlegend=False,
                )
                st.plotly_chart(fig_mod, use_container_width=True)

        # -- Dog vs Cat Comparison --
        st.subheader("Dog vs Cat: Adoption Speed Comparison")
        dog_cat = patterns_resp.get("dog_vs_cat", {})
        class_labels_list = class_dist.get("labels", [f"Class {i}" for i in range(5)])

        if dog_cat:
            fig_dc = go.Figure()
            for pet_label, color in [("Dog", "#3b82f6"), ("Cat", "#f97316")]:
                data = dog_cat.get(pet_label, {})
                fig_dc.add_trace(go.Bar(
                    x=class_labels_list,
                    y=data.get("percentages", []),
                    name=pet_label,
                    marker_color=color,
                ))
            fig_dc.update_layout(
                barmode="group",
                height=380,
                template="plotly_white",
                yaxis_title="Percentage (%)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_dc, use_container_width=True)


# =====================================================================
# Tab 3: Model Performance
# =====================================================================

with tab_perf:
    perf_data = api.explore_performance()
    if perf_data.get("error"):
        st.error(perf_data.get("message", "Failed to load performance data."))
    else:
        # -- Metric cards --
        agg = perf_data.get("aggregate_metrics", {})
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("QWK", f"{agg.get('qwk', 0):.4f}")
        with m2:
            st.metric("Accuracy", f"{agg.get('accuracy', 0):.1%}")
        with m3:
            st.metric("Weighted F1", f"{agg.get('weighted_f1', 0):.4f}")
        with m4:
            st.metric("Macro F1", f"{agg.get('macro_f1', 0):.4f}")

        st.caption(
            "The model is best at distinguishing pets that will be adopted quickly "
            "from those that may take longer. Individual class predictions are "
            "approximate -- use the confidence score and probability distribution "
            "to gauge reliability."
        )

        # -- Confusion Matrix Heatmap --
        st.subheader("Confusion Matrix")
        cm = perf_data.get("confusion_matrix", [])
        cm_labels = perf_data.get("class_labels", [])

        if cm:
            fig_cm = go.Figure(go.Heatmap(
                z=cm,
                x=cm_labels,
                y=cm_labels,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig_cm.update_layout(
                xaxis_title="Predicted Class",
                yaxis_title="Actual Class",
                height=500,
                template="plotly_white",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # -- Per-class metrics table --
        st.subheader("Per-Class Metrics")
        per_class = perf_data.get("per_class_metrics", [])
        if per_class:
            table_data = []
            for m in per_class:
                table_data.append({
                    "Class": m.get("label", ""),
                    "Precision": f"{m.get('precision', 0):.4f}",
                    "Recall": f"{m.get('recall', 0):.4f}",
                    "F1 Score": f"{m.get('f1', 0):.4f}",
                    "Support": m.get("support", 0),
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)
