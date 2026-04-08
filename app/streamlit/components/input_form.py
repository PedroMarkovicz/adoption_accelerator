"""Pet profile input form component.

Renders all 6 sections of the form and returns a dict that matches
the PetProfileRequest fields accepted by the FastAPI /predict endpoint.
All widget values are persisted in st.session_state via the key= parameter.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")

_SIZE_LABELS = {1: "Small", 2: "Medium", 3: "Large", 4: "Extra Large"}
_FUR_LABELS  = {1: "Short", 2: "Medium", 3: "Long"}
_HEALTH_OPTS = ["Yes", "No", "Not Sure"]
_HEALTH_COND = ["Healthy", "Minor Injury", "Serious Injury"]


@st.cache_data
def _load_breeds() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_ASSETS_DIR, "breed_labels.csv"))


@st.cache_data
def _load_colors() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_ASSETS_DIR, "color_labels.csv"))


@st.cache_data
def _load_states() -> pd.DataFrame:
    return pd.read_csv(os.path.join(_ASSETS_DIR, "state_labels.csv"))


def _init_session_state() -> None:
    defaults: dict = {
        "form_pet_type":     "Dog",
        "form_name":         "",
        "form_age_months":   6,
        "form_gender":       "Male",
        "form_breed1":       0,
        "form_breed2":       0,
        "form_color1":       0,
        "form_color2":       0,
        "form_color3":       0,
        "form_maturity_size": 2,
        "form_fur_length":   1,
        "form_vaccinated":   "Not Sure",
        "form_dewormed":     "Not Sure",
        "form_sterilized":   "Not Sure",
        "form_health":       "Healthy",
        "form_fee":          0.0,
        "form_quantity":     1,
        "form_state":        0,
        "form_video_amt":    0,
        "form_description":  "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _on_pet_type_change() -> None:
    """Reset breed selections when pet type switches between Dog and Cat."""
    st.session_state["form_breed1"] = 0
    st.session_state["form_breed2"] = 0


def render_input_form() -> tuple[dict | None, list[tuple[str, bytes]], bool]:
    """Render the pet profile form.

    Returns:
        (profile_dict, images, submitted)
        - profile_dict: None when not submitted.
        - images: list of (filename, raw_bytes) for uploaded photos.
        - submitted: True when the user pressed the predict button.
    """
    _init_session_state()

    breeds_df = _load_breeds()
    colors_df = _load_colors()
    states_df = _load_states()

    # Build breed option lists: [(id, name), ...]
    dog_breed_opts = [(0, "Mixed / Unknown")] + [
        (int(r["BreedID"]), r["BreedName"])
        for _, r in breeds_df[breeds_df["Type"] == 1].iterrows()
    ]
    cat_breed_opts = [(0, "Mixed / Unknown")] + [
        (int(r["BreedID"]), r["BreedName"])
        for _, r in breeds_df[breeds_df["Type"] == 2].iterrows()
    ]

    color_opts = [(0, "None / Unspecified")] + [
        (int(r["ColorID"]), r["ColorName"]) for _, r in colors_df.iterrows()
    ]
    state_opts = [(0, "Unspecified")] + [
        (int(r["StateID"]), r["StateName"]) for _, r in states_df.iterrows()
    ]

    # --- Section 1: Basic Info ---
    st.subheader("Basic Information")
    st.radio(
        "Pet Type",
        options=["Dog", "Cat"],
        horizontal=True,
        key="form_pet_type",
        on_change=_on_pet_type_change,
    )
    col_name, col_age, col_gender = st.columns(3)
    with col_name:
        st.text_input("Name", key="form_name")
    with col_age:
        st.number_input("Age (months)", min_value=0, max_value=255, step=1, key="form_age_months")
    with col_gender:
        st.selectbox("Gender", options=["Male", "Female", "Mixed"], key="form_gender")

    # --- Section 2: Appearance ---
    st.subheader("Appearance")
    breed_opts = dog_breed_opts if st.session_state.form_pet_type == "Dog" else cat_breed_opts
    breed_ids   = [b[0] for b in breed_opts]
    breed_names = {b[0]: b[1] for b in breed_opts}

    # Guard: if stored breed ID is invalid for current pet type, reset.
    if st.session_state.form_breed1 not in breed_ids:
        st.session_state.form_breed1 = 0
    if st.session_state.form_breed2 not in breed_ids:
        st.session_state.form_breed2 = 0

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.selectbox(
            "Primary Breed",
            options=breed_ids,
            format_func=lambda x: breed_names.get(x, str(x)),
            key="form_breed1",
        )
    with col_b2:
        st.selectbox(
            "Secondary Breed (optional)",
            options=breed_ids,
            format_func=lambda x: breed_names.get(x, str(x)),
            key="form_breed2",
        )

    color_ids   = [c[0] for c in color_opts]
    color_names = {c[0]: c[1] for c in color_opts}
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.selectbox(
            "Primary Color",
            options=color_ids,
            format_func=lambda x: color_names.get(x, str(x)),
            key="form_color1",
        )
    with col_c2:
        st.selectbox(
            "Secondary Color",
            options=color_ids,
            format_func=lambda x: color_names.get(x, str(x)),
            key="form_color2",
        )
    with col_c3:
        st.selectbox(
            "Tertiary Color",
            options=color_ids,
            format_func=lambda x: color_names.get(x, str(x)),
            key="form_color3",
        )

    col_sz, col_fur = st.columns(2)
    with col_sz:
        st.selectbox(
            "Maturity Size",
            options=[1, 2, 3, 4],
            format_func=lambda x: _SIZE_LABELS[x],
            key="form_maturity_size",
        )
    with col_fur:
        st.selectbox(
            "Fur Length",
            options=[1, 2, 3],
            format_func=lambda x: _FUR_LABELS[x],
            key="form_fur_length",
        )

    # --- Section 3: Health ---
    st.subheader("Health")
    col_v, col_d, col_s = st.columns(3)
    with col_v:
        st.selectbox("Vaccinated", options=_HEALTH_OPTS, key="form_vaccinated")
    with col_d:
        st.selectbox("Dewormed", options=_HEALTH_OPTS, key="form_dewormed")
    with col_s:
        st.selectbox("Sterilized", options=_HEALTH_OPTS, key="form_sterilized")
    st.selectbox("Health Condition", options=_HEALTH_COND, key="form_health")

    # --- Section 4: Listing Details ---
    st.subheader("Listing Details")
    col_fee, col_qty, col_vid = st.columns(3)
    with col_fee:
        st.number_input("Adoption Fee (MYR)", min_value=0.0, step=10.0, key="form_fee")
    with col_qty:
        st.number_input("Quantity", min_value=1, step=1, key="form_quantity")
    with col_vid:
        st.number_input("Video Count", min_value=0, step=1, key="form_video_amt")

    state_ids   = [s[0] for s in state_opts]
    state_names = {s[0]: s[1] for s in state_opts}
    st.selectbox(
        "State (Malaysia)",
        options=state_ids,
        format_func=lambda x: state_names.get(x, str(x)),
        key="form_state",
    )

    # --- Section 5: Description ---
    st.subheader("Description")
    description = st.text_area(
        "Pet description",
        key="form_description",
        height=120,
        placeholder="Describe the pet's personality, history, or any relevant details...",
    )
    st.caption(f"{len(st.session_state.form_description)} characters")

    # --- Section 6: Photos ---
    st.subheader("Photos")
    uploaded_files = st.file_uploader(
        "Upload pet photos (optional)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Photos improve prediction accuracy via image feature extraction.",
    )
    photo_count = len(uploaded_files) if uploaded_files else 0
    if photo_count > 0:
        st.caption(f"{photo_count} photo(s) selected")

    submitted = st.button(
        "Predict Adoption Speed",
        type="primary",
        use_container_width=True,
    )

    if not submitted:
        return None, [], False

    # Capture image bytes before they are discarded
    images: list[tuple[str, bytes]] = []
    if uploaded_files:
        for uf in uploaded_files:
            raw = uf.read()
            if raw:
                images.append((uf.name, raw))

    profile = {
        "pet_type":      st.session_state.form_pet_type,
        "name":          st.session_state.form_name,
        "age_months":    int(st.session_state.form_age_months),
        "gender":        st.session_state.form_gender,
        "breed1":        int(st.session_state.form_breed1),
        "breed2":        int(st.session_state.form_breed2),
        "color1":        int(st.session_state.form_color1),
        "color2":        int(st.session_state.form_color2),
        "color3":        int(st.session_state.form_color3),
        "maturity_size": int(st.session_state.form_maturity_size),
        "fur_length":    int(st.session_state.form_fur_length),
        "vaccinated":    st.session_state.form_vaccinated,
        "dewormed":      st.session_state.form_dewormed,
        "sterilized":    st.session_state.form_sterilized,
        "health":        st.session_state.form_health,
        "fee":           float(st.session_state.form_fee),
        "quantity":      int(st.session_state.form_quantity),
        "state":         int(st.session_state.form_state),
        "video_amt":     int(st.session_state.form_video_amt),
        "description":   st.session_state.form_description,
    }

    return profile, images, True
