"""Channel layout and shared FLIM controls on the Configuration page."""

import streamlit as st

from src.config import DEFAULT_2D_DECAY_DURATION_NS, default_laser_rate
from src.widgets.laser_rate_widget import laser_rate_input


_FLIM_SETTINGS_KEY = "_extraction_flim_settings"
_SHARED_FIELDS = (
    "laser_rate", "fit_free_calibration", "fluorescence_lifetime_standard_lifetime",
    "duration", "time_bins",
)


def channel_columns(count):
    """Use the same channel grouping for setup and extraction settings."""
    if count <= 4:
        return list(st.columns(count))
    with st.expander("Channels 1–4", expanded=False):
        first_group = st.columns(4)
    second_label = "Channel 5" if count == 5 else f"Channels 5–{count}"
    with st.expander(second_label, expanded=True):
        second_group = st.columns(count - 4)
    return list(first_group) + list(second_group)


def forget_shared_flim_settings(profile):
    """A recreated profile must start with fresh shared settings."""
    st.session_state.get(_FLIM_SETTINGS_KEY, {}).pop(profile, None)


def render_shared_flim_settings(cfg, profile, has_flim):
    """Edit shared settings, retaining hidden values in the profile's config.

    Widget state disappears when the section, input format, or profile changes.
    Keep only these shared fields in non-widget state, and restore them into cfg
    even while hidden so Update Configuration also saves their pending values.
    """
    drafts = st.session_state.setdefault(_FLIM_SETTINGS_KEY, {})
    draft = drafts.setdefault(profile, {})
    if "flim_decay_input_type" in draft:
        cfg["flim_decay_input_type"] = draft["flim_decay_input_type"]
    for input_type, fields in draft.get("formats", {}).items():
        cfg.setdefault(input_type, {}).update(fields)

    input_type = cfg["flim_decay_input_type"]
    if has_flim:
        # These controls precede the extractor pickers. Read their pending widget
        # values so format and extractor changes affect visibility on this rerun.
        input_type = st.session_state.get(f"flim_decay_input_type_{profile}", input_type)
        has_fit_free = False
        for i in range(cfg["num_channels"]):
            channel_key = f"ch{i+1}"
            channel = cfg[channel_key]
            if channel["imaging_modality"] != "FLIM":
                continue
            extractors = st.session_state.get(
                f"{input_type}_{channel_key}_feature_extractors_{profile}",
                channel.get(input_type, {}).get("selected_feature_extractors", []),
            )
            if "Lifetime fit free" in extractors:
                has_fit_free = True
                break
        with st.container(border=True):
            st.subheader("Shared FLIM settings", help="Applies to all FLIM channels.")
            cols = st.columns(3 if has_fit_free else 1)
            with cols[0]:
                input_type = st.selectbox(
                    "FLIM input format", cfg["flim_decay_input_types"],
                    index=cfg["flim_decay_input_types"].index(input_type),
                    help=(
                        "How your FLIM data is supplied:\n\n"
                        "- **Decay (3/4D)** — spatially-resolved decays stored as 3D/4D arrays in vendor formats (`.sdt`, `.ptu`), optionally with a channel dimension.\n"
                        "- **Decay (3/4D) pixel-prefitted** — per-pixel pre-fitted SPCImage outputs (`.asc`).\n"
                        "- **Decay (2D)** — a tabular CSV where each row is a cell and each column is a time bin."
                    ),
                    key=f"flim_decay_input_type_{profile}",
                )
            settings = cfg.setdefault(input_type, {})
            if has_fit_free:
                with cols[1]:
                    settings["laser_rate"] = laser_rate_input(
                        f"Laser rate **(MHz)** for {input_type}",
                        default_laser_rate(input_type, settings),
                        key=f"laser_rate_{input_type}_{profile}",
                    )
                with cols[2]:
                    options = ["IRF", "Fluorescence Lifetime Standard"]
                    method = settings.get("fit_free_calibration", "IRF")
                    settings["fit_free_calibration"] = st.radio(
                        "Fit free calibration method", options,
                        index=options.index(method) if method in options else 0,
                        key=f"fit_free_calibration_{input_type}_{profile}",
                    )
                    if settings["fit_free_calibration"] == "Fluorescence Lifetime Standard":
                        settings["fluorescence_lifetime_standard_lifetime"] = st.number_input(
                            "Fluorescence lifetime standard's lifetime **(ns)**",
                            value=settings.get("fluorescence_lifetime_standard_lifetime", 1.0),
                            min_value=0.1, max_value=20.0,
                            key=f"fluorescence_lifetime_standard_lifetime_{input_type}_{profile}",
                        )
                        st.caption("Provide channel-specific Fluorescence lifetime standard file suffixes below in the File suffixes section.")
            if input_type == "Decay (2D)":
                cols = st.columns(2)
                with cols[0]:
                    settings["duration"] = st.number_input(
                        f"{input_type} duration (**ns**)",
                        value=settings.get("duration", DEFAULT_2D_DECAY_DURATION_NS), min_value=0.0, max_value=100.0,
                        key=f"{input_type}_duration_{profile}",
                    )
                with cols[1]:
                    settings["time_bins"] = st.number_input(
                        f"{input_type} time bins", value=settings.get("time_bins", 1024),
                        min_value=10, key=f"{input_type}_time_bins_{profile}",
                    )

    cfg["flim_decay_input_type"] = input_type
    cfg.setdefault(input_type, {})
    draft["flim_decay_input_type"] = input_type
    draft["formats"] = {
        kind: {key: cfg[kind][key] for key in _SHARED_FIELDS if key in cfg[kind]}
        for kind in cfg["flim_decay_input_types"] if kind in cfg
    }
    return input_type, cfg[input_type].get("fit_free_calibration", "IRF")
