import streamlit as st
from src.navigation import render_top_menu

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    
    # Render the top menu on the main page
    render_top_menu()
    # Display the logo
    from pathlib import Path
    import sys

    def resource_path(rel: str) -> Path:
        """Return the absolute path to a bundled resource."""
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
        return base / rel

    logo_file = resource_path("logo.png")
    deployed_url = "https://flim-playground.streamlit.app/"
    github_repo_url = "https://github.com/skalalab/flim_playground"
    doc_github_url = "https://skalalab.github.io/flim_playground_doc/"
    doc_extraction_url = "https://skalalab.github.io/flim_playground_doc/data_extraction.html"
    doc_analysis_url = "https://skalalab.github.io/flim_playground_doc/data_analysis.html"

    # Load + migrate the extraction config and resolve the active profile so the
    # profile controls and the page body both read the active profile.
    from src.config import (
        load_config,
        save_config,
        _migrate_extraction_config_to_profiles,
        get_current_profile_name,
        list_profiles,
        set_current_profile,
        create_profile,
        delete_profile,
    )

    MAX_PROFILES = 10
    full_cfg = _migrate_extraction_config_to_profiles(load_config())
    active = get_current_profile_name()

    # Logo and the profile controls share one compact row; the "Configuration"
    # header sits above the three controls (sub-columns, not stacked). Quick links
    # live at the very bottom of the page, below the "Update Configuration" button.
    # On a fresh install there is no config.toml yet, so list_profiles() is
    # empty. Fall back to the active ("default") profile so the selectbox below
    # never renders with empty options (which would return None and crash on
    # set_current_profile(None)); the profile is seeded with app defaults further
    # down and persisted on the first "Update Configuration" click.
    profiles = list_profiles() or [active]
    logo_col, welcome_col, profile_col = st.columns([1.7, 1.8, 3.5], vertical_alignment="center")
    with logo_col:
        st.image(str(logo_file), width="stretch")
    with welcome_col:
        # Short welcome / orientation, sitting between the logo and the controls.
        st.markdown(
            f"**Welcome 👋 to [FLIM Playground]({doc_github_url})!** 🥳🎉🥂  \n"
            f"[**Data Extraction**]({doc_extraction_url}) pulls single-object features from raw microscopy data.  \n"
            f"[**Data Analysis**]({doc_analysis_url}) transforms tabular datasets into insights through visualization and statistical modeling."
        )
    with profile_col:
        st.markdown("##### ⚙️ Configuration")
        select_col, create_col = st.columns([1, 1], vertical_alignment="top")
        # --- Switch profile, with Delete stacked directly below it ---
        with select_col:
            selected_profile = st.selectbox(
                "Profile",
                options=profiles,
                index=profiles.index(active) if active in profiles else 0,
                key="extraction_profile_selector",
                help="Select which extraction profile to configure. Each profile is an independent setup.",
            )
            if selected_profile and selected_profile != active:
                set_current_profile(selected_profile)  # persist immediately, before rerun
                st.rerun()
            # --- Delete profile (never the last one) ---
            only_one = len(profiles) <= 1
            if st.button(
                "🗑️ Delete",
                key="delete_extraction_profile",
                disabled=only_one,
                help="Cannot delete the only profile" if only_one else f"Delete profile '{active}'",
            ):
                delete_profile(active)
                # The deleted name is no longer a valid option; drop the stored
                # selection so the switcher falls back to the new active profile.
                st.session_state.pop("extraction_profile_selector", None)
                st.rerun()
        # --- Create profile (blank; app defaults are seeded on render) ---
        with create_col:
            at_max = len(profiles) >= MAX_PROFILES
            with st.form("create_extraction_profile_form", clear_on_submit=True):
                new_profile_name = st.text_input(
                    "New profile",
                    placeholder="e.g. experiment-B",
                    key="new_extraction_profile_name",
                    disabled=at_max,
                )
                create_clicked = st.form_submit_button(
                    "➕ Create",
                    disabled=at_max,
                    help=f"Maximum {MAX_PROFILES} profiles reached" if at_max else "Create a new blank profile",
                )
                if create_clicked and new_profile_name:
                    new_profile_name = new_profile_name.strip()
                    if new_profile_name and new_profile_name not in profiles:
                        create_profile(new_profile_name)
                        # Reset the switcher so it re-derives from the new active
                        # profile next run (a stale value would flip us back).
                        st.session_state.pop("extraction_profile_selector", None)
                        st.rerun()
                    elif new_profile_name in profiles:
                        st.error(f"'{new_profile_name}' already exists!")

    # Point `cfg` at the active profile's sub-dict. The rest of the page edits it
    # in place; the final "Update Configuration" save persists the whole tree.
    full_cfg.setdefault("profiles", {}).setdefault(active, {})
    cfg = full_cfg["profiles"][active]
    error_msg = ""
    max_num_channels = 4
    all_flim_decay_input_types = ["Decay (3/4D)", "Decay (3/4D) pixel-prefitted", "Decay (2D)"]
    intensity_only_input_types = ["Intensity (2D)"]
    all_available_categorical_cols = ["experiment", "patient_id", "day", "hour", "cell_type", "media", "dish", "cell_line", "treatment", "condition", "replicate"]
    spc_output_suffix = {"a1": "_a1[%].asc", "t1": "_t1.asc", "a2": "_a2[%].asc", "t2": "_t2.asc", "a3": "_a3[%].asc", "t3": "_t3.asc"}
    all_feature_extractors = ["Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture"]
    if "all_feature_extractors" not in cfg:
        cfg["all_feature_extractors"] = all_feature_extractors
    
    # Initialization: 
    if "flim_decay_input_types" not in cfg:
        cfg["flim_decay_input_types"] = all_flim_decay_input_types

    if "categorical_cols" not in cfg:
        cfg["categorical_cols"] = all_available_categorical_cols

    if "spc_output_suffix" not in cfg:
        cfg["spc_output_suffix"] = spc_output_suffix

    if "flim_decay_input_type" not in cfg:
        cfg["flim_decay_input_type"] = all_flim_decay_input_types[0]

    # currently, the only option for intensity-only is 2D
    intensity_only_input_type = intensity_only_input_types[0]
    cfg["intensity_only_input_type"] = intensity_only_input_type
    if intensity_only_input_type not in cfg:
        cfg[intensity_only_input_type] = {}

    cols = st.columns(4)

    # Ask for the number of channels user needs
    with cols[0]:
        cfg["num_channels"] = st.selectbox("Number of channels", list(range(1, max_num_channels + 1)), index=cfg.get("num_channels", 1) - 1, help="Number of channels you have in your data", key=f"num_channels_{active}")
    with cols[1]:
        flim_decay_input_type = st.selectbox("FLIM Decay Input type", cfg["flim_decay_input_types"], index= cfg["flim_decay_input_types"].index(cfg["flim_decay_input_type"]), help=(
            "How your raw FLIM decay data is stored:\n\n"
            "- **Decay (3/4D)** — spatially-resolved decays stored as 3D/4D arrays in vendor formats (`.sdt`, `.ptu`), optionally with a channel dimension.\n"
            "- **Decay (3/4D) pixel-prefitted** — per-pixel pre-fitted SPCImage outputs (`.asc`).\n"
            "- **Decay (2D)** — a tabular CSV where each row is a cell and each column is a time bin."
        ), key=f"flim_decay_input_type_{active}")
        cfg["flim_decay_input_type"] = flim_decay_input_type
        if flim_decay_input_type not in cfg:
            cfg[flim_decay_input_type] = {}
        # later add input_type selection for other imaging modalities
    with cols[2]:
        cfg["unique_cell_id_col"] = st.text_input("Unique cell identifier column name", value=cfg.get("unique_cell_id_col", "cell_id"), help="Unique cell identifier column name", key=f"unique_cell_id_{active}")
    with cols[3]:
        cfg["fov_name_col"] = st.text_input("FOV column name", value=cfg.get("fov_name_col", "image_name"), key=f"fov_name_{active}")

    cols = st.columns(4)
    with cols[0]:
        laser_rate = st.number_input(f"Laser rate **(GHz)** for {flim_decay_input_type}", value=cfg.get(flim_decay_input_type, {}).get("laser_rate", 0.08), min_value=0.0, max_value=1.0, key=f"laser_rate_{flim_decay_input_type}_{active}")
        cfg[flim_decay_input_type]["laser_rate"] = laser_rate
    with cols[1]:
        # Get default value from config and find its index
        options = ["IRF", "Fluorescence Lifetime Standard"]
        default_value = cfg.get(flim_decay_input_type, {}).get("fit_free_calibration", "IRF")
        default_index = options.index(default_value) if default_value in options else 0
        fit_free_calibration = st.radio("Fit free calibration method", options, index=default_index, key=f"fit_free_calibration_{flim_decay_input_type}_{active}")
        cfg[flim_decay_input_type]["fit_free_calibration"] = fit_free_calibration
        if fit_free_calibration == "Fluorescence Lifetime Standard":
            with cols[3]:
                st.caption("Provide channel-specific Fluorescence lifetime standard file suffixes below in the File suffixes section.")
            with cols[2]:
                # get the fluorescence lifetime standard's lifetime (shared across channels)
                cfg[flim_decay_input_type]["fluorescence_lifetime_standard_lifetime"] = st.number_input("Fluorescence lifetime standard's lifetime **(ns)**", value=cfg.get(flim_decay_input_type, {}).get("fluorescence_lifetime_standard_lifetime", 1.0), min_value=0.1, max_value=20.0, key=f"fluorescence_lifetime_standard_lifetime_{flim_decay_input_type}_{active}")
       
        # feature extractor initialization
    if "available_feature_extractors" not in cfg[flim_decay_input_type]:
        if flim_decay_input_type == "Decay (2D)":
            cfg[flim_decay_input_type]["available_feature_extractors"] = ["Lifetime fit", "Lifetime fit free"]
        else:
            cfg[flim_decay_input_type]["available_feature_extractors"] = ["Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture"]
    if "available_feature_extractors" not in cfg[intensity_only_input_type]:
        cfg[intensity_only_input_type]["available_feature_extractors"] = ["Intensity morphology", "Intensity texture"]

    if flim_decay_input_type == "Decay (2D)":
        imaging_modalities = ["FLIM"]
    else:
        imaging_modalities = ["FLIM", "Intensity-only"]

    # init file types for each input type (more inclusive, will exclude some file types later based on the selected feature extractors)
    for input_type in all_flim_decay_input_types + intensity_only_input_types:
        if input_type not in cfg:
            cfg[input_type] = {}
        if "file_types" not in cfg[input_type]:
            if input_type == "Decay (3/4D)":
                cfg[input_type]["file_types"] = ["Decay", "IRF", "Mask",]
            elif input_type == "Decay (3/4D) pixel-prefitted":
                cfg[input_type]["file_types"] = ["Decay", "IRF", "Mask", "SPCImage t1"]
            elif input_type == "Decay (2D)":
                cfg[input_type]["file_types"] = ["Decay", "IRF"]
            elif input_type == "Intensity (2D)":
                cfg[input_type]["file_types"] = ["Intensity (2D)", "Mask"]
    
    cols = st.columns(cfg["num_channels"])
   
     # check for duplicate channel names
    channel_names = []
    for i, col in enumerate(cols):
        with col:
            channel_key = f"ch{i+1}"
            if channel_key not in cfg:
                cfg[channel_key] = {}
            imaging_modality = st.selectbox("Imaging modality", imaging_modalities, index=0, key=f"imaging_modality_{channel_key}_{active}")
            # get input type for this channel
            cfg[channel_key]["imaging_modality"] = imaging_modality
            if imaging_modality == "FLIM":
                input_type = flim_decay_input_type
            elif imaging_modality == "Intensity-only":
                input_type = intensity_only_input_type
            cfg[channel_key]["input_type"] = input_type
            if input_type not in cfg[channel_key]:
                cfg[channel_key][input_type] = {}

            # get custom channel name
            default_name = cfg[channel_key].get("channel_name", f"Channel {i+1}")
            custom_channel_name = st.text_input(f"Channel {i+1} name", value=default_name, key=f"channel_name_{channel_key}_{active}")
            if custom_channel_name in channel_names:
                error_msg = "Duplicate channel names found. Please change the names to be unique."
                st.error(error_msg)
                continue
            channel_names.append(custom_channel_name)
            cfg[channel_key]["channel_name"] = custom_channel_name
            # get selected feature extractors for this channel
            available_feature_extractors = cfg[input_type]["available_feature_extractors"]
            selected_feature_extractors = st.multiselect(f"Extract feature types from {custom_channel_name}", available_feature_extractors, default= cfg[channel_key][input_type].get("selected_feature_extractors", []), key=f"{input_type}_{channel_key}_feature_extractors_{active}")
            cfg[channel_key][input_type]["selected_feature_extractors"] = selected_feature_extractors
            if len(selected_feature_extractors) == 0: 
                error_msg = f"Please select at least one feature type for {custom_channel_name}. Or you can adjust the number of channels on the top. "
                st.error(error_msg)
                continue
              
            # get the number of components for each channel if Lifetime is in selected feature extractors and fit is in selected modules
            if "Lifetime fit" in selected_feature_extractors:
                num_components = st.number_input(f"Number of components for {custom_channel_name}", value=cfg[channel_key][input_type].get("num_components", 1), min_value=1, max_value=3, help="Number of components for the lifetime fit/fit free analysis", key=f"num_components_{channel_key}_{input_type}_{active}")
                cfg[channel_key][input_type]["num_components"] = num_components
                # Fixed-lifetime defaults (per component)
                if num_components > 1:
                    if "fixed_lifetimes" not in cfg[channel_key][input_type]:
                        cfg[channel_key][input_type]["fixed_lifetimes"] = {}
                    st.caption("Fix lifetime components (ns) — set 0 to fit freely:")
                    fix_cols = st.columns(num_components)
                    for comp_i in range(1, num_components + 1):
                        t_key = f"t{comp_i}"
                        existing = cfg[channel_key][input_type]["fixed_lifetimes"].get(t_key, 0.0) or 0.0
                        with fix_cols[comp_i - 1]:
                            fixed_val = st.number_input(
                                f"Fix τ{comp_i} (ns)",
                                value=float(existing),
                                min_value=0.0,
                                max_value=100.0,
                                step=0.01,
                                format="%.3f",
                                key=f"{channel_key}_{input_type}_fixed_t{comp_i}_{active}",
                                help=f"Set > 0 to fix τ{comp_i} to this value. 0 = free parameter."
                            )
                            cfg[channel_key][input_type]["fixed_lifetimes"][t_key] = fixed_val if fixed_val > 0 else None
                else:
                    # 1-component: no fixing needed, clear any stale config
                    cfg[channel_key][input_type]["fixed_lifetimes"] = {}
            # Initialize the input section for this channel if it doesn't exist
            if "input_suffixes" not in cfg[channel_key][input_type]:
                cfg[channel_key][input_type]["input_suffixes"] = {}

            st.subheader(f"File suffixes: {custom_channel_name}")
            for file_type in cfg[input_type]["file_types"]:
                if file_type == "Decay" and "prefitted" in input_type and len(selected_feature_extractors) == 1 and "Lifetime fit" in selected_feature_extractors:
                    continue
                # Skip t1 if no Lifetime fit extractors are selected
                if file_type == "SPCImage t1" and "Lifetime fit" not in selected_feature_extractors:
                    continue
                # Skip IRF if no Lifetime extractors OR if prefitted and no fit free extractors
                if file_type == "IRF" and (not any("Lifetime" in extractor for extractor in selected_feature_extractors) or 
                                          ("prefitted" in input_type and "Lifetime fit free" not in selected_feature_extractors)):
                    continue

                if file_type == "IRF" and ("Lifetime fit" not in selected_feature_extractors or "prefitted" in input_type) and "Lifetime fit free" in selected_feature_extractors and fit_free_calibration == "Fluorescence Lifetime Standard":
                    continue

                cfg[channel_key][input_type]["input_suffixes"][file_type] = st.text_input(f"{file_type}", value=cfg[channel_key][input_type]["input_suffixes"].get(file_type, ""), key=f"{channel_key}_{input_type}_{file_type}_{active}")

            # If using Fluorescence lifetime standard calibration for fit free on this channel, ask for channel-specific Fluorescence lifetime standard file suffix
            if "Lifetime fit free" in selected_feature_extractors and fit_free_calibration == "Fluorescence Lifetime Standard":
                cfg[channel_key][input_type]["input_suffixes"]["Fluorescence Lifetime Standard"] = st.text_input(
                    "Fluorescence lifetime standard file",
                    value=cfg[channel_key][input_type]["input_suffixes"].get("Fluorescence Lifetime Standard", ""),
                    key=f"{channel_key}_{input_type}_FluorescenceLifetimeStandard_{active}"
                )

    if imaging_modality == "FLIM" and flim_decay_input_type == "Decay (2D)":
        cols = st.columns(2)
        with cols[0]:
            # ask for k_flow duration and time bins
            cfg[flim_decay_input_type]["duration"] = st.number_input(f"{flim_decay_input_type} duration (**ns**)", value=cfg.get(flim_decay_input_type, {}).get("duration", 20.0), min_value=0.0, max_value=100.0, key=f"{flim_decay_input_type}_duration_{active}")
        with cols[1]:
            cfg[flim_decay_input_type]["time_bins"] = st.number_input(f"{flim_decay_input_type} time bins", value=cfg.get(flim_decay_input_type, {}).get("time_bins", 1024), min_value=10, key=f"{flim_decay_input_type}_time_bins_{active}")
      
       
    # render a multiselect for categorical columns
    categorical_cols = st.multiselect("Categorical columns (type to add more)", cfg.get("categorical_cols", []), default=cfg.get("categorical_cols", []),  accept_new_options=True, key=f"categorical_cols_{active}")
    cfg["categorical_cols"] = categorical_cols

    # Check if we should show a success message from previous update
    if st.session_state.get("config_updated", False):
        st.success("Configuration updated!")
        # Clear the flag so message doesn't persist indefinitely
        st.session_state.config_updated = False

    if error_msg == "":
        update_config_button = st.button("Update Configuration")
        if update_config_button:
            # `cfg` is full_cfg["profiles"][active] by reference, so all in-place
            # edits are already captured; persist the whole profile tree.
            save_config(full_cfg)
            # Set flag to show success message after rerun
            st.session_state.config_updated = True
            st.rerun()

    # Quick links, pinned to the bottom of the page on a single row.
    st.divider()
    st.markdown(
        f"[Documentation]({doc_github_url}) &nbsp;·&nbsp; "
        f"[GitHub Repo]({github_repo_url}) &nbsp;·&nbsp; "
        f"[Streamlit Cloud]({deployed_url}) *(Data Analysis only)*"
    )

if __name__ == "__main__":
    main()