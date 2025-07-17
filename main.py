import streamlit as st
from src.navigation import render_top_menu

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    
    # Render the top menu on the main page
    render_top_menu()
    left_column, center_column, right_column = st.columns([1.5, 1, 1.5])
    
    # Display the logo in the center column
    from pathlib import Path
    import sys
    
    def resource_path(rel: str) -> Path:
        """Return the absolute path to a bundled resource."""
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
        return base / rel
    
    logo_file = resource_path("logo.png")
    with center_column:
        st.image(str(logo_file))

    st.markdown("""
    <div style="font-size: 16px;">
        Welcome to <span style="font-size: 20px; font-weight: bold;">Fluorescence Lifetime Imaging Microscopy Playground!</span>.
    </div>
    """, unsafe_allow_html=True)
    
    deployed_url = "https://flim-playground.streamlit.app/"
    github_repo_url = "https://github.com/skalalab/flim_playground"
    doc_github_url = "https://github.com/skalalab/flim_playground_doc" # later update it to the deployed url
    generalInfo = f"""Please use the top menu to navigate to other pages. For detailed documentation of each playground, please refer to [documentation]({doc_github_url}). 
    This platform is open-source: code and releases for all major OS (Windows, Mac, Linux) are available at [Github Repo]({github_repo_url}). 
    The Visualization and Classification playground is also deployed on [Streamlit Cloud]({deployed_url}). """
    st.write(generalInfo)

    # configuration panel 
    st.title("Configuration")

    from src.config import load_config, save_config

    # Load the current user configuration
    cfg = load_config()
    error_msg = ""
    max_num_channels = 4
    imaging_modalities = ["FLIM"]
    all_flim_decay_input_types = ["Decay (3/4D)", "Decay (3/4D) pixel-prefitted", "Decay (2D)"]
    # in the future it will be other lists, one for each imaging modality
    spc_output_suffix = {"a1": "_a1[%].asc", "t1": "_t1.asc", "a2": "_a2[%].asc", "t2": "_t2.asc", "a3": "_a3[%].asc", "t3": "_t3.asc"}
    
    # Initialization: 
    if "flim_decay_input_types" not in cfg:
        cfg["flim_decay_input_types"] = all_flim_decay_input_types

    if "spc_output_suffix" not in cfg:
        cfg["spc_output_suffix"] = spc_output_suffix

    if "flim_decay_input_type" not in cfg:
        cfg["flim_decay_input_type"] = all_flim_decay_input_types[0]

    cols = st.columns(4)

    # Ask for the number of channels user needs
    with cols[0]:
        cfg["num_channels"] = st.selectbox("Number of channels you have in your data", list(range(1, max_num_channels + 1)), index=cfg.get("num_channels", 1) - 1, help="Number of channels you have in your data")
    with cols[1]:
        flim_decay_input_type = st.selectbox("FLIM Decay Input type", cfg["flim_decay_input_types"], index= cfg["flim_decay_input_types"].index(cfg["flim_decay_input_type"]))
        cfg["flim_decay_input_type"] = flim_decay_input_type
        if flim_decay_input_type not in cfg:
            cfg[flim_decay_input_type] = {}
        # later add input_type selection for other imaging modalities
    with cols[2]:
        cfg["unique_cell_id_col"] = st.text_input("Unique cell identifier column name", value=cfg.get("unique_cell_id_col", "cell_id"), help="Unique cell identifier column name")
    with cols[3]:
        laser_rate = st.number_input(f"Laser rate (GHz) for {flim_decay_input_type}", value=cfg.get(flim_decay_input_type, {}).get("laser_rate", 1.0), min_value=0.0, max_value=2.0, key=f"laser_rate_{flim_decay_input_type}")
        cfg[flim_decay_input_type]["laser_rate"] = laser_rate
        # feature extractor initialization
    if "available_feature_extractors" not in cfg[flim_decay_input_type]:
        if flim_decay_input_type == "Decay (2D)":
            cfg[flim_decay_input_type]["available_feature_extractors"] = ["Lifetime fit", "Lifetime fit free"]
        else:
            cfg[flim_decay_input_type]["available_feature_extractors"] = ["Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture"]

    # init file types for each input type
    for input_type in all_flim_decay_input_types:
        if "file_types" not in cfg[input_type]:
            if input_type == "Decay (3/4D)":
                cfg[input_type]["file_types"] = ["Decay", "IRF", "Mask"]
            elif input_type == "Decay (3/4D) pixel-prefitted":
                cfg[input_type]["file_types"] = ["Decay", "IRF", "Mask", "a1"]
            elif input_type == "Decay (2D)":
                cfg[input_type]["file_types"] = ["Decay", "IRF"]
    

    if "fov_name_col" not in cfg:
        if flim_decay_input_type == "Decay (2D)":
            cfg["fov_name_col"] = "exp_name"
        else:
            cfg["fov_name_col"] = "image_name"
   
    cols = st.columns(cfg["num_channels"])
   
     # check for duplicate channel names
    channel_names = []

    for i, col in enumerate(cols):
        with col:
            channel_key = f"ch{i+1}"
            if channel_key not in cfg:
                cfg[channel_key] = {}
            imaging_modality = "FLIM"
            if "imaging_modality" not in cfg[channel_key]:
                cfg[channel_key]["imaging_modality"] = imaging_modality
        #    imaging_modality = st.selectbox("Imaging modality", imaging_modalities, index=0, key=f"imaging_modality_{channel_key}")
            # get input type for this channel
            if imaging_modality == "FLIM":
                input_type = flim_decay_input_type
            cfg[channel_key]["input_type"] = input_type
            if input_type not in cfg[channel_key]:
                cfg[channel_key][input_type] = {}

            # get custom channel name
            default_name = cfg[channel_key].get("channel_name", f"Channel {i+1}")
            custom_channel_name = st.text_input(f"Channel {i+1} name", value=default_name)
            if custom_channel_name in channel_names:
                error_msg = f"Duplicate channel names found. Please change the names to be unique."
                st.error(error_msg)
                continue
            channel_names.append(custom_channel_name)
            cfg[channel_key]["channel_name"] = custom_channel_name
            # get selected feature extractors for this channel
            available_feature_extractors = cfg[input_type]["available_feature_extractors"]
            selected_feature_extractors = st.multiselect(f"Extract feature types from {custom_channel_name}", available_feature_extractors, default= cfg[channel_key][input_type].get("selected_feature_extractors", []), key=f"{input_type}_{channel_key}_feature_extractors")
            cfg[channel_key][input_type]["selected_feature_extractors"] = selected_feature_extractors
            if len(selected_feature_extractors) == 0: 
                error_msg = f"Please select at least one feature type for {custom_channel_name}. Or you can adjust the number of channels on the top. "
                st.error(error_msg)
                continue
              
            # get the number of components for each channel if Lifetime is in selected feature extractors and fit is in selected modules
            if "Lifetime fit" in selected_feature_extractors:
                num_components = st.number_input(f"Number of components for {custom_channel_name}", value=cfg[channel_key][input_type].get("num_components", 1), min_value=1, max_value=3, help="Number of components for the lifetime fit/fit free analysis")
                cfg[channel_key][input_type]["num_components"] = num_components
            # Initialize the input section for this channel if it doesn't exist
            if "input_suffixes" not in cfg[channel_key][input_type]:
                cfg[channel_key][input_type]["input_suffixes"] = {}

            st.subheader(f"File suffixes: {custom_channel_name}")
            for file_type in cfg[input_type]["file_types"]:
                # Skip a1 if no Lifetime fit extractors are selected
                if file_type == "a1" and not any("Lifetime fit" in extractor for extractor in selected_feature_extractors):
                    continue
                # Skip IRF if no Lifetime extractors OR if prefitted and no fit free extractors
                if file_type == "IRF" and (not any("Lifetime" in extractor for extractor in selected_feature_extractors) or 
                                          ("prefitted" in input_type and not any("fit free" in extractor for extractor in selected_feature_extractors))):
                    continue
                cfg[channel_key][input_type]["input_suffixes"][file_type] = st.text_input(f"{file_type}", value=cfg[channel_key][input_type]["input_suffixes"].get(file_type, ""), key=f"{channel_key}_{input_type}_{file_type}")

    if imaging_modality == "FLIM" and flim_decay_input_type == "Decay (2D)":
        cols = st.columns(2)
        with cols[0]:
            # ask for k_flow duration and time bins
            cfg[flim_decay_input_type]["duration"] = st.number_input(f"{flim_decay_input_type} duration (s)", value=cfg.get(flim_decay_input_type, {}).get("duration", 20.0), min_value=0.0, max_value=100.0, key=f"{flim_decay_input_type}_duration")
        with cols[1]:
            cfg[flim_decay_input_type]["time_bins"] = st.number_input(f"{flim_decay_input_type} time bins", value=cfg.get(flim_decay_input_type, {}).get("time_bins", 1024), min_value=256, max_value=2048, key=f"{flim_decay_input_type}_time_bins")
    if error_msg == "":
        update_config_button = st.button("Update Configuration")
        if update_config_button:
            save_config(cfg)
            st.success("Configuration updated!")

if __name__ == "__main__":
    main()