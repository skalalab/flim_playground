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
    available_feature_extractors = {"Lifetime": ["fit", "fit free"], "Intensity": ["morphology", "texture"]}
    available_input_types = ["ROI Summing Fit", "SPCImage", "K-Flow"]
    available_file_types = ["Mask", "Decay", "IRF", "Histogram", "a1"]
    spc_output_suffix = {"a1": "_a1[%].asc", "t1": "_t1.asc", "a2": "_a2[%].asc", "t2": "_t2.asc", "a3": "_a3[%].asc", "t3": "_t3.asc"}
    
    # Initialization: 
    # channel_names section if it doesn't exist
    if "channel_names" not in cfg:
        cfg["channel_names"] = {}
    # feature extractor initialization
    if "available_feature_extractors" not in cfg:
        cfg["available_feature_extractors"] = available_feature_extractors

    if "image_name_col" not in cfg:
        cfg["image_name_col"] = "image_name"

    if "laser_rate" not in cfg:
        cfg["laser_rate"] = {}

    if "available_input_types" not in cfg:
        cfg["available_input_types"] = available_input_types

    if "spc_output_suffix" not in cfg:
        cfg["spc_output_suffix"] = spc_output_suffix

    if "available_file_types" not in cfg:
        cfg["available_file_types"] = available_file_types

    # input section initialization
    if "inputSuffixes" not in cfg:
        cfg["inputSuffixes"] = {}
    # feature type for each channel initialization
    if "feature_extractors" not in cfg:
        cfg["feature_extractors"] = {}
    # num_components for each channel initialization
    if "num_components" not in cfg:
        cfg["num_components"] = {}

    col1, col2, col3 = st.columns(3)
    # Ask for the number of channels user needs
    with col1:
        cfg["num_channels"] = st.selectbox("Number of channels you have in your data", list(range(1, max_num_channels + 1)), index=cfg.get("num_channels", 1) - 1, help="Number of channels you have in your data")
    with col2:
        cfg["unique_cell_id_col"] = st.text_input("Unique cell identifier column name", value=cfg.get("unique_cell_id_col", "cell_id"), help="Unique cell identifier column name")
    with col3:
        prev_input_type = cfg.get("preferred_input_type", None)
        if prev_input_type is None:
            index = 0
        else:
            index = cfg["available_input_types"].index(prev_input_type)
        cfg["preferred_input_type"] = st.selectbox("Preferred input type", cfg["available_input_types"], index=index, help="Preferred input type")

    # Ask for the name for each channel
    cols = st.columns(cfg["num_channels"])
    for i, col in enumerate(cols):
        with col:
            channel_key = f"ch{i+1}"
            default_name = cfg.get("channel_names", {}).get(channel_key, f"Channel {i+1}")
            custom_channel_name = st.text_input(f"Channel {i+1} name", value=default_name)
            cfg["channel_names"][channel_key] = custom_channel_name
            
            default_feature_extractors = cfg["feature_extractors"].get(custom_channel_name, [])
            selected_feature_extractors = st.multiselect(f"Extract feature types from {custom_channel_name}", cfg["available_feature_extractors"].keys(), default=default_feature_extractors)
            if len(selected_feature_extractors) == 0: 
                error_msg = f"Please select at least one feature type for {custom_channel_name}. Or you can adjust the number of channels on the top. "
                st.error(error_msg)
                continue
            if custom_channel_name not in cfg["feature_extractors"]:
                cfg["feature_extractors"][custom_channel_name] = {}

            cols = st.columns(len(selected_feature_extractors))
            for i, col in enumerate(cols):
                with col:
                    feature_extractor = selected_feature_extractors[i]
                    # get available modules for each feature extractor
                    available_modules = cfg["available_feature_extractors"][feature_extractor]
                    default_modules = cfg["feature_extractors"][custom_channel_name].get(feature_extractor, [])
                    selected_modules = st.multiselect(f"Extract modules from {feature_extractor}", available_modules, default=default_modules, key=f"{custom_channel_name}_{feature_extractor}")
                    if len(selected_modules) == 0:
                        error_msg = f"Please select at least one module for {feature_extractor}. Or you can adjust the number of channels on the top. "
                        st.error(error_msg)
                        continue
                    cfg["feature_extractors"][custom_channel_name][feature_extractor] = selected_modules
            
            # remove feature extractors that are not in selected_feature_extractors
            for feature_extractor in list(cfg["feature_extractors"][custom_channel_name].keys()):
                if feature_extractor not in selected_feature_extractors:
                    cfg["feature_extractors"][custom_channel_name].pop(feature_extractor)
            # get the number of components for each channel if Lifetime is in selected feature extractors and fit is in selected modules
            if "Lifetime" in cfg["feature_extractors"][custom_channel_name] and "fit" in cfg["feature_extractors"][custom_channel_name]["Lifetime"]:
                num_components = st.number_input(f"Number of components for {custom_channel_name}", value=cfg.get("num_components", {}).get(custom_channel_name, 1), min_value=1, max_value=3, help="Number of components for the lifetime fit/fit free analysis")
                cfg["num_components"][custom_channel_name] = num_components
            else:
                cfg["num_components"][custom_channel_name] = 0

            # Initialize the input section for this channel if it doesn't exist
            if custom_channel_name not in cfg["inputSuffixes"]:
                cfg["inputSuffixes"][custom_channel_name] = {}

            asked_file_types = {}
            st.subheader(f"File suffixes: {custom_channel_name}")
            for input_type in cfg["available_input_types"]:
                if input_type not in cfg["inputSuffixes"][custom_channel_name]:
                    cfg["inputSuffixes"][custom_channel_name][input_type] = {}
                if input_type == "ROI Summing Fit" or input_type == "SPCImage":
                    if "Mask" not in asked_file_types:
                        cfg["inputSuffixes"][custom_channel_name][input_type]["Mask"] = st.text_input(f"Mask", value=cfg["inputSuffixes"][custom_channel_name][input_type].get("Mask", ""), key=f"{custom_channel_name}_{input_type}_mask")
                        asked_file_types["Mask"] = cfg["inputSuffixes"][custom_channel_name][input_type]["Mask"]
                    else:
                        cfg["inputSuffixes"][custom_channel_name][input_type]["Mask"] = asked_file_types["Mask"]
                    if "Decay" not in asked_file_types:
                        cfg["inputSuffixes"][custom_channel_name][input_type]["Decay"] = st.text_input(f"Decay", value=cfg["inputSuffixes"][custom_channel_name][input_type].get("Decay", ""), key=f"{custom_channel_name}_{input_type}_decay")
                        asked_file_types["Decay"] = cfg["inputSuffixes"][custom_channel_name][input_type]["Decay"]
                    else:
                        cfg["inputSuffixes"][custom_channel_name][input_type]["Decay"] = asked_file_types["Decay"]
                    if "Lifetime" in cfg["feature_extractors"][custom_channel_name]:
                        if input_type == "SPCImage":
                            if "a1" not in asked_file_types:
                                cfg["inputSuffixes"][custom_channel_name][input_type]["a1"] = st.text_input(f"a1", value=cfg["inputSuffixes"][custom_channel_name][input_type].get("a1", ""), key=f"{custom_channel_name}_{input_type}_a1")
                                asked_file_types["a1"] = cfg["inputSuffixes"][custom_channel_name][input_type]["a1"]
                            else:
                                cfg["inputSuffixes"][custom_channel_name][input_type]["a1"] = asked_file_types["a1"]
                            if "fit free" in cfg["feature_extractors"][custom_channel_name]["Lifetime"]:
                                if "IRF" not in asked_file_types:
                                    cfg["inputSuffixes"][custom_channel_name][input_type]["IRF"] = st.text_input(f"IRF", value=cfg["inputSuffixes"][custom_channel_name][input_type].get("IRF", ""), key=f"{custom_channel_name}_{input_type}_irf")
                                    asked_file_types["IRF"] = cfg["inputSuffixes"][custom_channel_name][input_type]["IRF"]
                                else:
                                    cfg["inputSuffixes"][custom_channel_name][input_type]["IRF"] = asked_file_types["IRF"]
                            # spc image fit only does not need irf
                        else:
                            if "IRF" not in asked_file_types:
                                cfg["inputSuffixes"][custom_channel_name][input_type]["IRF"] = st.text_input(f"IRF", value=cfg["inputSuffixes"][custom_channel_name][input_type].get("IRF", ""), key=f"{custom_channel_name}_{input_type}_irf")
                                asked_file_types["IRF"] = cfg["inputSuffixes"][custom_channel_name][input_type]["IRF"]
                            else:
                                cfg["inputSuffixes"][custom_channel_name][input_type]["IRF"] = asked_file_types["IRF"]

                elif input_type == "K-Flow":
                    if "Histogram" not in asked_file_types:
                        cfg["inputSuffixes"][custom_channel_name][input_type]["Histogram"] = st.text_input(f"Histogram", value=cfg["inputSuffixes"][custom_channel_name][input_type].get("Histogram", ""), key=f"{custom_channel_name}_{input_type}_histogram")
                        asked_file_types["Histogram"] = cfg["inputSuffixes"][custom_channel_name][input_type]["Histogram"]
                    else:
                        cfg["inputSuffixes"][custom_channel_name][input_type]["Histogram"] = asked_file_types["Histogram"]
                    if "Lifetime" in cfg["feature_extractors"][custom_channel_name]:
                        if "IRF" not in asked_file_types:
                            cfg["inputSuffixes"][custom_channel_name][input_type]["IRF"] = st.text_input(f"IRF", value=cfg["inputSuffixes"][custom_channel_name][input_type].get("IRF", ""), key=f"{custom_channel_name}_{input_type}_irf")
                            asked_file_types["IRF"] = cfg["inputSuffixes"][custom_channel_name][input_type]["IRF"]
                        else:
                            cfg["inputSuffixes"][custom_channel_name][input_type]["IRF"] = asked_file_types["IRF"]
    # laser rate 
    cols = st.columns(len(available_input_types))
    for i, col in enumerate(cols):
        with col:
            input_type = available_input_types[i]   
            cfg["laser_rate"][input_type] = st.number_input(f"Laser rate (GHz) for {input_type}", value=cfg.get("laser_rate", {}).get(input_type, 1.0), min_value=0.0, max_value=2.0, key=f"laser_rate_{input_type}")
    cols = st.columns(2)
    with cols[0]:
        # ask for k_flow duration and time bins
        cfg["k_flow_duration"] = st.number_input(f"K-Flow duration (s)", value=cfg.get("k_flow_duration", 20.0), min_value=0.0, max_value=100.0, key="k_flow_duration")
    with cols[1]:
        cfg["k_flow_time_bins"] = st.number_input(f"K-Flow time bins", value=cfg.get("k_flow_time_bins", 1024), min_value=256, max_value=2048, key="k_flow_time_bins")
    if error_msg == "":
        update_config_button = st.button("Update Configuration")
        if update_config_button:
            save_config(cfg)
            st.success("Configuration updated!")

if __name__ == "__main__":
    main()