import streamlit as st
from src.navigation import render_top_menu
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

# Initialization: 
# channel_names section if it doesn't exist
if "channel_names" not in cfg:
    cfg["channel_names"] = {}
# feature type initialization
if "available_feature_types" not in cfg:
    cfg["available_feature_types"] = ["Lifetime_Fit", "Lifetime_FitFree", "Intensity"]

if "available_input_types" not in cfg:
    cfg["available_input_types"] = ["ROI Summing Fit", "SPCImage", "K-Flow"]
# feature type input types initialization: for each feature type, the set of input types that are required to extract that feature type
if "required_file_types" not in cfg:
    cfg["required_file_types"] = {}

if "spc_output_suffix" not in cfg:
    cfg["spc_output_suffix"] = {}
    cfg["spc_output_suffix"]["a1"] = "_a1[%].asc"
    cfg["spc_output_suffix"]["t1"] = "_t1.asc"
    cfg["spc_output_suffix"]["a2"] = "_a2[%].asc"
    cfg["spc_output_suffix"]["t2"] = "_t2.asc"
    cfg["spc_output_suffix"]["a3"] = "_a3[%].asc"
    cfg["spc_output_suffix"]["t3"] = "_t3.asc"

for feature_type in cfg["available_feature_types"]:
    if feature_type not in cfg["required_file_types"]:
        cfg["required_file_types"][feature_type] = {}
    for input_type in cfg["available_input_types"]:
        if input_type not in cfg["required_file_types"][feature_type]:
            cfg["required_file_types"][feature_type][input_type] = []
        if input_type == "K-Flow":
            cfg["required_file_types"][feature_type][input_type].append("Histogram")
            if "Lifetime" in feature_type:
                cfg["required_file_types"][feature_type][input_type].append("IRF")
        else:
            cfg["required_file_types"][feature_type][input_type].append("Mask")
            cfg["required_file_types"][feature_type][input_type].append("Decay")
            if "Lifetime" in feature_type:
                if feature_type == "Lifetime_FitFree" or input_type == "ROI Summing Fit":
                    cfg["required_file_types"][feature_type][input_type].append("IRF")
                if input_type == "SPCImage":
                    cfg["required_file_types"][feature_type][input_type].append("a1")
        cfg["required_file_types"][feature_type][input_type] = list(set(cfg["required_file_types"][feature_type][input_type]))

# input section initialization
if "inputSuffixes" not in cfg:
    cfg["inputSuffixes"] = {}
# feature type for each channel initialization
if "feature_types" not in cfg:
    cfg["feature_types"] = {}
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
        new_name = st.text_input(f"Channel {i+1} name", value=default_name)
        cfg["channel_names"][channel_key] = new_name
        
        default_feature_types = cfg["feature_types"].get(new_name, [])
        new_feature_types = st.multiselect(f"Extracted feature types from {new_name}", cfg["available_feature_types"], default=default_feature_types)
        if len(new_feature_types) == 0: 
            error_msg = f"Please select at least one feature type for {new_name}. Or you can adjust the number of channels on the top. "
            st.error(error_msg)
            continue
        cfg["feature_types"][new_name] = new_feature_types
        # get the number of components for each channel if feature types has "Lifetime"
        if new_name not in cfg["num_components"]:
            cfg["num_components"][new_name] = 0
        if any("Lifetime" in feature_type for feature_type in new_feature_types):
            num_components = st.number_input(f"Number of components for {new_name}", value=1, min_value=1, max_value=3, help="Number of components for the lifetime fit/fit free analysis")
            cfg["num_components"][new_name] = num_components
        else:
            cfg["num_components"][new_name] = 0

        # Initialize the input section for this channel if it doesn't exist
        if new_name not in cfg["inputSuffixes"]:
            cfg["inputSuffixes"][new_name] = {}

        st.subheader(f"File suffixes: {new_name}")
        asked_file_types = {} # stores key-value pairs of file type and suffix
        for feature_type in new_feature_types:
            if feature_type not in cfg["inputSuffixes"][new_name]:
                cfg["inputSuffixes"][new_name][feature_type] = {} 
            for input_type in cfg["available_input_types"]:
                if input_type not in cfg["inputSuffixes"][new_name][feature_type]:
                    cfg["inputSuffixes"][new_name][feature_type][input_type] = {}
                for required_file_type in cfg["required_file_types"][feature_type][input_type]:
                    if required_file_type in asked_file_types:
                        cfg["inputSuffixes"][new_name][feature_type][input_type][required_file_type] = asked_file_types[required_file_type]
                    else:
                        default_suffix = cfg["inputSuffixes"][new_name][feature_type][input_type].get(required_file_type, "")
                        help_msg = "Ignore this field if you don't have this file type in your data." if required_file_type == "a1" or required_file_type == "Histogram" else None
                        new_suffix = st.text_input(f"{required_file_type}", value=default_suffix, key=f"{new_name}_{feature_type}_{input_type}_{required_file_type}", help=help_msg)
                        cfg["inputSuffixes"][new_name][feature_type][input_type][required_file_type] = new_suffix
                        asked_file_types[required_file_type] = new_suffix

if error_msg == "":
    update_config_button = st.button("Update Configuration")
    if update_config_button:
        save_config(cfg)
        st.success("Configuration updated!")