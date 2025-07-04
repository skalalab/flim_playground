import streamlit as st
from src.navigation import render_top_menu, titles
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
logo_file = resource_path("logo/FP_trans_320.png")
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

# Ask for channel names, using stored values as defaults if present
channels = ["blue", "green", "red"]
col1, col2, col3 = st.columns(3)

for i, channel in enumerate(channels):
    with [col1, col2, col3][i]:
        cfg["channel_name"][channel] = st.text_input(f"Enter the name of the {channel} channel", value=cfg.get("channel_name", {}).get(channel, "nadh"), key=f"channel_{channel}")

cfg["file_suffix"]["mask"] = st.text_input("Enter the suffix of the mask file", value=cfg.get("file_suffix", {}).get("mask", "_mask.tiff"))
for channel in channels:
    channel_name = cfg["channel_name"][channel]
    st.subheader(f"Suffix for input files of {channel_name} channel")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cfg["file_suffix"][f"{channel}_irf"] = st.text_input(f"IRF file suffix", value=cfg.get("file_suffix", {}).get(f"{channel}_irf", f"{channel}_irf.txt"), key=f"{channel}_irf")
    with col2:
        cfg["file_suffix"][f"{channel}_decay"] = st.text_input(f"Decay file suffix", value=cfg.get("file_suffix", {}).get(f"{channel}_decay", f"{channel}.sdt"), key=f"{channel}_decay")
    with col3:
        cfg["file_suffix"][f"{channel}_histogram"] = st.text_input(f"Histogram file suffix", value=cfg.get("file_suffix", {}).get(f"{channel}_histogram", f"{channel}_histogram.csv"), key=f"{channel}_histogram")
    with col4:
        cfg["file_suffix"][f"{channel}_a1"] = st.text_input(f"A1 file suffix", value=cfg.get("file_suffix", {}).get(f"{channel}_a1", f"{channel}_a1.asc"), key=f"{channel}_a1")

update_config_button = st.button("Update Configuration")
if update_config_button:
    save_config(cfg)
    st.success("Configuration updated!")