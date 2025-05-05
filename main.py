import streamlit as st
from src.navigation import render_top_menu, titles
from src.docs import docs
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
    Welcome to <span style="font-size: 20px; font-weight: bold;">Fluorescence Lifetime Imaging Microscopy Playground!</span> 
    For detailed instructions, you can come to me or read the docs below.
</div>
""", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center;'>Select a playground to know more</h4>", unsafe_allow_html=True)
col1, col2 = st.columns([0.5, 1])
with col1: 
    selected_playground = st.selectbox(
                    "Playgrounds", 
                    titles + ["General Info"], 
                    index=0, 
                    key="menu_steps",
    )
with col2: 
    st.markdown("<h5 style='text-align: center;'>Explanation</h5>", unsafe_allow_html=True)

    try:
        doc = docs[selected_playground]
        st.markdown(doc, unsafe_allow_html=True)
    except KeyError:
        st.markdown("<h5 style='text-align: center; color: red'>No doc available yet.</h5>", unsafe_allow_html=True)

    
# st.write("This is the main page. Use the top menu to navigate to other pages.")