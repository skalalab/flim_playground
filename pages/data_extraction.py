import streamlit as st
from navigation import render_top_menu
from diagram import flimGraph
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()
st.title("Data Extraction")
col1, col2 = st.columns([0.1, 1])
with col1:
    # Checkbox acts like a switch
    simplify = st.checkbox("Simplify", value=False)
    
    # Dropdown for analysis methods
    analysis_method = st.selectbox(
        "Analysis methods",
        ["Fitting", "Dimension Reduction", "Phasor", "all"]
    )
    
with col2:

    # Display the resulting graph
    st.graphviz_chart(flimGraph(lifetime_extraction_method=analysis_method, simplify=simplify))