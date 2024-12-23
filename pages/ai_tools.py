import streamlit as st
from navigation import render_top_menu
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu
render_top_menu()
st.write("to be developed.")
col1, col2 = st.columns([0.4, 1])
with col1:
    st.title("AI Tools")
    method = st.selectbox(
        "Select an AI tool",
        ["Finetune a segmentation model"],
    )

