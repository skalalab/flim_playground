import streamlit as st
from navigation import render_top_menu
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu on this page as well
render_top_menu()
st.title("Classification (better to be run offline)")
st.write("To be developed")