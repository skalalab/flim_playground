import streamlit as st
from navigation import render_top_menu
#from diagram import flimGraph
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()
st.title("Data Extraction")
