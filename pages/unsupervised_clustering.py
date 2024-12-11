import streamlit as st
from navigation import render_top_menu
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
# Render the top menu 
render_top_menu()

st.title("Page 1")
st.write("This is Page 1. Use the top menu to go to other pages.")