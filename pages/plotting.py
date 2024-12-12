import streamlit as st
from navigation import render_top_menu
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
render_top_menu()