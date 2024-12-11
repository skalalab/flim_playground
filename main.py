import streamlit as st
from navigation import render_top_menu

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Hide the sidebar completely with CSS
hide_sidebar_style = """
<style>
.css-1cpxqw2 {visibility: hidden;} /* This may need adjusting depending on Streamlit version */
</style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# Render the top menu on the main page
render_top_menu()

st.title("Home")
st.write("This is the main page. Use the top menu to navigate to other pages.")