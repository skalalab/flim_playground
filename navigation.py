import streamlit as st

"""
This module contains the navigation bar for the FLIM Playground app.
If new modules are added, they should be included in the `pages` list below.
"""

# page is the name of the playground python file without the .py extension
# title is the name of the playground as it will appear in the menu
page_1 = "data_extraction"
page_2 = "visualization"
page_3 = "classification"

pages = [page_1, page_2, page_3]
def link_2_name(link):    
    return link.replace("_", " ").title()

titles = [link_2_name(page) for page in pages]

def render_top_menu():

    st.markdown(
        """
        <style>
        /* Hide the default Streamlit burger menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True
    )

    menu_html = f"""
    <div style='background-color:#f0f0f0; padding:10px; border-bottom:1px solid #ccc;'>
    <a href='/' style='margin-right:20px; text-decoration:none; font-weight:bold;'>Index</a>"""

    for page in pages:
        menu_html += f"""
        <a href='/{page}' style='margin-right:20px; text-decoration:none; font-weight:bold;'>{link_2_name(page)}</a>"""
    
    menu_html += "</div>"

    st.markdown(menu_html, unsafe_allow_html=True)