import streamlit as st
page_1 = "region_props"
page_2 = "outlier_finder"
page_3 = "sdt_suite"
page_4 = "classification"

pages = [page_1, page_2, page_3, page_4]
def link_2_name(link):
    if link == "outlier_finder":
        return "Clustering & " + link.replace("_", " ").title()
    return link.replace("_", " ").title()

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