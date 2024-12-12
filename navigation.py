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
    # Adjust these links based on your actual page routes
    # The links below assume default multipage naming conventions:
    # main page: /
    # page 1: /Page1
    # page 2: /Page2
    # page 3: /Page3
    
    # You can determine these routes by running the app and copying the URLs from your browser.

    st.markdown(
        """
        <style>
        /* Hide the default Streamlit burger menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True
    )
    

    # Define the menu bar as a horizontal list of links
    # Use relative links that Streamlit sets for pages.
    # Typically, after running `streamlit run main.py`, visit each page from the sidebar
    # and note the URL. Adjust below accordingly.
    menu_html = f"""
    <div style='background-color:#f0f0f0; padding:10px; border-bottom:1px solid #ccc;'>
    <a href='/' style='margin-right:20px; text-decoration:none; font-weight:bold;'>Index</a>"""

    for page in pages:
        menu_html += f"""
        <a href='/{page}' style='margin-right:20px; text-decoration:none; font-weight:bold;'>{link_2_name(page)}</a>"""
    
    menu_html += "</div>"

    st.markdown(menu_html, unsafe_allow_html=True)