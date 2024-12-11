import streamlit as st

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
    menu_html = """
    <div style='background-color:#f0f0f0; padding:10px; border-bottom:1px solid #ccc;'>
    <a href='/' style='margin-right:20px; text-decoration:none; font-weight:bold;'>Home</a>
    <a href='/outlier_finder_fit' style='margin-right:20px; text-decoration:none; font-weight:bold;'>Page 1</a>
    <a href='/Page2' style='margin-right:20px; text-decoration:none; font-weight:bold;'>Page 2</a>
    <a href='/Page3' style='text-decoration:none; font-weight:bold;'>Page 3</a>
    </div>
    """
    st.markdown(menu_html, unsafe_allow_html=True)