import sys

import streamlit as st

from src.emojis import sad_emoji

"""
This module contains the navigation bar for the FLIM Playground app.
If new modules are added, they should be included in the `pages` list below.
"""

# page is the name of the playground python file without the .py extension
# title is the name of the playground as it will appear in the menu
page_1 = "data_extraction"
page_2 = "data_analysis"

pages = [page_1, page_2]
def link_2_name(link):
    return link.replace("_", " ").title()

def render_top_menu():

    # A quarantined .app opened in place runs read-only under macOS App
    # Translocation, so the first config save would crash with a redacted
    # OSError (real user report). See README "First launch".
    if "/AppTranslocation/" in sys.executable:
        # sys.executable is the throwaway read-only mount (xattr would fail
        # there), but its path still ends .../<real name>.app/Contents/... —
        # recover the name so the command targets the actual download.
        app_name = sys.executable.split("/Contents/")[0].rsplit("/", 1)[-1]
        st.error(
            "macOS opened this quarantined app read-only, so settings can't save. "
            f'Quit, run `xattr -dr com.apple.quarantine ~/Downloads/"{app_name}"` '
            f"in Terminal (adjust the path if the app is elsewhere), then reopen it. {sad_emoji}"
        )
        st.stop()

    st.markdown(
        """
        <style>
        /* Hide the default Streamlit burger menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True
    )

    menu_html = """
    <div style='background-color:#f0f0f0; padding:10px; border-bottom:1px solid #ccc;'>
    <a href='/' style='margin-right:20px; text-decoration:none; font-weight:bold;'>Home</a>"""

    for page in pages:
        menu_html += f"""
        <a href='/{page}' style='margin-right:20px; text-decoration:none; font-weight:bold;'>{link_2_name(page)}</a>"""

    menu_html += "</div>"

    st.markdown(menu_html, unsafe_allow_html=True)
