import html
import sys

import streamlit as st

from src.emojis import sad_emoji
from src.version import get_version_label

"""
This module contains the navigation bar for the FLIM Playground app.
If new modules are added, they should be included in the `pages` list below.
"""

# Page module names, without the .py extension.
page_1 = "data_extraction"
page_2 = "data_analysis"

pages = [page_1, page_2]
def link_2_name(link):
    return link.replace("_", " ").title()

def render_top_menu():

    # App Translocation makes the app read-only and prevents configuration saves.
    if "/AppTranslocation/" in sys.executable:
        # Recover the app name so the command targets the download, not the read-only mount.
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
    <div style='background-color:#f0f0f0; padding:10px; border-bottom:1px solid #ccc; display:flex; align-items:baseline;'>
    <a href='/' style='margin-right:20px; text-decoration:none; font-weight:bold;'>Home</a>"""

    for page in pages:
        menu_html += f"""
        <a href='/{page}' style='margin-right:20px; text-decoration:none; font-weight:bold;'>{link_2_name(page)}</a>"""

    # Right-align the version on the links' baseline. Avoid adding a source newline
    # that changes Markdown dedenting, and escape the version at the HTML boundary.
    menu_html += (
        "<span title='FLIM Playground version' "
        "style='margin-left:auto; color:#666; font-size:0.8em;'>"
        f"{html.escape(get_version_label())}</span></div>"
    )

    st.markdown(menu_html, unsafe_allow_html=True)
