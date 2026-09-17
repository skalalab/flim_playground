import html
import sys
from urllib.parse import urlparse

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


def current_page():
    """The page being rendered: a name from ``pages``, "home", or None when unknown.

    Read from the browser URL, which is None without a browser session (AppTest).
    The last path segment names a page; anything else, including a deployment
    base path, is the main page.
    """
    url = st.context.url
    if not url:
        return None
    last_segment = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    return last_segment if last_segment in pages else "home"


def _link_style(active):
    # Negative vertical margins cancel the padding so the pill does not grow the bar.
    style = "margin:-2px 4px -2px 0; padding:2px 8px; text-decoration:none;"
    if active:
        # The current page reads as a place, not a link: a bold white pill in the body text colour.
        style += " background-color:#fff; color:#31333f; font-weight:bold; border-radius:6px; box-shadow:0 1px 2px rgba(0,0,0,0.15);"
    return style

def render_top_menu(space_below="0"):
    """Render the navigation bar. ``space_below`` is CSS length of breathing room
    between the bar and the page's first element (the pages otherwise touch it)."""

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
        /* A column-opening "Fields of view:" heading sits level with the neighbouring
           widget label, which unlike a heading carries no top padding. */
        div[class*="st-key-fov_heading"] h5 {padding-top: 0;}
        </style>
        """, unsafe_allow_html=True
    )

    current = current_page()
    menu_html = f"""
    <div style='background-color:#f0f0f0; padding:10px; margin-bottom:{space_below}; border-bottom:1px solid #ccc; display:flex; align-items:baseline;'>
    <a href='/' style='{_link_style(current == "home")}'>Home</a>"""

    for page in pages:
        menu_html += f"""
        <a href='/{page}' style='{_link_style(current == page)}'>{link_2_name(page)}</a>"""

    # Right-align the version on the links' baseline. Avoid adding a source newline
    # that changes Markdown dedenting, and escape the version at the HTML boundary.
    menu_html += (
        "<span title='FLIM Playground version' "
        "style='margin-left:auto; color:#666; font-size:0.8em;'>"
        f"{html.escape(get_version_label())}</span></div>"
    )

    st.markdown(menu_html, unsafe_allow_html=True)
