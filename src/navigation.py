import html
import sys
from pathlib import Path
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

# Where a visitor who wants the pages this deployment cannot serve should go.
_DESKTOP_APP_URL = "https://github.com/skalalab/flim_playground#install"


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


def _only_page():
    """The single page this deployment serves, or None when the whole app runs.

    The online deployment runs ``pages/data_analysis.py`` as its entrypoint, so
    Streamlit has no ``pages/`` directory beside it and serves that one script at
    every path: the URL says ``/data_extraction`` while Data Analysis renders. The
    entrypoint stays the main script while a page renders, so the full app —
    ``main.py`` with ``pages/`` beside it, in the source tree and in the bundle —
    never matches. Anything unexpected reads as the full app and keeps every link.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        entry = Path(get_script_run_ctx().pages_manager.main_script_path)
    except (AttributeError, ImportError, TypeError):
        return None
    return entry.stem if entry.parent.name == "pages" and entry.stem in pages else None


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

    only = _only_page()
    menu_html = f"""
    <div style='background-color:#f0f0f0; padding:10px; margin-bottom:{space_below}; border-bottom:1px solid #ccc; display:flex; align-items:baseline;'>"""

    if only:
        # The page that rendered is the only place to be, whatever the URL says.
        # Offer the build that has the rest instead of links that lead back here.
        menu_html += f"""
    <a href='/{only}' style='{_link_style(True)}'>{link_2_name(only)}</a>"""
        menu_html += (
            f"<a href='{_DESKTOP_APP_URL}' target='_blank' rel='noopener' "
            "style='margin-left:auto; font-size:0.8em;'>"
            "Data Extraction: get the desktop app ↗</a>"
        )
        version_margin = "margin-left:12px"
    else:
        current = current_page()
        menu_html += f"""
    <a href='/' style='{_link_style(current == "home")}'>Home</a>"""

        for page in pages:
            menu_html += f"""
        <a href='/{page}' style='{_link_style(current == page)}'>{link_2_name(page)}</a>"""
        version_margin = "margin-left:auto"

    # Right-align the version on the links' baseline. Avoid adding a source newline
    # that changes Markdown dedenting, and escape the version at the HTML boundary.
    menu_html += (
        "<span title='FLIM Playground version' "
        f"style='{version_margin}; color:#666; font-size:0.8em;'>"
        f"{html.escape(get_version_label())}</span></div>"
    )

    st.markdown(menu_html, unsafe_allow_html=True)
