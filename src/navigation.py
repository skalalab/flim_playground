import html
import sys

import streamlit as st

from src.emojis import sad_emoji
from src.version import get_version_label

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
    <div style='background-color:#f0f0f0; padding:10px; border-bottom:1px solid #ccc; display:flex; align-items:baseline;'>
    <a href='/' style='margin-right:20px; text-decoration:none; font-weight:bold;'>Home</a>"""

    for page in pages:
        menu_html += f"""
        <a href='/{page}' style='margin-right:20px; text-decoration:none; font-weight:bold;'>{link_2_name(page)}</a>"""

    # Version label, right-aligned in the same grey bar -- the only chrome that
    # renders on all three pages, so a screenshot from any of them is taggable.
    # Three deliberate details:
    #  - `margin-left:auto` in a flex row pushes it to the right edge, and
    #    `align-items:baseline` sits it on the links' own baseline; `float`
    #    would align to the top of the line box and ride high above them.
    #  - Concatenated onto the closing </div> with NO new source line: markdown
    #    bodies are run through textwrap.dedent (streamlit/string_util.py:38),
    #    so a line at a different indent changes the common prefix for every
    #    other line in the bar.
    #  - Escaped at the interpolation point, per house rule: the string comes
    #    from a git tag or a bundled file and the browser reads this as markup.
    menu_html += (
        "<span title='FLIM Playground version' "
        "style='margin-left:auto; color:#666; font-size:0.8em;'>"
        f"{html.escape(get_version_label())}</span></div>"
    )

    st.markdown(menu_html, unsafe_allow_html=True)
