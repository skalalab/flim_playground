import html
import os
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from src import updater
from src.column_roles import code_span
from src.config import get_persistent_dir
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
DESKTOP_APP_URL = "https://github.com/skalalab/flim_playground#install"


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


def data_extraction_available():
    """Whether Data Extraction is part of this deployment.

    False online, where ``pages/data_analysis.py`` is the entrypoint and Data
    Extraction ships only with the desktop app. A deployed app always has a
    browser URL; AppTest has none, so page tests keep the desktop answer.
    """
    return not (st.context.url and _only_page() == page_2)


def _link_style(active):
    # Negative vertical margins cancel the padding so the pill does not grow the bar.
    style = "margin:-2px 4px -2px 0; padding:2px 8px; text-decoration:none;"
    if active:
        # The current page reads as a place, not a link: a bold white pill in the body text colour.
        style += " background-color:#fff; color:#31333f; font-weight:bold; border-radius:6px; box-shadow:0 1px 2px rgba(0,0,0,0.15);"
    return style


# Material Symbols "power_settings_new", in the link colour. At the version label's
# 0.8em it shares that label's top and baseline and centres on the page links' text.
_POWER_ICON = (
    "<svg viewBox='0 0 24 24' width='0.8em' height='0.8em' fill='currentColor' style='vertical-align:-0.1em'>"
    "<path d='M13 3h-2v10h2V3zm4.83 2.17l-1.42 1.42C17.99 7.86 19 9.81 19 12c0 3.87-3.13 7-7 7"
    "s-7-3.13-7-7c0-2.19 1.01-4.14 2.58-5.42L6.17 5.17C4.23 6.82 3 9.26 3 12c0 4.97 4.03 9 9 9"
    "s9-4.03 9-9c0-2.74-1.23-5.18-3.17-6.83z'/></svg>"
)

# After quitting, hide Streamlit's header (reconnect status, menu) and its
# "Connection error" dialog, which opens about 3 s after the server goes away.
_QUIT_STYLE = "<style>[data-testid='stHeader'], [data-testid='stDialog'] {display:none !important;}</style>"


def _update_page(tag, asset):
    """Offer ``asset``; on Update now, download it and hand over to ``updater.finish``,
    which replaces the app and reopens it. Ends the script either way."""
    st.subheader(
        f"Update to v{tag} · [What's new ↗]({updater.RELEASE_NOTES_URL})",
        anchor=False,
    )
    with st.container(horizontal=True, vertical_alignment="center"):
        update_clicked = st.button("Update now", type="primary")
        st.markdown("**(Running tasks in all tabs will stop)**", width="content")
    if not update_clicked:
        st.stop()
    progress = st.progress(0.0, text=f"Downloading v{tag}…")

    def on_progress(done, total):
        if done < total:
            progress.progress(done / total, text=f"Downloading v{tag}… {done / 1e6:.0f} of {total / 1e6:.0f} MB")
        else:
            progress.progress(1.0, text="Preparing the update…")

    try:
        new_app = updater.prepare(tag, asset, on_progress)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        st.error(
            f"The update stopped and nothing changed: {code_span(error)}. "
            f"You can [update manually]({DESKTOP_APP_URL}). {sad_emoji}"
        )
        st.stop()
    # Committed: from here a closed tab or a click must not stop the swap and exit.
    threading.Timer(1, updater.finish, (new_app,)).start()
    st.info(f"Installing v{tag}. FLIM Playground reopens in a new tab, so you can close this one.")
    st.markdown(_QUIT_STYLE, unsafe_allow_html=True)
    st.stop()


def render_top_menu(space_below="0"):
    """Render the navigation bar. ``space_below`` is CSS length of breathing room
    between the bar and the page's first element (the pages otherwise touch it)."""

    # Set only by launcher.py; without it (online, `streamlit run`) nothing may end the server.
    quit_token = os.environ.get("FLIM_PLAYGROUND_QUIT_TOKEN")

    # App Translocation makes the app read-only and prevents configuration saves.
    if "/AppTranslocation/" in sys.executable:
        # Recover the app name so the command targets the download, not the read-only mount.
        app_name = sys.executable.split("/Contents/")[0].rsplit("/", 1)[-1]
        st.error(
            "macOS opened this quarantined app read-only, so settings can't save, and it has quit. "
            f'Run `xattr -dr com.apple.quarantine ~/Downloads/"{app_name}"` '
            f"in Terminal (adjust the path if the app is elsewhere), then reopen it. {sad_emoji}"
        )
        if quit_token:
            # Quit for the user, so reopening starts a fresh, writable instance.
            st.markdown(_QUIT_STYLE, unsafe_allow_html=True)
            threading.Timer(1, os._exit, (0,)).start()
        st.stop()

    # The power button links here with this launch's token. `quit_token and` matters:
    # without a token, a page with no ?quit would compare None with None.
    if quit_token and st.query_params.get("quit") == quit_token:
        st.info("FLIM Playground has quit. You can close its tabs.")
        st.markdown(_QUIT_STYLE, unsafe_allow_html=True)
        # Exit once the goodbye has reached the browser.
        threading.Timer(1, os._exit, (0,)).start()
        st.stop()

    # A newer release is offered to this launch alone: its link carries the token too.
    update = updater.available_update() if quit_token else None

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
            f"<a href='{DESKTOP_APP_URL}' target='_blank' rel='noopener' "
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
        f"{html.escape(get_version_label())}</span>"
    )
    if update:
        menu_html += (
            f"<a href='/?update={quit_token}' target='_self' title='Install the latest release' "
            f"style='{_link_style(False)} margin-left:4px; font-size:0.8em;'>"
            f"Update to v{html.escape(update[0])}</a>"
        )
    if quit_token:
        # Without a target, st.markdown opens links in a new tab.
        menu_html += (
            f"<a href='/?quit={quit_token}' target='_self' title='Quit FLIM Playground' "
            f"aria-label='Quit' style='{_link_style(False)} margin-left:12px;'>{_POWER_ICON}</a>"
        )

    st.markdown(menu_html + "</div>", unsafe_allow_html=True)

    # macOS asks each new build whether it may use the folder holding its settings,
    # the one the .app is in. Until someone clicks Allow, every read there fails.
    if getattr(sys, "frozen", False) and sys.platform == "darwin":
        try:
            os.scandir(get_persistent_dir()).close()
        except PermissionError:
            st.warning(
                "macOS is asking whether FLIM Playground may access the folder it's in, where its "
                "settings are saved. Click **Allow**, then reload this page. If you clicked Don't "
                f"Allow, turn it on in System Settings → Privacy & Security → Files & Folders. {sad_emoji}"
            )
            st.stop()

    if failure := updater.pop_failure():
        st.warning(
            f"The last update didn't finish, so this is still {get_version_label()}: "
            f"{code_span(failure)}. You can [update manually]({DESKTOP_APP_URL}). {sad_emoji}"
        )
    if update and st.query_params.get("update") == quit_token:
        _update_page(*update)
