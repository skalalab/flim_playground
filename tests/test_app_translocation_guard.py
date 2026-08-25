"""A quarantined .app opened in place runs read-only under macOS App
Translocation, so the first config save used to crash with a redacted OSError
(real user report). render_top_menu() — run first on every page — checks
sys.executable for the translocation mount and stops with the xattr fix instead.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = str(Path(__file__).resolve().parents[1])

_MENU_SCRIPT = (
    "import sys\n"
    f"sys.path.insert(0, r'{_ROOT}')\n"
    "import streamlit as st\n"
    "from src.navigation import render_top_menu\n"
    "render_top_menu()\n"
    "st.write('PAGE-BODY-RENDERED')\n"
)

# The exact shape from the user report.
_TRANSLOCATED = (
    "/private/var/folders/xk/lqlzjrs12yb7vscb0r20jdy00000gn/T/AppTranslocation/"
    "448EDF53-C8C9-4A62-9CA1-6412113CDF60/d/Flim-Playground 2.app/"
    "Contents/MacOS/Flim-Playground"
)


def test_translocated_launch_stops_with_xattr_guidance(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "executable", _TRANSLOCATED)
    at = AppTest.from_string(_MENU_SCRIPT).run(timeout=60)

    assert not at.exception, f"guard must not raise: {[e.value for e in at.exception]}"
    errors = " ".join(e.value for e in at.error)
    # Full pasteable command targeting the real download (name recovered from
    # the translocated path), NOT the read-only mount in sys.executable.
    assert 'xattr -dr com.apple.quarantine ~/Downloads/"Flim-Playground 2.app"' in errors
    assert "/AppTranslocation/" not in errors
    # st.stop() fired: nothing after render_top_menu ran.
    assert "PAGE-BODY-RENDERED" not in " ".join(m.value for m in at.markdown)


def test_normal_launch_renders(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        sys, "executable",
        "/Users/foo/Downloads/Flim-Playground.app/Contents/MacOS/Flim-Playground",
    )
    at = AppTest.from_string(_MENU_SCRIPT).run(timeout=60)

    assert not at.exception
    assert not at.error, f"no error expected, got {[e.value for e in at.error]}"
    assert "PAGE-BODY-RENDERED" in " ".join(m.value for m in at.markdown)
