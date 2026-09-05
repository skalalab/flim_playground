"""A translocated macOS app stops before page rendering or config writes.
The notice gives an xattr command targeting the downloaded app.
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

# Representative translocated app path.
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
    # Target the downloaded app, using the name recovered from the translocated path.
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


def _patch_version(monkeypatch):
    """Patch src.navigation, where render_top_menu resolves its imported version helper."""
    import src.navigation as navigation

    monkeypatch.setattr(navigation, "get_version_label", lambda: "v9.9.9-test")


def test_version_label_renders_inside_the_menu_bar(monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(
        sys, "executable",
        "/Users/foo/Downloads/Flim-Playground.app/Contents/MacOS/Flim-Playground",
    )
    _patch_version(monkeypatch)
    at = AppTest.from_string(_MENU_SCRIPT).run(timeout=60)

    assert not at.exception
    assert not at.error, f"a version label must never error: {[e.value for e in at.error]}"
    bars = [m.value for m in at.markdown if "9.9.9-test" in m.value]
    assert len(bars) == 1, "exactly one element carries the version"
    # Keep the version in the navigation bar to avoid an extra vertical gap.
    assert "background-color:#f0f0f0" in bars[0]
    assert "Data Analysis" in bars[0], "nav links must survive the flex change"
    assert "v9.9.9-test</span></div>" in bars[0], "label closes the bar div"


def test_translocated_launch_shows_no_version(monkeypatch):
    """The translocation guard stops before either the version label or page body renders."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "executable", _TRANSLOCATED)
    _patch_version(monkeypatch)
    at = AppTest.from_string(_MENU_SCRIPT).run(timeout=60)

    assert "9.9.9-test" not in " ".join(m.value for m in at.markdown)
    assert not at.markdown, "a stopped page renders no markdown at all"
    assert "xattr -dr com.apple.quarantine" in " ".join(e.value for e in at.error)
