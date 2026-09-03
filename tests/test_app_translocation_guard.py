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


def _patch_version(monkeypatch):
    """Patch the name render_top_menu actually resolves.

    navigation.py does `from src.version import get_version_label`, so the
    binding lives in src.navigation's globals -- patching src.version would miss
    it and the test would silently assert against the real version, passing
    today and failing the day someone cuts a release.
    """
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
    # Same element as the grey bar, not a second markdown block beneath it
    # (which would render its own vertical gap under the nav).
    assert "background-color:#f0f0f0" in bars[0]
    assert "Data Analysis" in bars[0], "nav links must survive the flex change"
    assert "v9.9.9-test</span></div>" in bars[0], "label closes the bar div"


def test_translocated_launch_shows_no_version(monkeypatch):
    """The guard's st.stop() must precede the label.

    Not covered by the PAGE-BODY-RENDERED assertion above: that only watches the
    test script's own body, so a label emitted above the guard (or slipped into
    the <style> block) would leak on a quarantined mac launch with every
    existing test still green.
    """
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "executable", _TRANSLOCATED)
    _patch_version(monkeypatch)
    at = AppTest.from_string(_MENU_SCRIPT).run(timeout=60)

    assert "9.9.9-test" not in " ".join(m.value for m in at.markdown)
    assert not at.markdown, "a stopped page renders no markdown at all"
    assert "xattr -dr com.apple.quarantine" in " ".join(e.value for e in at.error)
