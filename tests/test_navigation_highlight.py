"""The navigation bar marks the page being rendered, read from the browser URL."""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = str(Path(__file__).resolve().parents[1])

_MENU_SCRIPT = (
    "import sys\n"
    f"sys.path.insert(0, r'{_ROOT}')\n"
    "from src.navigation import render_top_menu\n"
    "render_top_menu()\n"
)

_PILL = "background-color:#fff"


def _highlighted_links(monkeypatch, url):
    from streamlit.runtime.context import ContextProxy
    from streamlit.testing.v1 import AppTest

    # AppTest has no browser session; stand in for the URL the frontend reports.
    monkeypatch.setattr(ContextProxy, "url", property(lambda self: url))
    at = AppTest.from_string(_MENU_SCRIPT).run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    bars = [m.value for m in at.markdown if "background-color:#f0f0f0" in m.value]
    assert len(bars) == 1, "exactly one element is the navigation bar"
    links = re.findall(r"<a href='([^']*)' style='([^']*)'>([^<]*)</a>", bars[0])
    assert [label for _, _, label in links] == ["Home", "Data Extraction", "Data Analysis"]
    # Only the current page is bold; the other links read as plain links.
    assert [label for _, style, label in links if "font-weight:bold" in style] == \
        [label for _, style, label in links if _PILL in style]
    return [label for _, style, label in links if _PILL in style]


@pytest.mark.parametrize(("url", "expected"), [
    ("http://localhost:8501/data_extraction", "Data Extraction"),
    ("http://localhost:8501/data_analysis", "Data Analysis"),
    ("http://localhost:8501", "Home"),
    ("http://localhost:8501/", "Home"),
    # A deployment base path is not a page name: still the main page.
    ("https://example.org/flim", "Home"),
])
def test_current_page_link_is_highlighted(monkeypatch, url, expected):
    assert _highlighted_links(monkeypatch, url) == [expected]


def test_unknown_url_highlights_nothing(monkeypatch):
    assert _highlighted_links(monkeypatch, None) == []


def _bar_html(script):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_string(script).run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    bars = [m.value for m in at.markdown if "background-color:#f0f0f0" in m.value]
    assert len(bars) == 1
    return bars[0]


def test_bar_touches_the_page_unless_a_page_asks_for_room():
    # Home keeps its bar flush; Data Extraction and Data Analysis pass space_below="0.5rem".
    assert "margin-bottom:0;" in _bar_html(_MENU_SCRIPT)
    with_room = _MENU_SCRIPT.replace("render_top_menu()", "render_top_menu(space_below='0.5rem')")
    assert "margin-bottom:0.5rem;" in _bar_html(with_room)
