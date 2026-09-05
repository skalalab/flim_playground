"""Theme color is a server-side st.context.theme read with no widget or rerun."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

from src.vis.helpers import get_context_theme_color

ROOT = str(Path(__file__).resolve().parents[1])

# Multiple plots can read the theme in one run without distinct widget keys.
SCRIPT = f"""
import sys
sys.path.insert(0, {ROOT!r})
import streamlit as st
from src.vis.helpers import get_context_theme_color

first = get_context_theme_color()
second = get_context_theme_color()
st.text(f"{{first}}|{{second}}")
"""


def test_streamlit_theme_component_is_gone():
    import src.vis.helpers as helpers

    assert not hasattr(helpers, "st_theme")
    assert not hasattr(helpers, "get_theme_color")


def test_returns_a_plotly_usable_color():
    assert get_context_theme_color() in {"black", "white"}


def test_it_renders_no_widget_of_any_kind():
    # A component would show up as an element and would need a unique key per call.
    at = AppTest.from_string(SCRIPT).run()

    assert not at.exception
    assert len(at.selectbox) == 0
    assert len(at.checkbox) == 0
    assert len(at.button) == 0
