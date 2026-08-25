"""Theme color comes from `st.context.theme`, not a bidirectional component.

The old `st_theme()` component returned None on its first render and then reported the
theme back, forcing a second script run. Callers keyed it by the selected feature, so a
fresh component mounted — and the plot was rebuilt — on every pick. It also needed a
unique key per call site to avoid duplicate element IDs.

`get_context_theme_color()` is a plain server-side read: no key, no widget, no rerun.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from streamlit.testing.v1 import AppTest

from src.vis.helpers import get_context_theme_color

ROOT = str(Path(__file__).resolve().parents[1])

# Two calls in one run stands in for two channels / two plots on a page — the case that
# forced the old component's per-call-site keys.
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


def test_st_theme_package_is_no_longer_installed():
    # The `st-theme` dependency was dropped; nothing may import it back in.
    with pytest.raises(ModuleNotFoundError):
        import streamlit_theme  # noqa: F401


def test_returns_a_plotly_usable_color():
    assert get_context_theme_color() in {"black", "white"}


def test_repeated_calls_in_one_run_agree_and_raise_nothing():
    at = AppTest.from_string(SCRIPT).run()

    assert not at.exception
    first, second = at.text[0].value.split("|")
    assert first == second
    assert first in {"black", "white"}


def test_it_renders_no_widget_of_any_kind():
    # A component would show up as an element and would need a unique key per call.
    at = AppTest.from_string(SCRIPT).run()

    assert not at.exception
    assert len(at.selectbox) == 0
    assert len(at.checkbox) == 0
    assert len(at.button) == 0
