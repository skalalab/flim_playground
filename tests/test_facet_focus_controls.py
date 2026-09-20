"""Promotion state: what a click stores, and what a redraw is asked for."""

from types import SimpleNamespace

import plotly.graph_objects as go
import pytest
import streamlit as st

from src.widgets.visualization_widgets import (
    FACET_FOCUS_KEYS,
    facet_chart_suffix,
    facet_click_focus,
    facet_focus_selection,
)

FOCUS_KEY = FACET_FOCUS_KEYS["2D Feature Distribution"]


@pytest.fixture(autouse=True)
def _session():
    st.session_state.clear()
    yield
    st.session_state.clear()


def figure(keys=("Day 2", "Day 10")):
    fig = go.Figure()
    for slot in range(len(keys) + 1):
        fig.add_trace(go.Scatter(x=[1], y=[1],
                                 meta=dict(facet_role="points", facet_slot=slot)))
    fig.update_layout(meta={"facet_focus": {"keys": list(keys)}})
    return fig


def click(curve):
    return SimpleNamespace(selection={"points": [{"curve_number": curve, "point_index": 0}]})


def test_a_stored_promotion_is_used_when_the_grid_still_offers_it():
    st.session_state[FOCUS_KEY] = "Day 10"
    assert facet_focus_selection(figure(), FOCUS_KEY) == "Day 10"


def test_a_promotion_whose_level_vanished_is_forgotten():
    st.session_state[FOCUS_KEY] = "Day 99"
    assert facet_focus_selection(figure(), FOCUS_KEY) is None
    assert st.session_state[FOCUS_KEY] is None


def test_a_figure_without_a_grid_promotes_nothing():
    assert facet_focus_selection(go.Figure(), FOCUS_KEY) is None


def test_a_click_stores_the_promotion_and_asks_for_a_redraw():
    fig = figure()
    assert facet_click_focus(fig, click(2), FOCUS_KEY) is True
    assert st.session_state[FOCUS_KEY] == "Day 10"


def test_the_same_selection_returned_again_asks_for_nothing():
    fig = figure()
    facet_click_focus(fig, click(2), FOCUS_KEY)
    assert facet_click_focus(fig, click(2), FOCUS_KEY) is False


def test_clicking_the_promoted_panel_still_redraws_to_clear_the_selection():
    fig = figure()
    st.session_state[FOCUS_KEY] = "Day 10"
    # A different point of the same panel: the promotion does not move, but the
    # chart must be remounted or Plotly keeps dimming everything else.
    assert facet_click_focus(fig, click(2), FOCUS_KEY) is True
    assert st.session_state[FOCUS_KEY] == "Day 10"
    assert facet_chart_suffix(FOCUS_KEY) == "_1"


def test_every_handled_click_moves_the_chart_key():
    fig = figure()
    assert facet_chart_suffix(FOCUS_KEY) == "_0"
    facet_click_focus(fig, click(1), FOCUS_KEY)
    facet_click_focus(fig, click(2), FOCUS_KEY)
    assert facet_chart_suffix(FOCUS_KEY) == "_2"


def test_an_empty_event_never_redraws():
    assert facet_click_focus(figure(), None, FOCUS_KEY) is False
    assert facet_click_focus(figure(), SimpleNamespace(selection={"points": []}),
                             FOCUS_KEY) is False
    assert facet_chart_suffix(FOCUS_KEY) == "_0"


def test_both_grids_keep_their_own_promotion():
    assert FACET_FOCUS_KEYS["Dimension Reduction"] != FOCUS_KEY


def test_separating_by_another_column_forgets_the_promotion():
    fig = figure()
    fig.layout.meta["facet_focus"]["separate_by"] = "day"
    facet_focus_selection(fig, FOCUS_KEY)
    facet_click_focus(fig, click(2), FOCUS_KEY)
    assert facet_focus_selection(fig, FOCUS_KEY) == "Day 10"
    # Two columns can share a value, so the level alone cannot identify it.
    fig.layout.meta["facet_focus"]["separate_by"] = "treatment"
    assert facet_focus_selection(fig, FOCUS_KEY) is None
