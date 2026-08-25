"""Collapsible axis pickers for the 2D Feature Distribution module.

`twod_single_feature_select_widget` renders each axis as a plain grid of
feature-group selectboxes until that axis has a pick, then swaps the grid into a
collapsed `st.expander`. Choosing the container requires knowing the selection
*before* the selectboxes render, which is what `resolve_pending_selection` does.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest

from src.widgets.selection_widgets import resolve_pending_selection

GROUPS = {
    "Lifetime fit_nadh": ["Lifetime fit_nadh: t1", "Lifetime fit_nadh: t2"],
    "Derived Features": ["Derived: ratio"],
}


def test_returns_select_when_no_menu_keys_present():
    assert resolve_pending_selection(GROUPS, "2d_x", session_state={}) == "Select"


def test_returns_select_when_every_menu_holds_the_sentinel():
    state = {
        "2d_x_menu_Lifetime fit_nadh": "Select",
        "2d_x_menu_Derived Features": "Select",
    }
    assert resolve_pending_selection(GROUPS, "2d_x", session_state=state) == "Select"


def test_resolves_display_name_back_to_full_column():
    state = {"2d_x_menu_Lifetime fit_nadh": "t1"}
    assert (
        resolve_pending_selection(GROUPS, "2d_x", session_state=state)
        == "Lifetime fit_nadh: t1"
    )


def test_resolves_derived_features_group_to_its_derived_prefix():
    # The group is named "Derived Features" but its columns are "Derived: <name>",
    # so the mapping must key off the column, not off f"{group}: {name}".
    state = {"2d_x_menu_Derived Features": "ratio"}
    assert (
        resolve_pending_selection(GROUPS, "2d_x", session_state=state)
        == "Derived: ratio"
    )


def test_returns_select_when_stored_value_left_the_option_list():
    # x took t2, so twod_single_feature_select_widget removed it from the y grid's
    # options. Streamlit silently resets that selectbox to "Select" on render, so
    # the pre-render probe must agree instead of reporting a stale "t2".
    reduced = {
        "Lifetime fit_nadh": ["Lifetime fit_nadh: t1"],
        "Derived Features": ["Derived: ratio"],
    }
    state = {"2d_y_menu_Lifetime fit_nadh": "t2"}
    assert resolve_pending_selection(reduced, "2d_y", session_state=state) == "Select"


def test_key_prefix_scopes_the_lookup_to_one_axis():
    state = {"2d_x_menu_Lifetime fit_nadh": "t1"}
    assert resolve_pending_selection(GROUPS, "2d_y", session_state=state) == "Select"


def test_non_data_extraction_columns_map_to_themselves():
    groups = {"Uncategorized Features": ["raw_col"]}
    state = {"2d_x_menu_Uncategorized Features": "raw_col"}
    assert (
        resolve_pending_selection(
            groups, "2d_x", data_extraction=False, session_state=state
        )
        == "raw_col"
    )


HARNESS = str(Path(__file__).resolve().parent / "harness_twod_select.py")

X_MENU = "2d_x_menu_Lifetime fit_nadh"
Y_MENU = "2d_y_menu_Lifetime fit_nadh"


def _run():
    return AppTest.from_file(HARNESS).run()


def test_fresh_run_shows_the_x_grid_expanded_and_hides_the_y_grid():
    at = _run()
    assert len(at.expander) == 0
    assert len(at.selectbox) == 2  # one per group, x axis only
    assert at.text[0].value == "x=Select|y=Select"


def test_choosing_x_collapses_the_x_grid_and_reveals_the_y_grid():
    at = _run()
    at.selectbox(X_MENU).select("t1").run()

    assert len(at.expander) == 1
    assert at.expander[0].label == "X-axis — Lifetime fit_nadh: t1"
    assert at.expander[0].proto.expanded is False
    # 2 x-axis boxes now nested inside the expander, plus 2 fresh y-axis boxes.
    assert len(at.selectbox) == 4
    assert at.text[0].value == "x=Lifetime fit_nadh: t1|y=Select"


def test_choosing_y_collapses_the_y_grid():
    at = _run()
    at.selectbox(X_MENU).select("t1").run()
    at.selectbox(Y_MENU).select("t2").run()

    assert len(at.expander) == 2
    assert at.expander[0].label == "X-axis — Lifetime fit_nadh: t1"
    assert at.expander[1].label == "Y-axis — Lifetime fit_nadh: t2"
    assert all(e.proto.expanded is False for e in at.expander)
    assert at.text[0].value == "x=Lifetime fit_nadh: t1|y=Lifetime fit_nadh: t2"


def test_both_axes_chosen_renders_no_summary_box():
    # The two expander labels already carry the full column names, so a summary
    # would be a third restatement of the same pair.
    at = _run()
    at.selectbox(X_MENU).select("t1").run()
    at.selectbox(Y_MENU).select("t2").run()

    assert len(at.info) == 0


def test_stealing_ys_feature_for_x_reopens_the_y_grid():
    # x=t1, y=t2; then re-open x and take t2 for it. t2 leaves the y option list,
    # Streamlit resets y to "Select", so y must fall back to the expanded grid
    # rather than keep a collapsed expander labelled with a value it no longer holds.
    at = _run()
    at.selectbox(X_MENU).select("t1").run()
    at.selectbox(Y_MENU).select("t2").run()
    at.selectbox(X_MENU).select("t2").run()

    assert len(at.expander) == 1
    assert at.expander[0].label == "X-axis — Lifetime fit_nadh: t2"
    assert at.text[0].value == "x=Lifetime fit_nadh: t2|y=Select"


def test_clearing_x_restores_the_plain_grid_and_hides_y():
    at = _run()
    at.selectbox(X_MENU).select("t1").run()
    at.selectbox(X_MENU).select("Select").run()

    assert len(at.expander) == 0
    assert len(at.selectbox) == 2
    assert at.text[0].value == "x=Select|y=Select"


def test_derived_features_pick_labels_the_expander_with_its_real_column():
    at = _run()
    at.selectbox("2d_x_menu_Derived Features").select("ratio").run()

    assert at.expander[0].label == "X-axis — Derived: ratio"
    assert at.text[0].value == "x=Derived: ratio|y=Select"
