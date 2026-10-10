"""Feature pickers map shortened display names back to exact column names.
Group names can differ from column prefixes: Derived Features contains columns
named Derived: <name>. Both pickers use feature_display_to_column.
"""
import sys

import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.widgets.selection_widgets import feature_display_to_column
from src.widgets.multiselect_modes import ALL_LABEL, EXCEPT_LABEL


def test_normal_group_strips_prefix_and_maps_back_to_full_column():
    cols = ["Lifetime fit_nadh: t1", "Lifetime fit_nadh: a1"]
    mapping = feature_display_to_column(cols, "Lifetime fit_nadh", data_extraction=True)
    # Picker shows the short names...
    assert list(mapping.keys()) == ["nadh: t1", "nadh: a1"]
    # ...but each resolves back to the real column.
    assert mapping["nadh: t1"] == "Lifetime fit_nadh: t1"
    assert mapping["nadh: a1"] == "Lifetime fit_nadh: a1"


@pytest.mark.parametrize("separator", ["_", ".", ":", ": "])
@pytest.mark.parametrize("flag", [False, True])
def test_first_separator_shortens_generic_group_and_maps_back(separator, flag):
    columns = [f"sepal{separator}length", f"sepal{separator}width"]
    assert feature_display_to_column(columns, "sepal", flag) == {
        "length": columns[0], "width": columns[1]}


@pytest.mark.parametrize("column,expected", [
    ("A_x.y:z", "x.y:z"),
    ("A.x_y:z", "x_y:z"),
    ("A:x_y.z", "x_y.z"),
    ("A: x_y.z", "x_y.z"),
    ("A_x_y", "x_y"),
])
def test_earliest_separator_keeps_the_entire_remaining_name(column, expected):
    assert feature_display_to_column([column], "A measurements") == {expected: column}


@pytest.mark.parametrize("columns,group", [
    (["A_length", "A.length"], "A"),
    (["A:length", "A: length"], "A"),
    (["A_", "A_width"], "A"),
    (["A:   ", "A_width"], "A"),
    (["_length", "_width"], "A"),
    (["A_length", "B_width"], "A"),
    (["A_length", "plain"], "A"),
    (["A_length", "A_width"], "Renamed"),
])
def test_unsafe_first_separator_shortening_keeps_every_full_name(columns, group):
    assert feature_display_to_column(columns, group) == {column: column for column in columns}


def test_shared_prefix_can_use_different_separators_without_losing_columns():
    columns = ["sepal_length", "sepal.width", "sepal: area"]
    assert feature_display_to_column(columns, "sepal") == {
        "length": columns[0], "width": columns[1], "area": columns[2]}


@pytest.mark.parametrize("separator", ["_", ".", ":"])
@pytest.mark.parametrize("suffix", ["Select", ALL_LABEL, EXCEPT_LABEL])
def test_new_separators_do_not_shorten_into_picker_controls(separator, suffix):
    columns = [f"A{separator}{suffix}", f"A{separator}value"]
    assert feature_display_to_column(columns, "A") == {column: column for column in columns}


def test_derived_features_group_round_trips_to_real_derived_column():
    """The Derived Features group resolves columns with the Derived: prefix."""
    cols = ["Derived: redox_ratio", "Derived: C"]
    mapping = feature_display_to_column(cols, "Derived Features", data_extraction=True)
    assert list(mapping.keys()) == ["redox_ratio", "C"]
    # Resolve the actual column: Derived: C.
    assert mapping["C"] == "Derived: C"
    assert mapping["redox_ratio"] == "Derived: redox_ratio"


def test_uncategorized_group_is_identity():
    cols = ["nadh_offset", "nadh_reduced_chi_square"]
    mapping = feature_display_to_column(cols, "Uncategorized Features", data_extraction=True)
    assert mapping == {"nadh_offset": "nadh_offset",
                       "nadh_reduced_chi_square": "nadh_reduced_chi_square"}


def test_mixed_names_keep_full_labels_even_with_legacy_false_flag():
    # Mixed prefixed and ordinary columns keep their full names.
    cols = ["my col: raw", "another"]
    mapping = feature_display_to_column(cols, "Some Group", data_extraction=False)
    assert mapping == {"my col: raw": "my col: raw", "another": "another"}


def test_derived_name_with_only_one_colon_split():
    # Split only at the first separator to preserve the rest of an uploaded name.
    mapping = feature_display_to_column(["Derived: a_b_c"], "Derived Features")
    assert mapping == {"a_b_c": "Derived: a_b_c"}


def test_common_prefix_shortens_independent_of_legacy_source_flag():
    cols = ["Derived: ratio", "Derived: sum"]
    assert feature_display_to_column(cols, "Derived Features", False) == {
        "ratio": cols[0], "sum": cols[1]}


def test_ordinary_and_mixed_names_keep_all_columns_without_collisions():
    for cols, group in [(["Area", "Width"], "Morphology"),
                        (["A: value", "B: value"], "A measurements"),
                        (["A: value", "plain"], "A measurements"),
                        (["A: value", "A: other"], "Renamed"),
                        ([": value"], " measurements")]:
        assert feature_display_to_column(cols, group) == {c: c for c in cols}


def test_short_pending_axis_resolves_to_the_full_column_for_both_flags():
    from src.widgets.selection_widgets import resolve_pending_selection
    for flag in (True, False):
        assert resolve_pending_selection({"Derived Features": ["Derived: ratio"]},
                                         "2d_x", flag,
                                         {"2d_x_menu_Derived Features": "ratio"}) == "Derived: ratio"


def _picker_app(mode, flag, suffix=None, separator=": "):
    import streamlit as st
    from src.widgets.selection_widgets import (
        single_feature_select_widget, multi_feature_select_widget,
        twod_single_feature_select_widget,
    )
    groups = {"Derived Features": [f"Derived{separator}A" if suffix is None else f"Derived{separator}{suffix}",
                                  f"Derived{separator}B"]}
    if mode == "single":
        chosen = single_feature_select_widget(groups, data_extraction=flag)
    elif mode == "multiple":
        chosen = multi_feature_select_widget(groups, data_extraction=flag)
    else:
        chosen = twod_single_feature_select_widget(groups, data_extraction=flag)
    st.text(repr(chosen))


@pytest.mark.parametrize("separator", ["_", ".", ":", ": "])
def test_all_picker_modes_resolve_short_labels_for_both_legacy_flags(separator):
    from streamlit.testing.v1 import AppTest
    for flag in (False, True):
        for mode in ("single", "multiple", "2d"):
            at = AppTest.from_function(_picker_app, args=(mode, flag, None, separator)).run()
            assert not at.exception
            if mode == "multiple":
                at.multiselect[0].set_value(["B"]).run()
                expected = repr([f"Derived{separator}B"])
            elif mode == "single":
                at.selectbox[0].select("A").run()
                expected = repr(f"Derived{separator}A")
            else:
                next(w for w in at.selectbox if w.key == "2d_x_menu_Derived Features").select("A").run()
                next(w for w in at.selectbox if w.key == "2d_y_menu_Derived Features").select("B").run()
                expected = repr((f"Derived{separator}A", f"Derived{separator}B"))
            assert not at.exception, [e.value for e in at.exception]
            assert at.text[0].value == expected



@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("suffix", ["Select", ALL_LABEL, EXCEPT_LABEL])
def test_control_suffix_keeps_every_group_label_full_and_pending_resolves(suffix, flag):
    from src.widgets.selection_widgets import resolve_pending_selection
    columns = [f"Derived: {suffix}", "Derived: B"]
    assert feature_display_to_column(columns, "Derived Features", flag) == {c: c for c in columns}
    assert resolve_pending_selection({"Derived Features": columns}, "2d_x", flag,
                                     {"2d_x_menu_Derived Features": columns[0]}) == columns[0]


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("suffix", ["Select", ALL_LABEL, EXCEPT_LABEL])
@pytest.mark.parametrize("mode", ["single", "2d", "multiple"])
def test_control_named_feature_remains_selectable_in_every_picker(mode, suffix, flag):
    """Full labels separate real columns from picker placeholder/mode controls."""
    from streamlit.testing.v1 import AppTest
    column = f"Derived: {suffix}"
    at = AppTest.from_function(_picker_app, args=(mode, flag, suffix)).run()
    assert not at.exception, [e.value for e in at.exception]
    if mode == "multiple":
        assert column in at.multiselect[0].options
        assert "Derived: B" in at.multiselect[0].options
        at.multiselect[0].set_value([column]).run()
        expected = repr([column])
    elif mode == "single":
        assert column in at.selectbox[0].options
        assert "Derived: B" in at.selectbox[0].options
        at.selectbox[0].select(column).run()
        expected = repr(column)
    else:
        x = next(w for w in at.selectbox if w.key == "2d_x_menu_Derived Features")
        assert column in x.options
        assert "Derived: B" in x.options
        x.select("Derived: B").run()
        next(w for w in at.selectbox if w.key == "2d_y_menu_Derived Features").select(column).run()
        expected = repr(("Derived: B", column))
    assert not at.exception, [e.value for e in at.exception]
    assert at.text[0].value == expected
