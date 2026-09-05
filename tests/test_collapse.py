"""Collapse produces one point per replicate and layout group.
Numeric columns are averaged; other non-ID columns survive only when constant in
every group. The helper must also work when inlined into a standalone export.
"""
import inspect

import numpy as np
import pandas as pd
import pytest
from src.collapse import collapse_rows


def _paired_df():
    """Six dishes, each in one treatment; two FOVs per dish; one day per dish.

    dish  treatment  day    image_name        cells
    D1    Control    Day 1  F01, F02          4
    D2    Drug       Day 1  F03, F04          4
    ...
    So `treatment` and `day` are constant within a dish (coarser), while
    `image_name` varies inside every dish (finer).
    """
    rows = []
    for i, (dish, treatment, day) in enumerate([
        ("D1", "Control", "Day 1"), ("D2", "Drug", "Day 1"),
        ("D3", "Control", "Day 2"), ("D4", "Drug", "Day 2"),
        ("D5", "Control", "Day 3"), ("D6", "Drug", "Day 3"),
    ]):
        for j in range(4):
            rows.append({
                "cell_id": f"{dish}_c{j}",
                "dish": dish,
                "treatment": treatment,
                "day": day,
                "image_name": f"F{2 * i + j // 2 + 1:02d}",
                "tau": 100.0 * i + j,
                "intensity": 10.0 * i + j,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One dot per (Collapse by × Color by × Separate by) combination
# ---------------------------------------------------------------------------

def test_one_row_per_collapse_level_within_each_x_slot():
    df = _paired_df()
    out, _label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert len(out) == 6                       # six dishes, not six x two treatments
    assert out.groupby(["treatment", "dish"]).size().max() == 1


def test_an_x_slot_holds_as_many_dots_as_it_has_replicates():
    """Each treatment slot retains one dot per dish."""
    df = _paired_df()
    out, _label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    per_slot = out.groupby("treatment").size().to_dict()
    assert per_slot == {"Control": 3, "Drug": 3}   # 3 dishes each


def test_a_replicate_spanning_two_slots_contributes_one_dot_to_each():
    """A dish measured under both treatments retains one dot in each slot for pairing."""
    df = _paired_df()
    df.loc[df["dish"] == "D1", "treatment"] = ["Control", "Control", "Drug", "Drug"]

    out, _label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    d1 = out[out["dish"] == "D1"]
    assert len(d1) == 2
    assert sorted(d1["treatment"]) == ["Control", "Drug"]


def test_the_value_is_the_arithmetic_mean_over_the_cells():
    df = _paired_df()
    out, _label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    expected = df.groupby(["dish", "treatment"], sort=False)["tau"].mean()
    got = out.set_index(["dish", "treatment"])["tau"]
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(),
                                   check_names=False)


def test_every_numeric_column_is_averaged_not_just_one():
    """All numerical features are averaged so the collapsed frame supports feature changes."""
    df = _paired_df()
    out, _label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert {"tau", "intensity"} <= set(out.columns)
    assert out.loc[out["dish"] == "D1", "intensity"].iloc[0] == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# The survival rule, in both directions
# ---------------------------------------------------------------------------

def test_key_columns_always_survive():
    df = _paired_df()
    out, _label, _varied = collapse_rows(df, "dish", ["treatment", "day"], "cell_id")

    for col in ("dish", "treatment", "day"):
        assert col in out.columns


def test_a_coarser_column_survives_and_keeps_its_value():
    """A day constant within each dish survives for use as a point decoration."""
    df = _paired_df()
    out, _label, varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert "day" in out.columns
    assert "day" not in varied
    assert out.loc[out["dish"] == "D3", "day"].iloc[0] == "Day 2"


def test_a_finer_column_is_dropped_and_named():
    """`image_name` takes two values inside every dish, so the dot has no single
    value to carry."""
    df = _paired_df()
    out, _label, varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert "image_name" not in out.columns
    assert varied == ["image_name"]


def test_varying_in_any_one_group_drops_the_column_everywhere():
    """A column varying in any group is dropped from the entire collapsed frame."""
    df = _paired_df()
    df.loc[df["dish"] == "D1", "day"] = ["Day 1", "Day 1", "Day 9", "Day 9"]

    out, _label, varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert "day" not in out.columns
    assert set(varied) == {"day", "image_name"}


def test_a_column_that_is_constant_but_partly_missing_counts_as_varying():
    """Nulls count as values: a group containing both "A" and NaN is not constant."""
    df = _paired_df()
    df["batch"] = "A"
    df.loc[df.index[0], "batch"] = np.nan

    out, _label, varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert "batch" not in out.columns
    assert "batch" in varied


def test_varied_names_exactly_the_dropped_columns():
    df = _paired_df()
    out, label, varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    dropped = set(df.columns) - set(out.columns)
    assert dropped == set(varied) | {"cell_id"}      # the row id is replaced, not varied
    assert label in out.columns


# ---------------------------------------------------------------------------
# The identifier
# ---------------------------------------------------------------------------

def test_the_row_id_is_replaced_by_a_label_carrying_the_count():
    df = _paired_df()
    out, label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert "cell_id" not in out.columns
    assert label == "dish (n)"
    assert out.loc[out["dish"] == "D1", label].iloc[0] == "D1 (n=4)"


def test_the_label_name_is_suffixed_when_it_would_overwrite_a_real_column():
    df = _paired_df()
    df["dish (n)"] = "do not overwrite me"

    out, label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert label == "dish (n).1"
    assert label in out.columns


def test_a_missing_row_id_column_is_not_required():
    df = _paired_df().drop(columns=["cell_id"])
    out, label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert len(out) == 6
    assert label in out.columns


# ---------------------------------------------------------------------------
# Degenerate and defensive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("collapse_by", [None, "", "not_a_column"])
def test_no_collapse_column_returns_the_frame_untouched(collapse_by):
    df = _paired_df()
    out, label, varied = collapse_rows(df, collapse_by, ["treatment"], "cell_id")

    assert out is df
    assert label == "cell_id"
    assert varied == []


def test_none_and_duplicate_slot_columns_are_ignored():
    """The page passes [*color_by, separate_by] raw, so both are routine."""
    df = _paired_df()
    a, _l, _v = collapse_rows(df, "dish", ["treatment", None, "treatment", "dish"], "cell_id")
    b, _l, _v = collapse_rows(df, "dish", ["treatment"], "cell_id")

    pd.testing.assert_frame_equal(a, b)


def test_an_absent_slot_column_is_ignored():
    df = _paired_df()
    out, _label, _varied = collapse_rows(df, "dish", ["treatment", "nope"], "cell_id")

    assert len(out) == 6


def test_empty_color_by_collapses_on_the_replicate_alone():
    df = _paired_df()
    out, _label, _varied = collapse_rows(df, "dish", [], "cell_id")

    assert len(out) == 6
    assert "treatment" in out.columns     # coarser than dish, so it survives


def test_an_empty_frame_survives():
    df = _paired_df().iloc[:0]
    out, label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    assert len(out) == 0
    assert label in out.columns


def test_row_order_is_deterministic_across_calls():
    """The sina jitter is seeded per group and indexes by row position, so a
    reshuffle between two runs would move points for no reason."""
    df = _paired_df()
    a, _l, _v = collapse_rows(df, "dish", ["treatment"], "cell_id")
    b, _l, _v = collapse_rows(df, "dish", ["treatment"], "cell_id")

    pd.testing.assert_frame_equal(a, b)


def test_column_order_follows_the_input_frame():
    df = _paired_df()
    out, label, _varied = collapse_rows(df, "dish", ["treatment"], "cell_id")

    survivors = [c for c in df.columns if c in out.columns]
    assert list(out.columns) == survivors + [label]


# ---------------------------------------------------------------------------
# The inlining contract -- src/export_script.py copies this source verbatim
# ---------------------------------------------------------------------------

def test_the_module_imports_no_streamlit_and_nothing_from_src():
    """Export inlining strips project imports and does not resolve transitive dependencies."""
    import src.collapse as module

    source = inspect.getsource(module)
    assert "streamlit" not in source
    assert "from src." not in source
    assert "import src" not in source


def test_collapse_rows_calls_no_private_helper_of_its_own():
    """The exported function is self-contained because _extract_source does not copy helpers."""
    import src.collapse as module

    private = [name for name in vars(module)
               if name.startswith("_") and not name.startswith("__")
               and callable(getattr(module, name))]
    assert private == []
