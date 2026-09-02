"""Collapse by, from the page's side: the picker, the option lists, and the hover.

The pure rules live in `tests/test_collapse.py`; these cover the wiring only --
which method offers the control, what it takes away from the other channels, what
the plot is handed once it has run, and the caption that explains a channel the
collapse had to switch off.
"""
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from src.collapse import collapse_rows
from src.widgets import analysis_config_widgets as acw
from src.widgets import visualization_widgets as vw

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")
FEATURE = "Lifetime fit_ch1: T1"


def _replicate_frame():
    """Three dishes per treatment, two FOVs per dish, one day per dish."""
    rows = []
    for i, (dish, treatment, day) in enumerate([
        ("D1", "ctrl", "Day 1"), ("D2", "drug", "Day 1"),
        ("D3", "ctrl", "Day 2"), ("D4", "drug", "Day 2"),
        ("D5", "ctrl", "Day 3"), ("D6", "drug", "Day 3"),
    ]):
        for j in range(6):
            rows.append({
                "cell_id": f"{dish}_c{j}",
                "image_name": f"{dish}_f{j % 2}",
                "dish": dish,
                "treatment": treatment,
                "day": day,
                FEATURE: 0.4 + 0.01 * i + 0.001 * j,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def page(monkeypatch):
    """The real page, with the uploader bypassed -- AppTest cannot drive a file
    upload, so the loaded state is modelled by monkeypatching load_table."""
    from streamlit.testing.v1 import AppTest

    frame = _replicate_frame()
    monkeypatch.setattr(acw, "get_categorical_cols_analysis",
                        lambda *a, **k: ["treatment", "dish", "day", "image_name"])
    monkeypatch.setattr(acw, "get_fov_name_col_analysis", lambda *a, **k: "image_name")
    monkeypatch.setattr(dataset_io, "load_table", lambda *_a, **_k: (
        frame, {"Uncategorized Features": [FEATURE]}, True, ",", "cell_id"))

    def _run(**session):
        at = AppTest.from_file(PAGE)
        for key, value in session.items():
            at.session_state[key] = value
        at.run(timeout=90)
        assert not at.exception, at.exception
        return at

    return _run


def _options(at, label):
    return next(box.options for box in at.selectbox if box.label == label)


def _labels(at):
    return [box.label for box in at.selectbox]


# ---------------------------------------------------------------------------
# Which methods offer it
# ---------------------------------------------------------------------------

def test_feature_comparison_offers_collapse_by(page):
    at = page()
    assert "Collapse by" in _labels(at)


def test_feature_histogram_does_not(page):
    """Not point-based: there is no x slot for the replicates to sit in."""
    at = page()
    at.radio[1].set_value("Feature Histogram")
    at.run(timeout=90)
    assert not at.exception
    assert "Collapse by" not in _labels(at)


def test_the_2d_distribution_does_not(page):
    at = page()
    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    assert "Collapse by" not in _labels(at)


# ---------------------------------------------------------------------------
# What it takes away, and what it deliberately does not
# ---------------------------------------------------------------------------

def test_collapse_by_is_last_in_the_grouping_chain(page):
    """Separate by narrows Color by; the two of them narrow Collapse by -- never the
    reverse. Collapsing is DERIVED from the x layout, so changing the layout may retire a
    collapse column, but picking a replicate must leave the layout alone. Reading Collapse
    by first inverted that and silently reset the grouping."""
    at = page(**{vw.COLLAPSE_BY_KEY: "dish", "vis_encoding_color_by": ["treatment"]})

    assert "dish" in _options(at, "Separate by")
    assert "dish" in next(m.options for m in at.multiselect
                          if m.label in ("Color by", "Group by"))
    assert "treatment" not in _options(at, "Collapse by")


def test_grouping_on_the_collapsed_column_retires_the_collapse(page):
    """The yield goes downstream: one dot per x slot leaves nothing for a box or a test
    to describe, so the collapse gives way -- not the grouping."""
    at = page(**{vw.COLLAPSE_BY_KEY: "dish", "vis_encoding_color_by": ["dish"]})

    assert "dish" not in _options(at, "Collapse by")
    assert at.session_state[vw.COLLAPSE_BY_KEY] is None


def test_the_collapse_column_is_still_offered_to_the_decoration_channel(page):
    """Subcolor by = Collapse by is the SuperPlot -- one colour per replicate, held
    across every x group -- so striking it there too would remove the feature's most
    useful pairing. Feature Comparison merges the three decorations into one slot, so
    there is a single picker to check, named for whichever role is live."""
    at = page(**{vw.COLLAPSE_BY_KEY: "dish"})

    assert "dish" in _options(at, "Shape by")

    at = page(**{vw.COLLAPSE_BY_KEY: "dish", vw.AS_COLOUR_KEY: True})
    assert "dish" in _options(at, "Subcolor by")
    assert "dish" in _options(at, "Opacity by")


def test_shape_and_subcolor_share_a_slot_and_opacity_keeps_its_own(page):
    """Shape and subcolor are both NOMINAL, so either is a sensible encoding for the same
    column and the toggle between them is a real choice. Opacity is the one ORDINAL
    channel, so it competes with neither and keeps a column of its own -- on every
    point-based method, subcolor or no subcolor. Folding all three into one three-way
    switch was built and reverted: it wrapped to three lines in a column this narrow."""
    at = page()
    assert "Opacity by" in _labels(at) and "Shape by" in _labels(at)
    assert "Subcolor by" not in _labels(at)

    at = page(**{vw.AS_COLOUR_KEY: True})
    assert "Subcolor by" in _labels(at) and "Shape by" not in _labels(at)
    assert "Opacity by" in _labels(at)

    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    assert "Opacity by" in _labels(at) and "Shape by" in _labels(at)


def test_a_grouped_column_is_struck_from_every_decoration(page):
    """A decoration marks a point INSIDE the slot its grouping put it in, so a column
    already spent on Separate by or Color by would give every point in a slot the same
    mark. One rule for all three, and for the plain Shape/Opacity columns elsewhere."""
    at = page(**{"vis_encoding_color_by": ["treatment"]})
    assert "treatment" not in _options(at, "Shape by")
    assert "treatment" not in _options(at, "Opacity by")

    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    assert "treatment" not in _options(at, "Shape by")
    assert "treatment" not in _options(at, "Opacity by")


def _opacity(at):
    return next(box for box in at.selectbox if box.label == "Opacity by")


def _colour(at):
    return next(m for m in at.multiselect if m.label in ("Color by", "Group by"))


def _bivariate(page):
    """Where the decorations are NOT merged, so Opacity by is a column of its own."""
    at = page()
    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    return at


def test_opacity_survives_an_unrelated_change_to_color_by(page):
    """Striking the grouped columns put Opacity by's options under Color by's control, and
    an UNKEYED widget is identified by its arguments -- options included -- so grouping on
    ANY column remounted this picker and silently blanked a pick that was still offered.
    Feature Comparison never showed it: its opacity role goes through the merged picker,
    which has been keyed all along."""
    at = _bivariate(page)
    at = _opacity(at).select("day").run(timeout=90)
    assert not at.exception
    assert _opacity(at).value == "day"

    at = _colour(at).select("dish").run(timeout=90)
    assert not at.exception
    assert "day" in _options(at, "Opacity by")
    assert _opacity(at).value == "day"


def test_grouping_on_the_held_opacity_column_retires_it(page):
    """The other half of the same bargain: a key makes the pick survive an option list
    changing, and Streamlit then RAISES on a stored value the widget no longer offers, so
    the column Color by just claimed has to be pruned out before it renders."""
    at = _bivariate(page)
    at = _opacity(at).select("day").run(timeout=90)
    at = _colour(at).select("day").run(timeout=90)

    assert not at.exception
    assert "day" not in _options(at, "Opacity by")
    assert _opacity(at).value is None
    assert at.session_state[vw.OPACITY_BY_KEY] is None


def test_nothing_is_struck_when_no_column_is_collapsed(page):
    at = page()
    assert "dish" in _options(at, "Separate by")


# ---------------------------------------------------------------------------
# The caption -- the only thing that tells a user a channel went away
# ---------------------------------------------------------------------------

# The feature picker's key, so the Feature Comparison branch actually runs.
FEATURE_KEY = "_menu_Uncategorized Features"


def test_a_decoration_finer_than_the_replicate_is_dropped_and_explained(page):
    """`image_name` takes two values inside every dish, so a collapsed dot has no
    single symbol to carry -- and a silently missing channel reads as a bug."""
    at = page(**{FEATURE_KEY: FEATURE, vw.COLLAPSE_BY_KEY: "dish",
                 vw.PICKER_COL_KEY: "image_name"})

    captions = " ".join(c.value for c in at.caption)
    # Names the control to go and change, and ends in what to do about it -- "varies
    # within" was accurate and told nobody anything.
    assert "**Shape by** is off" in captions
    assert "covers several `image_name` values" in captions
    assert "cannot be further divided" in captions


def test_a_decoration_coarser_than_the_replicate_is_left_alone(page):
    """`day` is one value per dish, so it still marks the dots -- and the collapse says
    nothing, because there is nothing to explain."""
    at = page(**{FEATURE_KEY: FEATURE, vw.COLLAPSE_BY_KEY: "dish",
                 vw.PICKER_COL_KEY: "day"})

    assert "is off" not in " ".join(c.value for c in at.caption)


def test_no_collapse_says_nothing_at_all(page):
    """The note is reserved on every run; it must stay empty when nothing collapsed."""
    at = page(**{FEATURE_KEY: FEATURE, vw.PICKER_COL_KEY: "image_name"})

    assert "is off" not in " ".join(c.value for c in at.caption)


def test_the_page_survives_a_slot_left_with_a_single_dot(page):
    """Collapsing by a column finer than the x slot is legal and leaves one point per
    slot; the box, the KDE and the stats must all degrade rather than raise."""
    at = page(**{FEATURE_KEY: FEATURE, vw.COLLAPSE_BY_KEY: "day",
                 "vis_encoding_color_by": ["dish"]})

    assert not at.exception


# ---------------------------------------------------------------------------
# What the plot is handed
# ---------------------------------------------------------------------------

def test_the_hover_names_the_replicate_and_carries_its_count(monkeypatch):
    """`n` rides inside the identifier's VALUE rather than a second hover line: a
    second line means 2-D customdata, which would split feature_comparison_plot's
    FOV template away from bivar's."""
    import contextlib

    import streamlit as st

    from src.vis import univar

    monkeypatch.setattr(st, "columns", lambda spec, **kw: tuple(
        contextlib.nullcontext() for _ in (spec if isinstance(spec, list) else range(spec))))
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kw: value)
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "black")

    collapsed, label_col, _varied = collapse_rows(
        _replicate_frame(), "dish", ["treatment"], "cell_id")
    fig = univar.feature_comparison_plot(
        collapsed, label_col, None, FEATURE, color_by=["treatment"], row_id_label="dish")

    points = [t for t in fig.data if getattr(t, "mode", None)]
    assert points, "no point traces"
    assert "<b>dish:</b> %{text}" in points[0].hovertemplate
    assert any("(n=6)" in str(value) for t in points
               for value in (t.text if t.text is not None else []))


def test_the_fov_hover_line_survives_a_collapse_by_the_fov_column(monkeypatch):
    """resolve_effective_fov_col answers this with no new branch: the collapse drops
    the FOV column iff it varied, and collapsing BY it means it did not."""
    from src.dataset_io import resolve_effective_fov_col

    collapsed, _label, _varied = collapse_rows(
        _replicate_frame(), "image_name", ["treatment"], "cell_id")

    assert resolve_effective_fov_col(collapsed, "image_name") == "image_name"


def test_collapsing_across_fovs_drops_the_fov_hover_line(monkeypatch):
    from src.dataset_io import resolve_effective_fov_col

    collapsed, _label, _varied = collapse_rows(
        _replicate_frame(), "dish", ["treatment"], "cell_id")

    assert resolve_effective_fov_col(collapsed, "image_name") is None


# ---------------------------------------------------------------------------
# The statistics, once n is three rather than eighteen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [(1, 1), (1, 3), (3, 1)])
def test_cohens_d_needs_a_degree_of_freedom_in_BOTH_groups(sizes):
    """The pooled variance weights each group's own var(ddof=1). Guarding only the sum
    let n1=1, n2=3 through, where the answer was NaN solely because pandas returns NaN
    from Series.var of one element -- numpy warns instead, and the exported script is
    where that would print."""
    import numpy as np

    from src.vis.helpers import cohens_d

    n1, n2 = sizes
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        value = cohens_d(np.arange(n1, dtype=float), np.arange(n2, dtype=float) + 5, "Mean")

    assert np.isnan(value)


def test_glass_delta_needs_one_only_in_the_control_group():
    """Glass's delta divides by the CONTROL group's spread alone, so a single-point
    treatment group is legitimate and must not be guarded away."""
    import numpy as np

    from src.vis.helpers import glass_delta

    assert np.isnan(glass_delta(np.array([1.0]), np.array([5.0, 6.0, 7.0]), "Mean"))
    assert not np.isnan(glass_delta(np.array([1.0, 2.0, 3.0]), np.array([5.0]), "Mean"))


@pytest.mark.parametrize("method", ["Glass's Delta", "Absolute Cohen's d"])
def test_one_replicate_per_group_yields_no_effect_size_and_no_numpy_warning(method):
    """Below two points the pooled spread is undefined. Both functions already
    returned nan by accident, through two numpy RuntimeWarnings that print out of the
    exported script unexplained; the guards make that explicit and silent."""
    import numpy as np

    from src.vis.helpers import _calculate_effect_size

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        value = _calculate_effect_size(
            pd.Series([1.0]), pd.Series([2.0]), method, "Mean")

    assert np.isnan(value)


@pytest.mark.parametrize("method", ["Glass's Delta", "Absolute Cohen's d"])
def test_two_replicates_per_group_still_compute(method):
    """The guard must not swallow the case Collapse by is actually for."""
    import numpy as np

    from src.vis.helpers import _calculate_effect_size

    value = _calculate_effect_size(
        pd.Series([1.0, 1.2]), pd.Series([2.0, 2.4]), method, "Mean")

    assert not np.isnan(value)


# ---------------------------------------------------------------------------
# The notice: a group too thin for a spread, among the pairs the user picked
# ---------------------------------------------------------------------------

def _stats_frame(counts):
    """One column of values per group, with `counts[group]` rows each."""
    rows = []
    for group, n in counts.items():
        rows.extend({"grp": group, "val": float(i)} for i in range(n))
    return pd.DataFrame(rows)


def _run_annotations(monkeypatch, counts, pairs, *, effect_size="Absolute Cohen's d",
                     statistical_test="None", section_label=None, selected=None):
    """Call the real annotator with Streamlit's warning captured."""
    import plotly.graph_objects as go

    from src.vis import helpers

    warned = []
    monkeypatch.setattr(helpers.st, "warning", lambda msg, **kw: warned.append(msg))
    monkeypatch.setattr(helpers.st, "number_input", lambda *a, **kw: 0.0)

    helpers._add_effect_size_annotations(
        fig=go.Figure(), df=_stats_frame(counts), selected_var="val",
        compare_groups=list(counts), group_col_name="grp",
        all_possible_pairs=pairs, annotation_color="black",
        effect_size_method=effect_size, mean_or_median="Mean",
        selected_pairs=selected if selected is not None else pairs,
        threshold=0.0, statistical_test=statistical_test,
        section_label=section_label)
    return " ".join(warned)


def test_a_group_with_one_point_is_named(monkeypatch):
    """Both statistics return nan below two points and draw nothing, so without this
    there is no way to tell "no difference" from "not computable"."""
    warned = _run_annotations(monkeypatch, {"ctrl": 4, "drug": 1}, [("ctrl", "drug")])

    assert "`drug` (1 point)" in warned
    assert "at least two points" in warned
    assert "`ctrl`" not in warned


def test_the_notice_names_the_statistic_the_user_asked_for(monkeypatch):
    counts, pairs = {"ctrl": 4, "drug": 1}, [("ctrl", "drug")]

    effect = _run_annotations(monkeypatch, counts, pairs, statistical_test="None")
    pval = _run_annotations(monkeypatch, counts, pairs, effect_size="None",
                            statistical_test="Welch's t-test")
    both = _run_annotations(monkeypatch, counts, pairs,
                            statistical_test="Welch's t-test")

    assert "No effect size for" in effect
    assert "No p-value for" in pval
    assert "No effect size or p-value for" in both


def test_only_the_pairs_the_user_selected_are_considered(monkeypatch):
    """A thin group nobody asked to compare is not a problem to report."""
    warned = _run_annotations(
        monkeypatch, {"ctrl": 4, "drug": 4, "lonely": 1},
        [("ctrl", "drug"), ("ctrl", "lonely")], selected=[("ctrl", "drug")])

    assert warned == ""


def test_nothing_is_said_when_every_group_has_a_spread(monkeypatch):
    warned = _run_annotations(monkeypatch, {"ctrl": 4, "drug": 2}, [("ctrl", "drug")])

    assert warned == ""


def test_nothing_is_said_when_neither_statistic_was_requested(monkeypatch):
    warned = _run_annotations(monkeypatch, {"ctrl": 4, "drug": 1}, [("ctrl", "drug")],
                              effect_size="None", statistical_test="None")

    assert warned == ""


def test_the_notice_names_its_separate_by_section(monkeypatch):
    """The annotator runs once per section, so an unlabelled notice would repeat with
    no way to tell which section each one meant."""
    warned = _run_annotations(monkeypatch, {"ctrl": 4, "drug": 1}, [("ctrl", "drug")],
                              section_label="MCF7")

    assert "in MCF7" in warned


def test_several_thin_groups_are_named_in_x_axis_order(monkeypatch):
    warned = _run_annotations(
        monkeypatch, {"a": 1, "b": 4, "c": 1},
        [("a", "b"), ("b", "c"), ("a", "c")])

    assert warned.index("`a`") < warned.index("`c`")
    assert "`b`" not in warned


def test_an_empty_group_still_counts_as_thin(monkeypatch):
    """dropna() can empty a group the frame still lists; 0 < 2 covers it."""
    frame = _stats_frame({"ctrl": 4, "drug": 2})
    frame.loc[frame["grp"] == "drug", "val"] = float("nan")
    import plotly.graph_objects as go

    from src.vis import helpers

    warned = []
    monkeypatch.setattr(helpers.st, "warning", lambda msg, **kw: warned.append(msg))
    helpers._add_effect_size_annotations(
        fig=go.Figure(), df=frame, selected_var="val", compare_groups=["ctrl", "drug"],
        group_col_name="grp", all_possible_pairs=[("ctrl", "drug")],
        annotation_color="black", effect_size_method="Absolute Cohen's d",
        mean_or_median="Mean", selected_pairs=[("ctrl", "drug")], threshold=0.0)

    assert "`drug` (0 points)" in " ".join(warned)
