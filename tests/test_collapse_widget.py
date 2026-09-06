"""Collapse-control wiring, channel options, plot data, and hover text. Pure collapse rules
are covered in test_collapse.py.
"""
import ast
import sys
import warnings
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from src.collapse import collapse_rows
from src.column_roles import code_span
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


def test_feature_histogram_has_no_collapse_or_point_decorations(page):
    """Histogram preserves individual-unit variability and heterogeneity."""
    at = page()
    at.radio[1].set_value("Feature Histogram")
    at.run(timeout=90)
    assert not at.exception
    assert "Separate by" in _labels(at)
    assert "Collapse by" not in _labels(at)
    assert not {"Shape by", "Subcolor by", "Opacity by"}.intersection(_labels(at))


def test_the_2d_distribution_offers_collapse_by(page):
    at = page()
    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    assert "Collapse by" in _labels(at)


# ---------------------------------------------------------------------------
# Grouping and decoration options
# ---------------------------------------------------------------------------

def test_collapse_by_is_last_in_the_grouping_chain(page):
    """Grouping constrains Collapse by; selecting a collapse column must not change
    grouping.
    """
    at = page(**{vw.COLLAPSE_BY_KEY: "dish", "vis_encoding_color_by": ["treatment"]})

    assert "dish" in _options(at, "Separate by")
    assert "dish" in next(m.options for m in at.multiselect
                          if m.label in ("Color by", "Group by"))
    assert "treatment" not in _options(at, "Collapse by")


def test_grouping_on_the_collapsed_column_retires_the_collapse(page):
    """A grouping change that conflicts with Collapse by clears the collapse selection."""
    at = page(**{vw.COLLAPSE_BY_KEY: "dish", "vis_encoding_color_by": ["dish"]})

    assert "dish" not in _options(at, "Collapse by")
    assert at.session_state[vw.COLLAPSE_BY_KEY] is None


@pytest.mark.parametrize("mode", ["shape", "subcolor", "opacity"])
def test_the_collapse_column_is_still_offered_to_the_decoration_channel(page, mode):
    """The collapse column stays available for one decoration per replicate."""
    at = page(**{vw.COLLAPSE_BY_KEY: "dish", "vis_encoding_point_mode": mode})
    assert "dish" in _options(at, f"{mode.title()} by")


def test_feature_comparison_has_one_direct_three_way_selector(page):
    at = page()
    group = at.get("button_group")[0]
    assert group.label == "Point encoding"
    assert [option.content for option in group.options] == ["Opacity", "Subcolor", "Shape"]
    assert group.value == "shape"
    assert "Shape by" in _labels(at)
    assert not {"Subcolor by", "Opacity by"}.intersection(_labels(at))

    for mode in ("subcolor", "opacity", "shape"):
        at.get("button_group")[0].set_value([mode]).run(timeout=90)
        assert not at.exception
        assert set(_labels(at)).intersection({"Shape by", "Subcolor by", "Opacity by"}) == {
            f"{mode.title()} by"}
        assert _colour(at).label == ("Group by" if mode == "subcolor" else "Color by")

    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    group = at.get("button_group")[0]
    assert [option.content for option in group.options] == ["Opacity", "Shape"]
    assert group.value == "shape"
    assert set(_labels(at)).intersection({"Shape by", "Subcolor by", "Opacity by"}) == {
        "Shape by"}


@pytest.mark.parametrize("mode", ["shape", "subcolor", "opacity"])
def test_a_color_grouping_column_can_also_encode_points(page, mode):
    """One category may identify color groups and reinforce them through another encoding."""
    at = page(**{vw.COLOR_BY_KEY: ["treatment"], vw.POINT_MODE_KEY: mode,
                 vw.PICKER_COL_KEY: "treatment"})
    assert "treatment" in _options(at, f"{mode.title()} by")
    assert at.selectbox(vw.PICKER_COL_KEY).value == "treatment"

    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    for fd_mode in ("shape", "opacity"):
        picker = _point_mode(at, fd_mode)
        assert "treatment" in picker.options
        assert picker.value == "treatment"


def _opacity(at):
    return next(box for box in at.selectbox if box.label == "Opacity by")


def _colour(at):
    return next(m for m in at.multiselect if m.label in ("Color by", "Group by"))


def _point_mode(at, mode):
    at.get("button_group")[0].set_value([mode]).run(timeout=90)
    assert not at.exception
    return next(box for box in at.selectbox if box.label == f"{mode.title()} by")


def test_switches_preserve_the_latest_column_and_clear_stays_clear(page):
    at = page()
    next(box for box in at.selectbox if box.label == "Shape by").select("day").run(timeout=90)
    for mode in ("subcolor", "opacity", "shape"):
        assert _point_mode(at, mode).value == "day"

    _point_mode(at, "opacity").select("dish").run(timeout=90)
    assert _point_mode(at, "subcolor").value == "dish"
    _point_mode(at, "subcolor").set_value(None).run(timeout=90)
    assert _point_mode(at, "opacity").value is None
    at.run(timeout=90)
    assert not at.exception
    assert at.session_state[vw.PICKER_COL_KEY] is None


def test_clicking_the_active_segment_keeps_the_last_mode(page):
    at = page()
    _point_mode(at, "opacity")
    at.get("button_group")[0].set_value([]).run(timeout=90)
    assert not at.exception
    assert at.get("button_group")[0].value == "opacity"
    assert "Opacity by" in _labels(at)


def test_subcolor_without_groups_is_disabled_but_remembers_the_column(page):
    at = page(**{"vis_encoding_point_mode": "subcolor", vw.PICKER_COL_KEY: "day"})
    _colour(at).set_value([]).run(timeout=90)
    assert not at.exception
    picker = next(box for box in at.selectbox if box.label == "Subcolor by")
    assert picker.disabled
    assert picker.value == "day"
    assert _colour(at).label == "Group by"

    _colour(at).set_value(["treatment"]).run(timeout=90)
    assert not at.exception
    picker = next(box for box in at.selectbox if box.label == "Subcolor by")
    assert not picker.disabled
    assert picker.value == "day"


def test_other_methods_keep_independent_opacity_across_feature_comparison(page):
    at = page()
    at.radio[0].set_value("### **Multivariate**").run(timeout=90)
    assert not at.exception
    _opacity(at).select("day").run(timeout=90)
    next(box for box in at.selectbox if box.label == "Shape by").select("dish").run(timeout=90)

    at.radio[0].set_value("### **Univariate**").run(timeout=90)
    assert not at.exception
    _point_mode(at, "opacity").select("image_name").run(timeout=90)
    # An extra run exposes cleanup of widgets hidden by the method change.
    at.run(timeout=90)
    assert not at.exception

    at.radio[0].set_value("### **Multivariate**").run(timeout=90)
    assert not at.exception
    assert _opacity(at).value == "day"
    assert next(box for box in at.selectbox if box.label == "Shape by").value == "image_name"

    at.radio[0].set_value("### **Univariate**").run(timeout=90)
    assert not at.exception
    assert at.get("button_group")[0].value == "opacity"
    assert _opacity(at).value == "image_name"


@pytest.mark.parametrize("legacy, expected_mode, expected_column", [
    ({vw.AS_COLOUR_KEY: True, vw.PICKER_COL_KEY: "dish", vw.OPACITY_BY_KEY: "day"},
     "subcolor", "dish"),
    ({vw.AS_COLOUR_KEY: False, vw.PICKER_COL_KEY: "dish", vw.OPACITY_BY_KEY: "day"},
     "shape", "dish"),
    ({vw.OPACITY_BY_KEY: "day"}, "opacity", "day"),
])
def test_legacy_encoding_migrates_once(page, legacy, expected_mode, expected_column):
    at = page(**legacy)
    assert at.get("button_group")[0].value == expected_mode
    picker = next(box for box in at.selectbox if box.label == f"{expected_mode.title()} by")
    assert picker.value == expected_column
    picker.set_value(None).run(timeout=90)
    at.run(timeout=90)
    assert not at.exception
    assert at.session_state[vw.PICKER_COL_KEY] is None


def _bivariate(page):
    """Open 2D Feature Distribution with its merged point-encoding control."""
    at = page()
    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    return at


def test_opacity_survives_an_unrelated_change_to_color_by(page):
    """Changing grouping options preserves an opacity selection that is still valid."""
    at = _bivariate(page)
    at = _point_mode(at, "opacity").select("day").run(timeout=90)
    assert not at.exception
    assert _opacity(at).value == "day"

    at = _colour(at).select("dish").run(timeout=90)
    assert not at.exception
    assert "day" in _options(at, "Opacity by")
    assert _opacity(at).value == "day"


@pytest.mark.parametrize("mode", ["shape", "opacity"])
def test_grouping_on_the_held_point_encoding_column_preserves_it(page, mode):
    """Adding a point-encoding column to Color by keeps both selections active."""
    at = _bivariate(page)
    _point_mode(at, mode)
    key = vw.PICKER_COL_KEY
    at = at.selectbox(key).select("day").run(timeout=90)
    at = _colour(at).select("day").run(timeout=90)

    assert not at.exception
    assert "day" in _options(at, f"{mode.title()} by")
    assert at.selectbox(key).value == "day"
    assert at.session_state[key] == "day"


def test_nothing_is_struck_when_no_column_is_collapsed(page):
    at = page()
    assert "dish" in _options(at, "Separate by")


# ---------------------------------------------------------------------------
# Disabled-decoration captions
# ---------------------------------------------------------------------------

# The feature picker's key, so the Feature Comparison branch actually runs.
FEATURE_KEY = "_menu_Uncategorized Features"


def test_superplot_page_retains_source_and_exports_its_overlay(page, monkeypatch):
    from src import export_script
    from src.vis import univar

    frames, states, figures = [], [], []
    render = univar.feature_comparison_plot
    generate = export_script.generate_script

    def spy(df, *args, **kwargs):
        frames.append((df.copy(), kwargs.get("source_df")))
        figure = render(df, *args, **kwargs)
        figures.append(figure)
        return figure

    def capture(state):
        states.append(state)
        return generate(state)

    monkeypatch.setattr(univar, "feature_comparison_plot", spy)
    monkeypatch.setattr(export_script, "generate_script", capture)
    at = page(**{FEATURE_KEY: FEATURE, vw.COLOR_BY_KEY: ["treatment"],
                 vw.COLLAPSE_BY_KEY: "dish", vw.POINT_MODE_KEY: "subcolor",
                 vw.PICKER_COL_KEY: "dish"})
    overlay = next(box for box in at.selectbox if box.label == "Overlay")
    assert "SuperPlot" in overlay.options
    overlay.set_value("SuperPlot").run(timeout=90)
    assert not at.exception
    assert len(frames[-1][0]) == 6
    assert len(frames[-1][1]) == 36
    assert states[-1]["method_params"]["overlay"] == "SuperPlot"
    assert states[-1]["method_params"]["collapse_by"] == "dish"
    assert sum(len(t.y) for t in figures[-1].data if isinstance(t.meta, dict)
               and t.meta.get("superplot_role") == "observation") == 36


@pytest.mark.parametrize("mode", ["shape", "subcolor", "opacity"])
@pytest.mark.parametrize("column", ["day", "treatment", "image_name", None])
def test_page_exports_only_the_active_valid_decoration(page, monkeypatch, mode, column):
    """Exercise real page capture and generation after its collapse validation."""
    from src import export_script

    captured = []
    generate = export_script.generate_script

    def capture(state):
        script = generate(state)
        captured.append((state, script))
        return script

    monkeypatch.setattr(export_script, "generate_script", capture)
    at = page(**{FEATURE_KEY: FEATURE, vw.COLLAPSE_BY_KEY: "dish",
                 "vis_encoding_point_mode": mode, vw.PICKER_COL_KEY: column,
                 vw.OPACITY_BY_KEY: "day"})
    assert captured
    state, script = captured[-1]
    expected = {"shape_by": None, "subcolor_by": None, "opacity_by": None}
    if column in ("day", "treatment"):
        expected[f"{mode}_by"] = column
    assert {key: state[key] for key in expected} == expected
    assert state["method_params"]["collapse_by"] == "dish"

    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in ast.parse(script).body if isinstance(node, ast.Assign)
        if isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in {key.upper() for key in expected}
    }
    assert assignments == {key.upper(): value for key, value in expected.items()}
    assert at.get("button_group")[0].value == mode
    if column == "image_name":
        assert f"**{mode.title()} by** is off" in " ".join(c.value for c in at.caption)


def test_a_decoration_finer_than_the_replicate_is_dropped_and_explained(page):
    """A varying image_name is dropped and the caption explains why its decoration is disabled."""
    at = page(**{FEATURE_KEY: FEATURE, vw.COLLAPSE_BY_KEY: "dish",
                 vw.PICKER_COL_KEY: "image_name"})

    captions = " ".join(c.value for c in at.caption)
    # Explain which control was disabled and why the replicate cannot use it.
    assert "**Shape by** is off" in captions
    assert "covers several `image_name` values" in captions
    assert "cannot be further divided" in captions


def test_a_decoration_coarser_than_the_replicate_is_left_alone(page):
    """A day constant within each dish remains available without a disabled-decoration notice."""
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
    """The identifier value includes replicate count while keeping customdata one-
    dimensional.
    """
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
    """Collapsing by FOV retains that column for hover because it is constant in each group."""
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
# Statistics on replicate counts
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sizes", [(1, 1), (1, 3), (3, 1)])
def test_cohens_d_needs_a_degree_of_freedom_in_BOTH_groups(sizes):
    """Pooled variance requires at least two samples in each group."""
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
    """Insufficient replicate counts return NaN without emitting numpy warnings."""
    import numpy as np

    from src.vis.helpers import _calculate_effect_size

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        value = _calculate_effect_size(
            pd.Series([1.0]), pd.Series([2.0]), method, "Mean")

    assert np.isnan(value)


@pytest.mark.parametrize("method", ["Glass's Delta", "Absolute Cohen's d"])
def test_two_replicates_per_group_still_compute(method):
    """Groups with enough replicates still receive a finite effect size."""
    import numpy as np

    from src.vis.helpers import _calculate_effect_size

    value = _calculate_effect_size(
        pd.Series([1.0, 1.2]), pd.Series([2.0, 2.4]), method, "Mean")

    assert not np.isnan(value)


# ---------------------------------------------------------------------------
# Insufficient-replicate notices for selected comparisons
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
    """Warn when selected comparisons lack enough replicates to compute statistics."""
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
    """Separate-by warnings name the section whose comparison lacks replicates."""
    warned = _run_annotations(monkeypatch, {"ctrl": 4, "drug": 1}, [("ctrl", "drug")],
                              section_label="MCF7")

    # Section labels come from file values and must display literally in Markdown.
    assert f"in {code_span('MCF7')}" in warned


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
