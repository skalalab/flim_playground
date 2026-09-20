"""One overview beside a column of highlight maps, all fitted per level x color."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
import streamlit as st

from src.column_roles import code_span
from src.vis import bivar, helpers


def frame():
    rows = []
    for day, offset in [("Day 10", 10), ("Day 2", 0), (None, 20)]:
        for treatment in ["ctrl", "drug"]:
            if day is None and treatment == "drug":
                continue
            for i in range(12):
                rows.append(dict(id=f"cell{len(rows)}", day=day, treatment=treatment,
                                 dish=f"D{i % 3}", shape=f"s{i % 2}", opacity=f"o{i % 3}",
                                 x=1. + i + offset,
                                 y=2. + (i if day == "Day 2" else -i) + offset
                                 + (i % 3) * .2))
    rows.append(dict(rows[0], id="incomplete", day="Day 99", x=np.nan))
    return pd.DataFrame(rows)


def bimodal_frame():
    """The ``ctrl`` group of two levels splits far enough for BIC to pick two.

    ``frame()`` is deliberately tight and near-collinear, so every group fits a
    single component and every fit-role assertion made on it would be vacuous.
    """
    data = frame()
    for start in (0, 24):  # Day 10 / ctrl and Day 2 / ctrl.
        data.loc[data.index[start:start + 6], ["x", "y"]] += 60.
    return data


def plot(data, **kwargs):
    kwargs.setdefault("analysis_options",
                      dict(log_x=False, log_y=False, marginal_plot_type="gaussian fit",
                           fit_regression=True, fit_gmm=True,
                           max_components=2, min_weight_threshold=.1))
    return bivar.feature_2d_distribution_plot(
        data, "id", None, "x", "y", color_by=["treatment"],
        shape_by="shape", opacity_by="opacity", separate_by="day", **kwargs)


def role(fig, name, axes=None):
    traces = [t for t in fig.data if isinstance(t.meta, dict)
              and t.meta.get("distribution_role") == name]
    if axes is None:
        return traces
    return [t for t in traces if (getattr(t, "xaxis", None) or "x") == axes]


def panel_axes(fig):
    """Panel p draws on x{p+4}/y{p+4}, in panel order."""
    levels = fig.layout.meta["distribution_categories"]
    return {level: f"x{index + 4}" for index, level in enumerate(levels)}


@pytest.fixture(autouse=True)
def settings():
    st.session_state.clear()
    st.session_state.plot_show_group_counts = True
    yield
    st.session_state.clear()


def test_every_level_gets_one_panel_in_natural_order_with_a_right_hand_label():
    fig, _, result = plot(frame())
    assert fig.layout.meta["distribution_categories"] == ["Day 2", "Day 10", "N/A"]
    composition = fig.layout.meta["distribution_facet_layout"]
    assert [name for name in composition["axes"] if name.startswith("xaxis")] == [
        "xaxis", "xaxis2", "xaxis4", "xaxis5", "xaxis6"]
    labels = [annotation.text for annotation in fig.layout.annotations]
    assert labels == ["Day 2", "Day 10", "N/A"]
    assert all(annotation.xanchor == "left" and annotation.xref == "paper"
               for annotation in fig.layout.annotations)
    axes = panel_axes(fig)
    for level, axis in axes.items():
        expected = result[result.day.fillna("N/A") == level]
        drawn = sorted(i for t in role(fig, "points", axis) for i in t.text)
        assert drawn == sorted(expected.id)


def test_the_overview_shows_every_cell_in_colour_with_no_grey_and_no_models():
    fig, _, result = plot(bimodal_frame())
    overview = role(fig, "points", "x")
    assert sum(len(t.x) for t in overview) == len(result)
    # The figure must own models for "none on the overview" to mean anything.
    assert role(fig, "fit") and role(fig, "regression")
    assert not role(fig, "context", "x")
    assert not role(fig, "fit", "x")
    assert not role(fig, "regression", "x")


def test_each_panel_puts_its_own_models_over_grey_context_of_everything_else():
    fig, _, result = plot(bimodal_frame())
    axes = panel_axes(fig)
    fits = role(fig, "fit")
    # Two of the three levels carry ellipses, so the per-panel checks below are
    # neither vacuous for the levels that have them nor blind to a stray one.
    assert {t.meta["category"] for t in fits} == {"Day 2", "Day 10"}
    for level, axis in axes.items():
        expected = result[result.day.fillna("N/A") == level]
        context = role(fig, "context", axis)
        assert sum(len(t.x) for t in context) == len(result) - len(expected)
        assert all(t.marker.color == "#b8b8b8" and t.marker.opacity == .25 for t in context)
        assert all(t.hoverinfo == "skip" and not t.showlegend for t in context)
        groups = expected.treatment.nunique()
        assert len(role(fig, "regression", axis)) == groups
        assert all(t.line.width == 1 for t in role(fig, "regression", axis))
        # Every ellipse of this level draws on this level's axes, and only here.
        assert len(role(fig, "fit", axis)) == sum(t.meta["category"] == level for t in fits)
        assert all(t.line.width == 1 for t in role(fig, "fit", axis))
        assert {t.meta["category"] for t in role(fig, "fit", axis)} <= {level}


def test_marginals_describe_the_whole_dataset_once_per_colour_group():
    fig, _, result = plot(frame())
    marginals = role(fig, "marginal")
    assert len(marginals) == 2 * result.treatment.nunique()
    assert {t.yaxis for t in marginals if t.xaxis in (None, "x")} == {"y2"}
    assert {t.xaxis for t in marginals if t.xaxis == "x2"} == {"x2"}
    # One curve per colour over every cell, not one per level: "ctrl" spans all
    # three days (offsets 0, 10, 20), so a curve computed from only one day's
    # subset would be narrower than the group's true min/max across the frame.
    ctrl_curve = next(t for t in marginals if t.name == "ctrl_x_density")
    ctrl_rows = result[result.treatment == "ctrl"]
    assert ctrl_curve.x[0] == pytest.approx(ctrl_rows.x.min())
    assert ctrl_curve.x[-1] == pytest.approx(ctrl_rows.x.max())
    assert fig.layout.yaxis2.range is None
    assert fig.layout.xaxis2.range is None


def test_panels_share_the_overviews_ranges_so_zooming_one_moves_all():
    fig, _, _ = plot(frame())
    axes = panel_axes(fig)
    assert fig.layout.xaxis.range is not None and fig.layout.yaxis.range is not None
    for index in range(len(axes)):
        suffix = str(index + 4)
        assert fig.layout[f"xaxis{suffix}"].matches == "x"
        assert fig.layout[f"yaxis{suffix}"].matches == "y"
        assert not fig.layout[f"xaxis{suffix}"].showticklabels
        assert not fig.layout[f"yaxis{suffix}"].showticklabels


def test_one_shared_legend_counts_each_colour_over_the_whole_dataset():
    fig, _, result = plot(frame())
    styled = helpers.apply_plot_styling(go.Figure(fig), 9, 18, 14)
    legends = [t.name for t in styled.data if t.showlegend]
    for group, count in result.treatment.value_counts().items():
        assert sum(name.startswith(group) for name in legends) == 1
        assert any(name.startswith(group) and str(count) in name for name in legends)
    assert not [t for t in styled.data
                if t.showlegend and (getattr(t, "xaxis", None) or "x") != "x"
                and isinstance(t.meta, dict)]


def test_styling_shrinks_panel_points_and_sizes_row_labels_by_the_legend_font():
    fig, _, _ = plot(frame())
    styled = helpers.apply_plot_styling(go.Figure(fig), 9, 18, 14)
    axes = panel_axes(styled)
    for trace in role(styled, "points", "x"):
        assert trace.marker.size == 9
    for axis in axes.values():
        for trace in role(styled, "points", axis) + role(styled, "context", axis):
            assert trace.marker.size == 7
    assert all(annotation.font.size == 14 for annotation in styled.layout.annotations)


def test_statistics_list_every_level_in_panel_order_and_no_category_metadata():
    fig, table_md, _ = plot(frame())
    meta = fig.layout.meta
    for key in ("distribution_summaries", "distribution_statistics_summaries",
                "distribution_category", "distribution_summary"):
        assert key not in meta
    assert not hasattr(bivar, "select_distribution_category")
    statistics = meta["distribution_statistics"]
    heading = f"**{code_span('day')}: {{level}}**"
    for text in (table_md, statistics):
        positions = [text.index(heading.format(level=code_span(level)))
                    for level in ["Day 2", "Day 10", "N/A"]]
        assert positions == sorted(positions)
    assert statistics.count("Pearson r =") == 5
    assert "flim-gmm-table" not in statistics
    # BIC keeps every group of this fixture at a single component, so the split
    # of GMM tables from the statistics text -- and their panel order -- is
    # proved on a copy that is bimodal in two of the three levels.
    split_fig, split_md, _ = plot(bimodal_frame())
    split_meta = split_fig.layout.meta
    assert "flim-gmm-table" in split_md
    assert "flim-gmm-table" not in split_meta["distribution_statistics"]
    categories = [table["category"] for table in split_meta["gmm_component_tables"]]
    # Both bimodal levels contribute a table, and "Day 2" precedes "Day 10"
    # because panel order, not the frame's row order, decides.
    assert categories == ["Day 2", "Day 10"]
    assert categories == sorted(categories, key=split_meta["distribution_categories"].index)
    assert set(categories) <= {"Day 2", "Day 10", "N/A"}


def test_level_names_with_markdown_metacharacters_do_not_leak_formatting():
    """A level name carrying Markdown syntax stays literal text in the heading.

    Level names come from the file, and every level is joined into ONE markdown
    string. HTML-escaping alone does not neutralise Markdown: measured through a
    CommonMark renderer, a level named ``[a](http://evil)`` escaped with
    ``html.escape`` renders as a live ``<a href>``. Wrapping it in a code span
    does neutralise it. (A lone ``*`` does NOT in fact pair across levels --
    do not weaken this to an asterisk case and conclude it is covered.)
    """
    data = frame()
    linked, starred = "[a](http://evil)", "10uM *"
    data.loc[data["day"] == "Day 10", "day"] = linked
    data.loc[data["day"] == "Day 2", "day"] = starred
    fig, table_md, _ = plot(data)
    statistics = fig.layout.meta["distribution_statistics"]
    for text in (table_md, statistics):
        for name in (linked, starred):
            assert f"**{code_span('day')}: {code_span(name)}**" in text
            # Every occurrence of the raw name sits inside its code span, so
            # none is left as loose Markdown for the renderer to interpret.
            assert text.count(name) == text.count(code_span(name))


def test_models_fit_each_category_colour_group_and_labels_stay_qualified(monkeypatch):
    calls = []
    real = bivar._find_best_gmm

    def capture(values, **kwargs):
        calls.append(np.asarray(values).copy())
        return real(values, **kwargs)

    monkeypatch.setattr(bivar, "_find_best_gmm", capture)
    _fig, _, result = plot(frame())
    assert len(calls) == 5
    for (day, treatment), group in result.groupby([result.day.fillna("N/A"), "treatment"]):
        assert any(np.array_equal(group[["x", "y"]].to_numpy(), call) for call in calls)
        labels = group["2D_GMM_group"].dropna()
        assert all(label.startswith(f"{day}::{treatment}_group") for label in labels)


def test_source_frame_is_not_mutated_and_encodings_stay_global():
    data = frame()
    original = data.copy(deep=True)
    fig, _, result = plot(data)
    pd.testing.assert_frame_equal(data, original)
    assert len(result) == 60
    colors = {}
    for trace in role(fig, "points"):
        colors.setdefault(trace.legendgroup, trace.marker.color)
        assert trace.marker.color == colors[trace.legendgroup]
    assert {s for t in role(fig, "points") for s in t.marker.symbol} == {"circle", "square"}


def test_duplicate_indices_and_ids_do_not_merge_panel_memberships():
    data = frame()
    data.index = [0] * len(data)
    data.id = "same"
    fig, _, result = plot(data)
    for level, axis in panel_axes(fig).items():
        actual = sorted((x, y) for t in role(fig, "points", axis) for x, y in zip(t.x, t.y))
        expected = sorted(map(tuple, result.loc[result.day.fillna("N/A") == level,
                                                ["x", "y"]].to_numpy()))
        assert actual == expected


def test_constant_group_keeps_points_and_explains_the_missing_model():
    data = frame().iloc[:3].copy()
    data.y = 1.
    fig, table_md, _ = plot(data)
    assert sum(len(t.x) for t in role(fig, "points")) == 6  # overview plus its one panel
    assert not role(fig, "regression")
    assert len(role(fig, "marginal")) == 1
    assert "constant" in table_md.lower()


def test_webgl_overlays_use_the_foreground_renderer(monkeypatch):
    monkeypatch.setattr(helpers, "WEBGL_POINT_THRESHOLD", 0)
    fig, _, _ = plot(bimodal_frame())
    overlays = (role(fig, "points") + role(fig, "context")
                + role(fig, "regression") + role(fig, "fit"))
    assert role(fig, "fit")  # Ellipses must be among the traces being checked.
    assert all(t.type == "scattergl" for t in overlays)


@pytest.mark.parametrize("separator", [["day"], "absent", "treatment"])
def test_invalid_separator_rejected(separator):
    with pytest.raises(ValueError, match="Separate by"):
        bivar.feature_2d_distribution_plot(frame(), "id", None, "x", "y",
                                           color_by=["treatment"], separate_by=separator,
                                           analysis_options={})


def test_marginal_none_removes_the_strip_axes_and_fills_the_frame():
    fig, _, _ = bivar.feature_2d_distribution_plot(
        frame(), "id", None, "x", "y", color_by=["treatment"],
        analysis_options=dict(log_x=False, log_y=False, marginal_plot_type="None",
                              fit_regression=False, fit_gmm=False,
                              max_components=2, min_weight_threshold=.1))
    assert fig.layout.xaxis.domain == (0., 1.)
    assert fig.layout.yaxis.domain == (0., 1.)
    # Plotly's Layout only grows an "xaxis2"-style slot once something assigns to
    # it; the None case never does, so the absent key itself is the assertion.
    # (Accessing fig.layout["xaxis2"] here raises PlotlyKeyError, it does not
    # return an axis whose domain is None.)
    for axis in ("xaxis2", "yaxis2", "yaxis3"):
        assert axis not in fig.layout
    assert not [trace for trace in fig.data
                if getattr(trace, "yaxis", None) in ("y2", "y3")
                or getattr(trace, "xaxis", None) == "x2"]


def test_separated_marginal_none_gives_the_overview_the_whole_block():
    fig, _, _ = plot(frame(), analysis_options=dict(
        log_x=False, log_y=False, marginal_plot_type="None",
        fit_regression=False, fit_gmm=False, max_components=2, min_weight_threshold=.1))
    composition = fig.layout.meta["distribution_facet_layout"]
    overview_width = composition["axes"]["xaxis"][1]
    assert composition["axes"]["yaxis"] == [0., 1.]
    assert "xaxis2" not in composition["axes"]
    assert overview_width == pytest.approx(3 * .96 / 4)
    assert composition["plot_height"] == pytest.approx(overview_width)
