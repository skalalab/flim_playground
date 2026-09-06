"""Execute standalone Feature Comparison exports under Agg for SuperPlot parity."""

import contextlib
import re
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.collections import PathCollection
from matplotlib.markers import MarkerStyle
from scipy.stats import ttest_ind

from src.export_script import generate_script
from src.vis.helpers import _density_at_points, cohens_d


def _state(*, overlay="SuperPlot", separate_by="day", logged=False, collapse_by="dish"):
    return {
        "csv_filename": "superplot.csv", "unique_row_id_col": "cell_id",
        "fov_name_col": None, "method": "Feature Comparison",
        "categorical_filters": {"keep": ["yes"]}, "numerical_filters": [("other", "<", 10)],
        "color_by": ["treatment"], "separate_by": separate_by,
        "shape_by": None, "opacity_by": None, "subcolor_by": None,
        "point_size": 8, "axis_label_size": 12, "legend_size": 10,
        "show_group_counts": True, "colormap": "tab10",
        "categorical_cols": ["day", "treatment", "dish", "keep", "shape", "opacity", "varied"],
        "method_params": {
            "selected_var": "value", "overlay": overlay, "collapse_by": collapse_by,
            "log_y": logged, "effect_size_method": "None", "statistical_test": "None",
            "mean_or_median": "Mean", "effect_size_threshold": 0.0,
            "connect_means": False,
        },
    }


def _source(*, separate=True):
    rows = []
    for day, shift in ([("Day 10", 100), ("Day 2", 0)] if separate else [("Day 2", 0)]):
        for treatment, means in [("ctrl", [2., 4., 8.]), ("drug", [20., 24., 31.])]:
            for replicate, mean in enumerate(means):
                offsets = [[0.], [-1., 1.], [-3., -1., 1., 3.]][replicate]
                for cell, offset in enumerate(offsets):
                    rows.append({
                        "cell_id": f"cell{len(rows)}", "day": day,
                        "treatment": treatment, "dish": f"D{replicate}",
                        "shape": f"s{replicate % 2}", "opacity": f"o{replicate}",
                        "varied": f"cell{cell}", "value": shift + mean + offset,
                        "other": 1., "keep": "yes",
                    })
    for cell_id, value, keep, other in [
        ("missing-feature", np.nan, "yes", 1.),
        ("categorically excluded", -1000., "no", 1.),
        ("numerically excluded", -1000., "yes", 20.),
    ]:
        row = rows[0].copy()
        row.update(cell_id=cell_id, value=value, keep=keep, other=other)
        rows.append(row)
    return pd.DataFrame(rows)


def _retained(source):
    return source[(source["keep"] == "yes") & (source["other"] < 10)].dropna(subset=["value"])


def _run(tmp_path, monkeypatch, state, source, *, script_transform=None):
    monkeypatch.setattr(plt, "show", lambda: None)
    tmp_path.mkdir(parents=True, exist_ok=True)
    source.to_csv(tmp_path / state["csv_filename"], index=False)
    script = generate_script(state)
    if script_transform:
        script = script_transform(script)
    path = tmp_path / "analysis.py"
    path.write_text(script)
    monkeypatch.chdir(tmp_path)
    try:
        return runpy.run_path(str(path))
    finally:
        plt.close("all")


def _points(ns, zorder):
    return [artist for artist in ns["ax"].collections
            if isinstance(artist, PathCollection) and artist.get_zorder() == zorder
            and len(artist.get_offsets())]


def _offsets(ns, zorder):
    artists = _points(ns, zorder)
    return np.concatenate([artist.get_offsets() for artist in artists]) if artists else np.empty((0, 2))


@pytest.mark.parametrize("separate_by", [None, "day"])
def test_superplot_centers_primary_points_and_keeps_filtered_source_counts_and_means(
    tmp_path, monkeypatch, separate_by
):
    source = _source(separate=bool(separate_by))
    plain_state = _state(overlay="None", separate_by=separate_by)
    plain_state["method_params"]["connect_means"] = True
    plain = _run(tmp_path / "plain", monkeypatch, plain_state, source)
    super_state = _state(separate_by=separate_by)
    super_state["method_params"]["connect_means"] = True
    ns = _run(tmp_path / "super", monkeypatch, super_state, source)

    assert "source_df" in ns, "SuperPlot must preserve filtered cells before collapse"
    assert ns["OVERLAY"] == "SuperPlot"
    assert ns["source_df"]["cell_id"].tolist() == _retained(source)["cell_id"].tolist()
    pd.testing.assert_frame_equal(ns["df"], plain["df"])
    np.testing.assert_allclose(_offsets(ns, 2)[:, 1], _offsets(plain, 2)[:, 1])
    for x, _ in _offsets(ns, 2):
        assert min(abs(x - center) for center in ns["x_positions"].values()) <= 0.1
    assert len(_offsets(ns, 1)) == len(_retained(source))
    assert ns["group_counts"] == plain["group_counts"] == {
        "ctrl": 6 if separate_by else 3, "drug": 6 if separate_by else 3}
    assert [text.get_text() for text in ns["ax"].get_legend().get_texts()] == [
        text.get_text() for text in plain["ax"].get_legend().get_texts()]
    connected = [line for line in ns["ax"].lines if line.get_marker() == "o"]
    expected = [line for line in plain["ax"].lines if line.get_marker() == "o"]
    assert len(connected) == len(expected)
    for actual, before in zip(connected, expected):
        np.testing.assert_allclose(actual.get_xdata(), before.get_xdata())
        np.testing.assert_allclose(actual.get_ydata(), before.get_ydata())
    assert (tmp_path / "super" / "feature_comparison.svg").exists()


@pytest.mark.parametrize("logged", [False, True])
def test_superplot_summary_uses_sample_sem_of_replicate_means_and_draws_capped_intervals(
    tmp_path, monkeypatch, logged
):
    source = _source()
    ns = _run(tmp_path, monkeypatch, _state(logged=logged), source)
    assert "superplot_summary_df" in ns, "SuperPlot must compute summary from primary df"
    primary = _retained(source).groupby(["dish", "treatment", "day"], sort=False)["value"].mean()
    if logged:
        primary = np.log10(primary + 1e-6)
        np.testing.assert_allclose(ns["source_df"]["value"], np.log10(_retained(source)["value"] + 1e-6))
    expected = primary.reset_index().groupby(["day", "treatment"], sort=False)["value"].agg(
        count="count", mean="mean", sem="sem")
    actual = ns["superplot_summary_df"]
    np.testing.assert_allclose(actual.loc[expected.index, ["count", "mean", "sem"]], expected)
    assert all(1 < line.get_zorder() < 2 for line in ns["ax"].lines
               if line.get_linestyle() == "-")
    for (day, color), summary in actual.iterrows():
        x = ns["x_positions"][(day, color)]
        horizontal = [line for line in ns["ax"].lines
                      if len(line.get_xdata()) == 2 and np.ptp(line.get_xdata()) > 0
                      and np.allclose(line.get_ydata(), summary["mean"])]
        assert any(np.mean(line.get_xdata()) == pytest.approx(x) for line in horizontal)
        vertical = [line for line in ns["ax"].lines
                    if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), x)]
        assert any(np.allclose(sorted(line.get_ydata()),
                               [summary["mean"] - summary["sem"], summary["mean"] + summary["sem"]])
                   for line in vertical)
        for y in [summary["mean"] - summary["sem"], summary["mean"] + summary["sem"]]:
            assert any(len(line.get_xdata()) == 2 and np.ptp(line.get_xdata()) > 0
                       and np.mean(line.get_xdata()) == pytest.approx(x)
                       and np.allclose(line.get_ydata(), y) for line in ns["ax"].lines)
    assert ns["ax"].get_ylabel() == ("log₁₀(value)" if logged else "value")


def test_single_replicate_draws_mean_without_sem_and_explains_why(tmp_path, monkeypatch, capsys):
    source = _source(separate=False)
    source = source[source["dish"] == "D0"]
    ns = _run(tmp_path, monkeypatch, _state(separate_by=None), source)
    assert "superplot_summary_df" in ns
    assert ns["superplot_summary_df"]["count"].tolist() == [1, 1]
    assert ns["superplot_summary_df"]["sem"].isna().all()
    mean_lines = [line for line in ns["ax"].lines
                  if len(line.get_ydata()) == 2 and np.ptp(line.get_ydata()) == 0]
    assert len(mean_lines) == 2
    output = capsys.readouterr().out
    assert "SEM" in output and "one replicate" in output


def test_source_negative_disables_log_for_both_layers_and_export_label(tmp_path, monkeypatch, capsys):
    source = _source(separate=False)
    idx = source.index[(source["treatment"] == "ctrl") & (source["dish"] == "D1")]
    source.loc[idx, "value"] = [-2., 10.]
    ns = _run(tmp_path, monkeypatch, _state(separate_by=None, logged=True), source)
    assert ns["LOG_Y"] is False, "Raw negative values must refuse both layer transforms"
    np.testing.assert_allclose(ns["source_df"]["value"], _retained(source)["value"])
    expected = _retained(source).groupby(["dish", "treatment"], sort=False)["value"].mean()
    np.testing.assert_allclose(ns["df"]["value"], expected)
    assert ns["ax"].get_ylabel() == "value"
    assert "log₁₀(value)" not in (tmp_path / "feature_comparison.svg").read_text()
    assert "Cannot apply log" in capsys.readouterr().out


@pytest.mark.parametrize("collapse_by", [None, "missing_column"])
def test_superplot_requires_a_real_collapse_column_in_editable_scripts(
    tmp_path, monkeypatch, collapse_by
):
    with pytest.raises(ValueError, match="SuperPlot requires.*Collapse"):
        _run(tmp_path, monkeypatch, _state(collapse_by=collapse_by), _source())


def test_editing_overlay_in_an_uncollapsed_script_enforces_prerequisite(tmp_path, monkeypatch):
    state = _state(overlay="None", collapse_by=None)
    with pytest.raises(ValueError, match="SuperPlot requires.*Collapse"):
        _run(tmp_path, monkeypatch, state, _source(), script_transform=lambda script:
             script.replace("OVERLAY = 'None'", "OVERLAY = 'SuperPlot'"))


def test_script_without_collapse_logic_rejects_adding_superplot_constants(tmp_path, monkeypatch):
    state = _state(overlay="None", collapse_by=None)
    with pytest.raises(ValueError, match="SuperPlot requires.*Collapse.*[Rr]egenerate"):
        _run(tmp_path, monkeypatch, state, _source(), script_transform=lambda script:
             script.replace("OVERLAY = 'None'", "COLLAPSE_BY = 'dish'\nOVERLAY = 'SuperPlot'"))


@pytest.mark.parametrize("statistical_test", ["Independent t-test", "Welch's t-test"])
def test_statistics_stay_on_primary_data_while_brackets_clear_raw_values(
    tmp_path, monkeypatch, statistical_test
):
    source = _source(separate=False)
    selected = source.index[(source["treatment"] == "drug") & (source["dish"] == "D2")]
    source.loc[selected, "value"] = [-70., 31., 33., 130.]
    state = _state(separate_by=None)
    state["method_params"].update(statistical_test=statistical_test,
                                 effect_size_method="Absolute Cohen's d")
    ns = _run(tmp_path, monkeypatch, state, source)
    ctrl = ns["df"].loc[ns["df"]["treatment"] == "ctrl", "value"].to_numpy()
    drug = ns["df"].loc[ns["df"]["treatment"] == "drug", "value"].to_numpy()
    assert ns["es"] == pytest.approx(cohens_d(ctrl, drug, "Mean"))
    assert ns["pval"] == pytest.approx(ttest_ind(
        ctrl, drug, equal_var=statistical_test == "Independent t-test").pvalue)
    raw = _retained(source)["value"]
    assert ns["data_range"] == pytest.approx(raw.max() - raw.min())
    bracket = [line for line in ns["ax"].lines if len(line.get_xdata()) == 4]
    assert len(bracket) == 1
    assert min(bracket[0].get_ydata()) > raw.max()


@pytest.mark.parametrize("legacy", [False, True])
def test_overlay_defaults_from_legacy_boxplot_but_explicit_none_takes_precedence(
    tmp_path, monkeypatch, legacy
):
    source = _source(separate=False)
    state = _state(overlay="None", separate_by=None)
    state["method_params"].pop("overlay")
    state["method_params"]["add_boxplot"] = legacy
    ns = _run(tmp_path / "legacy", monkeypatch, state, source)
    assert ns.get("OVERLAY") == ("Boxplot" if legacy else "None")
    assert len(ns["ax"].patches) == (2 if legacy else 0)
    assert all(artist.get_sizes() == pytest.approx([state["point_size"]])
               for artist in _points(ns, 2))
    state["method_params"]["overlay"] = "None"
    plain = _run(tmp_path / "explicit", monkeypatch, state, source)
    assert not plain["ax"].patches
    np.testing.assert_allclose(_offsets(ns, 2), _offsets(plain, 2))


@pytest.mark.parametrize("channels", ["none", "opacity", "shape", "combined", "dropped"])
@pytest.mark.parametrize("point_size", [0.5, 8])
def test_layers_reuse_effective_styles_with_independent_group_density_jitter(
    tmp_path, monkeypatch, channels, point_size
):
    source = _source()
    state = _state()
    state["point_size"] = point_size
    state["method_params"]["custom_order"] = {
        "compare_groups": ["drug", "ctrl"], "separate_groups": ["Day 10", "Day 2"]}
    if channels in ("shape", "combined"):
        state["shape_by"] = "shape"
    if channels in ("opacity", "combined"):
        state["opacity_by"] = "opacity"
    if channels == "combined":
        state["subcolor_by"] = "dish"
    if channels == "dropped":
        state.update(shape_by="varied", opacity_by="varied", subcolor_by="varied")
    ns = _run(tmp_path, monkeypatch, state, source)
    assert len(_points(ns, 1)) > 0, "SuperPlot needs a raw background layer"
    labels = [text.get_text() for text in ns["ax"].get_legend().get_texts()]
    counts = ns["subcolor_counts"] if channels == "combined" else ns["group_counts"]
    assert set(label for label in labels if "\nn=" in label) == {
        f"{level}\nn={count}" for level, count in counts.items()}
    assert len([label for label in labels if "\nn=" in label]) == len(counts)
    if channels == "dropped":
        assert ns["SHAPE_BY"] is ns["OPACITY_BY"] is ns["SUBCOLOR_BY"] is None
        assert not ns["shape_map"] and not ns["opacity_map"] and not ns["subcolor_of"]
    for layer, zorder, diameter, opacity_scale in [
        (ns["source_df"], 1, max(3, point_size * 0.75), 0.3),
        (ns["df"], 2, point_size * 1.5, 1),
    ]:
        rendered = {}
        for artist in _points(ns, zorder):
            offsets, faces, paths = artist.get_offsets(), artist.get_facecolors(), artist.get_paths()
            for index, (x, y) in enumerate(offsets):
                rendered[(round(x, 9), round(y, 9))] = (
                    faces[index % len(faces)], np.sqrt(artist.get_sizes()[0]),
                    paths[index % len(paths)].vertices)
        assert len(rendered) == len(layer)
        for (day, color), group in layer.groupby(["day", "_color_group"], sort=False):
            y = group["value"].to_numpy()
            densities = _density_at_points(y)
            density_scale = densities / max(densities) if max(densities) > 0 else np.ones(len(y))
            jitter_width = 0.1 if zorder == 2 else 0.35
            expected_x = ns["x_positions"][(day, color)] + np.random.default_rng(42).uniform(
                -1, 1, len(y)) * density_scale * jitter_width
            for x, (_, row) in zip(expected_x, group.iterrows()):
                rgba, actual_diameter, vertices = rendered[(round(x, 9), round(row["value"], 9))]
                expected_rgb = (ns["subcolor_of"][row[ns["SUBCOLOR_BY"]]] if ns["subcolor_of"]
                                else ns["color_map"][color][:3])
                alpha = ns["opacity_map"][row[ns["OPACITY_BY"]]] if ns["OPACITY_BY"] else 1.0
                np.testing.assert_allclose(rgba, [*expected_rgb[:3], alpha * opacity_scale])
                assert actual_diameter == pytest.approx(diameter)
                marker = MarkerStyle(ns["shape_map"][row[ns["SHAPE_BY"]]] if ns["SHAPE_BY"] else "o")
                np.testing.assert_allclose(vertices, marker.get_path().transformed(marker.get_transform()).vertices)


def test_no_color_groups_keep_missing_sections_and_replicate_counts(tmp_path, monkeypatch):
    source = _source()
    source.loc[source["day"] == "Day 10", "day"] = None
    state = _state()
    state["color_by"] = []
    ns = _run(tmp_path, monkeypatch, state, source)
    assert list(ns["x_positions"]) == [("Day 2", "all_data"), ("N/A", "all_data")]
    assert len(ns["source_df"]) == len(_retained(source))
    assert len(ns["df"]) == 6
    assert ns["group_counts"] == {"all_data": 6}
    summary = ns["superplot_summary_df"].droplevel("_color_group")
    for day in ["Day 2", "N/A"]:
        values = ns["df"].loc[ns["df"]["day"] == day, "value"]
        assert summary.loc[day, "mean"] == pytest.approx(values.mean())
        assert summary.loc[day, "sem"] == pytest.approx(values.std(ddof=1) / np.sqrt(3))
        assert summary.loc[day, "count"] == 3
    assert [text.get_text() for text in ns["ax"].get_legend().get_texts()] == ["all_data\nn=6"]


def test_selected_statistical_only_bracket_clears_intermediate_group_in_its_section(
    tmp_path, monkeypatch
):
    source = _source()
    middle = _retained(source).loc[lambda frame: frame["treatment"] == "drug"].copy()
    middle["cell_id"] += "-middle"
    middle["treatment"] = "middle"
    middle["value"] += 1000
    source = pd.concat([source, middle], ignore_index=True)
    state = _state()
    state["method_params"].update(
        statistical_test="Independent t-test", selected_pairs=["ctrl vs drug"],
        custom_order={"compare_groups": ["ctrl", "middle", "drug"]})
    ns = _run(tmp_path, monkeypatch, state, source)
    brackets = [line for line in ns["ax"].lines if len(line.get_xdata()) == 4]
    assert len(brackets) == 2
    for day, line in zip(ns["ordered_separate_groups"], brackets):
        raw = ns["source_df"].loc[ns["source_df"]["day"] == day, "value"]
        assert min(line.get_ydata()) > raw.max()
        assert line.get_ydata()[1] == pytest.approx(raw.max() + .05 * ns["data_range"])
        assert sorted(set(line.get_xdata())) == [ns["x_positions"][(day, "ctrl")],
                                                 ns["x_positions"][(day, "drug")]]


def test_explicit_boxplot_works_with_stale_false_legacy_setting(tmp_path, monkeypatch):
    state = _state(overlay="Boxplot", separate_by=None)
    state["method_params"]["add_boxplot"] = False
    ns = _run(tmp_path, monkeypatch, state, _source(separate=False))
    assert ns["OVERLAY"] == "Boxplot"
    assert len(ns["ax"].patches) == 2
    assert "source_df" not in ns and "superplot_summary_df" not in ns


@pytest.mark.parametrize("subcolor_by", [None, "dish"])
@pytest.mark.parametrize("logged", [False, True])
def test_app_and_export_match_both_layers_and_sem_with_custom_section_order(
    tmp_path, monkeypatch, subcolor_by, logged
):
    import streamlit as st

    from src.collapse import collapse_rows
    from src.vis import univar
    from src.vis.helpers import apply_plot_styling

    source = _source()
    state = _state(logged=logged)
    custom_order = {"compare_groups": ["drug", "ctrl"], "separate_groups": ["Day 10", "Day 2"]}
    state.update(shape_by="shape", opacity_by="opacity", subcolor_by=subcolor_by)
    state["method_params"]["custom_order"] = custom_order
    ns = _run(tmp_path, monkeypatch, state, source)

    monkeypatch.setattr(st, "session_state", {"plot_show_group_counts": True})
    monkeypatch.setattr(st, "columns", lambda widths, **kwargs:
                        [contextlib.nullcontext() for _ in widths])
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kwargs:
                        logged if label == "Log Y" else value)
    monkeypatch.setattr(st, "selectbox", lambda label, options, **kwargs:
                        "SuperPlot" if label == "Overlay" else options[0])
    monkeypatch.setattr(st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "black")
    retained = _retained(source).reset_index(drop=True)
    primary, label, _ = collapse_rows(retained, "dish", ["treatment", "day"], "cell_id")
    app = univar.feature_comparison_plot(
        primary, label, None, "value", ["treatment"], shape_by="shape", opacity_by="opacity",
        separate_by="day", custom_order=custom_order, subcolor_by=subcolor_by,
        row_id_label="dish", collapse_by="dish", source_df=retained,
        source_row_id_col="cell_id", source_row_id_label="cell_id")
    app = apply_plot_styling(app, state["point_size"], state["axis_label_size"], state["legend_size"])

    for role, zorder in [("observation", 1), ("replicate", 2)]:
        app_points = []
        for trace in app.data:
            if not isinstance(trace.meta, dict) or trace.meta.get("superplot_role") != role:
                continue
            red, green, blue, alpha = map(float, re.findall(r"[\d.]+", trace.marker.color))
            opacities = np.broadcast_to(np.asarray(trace.marker.opacity), len(trace.x))
            app_points.extend((float(x), float(y), red / 255, green / 255, blue / 255,
                               alpha * opacity, trace.marker.size)
                              for x, y, opacity in zip(trace.x, trace.y, opacities))
        exported_points = []
        for artist in _points(ns, zorder):
            colors = artist.get_facecolors()
            exported_points.extend((float(x), float(y), *colors[index % len(colors)],
                                    np.sqrt(artist.get_sizes()[0]))
                                   for index, (x, y) in enumerate(artist.get_offsets()))
        app_points, exported_points = np.asarray(sorted(app_points)), np.asarray(sorted(exported_points))
        assert len(app_points) == len(retained if role == "observation" else primary)
        np.testing.assert_allclose(exported_points[:, :2], app_points[:, :2], rtol=1e-10, atol=1e-10)
        # Plotly serializes palette RGB channels as integer bytes; Matplotlib retains floats.
        np.testing.assert_allclose(exported_points[:, 2:5], app_points[:, 2:5], rtol=0, atol=1 / 255)
        np.testing.assert_allclose(exported_points[:, 5:], app_points[:, 5:])

    app_summary = {float(np.mean(trace.x[:2])): trace.meta for trace in app.data
                   if isinstance(trace.meta, dict) and trace.meta.get("superplot_role") == "summary"}
    assert len(app_summary) == len(ns["superplot_summary_df"]) == 4
    for key, row in ns["superplot_summary_df"].iterrows():
        expected = app_summary[ns["x_positions"][key]]
        for field in ["count", "mean", "sem"]:
            assert row[field] == pytest.approx(expected[field])
    assert list(app.layout.xaxis.tickvals) == ns["tick_positions"]
    assert app.layout.yaxis.title.text == ns["ax"].get_ylabel()


@pytest.mark.parametrize("separate_by", ["count", "mean", "sem"])
def test_section_column_names_cannot_overwrite_superplot_statistics(
    tmp_path, monkeypatch, separate_by
):
    source = _source().rename(columns={"day": separate_by})
    state = _state(separate_by=separate_by)
    state["categorical_cols"] = [separate_by if column == "day" else column
                                  for column in state["categorical_cols"]]
    state["method_params"].update(statistical_test="Independent t-test")
    ns = _run(tmp_path, monkeypatch, state, source)

    summary = ns["superplot_summary_df"]
    assert summary.index.names == [separate_by, "_color_group"]
    assert summary.columns.tolist() == ["count", "mean", "sem"]
    expected = (_retained(source).groupby(["dish", "treatment", separate_by], sort=False)["value"]
                .mean().reset_index().groupby([separate_by, "treatment"], sort=False)["value"]
                .agg(count="count", mean="mean", sem="sem"))
    np.testing.assert_allclose(summary.loc[expected.index], expected)
    assert set(ns["display_df"][separate_by]) == {"Day 2", "Day 10"}
    assert len(_offsets(ns, 1)) == len(_retained(source))
    assert len(_offsets(ns, 2)) == 12
    for section_and_color, row in summary.iterrows():
        x = ns["x_positions"][section_and_color]
        assert any(len(line.get_xdata()) == 2 and np.ptp(line.get_xdata()) > 0
                   and np.mean(line.get_xdata()) == pytest.approx(x)
                   and np.allclose(line.get_ydata(), row["mean"])
                   for line in ns["ax"].lines)
    brackets = [line for line in ns["ax"].lines if len(line.get_xdata()) == 4]
    assert len(brackets) == 2
    for section, line in zip(ns["ordered_separate_groups"], brackets):
        values = ns["source_df"].loc[ns["source_df"][separate_by] == section, "value"]
        assert min(line.get_ydata()) > values.max()
