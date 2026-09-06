"""Execute Histogram exports to verify stacked panels and analyzed-row parity."""

import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.export_script import generate_script
from src.vis.histogram import histogram_bin_edges, prepare_histogram


def _state(*, separate_by="day", color_by=None, gmm=False,
           logged=False, intersection=False, width=None, max_components=2):
    return {
        "csv_filename": "histogram.csv", "unique_row_id_col": "cell_id",
        "fov_name_col": None, "method": "Feature Histogram",
        "categorical_filters": {}, "numerical_filters": [],
        "color_by": ["treatment"] if color_by is None else color_by,
        "separate_by": separate_by, "shape_by": None, "opacity_by": None,
        "axis_label_size": 12, "legend_size": 10, "show_group_counts": True,
        "colormap": "tab10", "categorical_cols": ["day", "treatment", "dish", "keep"],
        "method_params": {
            "selected_var": "value",
            "log_x": logged, "apply_gmm": gmm, "bin_width": width,
            "intersection_threshold": intersection,
            "gmm_max_components": max_components, "gmm_min_weight_threshold": 0.1,
        },
    }


def _source():
    rng = np.random.default_rng(735)
    rows = []
    for day, shift in [("Day 10", 20), ("Day 2", 0), (None, 40)]:
        for treatment, offset in [("ctrl", 0), ("drug", 2)]:
            for replicate in range(12):
                for cell in range(2):
                    value = shift + offset + (2 if replicate < 6 else 8) + rng.normal(0, .08)
                    rows.append({
                        "cell_id": f"row-{len(rows)}", "day": day,
                        "treatment": treatment, "dish": f"dish{replicate}",
                        "value": value, "other": float(cell), "keep": "yes",
                    })
    missing = rows[0].copy()
    missing.update(cell_id="missing", value=np.nan, other=1000.)
    rows.append(missing)
    return pd.DataFrame(rows)


def _run(tmp_path, monkeypatch, state, source, *, save=False, script_transform=None):
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    source.to_csv(tmp_path / state["csv_filename"], index=False)
    script = generate_script(state)
    if script_transform:
        script = script_transform(script)
    if save:
        assert "SAVE_DERIVED_DATA = False" in script
        script = script.replace("SAVE_DERIVED_DATA = False", "SAVE_DERIVED_DATA = True")
    path = tmp_path / "analysis.py"
    path.write_text(script)
    monkeypatch.chdir(tmp_path)
    try:
        return runpy.run_path(str(path))
    finally:
        plt.close("all")


def _normalized(source):
    source = source.copy()
    source["day"] = source["day"].fillna("N/A")
    return source


def _legend_labels(axis):
    legend = axis.get_legend()
    assert legend is not None, "Every Histogram subplot needs its own legend"
    return [text.get_text() for text in legend.get_texts()]


def _group_legend(group, *, gmm=False, show_counts=True):
    label = group["color_group"] + (" GMM" if gmm else "")
    count = f" (n={group['count']})" if show_counts else ""
    if gmm:
        return f"{label}{count}"
    skewness = group["skewness"]
    skew = f"{skewness:.3f}" if np.isfinite(skewness) else "undefined"
    return f"{label}{count}\nskew={skew}"


def _assert_upper_right_legend(axis):
    box = axis.get_window_extent()
    legend = axis.get_legend()
    assert legend is not None
    legend_box = legend.get_window_extent()
    font_pixels = legend.get_texts()[0].get_fontsize() * axis.figure.dpi / 72
    assert 0 <= box.x1 - legend_box.x1 <= font_pixels
    assert 0 <= box.y1 - legend_box.y1 <= font_pixels


def _assert_outside_right_legend(axis):
    box = axis.get_window_extent()
    legend = axis.get_legend()
    assert legend is not None
    legend_box = legend.get_window_extent()
    assert legend_box.x0 >= box.x1
    assert legend_box.y1 == pytest.approx(box.y1, abs=1)


@pytest.mark.parametrize("color_by", [["treatment"], []])
def test_count_export_draws_all_natural_panels_with_shared_bins_ranges_and_local_legends(
    tmp_path, monkeypatch, capsys, color_by
):
    source = _source()
    ns = _run(tmp_path, monkeypatch, _state(color_by=color_by), source)
    expected = prepare_histogram(_normalized(source), "value", color_by, "day")
    axes = ns["fig"].axes
    assert len(axes) == 3
    assert [axis.get_title() for axis in axes] == ["Day 2", "Day 10", "N/A"]
    assert [axis.get_xlabel() for axis in axes] == ["", "", "value"]
    np.testing.assert_allclose(ns["bin_edges"], expected["bin_edges"])
    for axis, panel in zip(axes, expected["panels"]):
        assert axis.get_xlim() == pytest.approx(expected["x_range"])
        assert axis.get_ylim() == pytest.approx(expected["y_range"])
        assert axis.get_ylabel() == "Count"
        assert len(axis.lines) == len(panel["groups"])
        for line, group in zip(axis.lines, panel["groups"]):
            np.testing.assert_allclose(line.get_xdata(), expected["bin_centers"])
            np.testing.assert_array_equal(line.get_ydata(), group["counts"])
            assert sum(line.get_ydata()) == group["count"]
            assert line.get_color() == ns["color_map"][group["color_group"]][:3]
    ns["fig"].canvas.draw()
    bounds = [axis.get_position().bounds for axis in axes]
    assert all(bound[0] == pytest.approx(bounds[0][0]) for bound in bounds)
    assert all(bound[2] == pytest.approx(bounds[0][2]) for bound in bounds)
    assert bounds[0][1] > bounds[1][1] > bounds[2][1]
    assert [axis.xaxis.get_visible() for axis in axes] == [False, False, True]
    assert [axis.spines["bottom"].get_visible() for axis in axes] == [False, False, True]
    assert not ns["fig"].legends
    for axis, panel in zip(axes, expected["panels"]):
        _assert_upper_right_legend(axis)
        assert _legend_labels(axis) == [_group_legend(group) for group in panel["groups"]]
    output = capsys.readouterr().out
    for panel in expected["panels"]:
        for group in panel["groups"]:
            assert f"{group['label']} (n={group['count']})" in output
    svg = (tmp_path / "feature_histogram.svg").read_text()
    assert all(category in svg for category in ["Day 2", "Day 10", "N/A"])
    assert "day:" not in svg
    assert len(list(tmp_path.glob("*.svg"))) == 1


@pytest.mark.parametrize("gmm", [False, True])
@pytest.mark.parametrize("separate_by", [None, "day"])
def test_export_preserves_individual_units_after_filters_and_logs_despite_stale_collapse(
    tmp_path, monkeypatch, gmm, separate_by
):
    source = _source()
    excluded = source.iloc[0].copy()
    excluded["cell_id"], excluded["value"], excluded["keep"] = "excluded", 10000., "no"
    source = pd.concat([source, excluded.to_frame().T], ignore_index=True)
    state = _state(gmm=gmm, logged=True, separate_by=separate_by)
    state["method_params"]["collapse_by"] = "dish"  # Obsolete saved Histogram option.
    state["categorical_filters"] = {"keep": ["yes"]}
    state["numerical_filters"] = [("other", "<", 2000)]
    ns = _run(tmp_path, monkeypatch, state, source, save=gmm)
    assert "COLLAPSE_BY" not in ns
    assert "collapse_rows" not in ns
    assert ns.get("SEPARATE_BY") == separate_by
    expected = _normalized(source[source["keep"] == "yes"].dropna(subset=["value"])).astype(
        {"value": float, "other": float}).reset_index(drop=True)
    expected["value"] = np.log10(expected["value"] + 1e-6)
    analyzed = ns["df"]
    assert len(analyzed) == len(expected)
    assert list(analyzed.columns) == list(expected.columns) + (["GMM_group"] if gmm else [])
    pd.testing.assert_frame_equal(analyzed[expected.columns].reset_index(drop=True), expected)
    expected_edges = (np.histogram_bin_edges(expected["value"], bins=1) if gmm
                      else histogram_bin_edges(expected["value"]))
    assert ns["bin_edges"] == pytest.approx(expected_edges)
    assert len(ns["fig"].axes) == (3 if separate_by else 1)
    assert [axis.get_xlabel() for axis in ns["fig"].axes] == (
        ["", "", "log₁₀(value)"] if separate_by else ["log₁₀(value)"])
    assert [axis.xaxis.get_visible() for axis in ns["fig"].axes] == (
        [False, False, True] if separate_by else [True])
    assert analyzed["cell_id"].tolist() == expected["cell_id"].tolist()
    if gmm:
        saved = pd.read_csv(tmp_path / "gmm_grouped_data.csv")
        assert len(saved) == len(expected)
        assert saved["GMM_group"].fillna("").tolist() == analyzed["GMM_group"].fillna("").tolist()


@pytest.mark.parametrize("intersection", [False, True])
def test_gmm_export_fits_each_local_population_and_saves_every_qualified_assignment(
    tmp_path, monkeypatch, capsys, intersection
):
    source = _source()
    ns = _run(tmp_path, monkeypatch, _state(gmm=True, intersection=intersection), source, save=True)
    saved = pd.read_csv(tmp_path / "gmm_grouped_data.csv")
    assert len(saved) == len(source) - 1
    assert "GMM_group" in saved
    for day, treatment, label in saved[["day", "treatment", "GMM_group"]].itertuples(index=False):
        category = "N/A" if pd.isna(day) else day
        assert label.startswith(f"{category}::{treatment}_group")
    expected = prepare_histogram(_normalized(source), "value", ["treatment"], "day",
                                 apply_gmm=True, max_components=2,
                                 intersection_threshold=intersection)
    assert saved["GMM_group"].tolist() == expected["df"]["GMM_group"].tolist()
    axes = ns["fig"].axes
    assert len(axes) == 3
    for axis, panel in zip(axes, expected["panels"]):
        assert axis.get_ylabel() == "Density"
        assert axis.get_xlim() == pytest.approx(expected["x_range"])
        assert axis.get_ylim() == pytest.approx(expected["y_range"])
        curves = [line for line in axis.lines if len(line.get_xdata()) == 1000]
        assert len(curves) == 6
        for index, group in enumerate(panel["groups"]):
            main, *components = curves[index * 3:index * 3 + 3]
            np.testing.assert_allclose(main.get_xdata(), group["x"])
            np.testing.assert_allclose(main.get_ydata(), group["pdf"])
            for line, component in zip(components, group["components"]):
                assert line.get_color() == main.get_color()
                assert line.get_linestyle() != "-"
                np.testing.assert_allclose(line.get_ydata(), component["density"])
        assert not axis.texts  # Full threshold values stay in the category summaries.
        assert len(axis.lines) - len(curves) == (2 if intersection else 0)
    assert not ns["fig"].legends
    for axis, panel in zip(axes, expected["panels"]):
        expected_labels = []
        for group in panel["groups"]:
            expected_labels += [_group_legend(group, gmm=True),
                                f"{group['color_group']} Component 1",
                                f"{group['color_group']} Component 2"]
        assert _legend_labels(axis) == expected_labels
    output = capsys.readouterr().out
    assert output.count("H-index:") == 6
    assert output.count("| Component |") == 6
    assert output.count("Threshold between component") == (6 if intersection else 0)
    assert "def prepare_histogram(" in (tmp_path / "analysis.py").read_text()
    # Executing the embedded helper with duplicate indices must retain positional labels.
    duplicate = _normalized(source)
    duplicate.index = [0] * len(duplicate)
    repeated = ns["prepare_histogram"](duplicate, "value", ["treatment"], "day",
                                       apply_gmm=True, max_components=2,
                                       intersection_threshold=intersection)
    assert repeated["df"]["GMM_group"].tolist() == expected["df"]["GMM_group"].tolist()


def test_sparse_and_failed_gmm_groups_keep_rows_counts_and_local_notices(
    tmp_path, monkeypatch, capsys
):
    from sklearn.mixture import GaussianMixture
    import scipy.optimize

    original_fit = GaussianMixture.fit

    def fit_or_fail(self, values, *args, **kwargs):
        if values.min() > 100:
            raise ValueError("singular local fit")
        return original_fit(self, values, *args, **kwargs)

    def no_intersection(*args, **kwargs):
        raise ValueError("no bracket")

    monkeypatch.setattr(GaussianMixture, "fit", fit_or_fail)
    monkeypatch.setattr(scipy.optimize, "brentq", no_intersection)
    source = _source()
    source = source[source["day"].notna()].dropna(subset=["value"]).copy()
    source.loc[source["day"] == "Day 10", "value"] += 100
    sparse = source.iloc[0].copy()
    sparse["cell_id"], sparse["day"], sparse["value"], sparse["treatment"] = "sparse", None, 5., "sparse"
    source = pd.concat([source, sparse.to_frame().T], ignore_index=True)
    ns = _run(tmp_path, monkeypatch, _state(gmm=True, intersection=True), source, save=True)
    saved = pd.read_csv(tmp_path / "gmm_grouped_data.csv")
    assert len(saved) == len(source)
    assert saved.loc[saved["day"] == "Day 2", "GMM_group"].notna().all()
    assert saved.loc[saved["day"] != "Day 2", "GMM_group"].isna().all()
    assert len(ns["fig"].axes) == 3
    assert _legend_labels(ns["fig"].axes[-1]) == ["sparse GMM (n=1)"]
    output = capsys.readouterr().out
    assert "day=Day 10 | ctrl (n=24)" in output
    assert "day=N/A | sparse (n=1)" in output
    assert "skewness = undefined" in output
    assert "skewed" not in output and "symmetric" not in output
    assert "GMM fitting failed: singular local fit" in output
    assert "Intersection threshold is unavailable; using hard assignment" in output


def test_single_component_gmm_preserves_all_unassigned_rows_and_density_curves(
    tmp_path, monkeypatch, capsys
):
    source = _source()
    ns = _run(tmp_path, monkeypatch, _state(gmm=True, max_components=1), source, save=True)
    saved = pd.read_csv(tmp_path / "gmm_grouped_data.csv")
    assert len(saved) == len(source) - 1
    assert "GMM_group" in saved
    assert saved["GMM_group"].isna().all()
    assert len(ns["fig"].axes) == 3
    assert all(len(axis.lines) == 2 for axis in ns["fig"].axes)
    for axis, panel in zip(ns["fig"].axes, ns["histogram_data"]["panels"]):
        assert _legend_labels(axis) == [_group_legend(group, gmm=True) for group in panel["groups"]]
    assert capsys.readouterr().out.count("H-index: 0.000") == 6


@pytest.mark.parametrize("width", [None, 100., 0., -1.])
def test_short_population_uses_the_same_validated_bin_width_as_the_app(
    tmp_path, monkeypatch, width
):
    source = _source().iloc[:2].copy()
    source["value"] = [1., 2.]
    ns = _run(tmp_path, monkeypatch, _state(width=width), source)
    assert ns["bin_edges"] == pytest.approx(histogram_bin_edges(source["value"], width))
    assert len(ns["fig"].axes) == 1
    assert ns["fig"].axes[0].get_title() == "Day 10"


@pytest.mark.parametrize("values,logged", [([1., 3.], True), ([-1., 3.], False)])
def test_log_guard_uses_individual_values_and_labels_only_an_applied_transform(
    tmp_path, monkeypatch, values, logged
):
    source = _source().iloc[:2].copy()
    source["value"] = values
    state = _state(logged=True)
    state["method_params"]["collapse_by"] = "dish"
    ns = _run(tmp_path, monkeypatch, state, source)
    expected = np.log10(np.array(values) + 1e-6) if logged else values
    assert ns["df"]["value"].tolist() == pytest.approx(expected)
    assert ns["LOG_X"] is logged
    assert ns["fig"].axes[0].get_xlabel() == ("log₁₀(value)" if logged else "value")


def test_constant_count_population_has_a_visible_marker_and_exact_count(
    tmp_path, monkeypatch, capsys
):
    source = _source().iloc[:4].copy()
    source["value"] = 5.
    ns = _run(tmp_path, monkeypatch, _state(), source)
    line, = ns["fig"].axes[0].lines
    assert line.get_marker() == "o"
    np.testing.assert_array_equal(line.get_xdata(), [5.])
    np.testing.assert_array_equal(line.get_ydata(), [4])
    assert ns["fig"].axes[0].get_legend().legend_handles[0].get_marker() == "o"
    assert _legend_labels(ns["fig"].axes[0]) == ["ctrl (n=4)\nskew=undefined"]
    output = capsys.readouterr().out
    assert "skewness = undefined" in output
    assert "constant observations" not in output


def test_count_script_can_enable_gmm_by_editing_its_configuration(tmp_path, monkeypatch):
    ns = _run(
        tmp_path, monkeypatch, _state(), _source(),
        script_transform=lambda script: script.replace("APPLY_GMM = False", "APPLY_GMM = True"))
    assert ns["APPLY_GMM"] is True
    assert ns["SAVE_DERIVED_DATA"] is False
    assert ns["df"]["GMM_group"].notna().all()
    assert len(ns["fig"].axes) == 3
    assert ns["fig"].axes[0].get_ylabel() == "Density"
    assert not (tmp_path / "gmm_grouped_data.csv").exists()


def test_default_off_grouping_can_be_enabled_in_the_export_configuration(tmp_path, monkeypatch):
    def enable_grouping(script):
        assert "SEPARATE_BY = None" in script
        assert "COLLAPSE_BY" not in script
        return script.replace("SEPARATE_BY = None", "SEPARATE_BY = 'day'")

    state = _state(separate_by=None)
    state["axis_label_size"] = 28
    state["legend_size"] = 20
    ns = _run(tmp_path, monkeypatch, state, _source(), script_transform=enable_grouping)
    assert len(ns["df"]) == 144
    assert len(ns["fig"].axes) == 3
    ns["fig"].canvas.draw()
    assert not ns["fig"].legends
    for axis in ns["fig"].axes:
        _assert_upper_right_legend(axis)


@pytest.mark.parametrize("width", [None, .25, 10.])
def test_gmm_export_ignores_the_saved_count_bin_width(tmp_path, monkeypatch, width):
    source = _source()
    ns = _run(tmp_path, monkeypatch, _state(gmm=True, width=width), source)
    values = source["value"].dropna()
    # CSV parsing can round the source extrema by one floating-point unit.
    np.testing.assert_allclose(ns["bin_edges"], [values.min(), values.max()], rtol=1e-14)
    for axis in ns["fig"].axes:
        assert axis.get_xlim() == pytest.approx([values.min(), values.max()])


@pytest.mark.parametrize("gmm", [False, True])
@pytest.mark.parametrize("show_counts", [False, True])
def test_local_legends_have_optional_counts_and_skewness_only_in_count_mode(
    tmp_path, monkeypatch, capsys, gmm, show_counts
):
    rows = []
    populations = [("Day 10", "ctrl", [1., 2., 3., 4., 5.]),
                   ("Day 10", "drug", [2., 2.]),
                   ("Day 2", "ctrl", [1., 1., 1., 5., np.nan]),
                   ("Day 2", "drug", [1., 2., 3.]),
                   (None, "ctrl", [8.])]
    for day, treatment, values in populations:
        for value in values:
            rows.append(dict(cell_id=f"row-{len(rows)}", day=day, treatment=treatment,
                             dish="one", keep="yes", other=0., value=value))
    state = _state(gmm=gmm, max_components=1)
    state["show_group_counts"] = show_counts
    ns = _run(tmp_path, monkeypatch, state, pd.DataFrame(rows))
    axes = ns["fig"].axes
    assert not ns["fig"].legends
    assert [axis.get_title() for axis in axes] == ["Day 2", "Day 10", "N/A"]
    suffix = " GMM" if gmm else ""
    expected = [[("ctrl", 4, "2.000"), ("drug", 3, "0.000")],
                [("ctrl", 5, "0.000"), ("drug", 2, "undefined")],
                [("ctrl", 1, "undefined")]]
    ns["fig"].canvas.draw()
    for axis, panel in zip(axes, expected):
        (_assert_outside_right_legend if gmm else _assert_upper_right_legend)(axis)
        assert _legend_labels(axis) == [
            f"{color}{suffix}" + (f" (n={count})" if show_counts else "")
            + (f"\nskew={skew}" if not gmm else "")
            for color, count, skew in panel]
    assert len(ns["df"]) == 15
    output = capsys.readouterr().out
    assert "skewness = 2.000" in output
    assert "skewed" not in output and "symmetric" not in output
    assert "def histogram_legend_label(" in (tmp_path / "analysis.py").read_text()


@pytest.mark.parametrize("legend_size", [10, 24, 32])
def test_gmm_legends_fit_outside_aligned_axes_without_clipping_or_neighbor_overlap(
    tmp_path, monkeypatch, legend_size
):
    state = _state(gmm=True, intersection=True)
    state["axis_label_size"] = 28
    state["legend_size"] = legend_size
    ns = _run(tmp_path, monkeypatch, state, _source())
    fig = ns["fig"]
    fig.canvas.draw()
    axes = fig.axes
    axes_boxes = [axis.get_window_extent() for axis in axes]
    figure_box = fig.get_window_extent()
    assert not fig.legends
    for index, axis in enumerate(axes):
        _assert_outside_right_legend(axis)
        legend_box = axis.get_legend().get_window_extent()
        assert axes_boxes[index].x0 == pytest.approx(axes_boxes[0].x0)
        assert axes_boxes[index].x1 == pytest.approx(axes_boxes[0].x1)
        assert legend_box.x1 <= figure_box.x1
        assert legend_box.y1 <= figure_box.y1
        assert legend_box.y0 >= figure_box.y0
        if index < len(axes) - 1:
            assert legend_box.y0 > axes_boxes[index + 1].y1
        assert len(_legend_labels(axis)) == 6
        assert all("skew=" not in label for label in _legend_labels(axis))
