"""The page and exported 2D analysis use the same replicate observations."""
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import gaussian_kde

from src import dataset_io, export_script
from src.vis import bivar
from src.vis.helpers import _find_best_gmm
from src.widgets import analysis_config_widgets as acw
from src.widgets import visualization_widgets as vw

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")
X, Y = "feature_x", "feature_y"
CATEGORIES = ["treatment", "dish", "day", "image_name"]


def _frame():
    """Paired dishes with unequal cell counts and nontrivial correlations."""
    rows = []
    for treatment, ys in [("ctrl", [2, 6, 3, 9]), ("drug", [7, 4, 10, 8])]:
        for i, (x, y) in enumerate(zip([1, 2, 4, 7], ys)):
            for j in range(i + 2):
                offset = j - (i + 1) / 2
                rows.append({
                    "cell_id": f"{treatment}_{i}_{j}",
                    "treatment": treatment, "dish": f"D{i}",
                    "day": f"Day {i % 2}", "image_name": f"{treatment}_{i}_f{j % 2}",
                    X: x + 0.15 * offset, Y: y - 0.4 * offset,
                })
    # Neither incomplete pair should contribute to means or hover counts.
    rows.extend([dict(rows[0], cell_id="missing_x", **{X: np.nan, Y: 99}),
                 dict(rows[0], cell_id="missing_y", **{X: 99, Y: np.nan})])
    return pd.DataFrame(rows)


def _means(frame, color_by, logged=False):
    result = (frame.dropna(subset=[X, Y])
              .groupby(["dish", *color_by], sort=False)[[X, Y]].mean().reset_index())
    if logged:
        result[[X, Y]] = np.log10(result[[X, Y]] + 1e-6)
    return result


@pytest.fixture
def page(monkeypatch):
    from streamlit.testing.v1 import AppTest

    frame = _frame()
    seen = {"pearson": [], "gmm": []}
    monkeypatch.setattr(acw, "get_categorical_cols_analysis", lambda *a, **k: CATEGORIES)
    monkeypatch.setattr(acw, "get_fov_name_col_analysis", lambda *a, **k: "image_name")
    monkeypatch.setattr(acw, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    monkeypatch.setattr(dataset_io, "load_table", lambda *a, **k: (
        frame.copy(), {"Uncategorized Features": [X, Y]}, True, ",", "cell_id"))

    plot, generate = bivar.feature_2d_distribution_plot, export_script.generate_script
    pearson, gmm = bivar.pearsonr, bivar._find_best_gmm

    def capture_plot(df, **kwargs):
        seen["input"], seen["kwargs"] = df.copy(), kwargs
        result = plot(df, **kwargs)
        seen["fig"], seen["table"], seen["data"] = result
        return result

    def capture_export(state):
        seen["state"] = state
        seen["script"] = generate(state)
        return seen["script"]

    def capture_pearson(x, y):
        seen["pearson"].append(np.column_stack([x, y]))
        return pearson(x, y)

    def capture_gmm(data, **kwargs):
        seen["gmm"].append(np.asarray(data).copy())
        return gmm(data, **kwargs)

    monkeypatch.setattr(bivar, "feature_2d_distribution_plot", capture_plot)
    monkeypatch.setattr(export_script, "generate_script", capture_export)
    monkeypatch.setattr(bivar, "pearsonr", capture_pearson)
    monkeypatch.setattr(bivar, "_find_best_gmm", capture_gmm)

    def run(*, collapse="dish", color_by=None, logged=False, marginal="gaussian fit",
            separate=None, point_mode="shape", point_column="day", repeat_days=False):
        nonlocal frame
        if repeat_days:
            frame = pd.concat([frame, frame.assign(
                cell_id=frame.cell_id + "_repeat", day="Day 10",
                **{X: frame[X] + 20, Y: frame[Y] * 2 + 15})], ignore_index=True)
        at = AppTest.from_file(PAGE).run(timeout=90)
        assert not at.exception
        at.radio[0].set_value("### **Bivariate**")
        settings = {
            "2d_x_menu_Uncategorized Features": X,
            "2d_y_menu_Uncategorized Features": Y,
            vw.COLOR_BY_KEY: ["treatment"] if color_by is None else color_by,
            vw.COLLAPSE_BY_KEY: collapse,
            vw.PICKER_COL_KEY: point_column, vw.OPACITY_BY_KEY: "image_name",
            "vis_encoding_fd_point_mode": point_mode,
            "vis_encoding_fd_separate_by": separate,
            f"fit_regression_2d_{X}_{Y}": True,
            f"fit_gmm_2d_{X}_{Y}": True,
            f"log_x_2d_{X}_{Y}": logged, f"log_y_2d_{X}_{Y}": logged,
            f"marginal_plot_type_selector_{X}_{Y}": marginal,
            "plot_show_group_counts": True,
        }
        for key, value in settings.items():
            at.session_state[key] = value
        at.run(timeout=90)
        assert not at.exception, at.exception
        assert "fig" in seen, "The 2D feature selection did not render a plot."
        return at, seen, frame

    return run


@pytest.mark.parametrize("marginal, logged", [
    ("gaussian fit", False), ("gaussian fit", True), ("boxplot", False), ("violin", False),
])
def test_every_2d_statistic_uses_the_colored_replicate_means(page, marginal, logged):
    at, seen, frame = page(marginal=marginal, logged=logged)
    expected = _means(frame, ["treatment"], logged)
    assert len(seen["data"]) == 8
    assert seen["kwargs"]["row_id_label"] == "dish"
    assert seen["kwargs"]["fov_name_col"] is None
    assert seen["kwargs"]["shape_by"] == "day"
    assert seen["kwargs"]["opacity_by"] is None
    assert "**Opacity by** is off" not in " ".join(c.value for c in at.caption)
    assert seen["state"]["method_params"]["collapse_by"] == "dish"
    assert seen["state"]["opacity_by"] is None
    assert len(seen["pearson"]) == len(seen["gmm"]) == 2

    regression_lines = [t for t in seen["fig"].data
                        if "Regression Line" in (t.hovertemplate or "")]
    assert len(regression_lines) == 2
    for i, (group, means) in enumerate(expected.groupby("treatment", sort=True)):
        values = means[[X, Y]].to_numpy()
        np.testing.assert_allclose(seen["pearson"][i], values)
        np.testing.assert_allclose(seen["gmm"][i], values)
        slope, intercept = np.polyfit(means[X], means[Y], 1)
        line = regression_lines[i]
        np.testing.assert_allclose(line.y, slope * np.asarray(line.x) + intercept)
        for axis, feature in [("x", X), ("y", Y)]:
            trace = next(t for t in seen["fig"].data
                         if str(t.name).startswith(f"{group}_{axis}_"))
            if marginal == "gaussian fit":
                grid = np.linspace(means[feature].min(), means[feature].max(), 200)
                density = gaussian_kde(means[feature])(grid)
                np.testing.assert_allclose(trace.x if axis == "x" else trace.y, grid)
                np.testing.assert_allclose(trace.y if axis == "x" else trace.x, density)
            else:
                np.testing.assert_allclose(trace.x if axis == "x" else trace.y, means[feature])

    points = [t for t in seen["fig"].data if t.text is not None]
    assert sum(len(t.text) for t in points) == 8
    assert all("<b>dish:</b>" in t.hovertemplate for t in points)
    assert any("D0 (n=2)" in t.text for t in points)


def test_2d_collapse_without_color_keeps_one_point_per_replicate(page):
    _, seen, frame = page(color_by=[])
    expected = _means(frame, [])
    assert len(seen["data"]) == 4
    np.testing.assert_allclose(seen["pearson"][0], expected[[X, Y]])
    np.testing.assert_allclose(seen["gmm"][0], expected[[X, Y]])


def test_2d_models_fit_two_replicates_within_each_combination_of_color_columns(page):
    _, seen, frame = page(color_by=["treatment", "day"])
    expected = _means(frame, ["treatment", "day"])
    assert len(seen["data"]) == 8
    assert len(seen["pearson"]) == len(seen["gmm"]) == 4
    for i, (_, group) in enumerate(expected.groupby(["treatment", "day"], sort=True)):
        np.testing.assert_allclose(seen["pearson"][i], group[[X, Y]])
        np.testing.assert_allclose(seen["gmm"][i], group[[X, Y]])


def test_2d_keeps_single_replicate_points_without_fitting_undefined_statistics(page):
    _, seen, _ = page(collapse="day", color_by=["treatment", "dish"])
    assert len(seen["data"]) == 8
    assert seen["pearson"] == seen["gmm"] == []
    assert sum(len(t.text) for t in seen["fig"].data if t.text is not None) == 8


def test_clearing_2d_collapse_restores_cell_level_analysis(page):
    at, seen, frame = page()
    next(w for w in at.selectbox if w.label == "Collapse by").set_value(None).run(timeout=90)
    assert not at.exception
    assert len(seen["data"]) == len(frame.dropna(subset=[X, Y]))
    assert seen["kwargs"]["opacity_by"] is None
    assert seen["kwargs"]["fov_name_col"] == "image_name"
    assert seen["state"]["method_params"]["collapse_by"] is None


def test_2d_collapse_control_follows_color_by_and_survives_method_changes(page):
    from streamlit.testing.v1.element_tree import Block

    at, _, _ = page()
    # Find the encoding row rather than the enclosing analysis column.
    row = next(block for block in at.main
               if isinstance(block, Block) and len(block.children) == 4
               and any(w.label == "Collapse by" for w in block.selectbox))
    assert row.children[0].selectbox[0].label == "Separate by"
    assert row.children[1].multiselect[0].label == "Color by"
    assert row.children[2].selectbox[0].label == "Collapse by"
    at.radio[0].set_value("### **Univariate**").run(timeout=90)
    assert not at.exception
    assert at.selectbox(vw.COLLAPSE_BY_KEY).value == "dish"
    at.radio[0].set_value("### **Bivariate**").run(timeout=90)
    assert not at.exception
    assert at.selectbox(vw.COLLAPSE_BY_KEY).value == "dish"


@pytest.mark.parametrize("logged", [False, True])
def test_separated_page_collapses_reused_dishes_inside_each_category_and_color(page, logged):
    at, seen, frame = page(separate="day", repeat_days=True, logged=logged)
    expected = (frame.dropna(subset=[X, Y]).groupby(["dish", "treatment", "day"], sort=False)
                [[X, Y]].mean().reset_index())
    if logged:
        expected[[X, Y]] = np.log10(expected[[X, Y]] + 1e-6)
    assert seen["kwargs"]["separate_by"] == "day"
    pd.testing.assert_frame_equal(seen["data"][expected.columns], expected)
    assert len(seen["data"]) == 16
    assert seen["state"]["separate_by"] == "day"
    assert seen["state"]["method_params"]["distribution_category"] == "Day 0"
    for _, group in expected.groupby(["day", "treatment"]):
        values = group[[X, Y]].to_numpy()
        assert any(np.array_equal(call, values) for call in seen["pearson"])
        assert any(np.array_equal(call, values) for call in seen["gmm"])
    selector = next(w for w in at.button_group if w.key == "vis_encoding_fd_category")
    selector.set_value("Day 10").run(timeout=90)
    assert not at.exception
    assert seen["state"]["method_params"]["distribution_category"] == "Day 10"


def test_only_active_merged_decoration_is_disabled_when_collapse_drops_it(page):
    at, seen, _ = page(separate="day", point_mode="opacity", point_column="image_name")
    assert seen["kwargs"]["opacity_by"] is None
    assert seen["kwargs"]["shape_by"] is None
    assert "**Opacity by** is off" in " ".join(c.value for c in at.caption)
    assert seen["state"]["opacity_by"] is None
    assert seen["state"]["shape_by"] is None


def _export_state(**params):
    return {
        "csv_filename": "data.csv", "unique_row_id_col": "cell_id",
        "fov_name_col": "image_name", "method": "2D Feature Distribution",
        "categorical_cols": CATEGORIES, "color_by": ["treatment"],
        "shape_by": "day", "opacity_by": "image_name", "show_group_counts": True,
        "method_params": {
            "selected_x": X, "selected_y": Y, "collapse_by": "dish",
            "fit_regression": True, "fit_gmm_2d": True, **params,
        },
    }


def _run_export(tmp_path, monkeypatch, frame, state):
    frame.to_csv(tmp_path / "data.csv", index=False)
    script = export_script.generate_script(state).replace(
        "SAVE_DERIVED_DATA = False", "SAVE_DERIVED_DATA = True")
    path = tmp_path / "analysis.py"
    path.write_text(script)
    monkeypatch.chdir(tmp_path)
    try:
        return runpy.run_path(str(path))
    finally:
        plt.close("all")


@pytest.mark.parametrize("logged", [False, True])
def test_2d_export_collapses_complete_pairs_before_logging_and_model_fits(tmp_path, monkeypatch, logged):
    frame = _frame()
    ns = _run_export(tmp_path, monkeypatch, frame, _export_state(log_x=logged, log_y=logged))
    expected = _means(frame, ["treatment"], logged)
    assert len(ns["df"]) == 8
    pd.testing.assert_frame_equal(ns["df"][expected.columns], expected)
    assert ns["OPACITY_BY"] is None
    assert ns["SHAPE_BY"] == "day"
    assert ns["df"].groupby("_color_group").size().to_dict() == {"ctrl": 4, "drug": 4}
    drug = expected[expected["treatment"] == "drug"]
    np.testing.assert_allclose(ns["X_reg"], drug[[X]])
    np.testing.assert_allclose(ns["X_gmm"], drug[[X, Y]])
    saved = pd.read_csv(tmp_path / "2D_gmm_data.csv")
    pd.testing.assert_frame_equal(saved[expected.columns], expected)
    assert saved["2D_GMM_group"].notna().all()


@pytest.mark.parametrize("n_samples", [1, 2])
def test_gmm_component_search_respects_replicate_count(n_samples):
    data = np.array([[1.0, 2.0], [3.0, 5.0]])[:n_samples]
    model = _find_best_gmm(data, max_components=3)
    if n_samples == 1:
        assert model is None
    else:
        assert 1 <= model.n_components <= n_samples


def test_2d_export_fits_gmm_with_only_two_replicates_per_color(tmp_path, monkeypatch):
    frame = _frame().query("dish in ['D0', 'D1']")
    ns = _run_export(tmp_path, monkeypatch, frame, _export_state())
    assert len(ns["df"]) == 4
    assert ns["df"]["2D_GMM_group"].notna().all()
