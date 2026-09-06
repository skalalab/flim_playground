"""Execute standalone FD exports to verify category-view and saved-data parity."""

import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import gaussian_kde

from src.export_script import generate_script


def _state(*, category="Day 2", separate_by="day", marginal="gaussian fit",
           regression=True, gmm=False, collapse=False, logged=False):
    return {
        "csv_filename": "distribution.csv",
        "unique_row_id_col": "cell_id",
        "fov_name_col": "image_name",
        "method": "2D Feature Distribution",
        "categorical_filters": {},
        "numerical_filters": [],
        "color_by": ["treatment"],
        "shape_by": "cell_line",
        "opacity_by": "dose",
        "separate_by": separate_by,
        "point_size": 5,
        "axis_label_size": 12,
        "legend_size": 10,
        "show_group_counts": True,
        "colormap": "tab10",
        "categorical_cols": ["treatment", "cell_line", "dose", "day", "dish"],
        "method_params": {
            "selected_x": "feature_x",
            "selected_y": "feature_y",
            "distribution_category": category,
            "marginal_plot_type": marginal,
            "fit_regression": regression,
            "fit_gmm_2d": gmm,
            "gmm_max_components": 2,
            "gmm_min_weight_threshold": 0.1,
            "collapse_by": "dish" if collapse else None,
            "log_x": logged,
            "log_y": logged,
        },
    }


def _source():
    rows = []
    for day, slope, shift in [("Day 10", -3, 200), ("Day 2", 2, 5), (None, 1, 40)]:
        for treatment, offset in [("ctrl", 0), ("drug", 2)]:
            for i in range(6):
                for cell in range(2):
                    x = 1.0 + i + cell * 0.2 + offset
                    rows.append({
                        "cell_id": f"row-{len(rows)}", "image_name": "image",
                        "day": day, "treatment": treatment, "dish": f"dish{i}",
                        "cell_line": "A" if cell else "B",
                        "dose": "low" if cell else "high",
                        "feature_x": x, "feature_y": slope * x + shift,
                        "metadata": f"meta-{len(rows)}",
                    })
    for col in ["feature_x", "feature_y"]:
        missing = rows[0].copy()
        missing.update(cell_id=f"missing-{col}", **{col: np.nan})
        rows.append(missing)
    return pd.DataFrame(rows)


def _run(tmp_path, monkeypatch, state, source, *, save=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source.to_csv(tmp_path / state["csv_filename"], index=False)
    script = generate_script(state)
    compile(script, "analysis.py", "exec")
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


def _points(ax):
    return [collection for collection in ax.collections if len(collection.get_offsets())]


def _foreground(ax):
    return [collection for collection in _points(ax)
            if not (np.isscalar(collection.get_alpha()) and collection.get_alpha() == 0.18)]


def test_category_view_retains_global_encodings_and_draws_local_points_and_fits(
    tmp_path, monkeypatch, capsys
):
    from sklearn.linear_model import LinearRegression

    fit_rows = []
    original_fit = LinearRegression.fit

    def record_fit(self, x, y, **kwargs):
        fit_rows.append(len(x))
        return original_fit(self, x, y, **kwargs)

    monkeypatch.setattr(LinearRegression, "fit", record_fit)
    ns = _run(tmp_path, monkeypatch, _state(category="Day 10"), _source())
    assert ns["distribution_category"] == "Day 10"
    assert sorted(fit_rows) == [12] * 6
    assert len(ns["df"]) == 72
    assert set(ns["color_map"]) == {"ctrl", "drug"}
    assert set(ns["shape_map"]) == {"A", "B"}
    assert ns["opacity_map"] == {"high": 0.3, "low": 1.0}
    assert ns["BASE_ALPHA"] == 0.8
    foreground = _foreground(ns["ax_main"])
    assert sum(len(collection.get_offsets()) for collection in foreground) == 24
    assert all(collection.get_sizes().tolist() == [25] for collection in foreground)
    assert {alpha for collection in foreground for alpha in collection.get_alpha()} == {0.3, 1.0}
    background = [collection for collection in _points(ns["ax_main"])
                  if np.isscalar(collection.get_alpha()) and collection.get_alpha() == 0.18]
    assert len(background) == 1
    assert len(background[0].get_offsets()) == 48
    assert background[0].get_facecolors()[0, :3] == pytest.approx(
        matplotlib.colors.to_rgb("#b8b8b8"))
    labels = [text.get_text() for text in ns["ax_main"].get_legend().get_texts()]
    assert labels[:2] == ["ctrl\nn=12", "drug\nn=12"]
    regressions = [line for line in ns["ax_main"].lines if line.get_linestyle() == "--"]
    assert len(regressions) == 2
    for line in regressions:
        assert np.polyfit(line.get_xdata(), line.get_ydata(), 1)[0] == pytest.approx(-3)
    output = capsys.readouterr().out
    assert output.count("Pearson r=") == 2
    assert "day=Day 10 | ctrl" in output
    assert "slope=-3.0000" in output
    assert "day=Day 2 |" not in output
    assert "day=N/A |" not in output
    assert "day: Day 10" in (tmp_path / "2d_feature_distribution.svg").read_text()


@pytest.mark.parametrize("category", [None, "missing", "N/A"])
def test_category_fallback_is_natural_and_missing_values_form_a_category(
    tmp_path, monkeypatch, category
):
    ns = _run(tmp_path, monkeypatch, _state(category=category), _source())
    assert ns["distribution_category"] == ("N/A" if category == "N/A" else "Day 2")
    assert ns["panel_levels"] == ["Day 2", "Day 10", "N/A"]


def test_marginals_use_local_samples_with_global_coordinate_and_density_scales(
    tmp_path, monkeypatch
):
    source = _source()
    views = [_run(tmp_path / day, monkeypatch, _state(category=day), source)
             for day in ["Day 2", "Day 10"]]
    for ns in views:
        assert len(ns["fig"].axes) == 3
        ns["fig"].canvas.draw()
        main_box = ns["ax_main"].get_window_extent()
        top_box = ns["ax_top"].get_window_extent()
        right_box = ns["ax_right"].get_window_extent()
        assert main_box.width == pytest.approx(main_box.height, abs=0.1)
        assert top_box.x0 == pytest.approx(main_box.x0, abs=0.1)
        assert top_box.x1 == pytest.approx(main_box.x1, abs=0.1)
        assert right_box.y0 == pytest.approx(main_box.y0, abs=0.1)
        assert right_box.y1 == pytest.approx(main_box.y1, abs=0.1)
        assert len(ns["ax_top"].lines) == len(ns["ax_right"].lines) == 2
        selected = source[source["day"] == ns["distribution_category"]].dropna(
            subset=["feature_x", "feature_y"])
        for i, treatment in enumerate(["ctrl", "drug"]):
            group = selected[selected["treatment"] == treatment]
            xline, yline = ns["ax_top"].lines[i], ns["ax_right"].lines[i]
            np.testing.assert_allclose(xline.get_ydata(), gaussian_kde(group["feature_x"])(xline.get_xdata()))
            np.testing.assert_allclose(yline.get_xdata(), gaussian_kde(group["feature_y"])(yline.get_ydata()))
    for name, getter in [("ax_main", "get_xlim"), ("ax_main", "get_ylim"),
                         ("ax_top", "get_ylim"), ("ax_right", "get_xlim")]:
        assert getattr(views[0][name], getter)() == pytest.approx(getattr(views[1][name], getter)())
    assert views[0]["ax_main"].get_xlim() != views[0]["ax_main"].get_ylim()
    assert views[0]["color_map"] == views[1]["color_map"]


@pytest.mark.parametrize("logged", [False, True])
def test_collapse_keeps_reused_replicates_distinct_by_category_before_logs(
    tmp_path, monkeypatch, logged
):
    source = _source()
    ns = _run(tmp_path, monkeypatch, _state(collapse=True, logged=logged), source)
    assert len(ns["df"]) == 36
    expected = source.dropna(subset=["feature_x", "feature_y"]).copy()
    expected["day"] = expected["day"].fillna("N/A")
    keys = ["dish", "treatment", "day"]
    expected = expected.groupby(keys)[["feature_x", "feature_y"]].mean().sort_index()
    if logged:
        expected = np.log10(expected + 1e-6)
    actual = ns["df"].set_index(keys)[expected.columns].sort_index()
    pd.testing.assert_frame_equal(actual, expected)
    assert ns["SHAPE_BY"] is ns["OPACITY_BY"] is None
    assert not ns["shape_map"] and not ns["opacity_map"]
    assert len(ns["distribution_results"]) == 6
    assert all(len(result["positions"]) == 6 for result in ns["distribution_results"])
    assert sum(len(collection.get_offsets()) for collection in _foreground(ns["ax_main"])) == 12
    if logged:
        assert ns["ax_main"].get_xlabel() == "log₁₀(feature_x)"
        assert ns["ax_main"].get_ylabel() == "log₁₀(feature_y)"


def _gmm_source():
    rng = np.random.default_rng(730)
    source = _source().dropna(subset=["feature_x", "feature_y"]).copy()
    for positions in source.groupby(source["day"].fillna("N/A")).groups.values():
        for treatment in ["ctrl", "drug"]:
            selected = [i for i in positions if source.loc[i, "treatment"] == treatment]
            for i, position in enumerate(selected):
                center = 2 if i < 6 else 15
                source.loc[position, ["feature_x", "feature_y"]] = (
                    [center, center * 2] + rng.normal(0, 0.06, 2))
    return source


@pytest.mark.parametrize("collapse,logged", [(False, False), (True, False), (True, True)])
def test_gmm_csv_contains_all_categories_and_qualified_labels_with_only_local_ellipses(
    tmp_path, monkeypatch, capsys, collapse, logged
):
    source = _gmm_source()
    state = _state(category="Day 2", gmm=True, collapse=collapse, logged=logged)
    ns = _run(tmp_path, monkeypatch, state, source, save=True)
    saved = pd.read_csv(tmp_path / "2D_gmm_data.csv", keep_default_na=False)
    assert len(saved) == (36 if collapse else 72)
    assert set(saved["day"]) == {"Day 2", "Day 10", "N/A"}
    assert all(label.startswith(f"{day}::{treatment}_group")
               for day, treatment, label in saved[["day", "treatment", "2D_GMM_group"]].itertuples(index=False))
    assert not any(column.startswith("_color_group") for column in saved)
    assert not any(column.startswith("__distribution") for column in saved)
    assert len(ns["ax_main"].patches) == 4
    assert all(len(result["components"]) == 2 for result in ns["distribution_results"])
    output = capsys.readouterr().out
    assert "day=Day 10 |" not in output and "day=N/A |" not in output
    assert "Weight" in output and "Component" in output
    expected = source.copy()
    expected["day"] = expected["day"].fillna("N/A")
    if collapse:
        expected = expected.groupby(["dish", "treatment", "day"], sort=False)[["feature_x", "feature_y"]].mean().reset_index()
    if logged:
        expected[["feature_x", "feature_y"]] = np.log10(expected[["feature_x", "feature_y"]] + 1e-6)
    keys = ["dish", "treatment", "day"] if collapse else ["cell_id"]
    pd.testing.assert_frame_equal(
        saved.set_index(keys)[["feature_x", "feature_y"]].sort_index(),
        expected.set_index(keys)[["feature_x", "feature_y"]].sort_index(),
    )


def test_sparse_and_constant_groups_keep_points_and_available_marginal_with_notices(
    tmp_path, monkeypatch, capsys
):
    source = _source()
    source = source[source["day"].notna()].copy()
    source.loc[(source["day"] == "Day 2") & (source["treatment"] == "ctrl"), "feature_y"] = 7
    drug = (source["day"] == "Day 2") & (source["treatment"] == "drug")
    source = source.drop(source[drug].index[1:])
    ns = _run(tmp_path, monkeypatch, _state(gmm=True), source, save=True)
    assert sum(len(collection.get_offsets()) for collection in _foreground(ns["ax_main"])) == 13
    assert len(ns["ax_top"].lines) == 1 and len(ns["ax_right"].lines) == 0
    assert not ns["ax_main"].lines and not ns["ax_main"].patches
    output = capsys.readouterr().out
    assert "constant X or Y" in output and "fewer than two observations" in output
    assert "day=Day 10 |" not in output
    saved = pd.read_csv(tmp_path / "2D_gmm_data.csv")
    assert saved.loc[saved["day"] == "Day 2", "2D_GMM_group"].isna().all()


def test_no_color_by_uses_one_local_population_and_opaque_color_mapping(tmp_path, monkeypatch):
    state = _state()
    state.update(color_by=[], shape_by=None, opacity_by=None)
    ns = _run(tmp_path, monkeypatch, state, _source())
    assert set(ns["color_map"]) == {"all_data"}
    assert len(ns["distribution_results"]) == 3
    assert len(_foreground(ns["ax_main"])) == 1
    assert _foreground(ns["ax_main"])[0].get_alpha() == 0.8
    assert _foreground(ns["ax_main"])[0].get_label() == "all_data\nn=24"


def test_internal_group_columns_preserve_uploaded_names_and_categories(tmp_path, monkeypatch):
    source = _gmm_source().rename(columns={"feature_x": "_color_group", "day": "_color_group_"})
    source["_color_group__"] = "original metadata"
    state = _state(gmm=True, separate_by="_color_group_")
    state["categorical_cols"] = ["_color_group_" if col == "day" else col for col in state["categorical_cols"]]
    state["method_params"]["selected_x"] = "_color_group"
    ns = _run(tmp_path, monkeypatch, state, source, save=True)
    saved = pd.read_csv(tmp_path / "2D_gmm_data.csv", keep_default_na=False)
    np.testing.assert_allclose(saved["_color_group"], source["_color_group"])
    assert saved["_color_group_"].tolist() == source["_color_group_"].fillna("N/A").tolist()
    assert saved["_color_group__"].tolist() == ["original metadata"] * len(source)
    assert "_color_group___" not in saved
    assert ns["distribution_category"] == "Day 2"


@pytest.mark.parametrize("separator,match", [
    (["day"], "Separate by must be one"),
    ("missing", "Separate by must be one"),
    ("treatment", "cannot also be used for Color by"),
])
def test_invalid_separators_fail_before_collapse(tmp_path, monkeypatch, separator, match):
    with pytest.raises(ValueError, match=match):
        _run(tmp_path, monkeypatch, _state(separate_by=separator, collapse=True), _source())


@pytest.mark.parametrize("marginal", ["gaussian fit", "boxplot", "violin", "none"])
def test_each_marginal_mode_executes_in_a_standalone_script(tmp_path, monkeypatch, marginal):
    state = _state(marginal=marginal, regression=False)
    ns = _run(tmp_path, monkeypatch, state, _source())
    assert ns["distribution_category"] == "Day 2"
    assert len(ns["fig"].axes) == (1 if marginal == "none" else 3)
    assert "from src." not in generate_script(state)


def test_unseparated_export_emits_none_separator_and_uses_fd_point_alpha(tmp_path, monkeypatch):
    state = _state(separate_by=None, category=None)
    state.update(shape_by=None, opacity_by=None)
    ns = _run(tmp_path, monkeypatch, state, _source())
    assert ns["SEPARATE_BY"] is ns["DISTRIBUTION_CATEGORY"] is None
    assert {collection.get_alpha() for collection in _points(ns["ax_main"])} == {0.8}
