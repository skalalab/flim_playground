"""Execute standalone scripts to verify derived CSV label customization."""

import ast
import runpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.export_script import generate_script


METHODS = [
    ("Feature Histogram", "GMM_group", "gmm_grouped_data.csv"),
    ("2D Feature Distribution", "2D_GMM_group", "2D_gmm_data.csv"),
]


def _source():
    rng = np.random.default_rng(180)
    rows = []
    for day in ["Day 1", "Day 2"]:
        for treatment in ["ctrl", "drug"]:
            for mode in [0, 1]:
                for _ in range(10):
                    x, y = rng.normal(2 + 12 * mode, 0.08, 2)
                    rows.append({
                        "cell_id": f"cell-{len(rows)}", "day": day,
                        "treatment": treatment, "metadata": f"original-{len(rows)}",
                        "feature_x": x, "feature_y": y,
                    })
    # One retained observation cannot be fitted. Its assignment remains null.
    rows.append({
        "cell_id": "sparse", "day": "Day 1", "treatment": "sparse",
        "metadata": "keep-sparse", "feature_x": 4.0, "feature_y": 3.0,
    })
    return pd.DataFrame(rows)


def _state(method, separate_by):
    params = {
        "Feature Histogram": {
            "selected_var": "feature_x", "apply_gmm": True,
            "gmm_max_components": 2, "gmm_min_weight_threshold": 0.1,
        },
        "2D Feature Distribution": {
            "selected_x": "feature_x", "selected_y": "feature_y",
            "marginal_plot_type": "none", "fit_gmm_2d": True,
            "gmm_max_components": 2, "gmm_min_weight_threshold": 0.1,
            "distribution_category": "Day 1",
        },
    }
    return {
        "method": method, "csv_filename": "source.csv",
        "unique_row_id_col": "cell_id", "fov_name_col": None,
        "color_by": ["treatment"], "separate_by": separate_by,
        "categorical_cols": ["day", "treatment"],
        "method_params": params[method],
    }


def _run(tmp_path, monkeypatch, state, source, *, save=True):
    source.to_csv(tmp_path / state["csv_filename"], index=False)
    script = generate_script(state)
    assert "SAVE_DERIVED_DATA = False" in script
    for node in ast.walk(ast.parse(script)):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("src")
        elif isinstance(node, ast.Import):
            assert all(not name.name.startswith("src") for name in node.names)
    if save:
        script = script.replace("SAVE_DERIVED_DATA = False", "SAVE_DERIVED_DATA = True")
    path = tmp_path / "analysis.py"
    path.write_text(script, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "1")
    try:
        return runpy.run_path(str(path))
    finally:
        plt.close("all")


@pytest.mark.parametrize("method,default_column,filename", METHODS)
@pytest.mark.parametrize("separate_by", [None, "day"])
@pytest.mark.parametrize("extra_color", [False, True])
def test_custom_csv_labels_preserve_internal_labels_unmapped_values_and_nulls(
    tmp_path, monkeypatch, method, default_column, filename, separate_by, extra_color
):
    source = _source()
    state = _state(method, separate_by)
    if extra_color:
        source["dose"] = "high_dose"
        state["color_by"].append("dose")
        state["categorical_cols"].append("dose")
    color_suffix = "::high_dose" if extra_color else ""
    column = ' Population "state", α\nlabel '
    value = ' Shared "state", β\nvalue '
    prefixes = ["Day 1::", "Day 2::"] if separate_by else [""]
    value_names = {
        f"{prefix}ctrl{color_suffix}_group{group}": value
        for prefix in prefixes for group in [1, 2]
    }
    state["derived_export"] = {"column_name": column, "value_names": value_names}

    ns = _run(tmp_path, monkeypatch, state, source)
    saved = pd.read_csv(tmp_path / filename)
    original_labels = ns["df"][default_column].reset_index(drop=True)
    expected_labels = original_labels.map(
        lambda label: value_names.get(label, label).strip() if pd.notna(label) else label
    ).rename(column.strip())
    # CSV represents both None and NaN assignments as an empty field.
    expected_labels = expected_labels.where(expected_labels.notna(), np.nan)

    assert column.strip() in saved.columns
    assert default_column not in saved.columns
    assert column.strip() not in ns["df"].columns
    pd.testing.assert_series_equal(saved[column.strip()], expected_labels)
    assert value.strip() in set(saved[column.strip()].dropna())
    assert original_labels.str.contains(f"drug{color_suffix}_group", regex=False).any()
    assert original_labels.isna().sum() == saved[column.strip()].isna().sum() == 1
    assert saved["cell_id"].tolist() == source["cell_id"].tolist()
    assert saved["metadata"].tolist() == source["metadata"].tolist()
    assert not any(name.startswith("_color_group") for name in saved.columns)


@pytest.mark.parametrize("method,default_column,filename", METHODS)
@pytest.mark.parametrize("separate_by", [None, "day"])
@pytest.mark.parametrize("color_by", [["treatment"], []])
def test_default_export_uses_unused_column_and_preserves_previous_annotations(
    tmp_path, monkeypatch, method, default_column, filename, separate_by, color_by
):
    source = _source()
    source[default_column] = [f"previous-{index}" for index in source.index]
    source[f"{default_column}_2"] = [f"earlier-{index}" for index in source.index]

    state = _state(method, separate_by)
    state["color_by"] = color_by
    ns = _run(tmp_path, monkeypatch, state, source)
    saved = pd.read_csv(tmp_path / filename)

    for column in [default_column, f"{default_column}_2"]:
        pd.testing.assert_series_equal(saved[column], source[column])
        pd.testing.assert_series_equal(ns["df"][column], source[column])
    result_column = f"{default_column}_3"
    assert result_column in saved.columns
    assert saved[result_column].str.contains("_group").any()
    assert saved[result_column].isna().sum() == (1 if color_by else 0)
    if not color_by:
        if separate_by:
            assert all(label.startswith(f"{category}_group")
                       for category, label in saved[[separate_by, result_column]].itertuples(index=False))
        else:
            assert saved[result_column].str.startswith("all_data_group").all()
    labels = ns["df"][result_column]
    pd.testing.assert_series_equal(saved[result_column], labels.where(labels.notna(), np.nan))


@pytest.mark.parametrize("method,default_column,filename", METHODS)
@pytest.mark.parametrize("separate_by", [None, "day"])
def test_custom_name_collision_fails_before_writing_csv(
    tmp_path, monkeypatch, method, default_column, filename, separate_by
):
    state = _state(method, separate_by)
    state["derived_export"] = {"column_name": " metadata ", "value_names": {}}

    with pytest.raises(ValueError, match="(?i)(exist|collision|another|already)"):
        _run(tmp_path, monkeypatch, state, _source())

    assert not (tmp_path / filename).exists()


@pytest.mark.parametrize("settings", [
    {"column_name": " \t\n", "value_names": {}},
    {"column_name": "population", "value_names": {"ctrl_group1": " \t\n"}},
])
def test_blank_export_names_fail_before_writing_csv(tmp_path, monkeypatch, settings):
    state = _state("Feature Histogram", None)
    state["derived_export"] = settings

    with pytest.raises(ValueError, match="(?i)(blank|empty|non.?empty)"):
        _run(tmp_path, monkeypatch, state, _source())

    assert not (tmp_path / "gmm_grouped_data.csv").exists()


@pytest.mark.parametrize("method,default_column,filename", METHODS)
@pytest.mark.parametrize("separate_by", [None, "day"])
def test_label_settings_keep_derived_csv_writing_opt_in(
    tmp_path, monkeypatch, method, default_column, filename, separate_by
):
    state = _state(method, separate_by)
    state["derived_export"] = {"column_name": "population", "value_names": {}}

    ns = _run(tmp_path, monkeypatch, state, _source(), save=False)

    assert default_column in ns["df"].columns
    assert "population" not in ns["df"].columns
    assert not (tmp_path / filename).exists()


@pytest.mark.parametrize("method,default_column,filename", METHODS[1:])
def test_unseparated_csv_roundtrip_preserves_annotations_named_like_color_helpers(
    tmp_path, monkeypatch, method, default_column, filename
):
    source = _source()
    source["_color_group_2"] = [f"earlier-{index}" for index in source.index]
    state = _state(method, None)
    state["derived_export"] = {"column_name": "_color_group", "value_names": {}}
    _run(tmp_path, monkeypatch, state, source)
    annotated = pd.read_csv(tmp_path / filename)

    ns = _run(tmp_path, monkeypatch, _state(method, None), annotated)
    saved = pd.read_csv(tmp_path / filename)

    assert "_color_group" in saved.columns
    assert "_color_group_2" in saved.columns
    pd.testing.assert_frame_equal(saved[annotated.columns], annotated)
    assert saved[default_column].notna().sum() == len(source) - 1
    assert ns["df"][default_column].isna().sum() == 1
    assert "_color_group_3" not in saved.columns
