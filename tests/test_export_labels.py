"""Export names describe assignments without changing source data or model labels."""
import numpy as np
import pandas as pd
import pytest


def frame():
    return pd.DataFrame({"id": ["b", "a", "c", "d"], "x": [1., 3., 8., 7.],
                         "labels": ["ctrl_group1", "drug_group2", None, "new_group1"]},
                        index=[4, 4, 2, 0])


def test_export_mapping_is_simultaneous_and_preserves_source_and_missing_values():
    from src.export_labels import apply_export_labels
    source = frame()
    original = source.copy(deep=True)
    result = apply_export_labels(source, "labels", {
        "column_name": "  Cell state α  ",
        "value_names": {"ctrl_group1": '  Low, "α"  ', "drug_group2": 'Low, "α"'},
    })
    assert list(result) == ["id", "x", "Cell state α"]
    assert result["Cell state α"].tolist() == ['Low, "α"', 'Low, "α"', None, "new_group1"]
    pd.testing.assert_frame_equal(source, original)
    pd.testing.assert_frame_equal(result[["id", "x"]], original[["id", "x"]])
    swapped = apply_export_labels(source, "labels", {
        "column_name": "labels", "value_names": {"ctrl_group1": "drug_group2", "drug_group2": "ctrl_group1"}})
    assert swapped.labels.tolist()[:2] == ["drug_group2", "ctrl_group1"]


@pytest.mark.parametrize("settings,match", [
    ({"column_name": " \t", "value_names": {}}, "column name"),
    ({"column_name": " x ", "value_names": {}}, "already exists"),
    ({"column_name": "labels", "value_names": {"ctrl_group1": " "}}, "value name"),
])
def test_invalid_names_are_rejected(settings, match):
    from src.export_labels import apply_export_labels
    with pytest.raises(ValueError, match=match):
        apply_export_labels(frame(), "labels", settings)


def test_defaults_and_absent_assignments_return_independent_copies():
    from src.export_labels import apply_export_labels, available_label_column
    data = frame()
    for source_column in ["labels", "absent"]:
        out = apply_export_labels(data, source_column)
        pd.testing.assert_frame_equal(out, data)
        assert out is not data
    assert available_label_column(["GMM_group", "GMM_group_2"], "GMM_group") == "GMM_group_3"
    assert available_label_column([], "GMM_group") == "GMM_group"


@pytest.mark.parametrize("method,legacy", [
    ("histogram", "GMM_group"), ("2d", "2D_GMM_group")])
@pytest.mark.parametrize("separate_by", [None, "day"])
def test_model_outputs_preserve_previously_exported_columns(method, legacy, separate_by):
    from src.vis.histogram import prepare_histogram
    from src.vis.bivar import feature_2d_distribution_plot
    rng = np.random.default_rng(9)
    centers = np.tile(np.repeat([.2, .8], 20), 2)
    data = pd.DataFrame({"id": [str(i) for i in range(80)], "day": np.repeat(["D1", "D2"], 40),
                         "x": centers + rng.normal(0, .01, 80),
                         "y": centers / 2 + rng.normal(0, .01, 80), legacy: "previous"})
    data["Lifetime fit free_ch1: G(1st)"] = data.x
    data["Lifetime fit free_ch1: S(1st)"] = data.y
    original = data.copy(deep=True)
    if method == "histogram":
        result = prepare_histogram(data, "x", separate_by=separate_by, apply_gmm=True, max_components=2)["df"]
    else:
        _, _, result = feature_2d_distribution_plot(data, "id", None, "x", "y", separate_by=separate_by,
            analysis_options={"fit_gmm": True, "max_components": 2})
    assert result[legacy].eq("previous").all()
    assert result[f"{legacy}_2"].notna().all()
    pd.testing.assert_frame_equal(data, original)


def test_phasor_retains_all_complete_rows_without_adding_an_export_column():
    from src.vis.bivar import phasor_plot
    data = pd.DataFrame({"id": list("abcde"), "group": ["fit"] * 4 + ["skip"],
        "Lifetime fit free_ch1: G(1st)": [.1, .11, .8, .81, .5],
        "Lifetime fit free_ch1: S(1st)": [.1, .11, .3, .31, .2]}, index=[0] * 5)
    _, result = phasor_plot(data, "id", None, "ch1", color_by=["group"])
    pd.testing.assert_frame_equal(result, data)
