"""The FOV column is optional and never invented; a present one is an ordinary
categorical, stringified and "N/A"-filled like any other.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from src.widgets import analysis_config_widgets as acw


def _profile(**cfg):
    """A stand-in analysis profile; the real one is read from analysis_config.toml."""
    return lambda *a, **k: cfg


def test_user_table_branch_lists_the_fov_column_as_categorical(monkeypatch):
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config",
                        _profile(categorical_cols=["treatment"], fov_name_col="well"))
    assert acw.get_categorical_cols_analysis(use_data_extraction=False) == ["treatment", "well"]


def test_a_blank_fov_name_is_not_added_as_a_categorical(monkeypatch):
    """The user-table profile's FOV field is optional, so "" is a reachable value."""
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config",
                        _profile(categorical_cols=["treatment"], fov_name_col=""))
    cats = acw.get_categorical_cols_analysis(use_data_extraction=False)
    assert cats == ["treatment"]


def test_a_repeat_in_the_stored_list_is_returned_once(monkeypatch):
    """Repeated configured categoricals are deduplicated so df[col] remains a Series."""
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config",
                        _profile(categorical_cols=["treatment", "well", "treatment"],
                                 fov_name_col="well"))
    cats = acw.get_categorical_cols_analysis(use_data_extraction=False)
    assert cats == ["treatment", "well"]


@pytest.mark.parametrize("use_data_extraction", [True, False])
def test_only_explicit_categories_and_fov_are_included(monkeypatch, use_data_extraction):
    """Export headers have no special role, but explicit category choices are kept."""
    configured = ["treatment", "GMM_group", "Cell state α", "well", "treatment"]
    monkeypatch.setattr(acw, "get_categorical_cols", lambda: configured)
    monkeypatch.setattr(acw, "get_fov_name_col", lambda: "well")
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config",
                        _profile(categorical_cols=configured, fov_name_col="well"))

    assert acw.get_categorical_cols_analysis(use_data_extraction) == [
        "treatment", "GMM_group", "Cell state α", "well"]
    assert configured == ["treatment", "GMM_group", "Cell state α", "well", "treatment"]


def test_reading_the_categoricals_does_not_write_to_the_profile(monkeypatch):
    """Reading categorical columns leaves the stored profile list unchanged."""
    stored = {"categorical_cols": ["treatment"], "fov_name_col": ""}
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config", lambda *a, **k: stored)
    acw.get_categorical_cols_analysis(use_data_extraction=False)
    assert stored == {"categorical_cols": ["treatment"], "fov_name_col": ""}


def test_a_numeric_fov_column_is_a_label_not_a_duplicated_feature(monkeypatch):
    """A numeric FOV column must be a label, and must appear exactly once."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "row")
    monkeypatch.setattr(dataset_io, "get_fov_name_col_analysis", lambda *a, **k: "well")

    # Use the accessor's categorical list to exercise FOV deduplication end to end.
    monkeypatch.setattr(acw, "_get_current_profile", lambda: "p")
    monkeypatch.setattr(acw, "_get_profile_config",
                        _profile(categorical_cols=[], fov_name_col="well"))
    cats = acw.get_categorical_cols_analysis(use_data_extraction=False)

    df = pd.DataFrame({"row": ["a", "b"], "well": [1, 2], "feat": [0.1, 0.2]})
    fixed, _warning, error = dataset_io.check_and_fix_df(df, cats, "row", "well")
    assert error == ""
    out, groups, _w, _e = dataset_io.get_features(fixed, cats, use_data_extraction=False)

    assert not out.columns.duplicated().any()
    assert list(out.columns) == ["row", "well", "feat"]
    assert groups == {"Uncategorized Features": ["feat"]}
    assert out["well"].tolist() == ["1", "2"]  # a label, not a measurement


def _no_fov_frame():
    return pd.DataFrame({"cell_id": ["a", "b"], "treatment": ["ctrl", "drug"],
                         "Lifetime fit_ch1: T1": [0.4, 0.5]})


def test_get_features_keeps_a_fov_less_frame(monkeypatch):
    """No FOV column is a valid analysis, not a KeyError in the prune."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    out, groups, _w, error = dataset_io.get_features(
        _no_fov_frame(), ["treatment"], use_data_extraction=True)
    assert error == ""
    assert list(out.columns) == ["cell_id", "treatment", "Lifetime fit_ch1: T1"]


def test_get_features_keeps_a_present_fov_column_exactly_once(monkeypatch):
    """The FOV column is retained exactly once."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    df = _no_fov_frame()
    df["image_name"] = ["f1", "f2"]
    out, _g, _w, _e = dataset_io.get_features(
        df, ["treatment", "image_name"], use_data_extraction=True)
    assert list(out.columns).count("image_name") == 1


def test_resolve_effective_fov_col():
    df = pd.DataFrame({"cell_id": ["a"], "image_name": ["f"]})
    assert dataset_io.resolve_effective_fov_col(df, "image_name") == "image_name"
    assert dataset_io.resolve_effective_fov_col(df, "missing") is None
    assert dataset_io.resolve_effective_fov_col(df, "") is None
    assert dataset_io.resolve_effective_fov_col(None, "image_name") is None


def test_an_absent_fov_column_is_not_invented():
    df = pd.DataFrame({"cell_id": ["a_1", "b_2"], "feat": [1.0, 2.0]})
    fixed, _warning, error = dataset_io.check_and_fix_df(df, [], "cell_id", "image_name")
    assert error == ""
    assert "image_name" not in fixed.columns


def test_blank_fov_cells_read_na_like_every_other_categorical():
    df = pd.DataFrame({"cell_id": ["a", "b"], "image_name": ["f1", None],
                       "feat": [1.0, 2.0]})
    fixed, _warning, _error = dataset_io.check_and_fix_df(
        df, ["image_name"], "cell_id", "image_name")
    assert fixed["image_name"].tolist() == ["f1", "N/A"]


def test_a_fov_column_blank_in_every_row_is_dropped_and_named():
    """The empty-column rule already handles it; there is no FOV special case."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "image_name": [None, None],
                       "feat": [1.0, 2.0]})
    fixed, warning, error = dataset_io.check_and_fix_df(
        df, ["image_name"], "cell_id", "image_name")
    assert error == ""
    assert "image_name" not in fixed.columns
    assert "image_name" in warning and "all empty" in warning


def test_the_fov_column_is_stringified_even_if_the_caller_omits_it():
    """The export-inlined loader deduplicates categorical names independently of config accessors."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "well": [1, 2], "feat": [1.0, 2.0]})
    fixed, _warning, _error = dataset_io.check_and_fix_df(df, [], "cell_id", "well")
    assert fixed["well"].tolist() == ["1", "2"]


def test_a_blank_configured_fov_name_creates_no_column():
    """A blank configured FOV name names no column."""
    df = pd.DataFrame({"row": ["a", "b"], "feat": [1.0, 2.0]})
    fixed, _warning, error = dataset_io.check_and_fix_df(df, [], "row", "")
    assert error == ""
    assert list(fixed.columns) == ["row", "feat"]


def test_an_all_empty_fov_column_is_dropped_before_it_is_resolved():
    """Resolve the effective FOV after normalization removes all-empty columns."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "image_name": [None, None],
                       "feat": [1.0, 2.0]})
    # Before: the configured name is present, so a naive check would say it exists.
    assert dataset_io.resolve_effective_fov_col(df, "image_name") == "image_name"

    fixed, _warning, _error = dataset_io.check_and_fix_df(
        df, [], "cell_id", "image_name")
    # After: the column was all-empty and dropped, so the analysis has no FOV.
    assert "image_name" not in fixed.columns
    assert dataset_io.resolve_effective_fov_col(fixed, "image_name") is None


def test_load_table_warns_when_the_configured_fov_column_is_absent(monkeypatch):
    """load_table names the configured FOV column when the loaded frame has none."""
    from tests.test_table_formats import _uploaded_file

    rendered = []
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    monkeypatch.setattr(dataset_io, "get_fov_name_col_analysis", lambda *a, **k: "image_name")
    monkeypatch.setattr(dataset_io.st, "markdown", lambda msg, **k: rendered.append(msg))
    monkeypatch.setattr(dataset_io.st, "write", lambda msg, **k: rendered.append(msg))

    # image_name is in categorical_cols, as get_categorical_cols_analysis
    # always supplies it -- real usage never omits the configured FOV column here.
    raw = b"cell_id,treatment,Lifetime fit_ch1: T1\na,ctrl,0.40\nb,drug,0.55\n"
    upload = _uploaded_file(raw, "no_fov.csv")
    df, _groups, complete, _delimiter, _row_id = dataset_io.load_table(
        upload, ["treatment", "image_name"])

    assert complete is True
    assert "image_name" not in df.columns
    # _render_warning routes the plain-text message through _as_html, which HTML
    # -escapes it -- so "image_name" is not quoted with a literal apostrophe here.
    shown = " ".join(rendered)
    assert "the FOV column" in shown and "image_name" in shown and "was not found" in shown
    assert "left out of hover text" in shown


def test_user_table_branch_stays_silent_about_a_missing_fov_column(monkeypatch):
    """A user table silently ignores a configured FOV name absent from its columns.
    Exercise read_table and interpret_table, the user-table path through review.
    """
    from tests.test_table_formats import _uploaded_file

    rendered = []
    monkeypatch.setattr(dataset_io.st, "markdown", lambda msg, **k: rendered.append(msg))
    monkeypatch.setattr(dataset_io.st, "write", lambda msg, **k: rendered.append(msg))

    raw = b"wine_id,treatment,Lifetime fit_ch1: T1\na,ctrl,0.40\nb,drug,0.55\n"
    upload = _uploaded_file(raw, "no_fov.csv")
    table, _meta, _delimiter, scope_warning, error = dataset_io.read_table(upload)
    assert error == ""
    df, _groups, complete, _row_id = dataset_io.interpret_table(
        table, ["treatment"], "wine_id", "image_name", feature_groups={},
        scope_warning=scope_warning, use_data_extraction=False)

    assert complete is True
    assert "image_name" not in df.columns
    shown = " ".join(rendered)
    assert "was not found" not in shown
    assert "left out of hover text" not in shown


def test_a_column_the_numeric_rule_rejects_is_named_not_silently_dropped(monkeypatch):
    """25% non-numeric, so the 1% rule leaves it as text -- and the prune cuts it."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    df = pd.DataFrame({"cell_id": list("abcd"),
                       "mostly_num": ["1", "2", "3", "oops"],
                       "Lifetime fit_ch1: T1": [1.0, 2.0, 3.0, 4.0]})
    out, _groups, warning, error = dataset_io.get_features(
        df, [], use_data_extraction=True)
    assert error == ""
    assert "mostly_num" not in out.columns
    assert "mostly_num" in warning and "not analysed" in warning


def test_the_prune_warning_is_singular_for_exactly_one_dropped_column(monkeypatch):
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    df = pd.DataFrame({"cell_id": ["a", "b"], "text_col": ["x", "y"],
                       "Lifetime fit_ch1: T1": [1.0, 2.0]})
    _out, _groups, warning, _error = dataset_io.get_features(
        df, [], use_data_extraction=True)
    assert "Warning: 1 column was not analysed: text_col.\n" in warning


def test_the_prune_warning_truncates_and_pluralizes_past_five_dropped_columns(monkeypatch):
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    bad_cols = {f"bad{i}": ["x", "y"] for i in range(7)}
    df = pd.DataFrame({"cell_id": ["a", "b"], **bad_cols,
                       "Lifetime fit_ch1: T1": [1.0, 2.0]})
    _out, _groups, warning, _error = dataset_io.get_features(
        df, [], use_data_extraction=True)
    assert "Warning: 7 columns were not analysed: " in warning
    assert "and 2 more." in warning


def test_a_datetime_column_is_named_too(monkeypatch):
    """Dates are deliberately skipped from numeric coercion, then pruned."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    df = pd.DataFrame({"cell_id": ["a", "b"],
                       "acquired": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                       "Lifetime fit_ch1: T1": [1.0, 2.0]})
    _out, _groups, warning, _error = dataset_io.get_features(
        df, [], use_data_extraction=True)
    assert "acquired" in warning


def test_nothing_is_reported_when_the_prune_drops_nothing(monkeypatch):
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    df = pd.DataFrame({"cell_id": ["a", "b"], "treatment": ["ctrl", "drug"],
                       "Lifetime fit_ch1: T1": [1.0, 2.0]})
    _out, _groups, warning, _error = dataset_io.get_features(
        df, ["treatment"], use_data_extraction=True)
    assert "not analysed" not in warning


def test_point_traces_build_without_a_customdata_column():
    import plotly.graph_objects as go

    from src.vis.helpers import add_interleaved_points_trace

    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0],
                       "cell_id": ["a", "b", "c"],
                       "_color_group": ["g", "g", "g"]})
    grouped = [(("g", None, None), df)]
    fig = go.Figure()
    add_interleaved_points_trace(
        fig=fig, grouped=grouped, color_map={"g": "#1f77b4"},
        shape_map=None, opacity_map=None, axis_labels=["x", "y"],
        text_col="cell_id", customdata_col=None,
        hovertemplate="<b>%{text}</b>",
    )
    assert len(fig.data) >= 1
    assert all(trace.customdata is None for trace in fig.data)


def test_the_point_plots_render_without_a_fov_column():
    """Every plot omits FOV hover references when no FOV column is resolved."""
    from src.vis.bivar import feature_2d_distribution_plot
    from src.vis.univar import feature_comparison_plot

    df = pd.DataFrame({
        "cell_id": [f"c{i}" for i in range(12)],
        "treatment": ["ctrl", "drug"] * 6,
        "Lifetime fit_ch1: T1": [0.4, 0.5, 0.6, 0.45, 0.55, 0.65] * 2,
        "Lifetime fit_ch1: T2": [2.0, 2.2, 2.4, 2.1, 2.3, 2.5] * 2,
    })

    fig = feature_comparison_plot(
        df, unique_row_id_col="cell_id", fov_name_col=None,
        selected_var="Lifetime fit_ch1: T1", color_by=["treatment"])
    assert fig is not None
    # Plotly displays dangling customdata references literally, so inspect hover
    # templates as well as checking that each figure builds.
    for trace in fig.data:
        assert trace.customdata is None
        hovertemplate = trace.hovertemplate or ""
        assert "<b>fov:</b>" not in hovertemplate
        assert "<b>Image:</b>" not in hovertemplate

    fig2, _table, _gmm = feature_2d_distribution_plot(
        df, unique_row_id_col="cell_id", fov_name_col=None,
        selected_x="Lifetime fit_ch1: T1", selected_y="Lifetime fit_ch1: T2",
        color_by=["treatment"])
    assert fig2 is not None
    for trace in fig2.data:
        assert trace.customdata is None
        hovertemplate = trace.hovertemplate or ""
        assert "<b>fov:</b>" not in hovertemplate
        assert "<b>Image:</b>" not in hovertemplate


def test_the_generated_script_bakes_a_none_fov_column():
    from src.export_script import generate_script

    state = {
        "csv_filename": "data.csv", "delimiter": ",", "unique_row_id_col": "cell_id",
        "fov_name_col": None, "method": "Feature Comparison",
        "categorical_filters": {}, "numerical_filters": [], "color_by": ["treatment"],
        "opacity_by": None, "shape_by": None, "separate_by": None, "subcolor_by": None,
        "point_size": 5, "axis_label_size": 12, "legend_size": 10, "colormap": "tab10",
        "categorical_cols": ["treatment"],
        "analysis_columns": ["cell_id", "treatment", "Lifetime fit_ch1: T1"],
        "method_params": {"selected_var": "Lifetime fit_ch1: T1",
                          "effect_size_method": "None", "statistical_test": "None"},
    }
    script = generate_script(state)
    assert "FOV_NAME_COL = None" in script
    assert "image_name" not in script


def test_fresh_visit_shows_phasor_plot(monkeypatch):
    """Before upload, extraction analysis offers Phasor Plot and the expected univariate methods."""
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(acw, "get_fov_name_col_analysis",
                        lambda use_data_extraction=True: "image_name" if use_data_extraction else "")

    page = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")
    at = AppTest.from_file(page)
    at.run(timeout=90)
    assert not at.exception
    assert at.radio[1].options == ["Feature Comparison", "Feature Histogram"]

    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    assert "Phasor Plot" in at.radio[1].options


def test_phasor_hides_when_loaded_feature_groups_lack_a_complete_gs_pair(monkeypatch):
    """A loaded frame with G but no matching S hides Phasor Plot."""
    from streamlit.testing.v1 import AppTest

    df = pd.DataFrame({
        "cell_id": ["a", "b", "c", "d"], "treatment": ["ctrl", "drug", "ctrl", "drug"],
        "feat1": [0.1, 0.2, 0.3, 0.4], "feat2": [1.0, 1.1, 1.2, 1.3],
        "Lifetime fit free_ch1: G(1st)": [0.1, 0.2, 0.3, 0.4],
    })
    feature_groups = {
        "Uncategorized Features": ["feat1", "feat2"],
        "Lifetime fit free_ch1": ["Lifetime fit free_ch1: G(1st)"],
    }
    monkeypatch.setattr(dataset_io, "load_table", lambda *_a, **_k: (df, feature_groups, True, ",", "cell_id"))

    page = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")
    at = AppTest.from_file(page)
    at.run(timeout=90)
    assert not at.exception
    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    assert "Phasor Plot" not in at.radio[1].options
    assert at.session_state["phasor_available"] is False


def test_phasor_shows_when_loaded_feature_groups_have_a_complete_gs_pair(monkeypatch):
    """A complete G/S pair for one harmonic makes the channel plottable."""
    from streamlit.testing.v1 import AppTest

    df = pd.DataFrame({
        "cell_id": ["a", "b", "c", "d"], "treatment": ["ctrl", "drug", "ctrl", "drug"],
        "feat1": [0.1, 0.2, 0.3, 0.4], "feat2": [1.0, 1.1, 1.2, 1.3],
        "Lifetime fit free_ch1: G(1st)": [0.1, 0.2, 0.3, 0.4],
        "Lifetime fit free_ch1: S(1st)": [0.4, 0.3, 0.2, 0.1],
    })
    feature_groups = {
        "Uncategorized Features": ["feat1", "feat2"],
        "Lifetime fit free_ch1": ["Lifetime fit free_ch1: G(1st)", "Lifetime fit free_ch1: S(1st)"],
    }
    monkeypatch.setattr(dataset_io, "load_table", lambda *_a, **_k: (df, feature_groups, True, ",", "cell_id"))

    page = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")
    at = AppTest.from_file(page)
    at.run(timeout=90)
    assert not at.exception
    at.radio[0].set_value("### **Bivariate**")
    at.run(timeout=90)
    assert not at.exception
    assert "Phasor Plot" in at.radio[1].options
    assert at.session_state["phasor_available"] is True


def test_transition_rerun_fires_once_and_then_settles(monkeypatch):
    """A change in phasor availability triggers at most one rerun; settled state triggers none."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(dataset_io, "load_table", lambda *_a, **_k: (
        _no_fov_frame(), {"Uncategorized Features": ["Lifetime fit_ch1: T1"]}, True, ",", "cell_id"))

    rerun_calls = []
    real_rerun = st.rerun
    def counting_rerun(*a, **k):
        rerun_calls.append(1)
        return real_rerun(*a, **k)
    monkeypatch.setattr(st, "rerun", counting_rerun)

    page = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")
    at = AppTest.from_file(page)
    at.run(timeout=90)
    assert not at.exception
    assert len(rerun_calls) <= 1
    settled_options = at.radio[1].options
    assert settled_options == ["Feature Comparison", "Feature Histogram"]

    calls_after_settling = len(rerun_calls)
    at.run(timeout=90)
    assert not at.exception
    assert len(rerun_calls) == calls_after_settling
    assert at.radio[1].options == settled_options


def test_extraction_happy_path_causes_no_rerun(monkeypatch):
    """A complete G/S pair matches the preload default, needs no rerun, and retains FOV hover."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(acw, "get_fov_name_col_analysis",
                        lambda use_data_extraction=True: "image_name" if use_data_extraction else "")

    df = pd.DataFrame({
        "cell_id": ["a", "b", "c", "d"], "image_name": ["f1", "f1", "f2", "f2"],
        "treatment": ["ctrl", "drug", "ctrl", "drug"],
        "Lifetime fit free_ch1: G(1st)": [0.1, 0.2, 0.3, 0.4],
        "Lifetime fit free_ch1: S(1st)": [0.4, 0.3, 0.2, 0.1],
    })
    feature_groups = {"Lifetime fit free_ch1": [
        "Lifetime fit free_ch1: G(1st)", "Lifetime fit free_ch1: S(1st)"]}
    monkeypatch.setattr(dataset_io, "load_table", lambda *_a, **_k: (df, feature_groups, True, ",", "cell_id"))

    rerun_calls = []
    real_rerun = st.rerun
    def counting_rerun(*a, **k):
        rerun_calls.append(1)
        return real_rerun(*a, **k)
    monkeypatch.setattr(st, "rerun", counting_rerun)

    page = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")
    at = AppTest.from_file(page)
    at.run(timeout=90)
    assert not at.exception
    assert rerun_calls == []
    assert at.radio[1].options == ["Feature Comparison", "Feature Histogram"]
    assert at.session_state["effective_fov_name_col"] == "image_name"
