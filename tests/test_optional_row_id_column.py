"""Optional identifiers are generated when absent. Hover labels use the configured column
name, or ID for generated row numbers.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from src.vis.bivar import feature_2d_distribution_plot, phasor_plot
from src.vis.helpers import hover_field
from src.vis.multivar import dimension_reduction_plot
from src.vis.univar import feature_comparison_plot


def _frame():
    return pd.DataFrame({
        "flower_id": ["a", "b", "c", "d"],
        "species": ["setosa", "setosa", "virginica", "virginica"],
        "Sepal length": [5.1, 4.9, 6.3, 5.8],
    })


# ---------------------------------------------------------------- check_and_fix_df


def test_a_blank_row_id_is_not_an_error():
    """No identifier configured is a valid table."""
    df = _frame().drop(columns=["flower_id"])
    fixed, _warning, error = dataset_io.check_and_fix_df(df, ["species"], "", None)
    assert error == ""
    assert list(fixed.columns) == ["species", "Sepal length"]


def test_a_none_row_id_is_not_an_error():
    """`None` reaches here from an exported script's baked literal, "" from the config
    text box. Both mean the same thing."""
    df = _frame().drop(columns=["flower_id"])
    _fixed, _warning, error = dataset_io.check_and_fix_df(df, ["species"], None, None)
    assert error == ""


def test_a_named_but_missing_row_id_is_still_an_error():
    """Naming a column the file lacks is a mistake, not a choice -- only a blank name
    means "this table has no identifier"."""
    df = _frame().drop(columns=["flower_id"])
    fixed, _warning, error = dataset_io.check_and_fix_df(df, ["species"], "flower_id", None)
    assert "flower_id column is missing" in error
    assert fixed is None


def test_a_repeated_row_id_is_refused_rather_than_repaired():
    """Repeated identifiers reject the file without dropping rows and changing the
    analysis.
    """
    df = _frame()
    df["flower_id"] = ["a", "a", "c", "d"]
    fixed, _warning, error = dataset_io.check_and_fix_df(df, ["species"], "flower_id", None)
    assert fixed is None
    assert "flower_id" in error and "'a' appears 2 times" in error, error


@pytest.mark.parametrize("ids", [
    [1, "1", "c", "d"],
    [1.5, "1.5", "c", "d"],
])
def test_row_ids_that_collide_as_strings_are_refused(ids):
    df = _frame().assign(flower_id=ids)
    original = df.copy(deep=True)
    roles = {"flower_id": "row_id", "Sepal length": "numerical"}

    gate_error = dataset_io.review_blocking_reason(df, roles)
    fixed, _warning, loader_error = dataset_io.check_and_fix_df(
        df, ["species"], "flower_id", None)

    assert "appears 2 times" in gate_error, gate_error
    assert "appears 2 times" in loader_error, loader_error
    assert fixed is None
    pd.testing.assert_frame_equal(df, original)


def test_unique_mixed_type_row_ids_keep_their_labels():
    df = _frame().assign(flower_id=[1, "01", "c", "d"])
    roles = {"flower_id": "row_id", "Sepal length": "numerical"}

    assert dataset_io.review_blocking_reason(df, roles) == ""
    fixed, _warning, error = dataset_io.check_and_fix_df(
        df, ["species"], "flower_id", None)

    assert error == ""
    assert fixed.flower_id.tolist() == ["1", "01", "c", "d"]
    assert len(fixed) == len(df)


def test_a_row_id_blank_in_some_rows_is_refused():
    """Reject missing identifiers before string conversion could turn them into "nan" labels."""
    df = _frame()
    df["flower_id"] = ["a", None, "c", None]
    fixed, _warning, error = dataset_io.check_and_fix_df(df, ["species"], "flower_id", None)
    assert fixed is None
    assert "blank in 2 of 4 rows" in error, error


def test_the_loader_and_the_review_gate_agree_on_what_an_identifier_is():
    """The gate and loader must agree. The export-inlined loader cannot import the gate
    helper.
    """
    from src.column_roles import ROLE_NUMERICAL, ROLE_ROW_ID
    from src.dataset_io import review_blocking_reason

    cases = {
        "unique": ["a", "b", "c", "d"],
        "repeated": ["a", "a", "c", "d"],
        "blank": ["a", None, "c", "d"],
        "all blank": [None, None, None, None],
    }
    for name, ids in cases.items():
        df = _frame()
        df["flower_id"] = ids
        roles = {"flower_id": ROLE_ROW_ID, "Sepal length": ROLE_NUMERICAL}
        loader_refused = dataset_io.check_and_fix_df(df, [], "flower_id", None)[2] != ""
        gate_refused = review_blocking_reason(df, roles) != ""
        assert loader_refused == gate_refused, f"{name}: loader {loader_refused}, gate {gate_refused}"


def test_a_blank_row_id_skips_deduplication_entirely():
    """Row numbers cannot repeat, so nothing is dropped -- including rows that are
    identical in every column the file actually has."""
    df = pd.DataFrame({"species": ["setosa"] * 3, "Sepal length": [5.1, 5.1, 5.1]})
    fixed, warning, error = dataset_io.check_and_fix_df(df, ["species"], "", None)
    assert error == ""
    assert len(fixed) == 3
    assert "duplicate" not in warning


# --------------------------------------------------------------- resolve_row_id_col


def test_a_configured_row_id_is_returned_untouched():
    df = _frame()
    out, row_id_col = dataset_io.resolve_row_id_col(df, "flower_id")
    assert row_id_col == "flower_id"
    assert list(out.columns) == list(_frame().columns)


def test_an_absent_row_id_is_invented_and_numbered_from_one():
    df = _frame().drop(columns=["flower_id"])
    out, row_id_col = dataset_io.resolve_row_id_col(df, "")
    assert row_id_col == "Row number"
    # str, like every configured identifier after check_and_fix_df -- so hover text
    # and downloaded CSVs cannot render "1" as "1.0".
    assert out["Row number"].tolist() == ["1", "2", "3", "4"]
    assert next(iter(out.columns)) == "Row number"


def test_a_real_row_number_column_is_never_overwritten():
    df = pd.DataFrame({"Row number": ["x", "y"], "feat": [1.0, 2.0]})
    out, row_id_col = dataset_io.resolve_row_id_col(df, "")
    assert row_id_col == "Row number.1"
    assert out["Row number"].tolist() == ["x", "y"]      # the user's data survives
    assert out["Row number.1"].tolist() == ["1", "2"]


# ---------------------------------------------------------------------- get_features


def test_an_invented_row_id_is_kept_but_is_not_a_feature():
    """Generated string row numbers skip numeric coercion and feature selection."""
    df, row_id_col = dataset_io.resolve_row_id_col(
        _frame().drop(columns=["flower_id"]), "")
    out, groups, _w, error = dataset_io.get_features(
        df, ["species"], use_data_extraction=False, unique_row_id_col=row_id_col)

    assert error == ""
    assert list(out.columns) == ["Row number", "species", "Sepal length"]
    assert groups == {"Uncategorized Features": ["Sepal length"]}
    assert out["Row number"].tolist() == ["1", "2", "3", "4"]


def test_get_features_still_reads_the_config_when_no_name_is_passed(monkeypatch):
    """The default that keeps every other caller working."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "flower_id")
    out, _groups, _w, error = dataset_io.get_features(
        _frame(), ["species"], use_data_extraction=False)
    assert error == ""
    assert list(out.columns) == ["flower_id", "species", "Sepal length"]


def test_a_column_named_as_both_the_identifier_and_a_categorical_is_kept_once():
    """Overlapping identifier and categorical settings retain one Series per column for
    filters.
    """
    out, _groups, warning, error = dataset_io.get_features(
        _frame(), ["species", "flower_id"], use_data_extraction=False,
        unique_row_id_col="flower_id")

    assert error == ""
    assert not out.columns.duplicated().any()
    assert list(out.columns) == ["flower_id", "species", "Sepal length"]
    # Filters can still use the column, and no data-loss notice is needed.
    assert out["flower_id"].nunique() > 1
    assert "not analysed" not in warning


# ------------------------------------------------------------------------ load_table


def _silent(monkeypatch, row_id, fov=""):
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: row_id)
    monkeypatch.setattr(dataset_io, "get_fov_name_col_analysis", lambda *a, **k: fov)
    monkeypatch.setattr(dataset_io.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(dataset_io.st, "write", lambda *a, **k: None)


IRIS = (b"Sepal length,Sepal width,species\n"
        b"5.1,3.5,setosa\n4.9,3.0,setosa\n6.3,3.3,virginica\n")


def test_load_table_accepts_an_iris_table_with_no_identifier(monkeypatch):
    """With a blank configured identifier, load_table returns the generated identifier
    name.
    """
    from tests.test_table_formats import _uploaded_file

    _silent(monkeypatch, row_id="")
    df, groups, complete, _delimiter, row_id_col = dataset_io.load_table(
        _uploaded_file(IRIS, "iris.csv"), ["species"])

    assert complete is True
    assert row_id_col == "Row number"
    assert df["Row number"].tolist() == ["1", "2", "3"]
    assert "Row number" not in [c for cols in groups.values() for c in cols]


def test_load_table_hands_back_the_configured_name_when_there_is_one(monkeypatch):
    from tests.test_table_formats import _uploaded_file

    _silent(monkeypatch, row_id="flower_id")
    raw = b"flower_id,Sepal length,species\n1,5.1,setosa\n2,4.9,setosa\n"
    df, _groups, complete, _delimiter, row_id_col = dataset_io.load_table(
        _uploaded_file(raw, "iris.csv"), ["species"])

    assert complete is True
    assert row_id_col == "flower_id"
    assert "Row number" not in df.columns


def test_load_table_returns_the_configured_name_when_the_upload_is_rejected(monkeypatch):
    """The early returns hand back a name too, so the caller never unpacks a short
    tuple on the path where nothing loaded."""
    from tests.test_table_formats import _uploaded_file

    _silent(monkeypatch, row_id="flower_id")
    # A .csv whose bytes are a zip: rejected by _name_content_mismatch, the first
    # early return in load_table.
    upload = _uploaded_file(b"PK\x03\x04rest of a workbook", "iris.csv")
    df, _groups, complete, _delimiter, row_id_col = dataset_io.load_table(
        upload, ["species"])

    assert complete is False and df is None
    assert row_id_col == "flower_id"


# ---------------------------------------------------------------------- hover labels


def _hover_templates(fig):
    return [t.hovertemplate for t in fig.data if getattr(t, "hovertemplate", None)]


def _plot_frame():
    return pd.DataFrame({
        "cell_id": ["a", "b", "c", "d"],
        "image_name": ["f1", "f1", "f2", "f2"],
        "species": ["setosa", "setosa", "virginica", "virginica"],
        "Sepal length": [5.1, 4.9, 6.3, 5.8],
        "Sepal width": [3.5, 3.0, 3.3, 3.1],
    })


def _figure(result):
    return result if isinstance(result, go.Figure) else result[0]


def _feature_comparison(label, fov="image_name"):
    return _figure(feature_comparison_plot(
        _plot_frame(), unique_row_id_col="cell_id", fov_name_col=fov,
        selected_var="Sepal length", color_by=["species"], row_id_label=label))


def _2d(label, fov="image_name"):
    return _figure(feature_2d_distribution_plot(
        _plot_frame(), unique_row_id_col="cell_id", fov_name_col=fov,
        selected_x="Sepal length", selected_y="Sepal width",
        color_by=["species"], row_id_label=label))


def _dimension_reduction(label):
    return _figure(dimension_reduction_plot(
        _plot_frame(), unique_row_id_col="cell_id", fov_name_col="image_name",
        selected_features=["Sepal length", "Sepal width"], colored_by=["species"],
        method="PCA", row_id_label=label))


def _phasor(label):
    df = _plot_frame()
    df["Lifetime fit free_ch1: G(1st)"] = [0.4, 0.5, 0.6, 0.45]
    df["Lifetime fit free_ch1: S(1st)"] = [0.3, 0.35, 0.4, 0.3]
    return _figure(phasor_plot(
        df, unique_row_id_col="cell_id", fov_name_col="image_name",
        selected_channel="ch1", color_by=["species"], row_id_label=label))


ALL_PLOTS = (_feature_comparison, _2d, _dimension_reduction, _phasor)


def test_every_point_plot_labels_the_identifier_with_its_column_name():
    """Hover names the identifier by its column, on all four plots -- one hover rule."""
    for plot in ALL_PLOTS:
        templates = _hover_templates(plot("flower_id"))
        assert templates, plot.__name__
        assert all("<b>flower_id:</b> %{text}" in t for t in templates), plot.__name__
        assert not any("Cell ID" in t for t in templates), plot.__name__


def test_every_point_plot_says_id_for_an_invented_row_number():
    for plot in ALL_PLOTS:
        templates = _hover_templates(plot("ID"))
        assert templates, plot.__name__
        assert all("<b>ID:</b> %{text}" in t for t in templates), plot.__name__


def test_the_extraction_branch_still_says_cell_id():
    """pages/data_analysis.py passes "Cell ID" whenever use_data_extraction is on, so
    the main FLIM workflow's hover is unchanged."""
    for plot in ALL_PLOTS:
        templates = _hover_templates(plot("Cell ID"))
        assert templates, plot.__name__
        assert all("<b>Cell ID:</b> %{text}" in t for t in templates), plot.__name__


def test_the_fov_line_reads_the_fov_columns_name():
    """One column, one label: the FOV line reads the FOV column's own name in both
    modules."""
    for plot in (_feature_comparison, _2d):
        templates = _hover_templates(plot("cell_id"))
        assert templates, plot.__name__
        assert all("<b>image_name:</b> %{customdata}" in t for t in templates), plot.__name__
        assert not any("<b>fov:</b>" in t or "<b>Image:</b>" in t for t in templates)


def test_no_fov_column_leaves_the_fov_line_out():
    for plot in (_feature_comparison, _2d):
        templates = _hover_templates(plot("cell_id", fov=None))
        assert templates, plot.__name__
        assert not any("customdata" in t for t in templates), plot.__name__


def test_a_column_name_carrying_markup_cannot_break_the_hover_line():
    """Escape configured column names before including them in HTML hover templates."""
    assert hover_field("a<b", "%{text}") == "<b>a&lt;b:</b> %{text}<br>"
    templates = _hover_templates(_feature_comparison("a<b"))
    assert templates
    assert all("<b>a&lt;b:</b> %{text}" in t for t in templates)


def test_a_feature_column_carrying_markup_is_escaped_too():
    """Every label on the line is a column name, the measured feature included, so the
    axis-value lines go through hover_field as well."""
    df = _plot_frame().rename(columns={"Sepal length": "a<b"})
    fig = _figure(feature_comparison_plot(
        df, unique_row_id_col="cell_id", fov_name_col="image_name",
        selected_var="a<b", color_by=["species"], row_id_label="flower_id"))
    templates = _hover_templates(fig)
    assert templates
    assert all("<b>a&lt;b:</b> %{y:.3f}" in t for t in templates)


# ------------------------------------------------------------------ page wiring

def _iris_frame_with_row_numbers():
    """What load_table hands the page for a table with no identifier."""
    return pd.DataFrame({
        "Row number": ["1", "2", "3", "4"],
        "species": ["setosa", "setosa", "virginica", "virginica"],
        "Sepal length": [5.1, 4.9, 6.3, 5.8],
    })


def _run_page(monkeypatch, row_id_col, configured, use_extraction):
    """Run the analysis page with load_table stubbed because AppTest cannot upload files.
    """
    from streamlit.testing.v1 import AppTest

    from src.widgets import analysis_config_widgets as acw

    df = _iris_frame_with_row_numbers() if row_id_col == "Row number" else _frame()
    monkeypatch.setattr(acw, "get_unique_row_id_col",
                        lambda use_data_extraction=True: configured)
    monkeypatch.setattr(acw, "get_fov_name_col_analysis",
                        lambda use_data_extraction=True: "")
    monkeypatch.setattr(dataset_io, "load_table", lambda *_a, **_k: (
        df, {"Uncategorized Features": ["Sepal length"]}, True, ",", row_id_col))

    page = str(Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py")
    at = AppTest.from_file(page)
    at.session_state["_use_data_extraction"] = use_extraction
    at.run(timeout=90)
    return at


def test_the_page_loads_a_table_with_an_invented_row_id(monkeypatch):
    """The resolved identifier reaches both the plotted frame and the export column snapshot."""
    at = _run_page(monkeypatch, "Row number", configured="", use_extraction=False)
    assert not at.exception
    # The prune snapshot the export replays must carry the invented column too.
    assert "Row number" in at.session_state["analysis_columns"]
    assert "Row number" in at.session_state["vis_df"].columns


def test_the_page_keeps_a_configured_row_id(monkeypatch):
    at = _run_page(monkeypatch, "flower_id", configured="flower_id", use_extraction=False)
    assert not at.exception
    assert "Row number" not in at.session_state["analysis_columns"]
    assert "flower_id" in at.session_state["analysis_columns"]


def test_an_exported_script_reinvents_the_row_id_and_runs(tmp_path):
    """An export with a blank configured identifier must generate the identifier its plot
    uses.
    """
    import subprocess

    from src.export_script import generate_script

    (tmp_path / "iris.csv").write_text(
        "Sepal length,Sepal width,species\n"
        "5.1,3.5,setosa\n4.9,3.0,setosa\n6.3,3.3,virginica\n5.8,2.7,virginica\n")
    state = {
        "csv_filename": "iris.csv", "delimiter": ",",
        "unique_row_id_col": "", "fov_name_col": None,
        "method": "Feature Comparison",
        "categorical_filters": {}, "numerical_filters": [],
        "color_by": ["species"], "opacity_by": None, "shape_by": None,
        "separate_by": None, "subcolor_by": None, "categorical_cols": ["species"],
        "analysis_columns": ["Row number", "species", "Sepal length", "Sepal width"],
        "point_size": 5, "axis_label_size": 12, "legend_size": 10, "colormap": "tab10",
        "method_params": {"selected_var": "Sepal length",
                          "effect_size_method": "None", "statistical_test": "None"},
    }
    script = tmp_path / "analysis.py"
    script.write_text(generate_script(state))
    assert "UNIQUE_ROW_ID_COL = ''" in script.read_text()

    run = subprocess.run([sys.executable, "analysis.py"], cwd=tmp_path,
                         capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    # The prune warns by name for any ANALYSIS_COLUMNS entry the frame lacks, so this
    # fails if the script invented a different name than the app did.
    assert "missing from the data file" not in run.stdout
    assert (tmp_path / "feature_comparison.svg").exists()


# ------------------------------------------------------------------- profile defaults

def test_a_fresh_install_creates_no_analysis_profile_at_all(monkeypatch, tmp_path):
    """Every accessor supports an absent config without creating a profile or writing a
    file.
    """
    from src.widgets import analysis_config_widgets as acw

    monkeypatch.setattr(acw, "_ANALYSIS_CONFIG_PATH", tmp_path / "analysis_config.toml")
    acw.st.session_state.pop("current_profile", None)
    # A configured extraction setup that must not leak on to this branch.
    monkeypatch.setattr(acw, "get_unique_cell_id_col", lambda: "cell_id")
    monkeypatch.setattr(acw, "get_fov_name_col", lambda: "image_name")
    monkeypatch.setattr(acw, "get_categorical_cols", lambda: ["treatment"])

    assert acw._get_current_profile() == ""
    assert acw.list_profiles() == []
    assert not (tmp_path / "analysis_config.toml").exists(), "reading the config wrote it"

    assert acw._get_profile_config() == {}
    assert acw.get_unique_row_id_col(use_data_extraction=False) == ""
    assert acw.get_fov_name_col_analysis(use_data_extraction=False) == ""
    assert acw.get_categorical_cols_analysis(use_data_extraction=False) == []
    assert not (tmp_path / "analysis_config.toml").exists(), "reading the config wrote it"
