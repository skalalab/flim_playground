"""Generated scripts and the app open each supported table format into the same frame.
The generated read call preserves suffix dispatch, header normalization, and the
app's detected delimiter.
"""
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset_io import check_and_fix_df, coerce_majority_numeric_cols
from src.export_script import _build_read_call, generate_script


def _frame():
    return pd.DataFrame({
        "cell_id": ["fov1_1", "fov1_2", "fov2_1", "fov2_2"],
        "image_name": ["fov1", "fov1", "fov2", "fov2"],
        "treatment": ["control", "drug", "control", "drug"],
        "Lifetime fit_ch1: T1": [0.40, 0.55, 0.61, 0.48],
        "Lifetime fit_ch1: T2": [2.10, 2.35, 2.51, 2.28],
    })


def _state(filename, delimiter=","):
    df = _frame()
    return {
        "csv_filename": filename,
        "delimiter": delimiter,
        "unique_row_id_col": "cell_id",
        "fov_name_col": "image_name",
        "method": "Feature Comparison",
        "categorical_filters": {},
        "numerical_filters": [],
        "color_by": ["treatment"],
        "opacity_by": None,
        "shape_by": None,
        "separate_by": None,
        "subcolor_by": None,
        "point_size": 5,
        "axis_label_size": 12,
        "legend_size": 10,
        "colormap": "tab10",
        "categorical_cols": ["treatment"],
        "analysis_columns": list(df.columns),
        "method_params": {"selected_var": "Lifetime fit_ch1: T1",
                          "effect_size_method": "None",
                          "statistical_test": "None"},
    }


def _write(df, path, delimiter=","):
    suffix = path.suffix.lower()
    if suffix == ".ods":
        df.to_excel(path, index=False, engine="odf")
    elif suffix in (".xlsx", ".xlsm"):
        df.to_excel(path, index=False, engine="openpyxl")
    else:
        df.to_csv(path, index=False, sep=delimiter)


def _app_frame(path, delimiter=","):
    """What the app holds in vis_df, via src.dataset_io's own read branch."""
    from src.dataset_io import _read_table_cached, suffix_of

    class _Upload:
        name = path.name

        def __init__(self, handle):
            self._handle = handle

        def __getattr__(self, attr):
            return getattr(self._handle, attr)

    with open(path, "rb") as handle:
        upload = _Upload(handle)
        df, _meta = _read_table_cached.__wrapped__(upload, suffix_of(upload))
    df, _warning, error = check_and_fix_df(df, ["treatment"], "cell_id", "image_name")
    assert error == "", error
    df, _coerce = coerce_majority_numeric_cols(df, {"cell_id", "image_name", "treatment"})
    return df


@pytest.mark.parametrize("filename,delimiter", [
    ("data.csv", ","),
    ("data.csv", ";"),
    ("data.tsv", "\t"),
    ("data.txt", ";"),
    ("data.txt", "|"),
    ("data.xlsx", ","),
    ("data.ods", ","),
])
def test_exported_script_loads_the_same_frame_the_app_did(tmp_path, monkeypatch,
                                                          filename, delimiter):
    path = tmp_path / filename
    _write(_frame(), path, delimiter)

    script_path = tmp_path / "analysis.py"
    script_path.write_text(generate_script(_state(filename, delimiter)))
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(script_path))
    finally:
        plt.close("all")

    # Plotting adds derived columns; compare every column loaded by the app.
    app_df = _app_frame(path, delimiter)
    exported = namespace["df"]
    missing = [col for col in app_df.columns if col not in exported.columns]
    assert not missing, f"exported script lost {missing}"
    pd.testing.assert_frame_equal(
        exported[app_df.columns].reset_index(drop=True), app_df.reset_index(drop=True))


def test_spreadsheet_read_call_never_uses_read_csv():
    """Spreadsheet exports use the Excel reader and normalize headers to strings."""
    for filename in ("book.xlsx", "book.xlsm", "book.xlsb", "book.xls", "book.ods"):
        call = _build_read_call(filename)
        assert "read_excel" in call and "read_csv" not in call
        assert 'engine="calamine"' in call
        # Stringify spreadsheet headers for categorical membership and df[name] lookups.
        assert "str(col) for col in df.columns" in call


def test_a_csv_bakes_in_the_separator_the_app_detected():
    """`.csv` shares one detection rule with .tsv/.txt, so the script must carry
    the answer rather than default to a comma — a `;`-separated .csv is valid.
    """
    comma = _build_read_call("data.csv", ",")
    assert "SEPARATOR = ','" in comma
    assert "sep=SEPARATOR" in comma and "index_col=False" in comma and "low_memory=False" in comma

    semicolon = _build_read_call("data.csv", ";")
    assert "SEPARATOR = ';'" in semicolon


def test_a_semicolon_csv_round_trips_through_the_exported_script(tmp_path, monkeypatch):
    """A semicolon-separated CSV exports with the delimiter detected by the app."""
    path = tmp_path / "euro.csv"
    _frame().to_csv(path, index=False, sep=";")
    (tmp_path / "analysis.py").write_text(generate_script(_state("euro.csv", ";")))
    monkeypatch.chdir(tmp_path)
    try:
        namespace = runpy.run_path(str(tmp_path / "analysis.py"))
    finally:
        plt.close("all")
    app_df = _app_frame(path, ";")
    pd.testing.assert_frame_equal(
        namespace["df"][app_df.columns].reset_index(drop=True), app_df.reset_index(drop=True))


def test_delimited_read_call_bakes_in_the_apps_answer():
    call = _build_read_call("data.txt", "|")
    assert "SEPARATOR = '|'" in call
    # It must not re-detect: the script has to reproduce the plot it came from,
    # not re-decide how to read the file.
    assert "sep=None" not in call and "Sniffer" not in call
