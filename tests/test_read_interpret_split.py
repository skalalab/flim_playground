"""read_table validates structure without consulting config; interpret_table applies
explicit column roles and groups. Review can inspect raw headers between the two.
load_table composes them for the Data Extraction branch.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from tests.test_table_formats import _upload, _uploaded_file


def _frame():
    return pd.DataFrame({"cell_id": ["a", "b"], "treatment": ["DMSO", "PBS"],
                         "feat": [1.0, 2.0]})


def _real_upload(df, suffix=".csv"):
    """Use a real UploadedFile for Streamlit's dedicated buffer hasher.
    read_table calls the cached reader; a named BytesIO uses the file-path hasher.
    """
    buf = _upload(df, suffix)
    return _uploaded_file(buf.getvalue(), buf.name)


def _explode(*_a, **_k):
    raise AssertionError("config was read")


# ------------------------------------------------------------- read_table

def test_read_table_reads_no_config(monkeypatch):
    """read_table returns the frame even when every config accessor raises."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", _explode)
    monkeypatch.setattr(dataset_io, "get_fov_name_col_analysis", _explode)

    df, _meta, delimiter, warning, error = dataset_io.read_table(_real_upload(_frame()))

    assert error == ""
    assert warning == ""
    assert delimiter == ","
    assert list(df.columns) == ["cell_id", "treatment", "feat"]


def test_read_table_hands_back_the_files_own_headers():
    """Raw file headers and values are available for profile matching and review."""
    df, _meta, _delim, _warning, error = dataset_io.read_table(
        _real_upload(pd.DataFrame({"a": [1], "b c": [2], "IL-18": [3]})))
    assert error == ""
    assert list(df.columns) == ["a", "b c", "IL-18"]


def test_read_table_reports_a_structural_problem_without_rendering():
    """A rejection comes back as text; the caller decides where it goes.

    CSV bytes under a spreadsheet name -- the name/content mismatch branch, which is
    the earliest thing read_table can reject on.
    """
    df, _meta, _delim, _warning, error = dataset_io.read_table(
        _uploaded_file(b"a,b\n1,2\n", "table.xlsx"))
    assert df is None
    assert error != ""


# --------------------------------------------------------- interpret_table

def test_interpret_table_takes_roles_as_arguments_not_from_config(monkeypatch):
    """Roles arrive as arguments, so a UI-picked name works exactly like a saved one."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", _explode)
    monkeypatch.setattr(dataset_io, "get_fov_name_col_analysis", _explode)

    out, groups, complete, row_id = dataset_io.interpret_table(
        _frame(), ["treatment"], "cell_id", "", use_data_extraction=False)

    assert complete is True
    assert row_id == "cell_id"
    assert "feat" in [col for cols in groups.values() for col in cols]
    assert list(out.columns) == ["cell_id", "treatment", "feat"]


def test_the_grouping_is_an_argument_like_the_roles():
    """Unsaved working-copy groups reach interpretation through explicit arguments."""
    _out, groups, complete, _row_id = dataset_io.interpret_table(
        _frame(), ["treatment"], "cell_id", "",
        feature_groups={"lifetime": ["feat"]}, use_data_extraction=False)

    assert complete is True
    assert groups == {"lifetime": ["feat"]}


def test_the_ignored_columns_are_an_argument_too():
    """Numeric Ignore roles reach get_features and exclude columns that otherwise qualify."""
    df = pd.DataFrame({"cell_id": ["a", "b"], "treatment": ["DMSO", "PBS"],
                       "feat": [1.0, 2.0], "plate": [3, 4]})

    out, groups, _complete, _row_id = dataset_io.interpret_table(
        df, ["treatment"], "cell_id", "", ignored_cols=["plate"],
        use_data_extraction=False)

    assert [col for cols in groups.values() for col in cols] == ["feat"]
    assert "plate" not in out.columns


def test_an_empty_group_map_puts_everything_in_uncategorized():
    """No groups is an answer, and there is nowhere else to look for one."""
    _out, groups, _complete, _row_id = dataset_io.interpret_table(
        _frame(), ["treatment"], "cell_id", "", feature_groups={},
        use_data_extraction=False)

    assert groups == {"Uncategorized Features": ["feat"]}


def test_omitting_the_grouping_reads_no_config_at_all():
    """Omitted groups mean no groups, without consulting the active profile or widget modules."""
    _out, groups, _complete, _row_id = dataset_io.interpret_table(
        _frame(), ["treatment"], "cell_id", "", use_data_extraction=False)

    assert groups == {"Uncategorized Features": ["feat"]}
    assert not hasattr(dataset_io, "get_all_feature_groups")


def test_interpret_table_invents_a_row_id_when_none_is_named():
    """A blank name still means "number the rows", the same as through load_table."""
    out, _groups, complete, row_id = dataset_io.interpret_table(
        _frame(), ["treatment"], "", "", use_data_extraction=False)

    assert complete is True
    assert row_id == "Row number"
    assert list(out["Row number"]) == ["1", "2"]


def test_interpret_table_rejects_a_named_row_id_the_frame_lacks():
    out, groups, complete, _row_id = dataset_io.interpret_table(
        _frame(), ["treatment"], "missing_id", "", use_data_extraction=False)

    assert complete is False
    assert out is None and groups is None


# --------------------------------------------------------------- composition

@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".xlsx"])
def test_load_table_still_returns_what_the_two_halves_produce(suffix, monkeypatch):
    """Extraction load_table matches read_table plus interpret_table on every read branch."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    monkeypatch.setattr(dataset_io, "get_fov_name_col_analysis", lambda *a, **k: "")

    combined = dataset_io.load_table(_real_upload(_frame(), suffix), ["treatment"])
    df, _meta, delimiter, scope_warning, error = dataset_io.read_table(_real_upload(_frame(), suffix))
    assert error == ""
    stepwise = dataset_io.interpret_table(df, ["treatment"], "cell_id", "",
                                          scope_warning=scope_warning)

    assert combined[2] is stepwise[2] is True          # upload_complete
    assert combined[3] == delimiter                    # separator baked into exports
    assert combined[4] == stepwise[3] == "cell_id"     # resolved row id
    assert list(combined[0].columns) == list(stepwise[0].columns)
