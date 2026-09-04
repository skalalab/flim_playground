"""load_table is two halves: read_table judges structure, interpret_table judges meaning.

The split exists so a caller can stop between them -- holding the file's own headers,
before any profile has been consulted. Two properties make that possible, and both fail
silently if lost:

- `read_table` reads no config, so it cannot be blocked by a profile that does not fit
  the file. Proved here by mutation: the config accessors are replaced with ones that
  raise, and read_table still returns the frame.
- `interpret_table` takes every column role as an *argument*, so the same code serves a
  name that came from the saved profile and one the user just picked in the UI. The
  review table's other column, the feature groups, travels the same way and has no
  fall-back to the *active* profile, which under "the file picks the profile" need not
  be the profile the file matched at all.

`load_table` composes the two for the Data Extraction branch, and only for it, because
a user's table has to stop between the halves by construction: the gate shows the file's
own headers before any profile has been applied.
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
    """A genuine UploadedFile, which st.cache_data has a dedicated hasher branch for.

    read_table goes through the *decorated* _read_table_cached, unlike the reader
    tests in test_table_formats, which call its __wrapped__ body. A bare BytesIO
    sends the hasher down its file-path branch and it stats the name.
    """
    buf = _upload(df, suffix)
    return _uploaded_file(buf.getvalue(), buf.name)


def _explode(*_a, **_k):
    raise AssertionError("config was read")


# ------------------------------------------------------------- read_table

def test_read_table_reads_no_config(monkeypatch):
    """The property the whole redesign rests on, proved by breaking config."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", _explode)
    monkeypatch.setattr(dataset_io, "get_fov_name_col_analysis", _explode)

    df, _meta, delimiter, warning, error = dataset_io.read_table(_real_upload(_frame()))

    assert error == ""
    assert warning == ""
    assert delimiter == ","
    assert list(df.columns) == ["cell_id", "treatment", "feat"]


def test_read_table_hands_back_the_files_own_headers():
    """What step 3's matching and the review table are built from."""
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
    """Not a role -- the review table's other column. Same argument treatment.

    A caller holding a session-local working copy has to be able to say "group feat
    under lifetime" without that edit having been saved to disk first.
    """
    _out, groups, complete, _row_id = dataset_io.interpret_table(
        _frame(), ["treatment"], "cell_id", "",
        feature_groups={"lifetime": ["feat"]}, use_data_extraction=False)

    assert complete is True
    assert groups == {"lifetime": ["feat"]}


def test_the_ignored_columns_are_an_argument_too():
    """The Ignore role reaches get_features through here, or it is inert.

    A *numeric* ignored column is the case that matters: it passes the dtype test, so
    only exclusion by name keeps it out of the features. Left unplumbed, the role
    would be recorded in the profile and silently do nothing.
    """
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
    """The same answer as `{}` -- there is no fall-back to the active profile.

    Left in, that fall-back was reachable only from a caller that passed nothing, and
    under "the file picks the profile" the active profile need not be the matched one:
    a file described by `pdl1` would take its groups from whatever was saved last. It
    was also `dataset_io`'s second reason to import the widget modules, which is why
    the accessor is gone from this module rather than merely unused.
    """
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
    """The composition matches the halves called by hand, on every read branch.

    Both sides are the extraction branch now -- load_table has no other -- so the
    stepwise call leaves `use_data_extraction` at its default rather than passing False.
    """
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
