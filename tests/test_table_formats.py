"""Supported table formats produce consistent frames and explain rejected shapes.
Spreadsheet fixtures use openpyxl and odfpy writer engines.
"""
import html
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dataset_io
from src.dataset_io import (
    _SAMPLE_BYTES,
    DELIMITED_SUFFIXES,
    SPREADSHEET_SUFFIXES,
    SUPPORTED_SUFFIXES,
    _as_html,
    _comma_decimal_hint,
    _content_is_spreadsheet,
    _diagnose_table,
    _name_content_mismatch,
    _read_table_cached,
    _resolve_delimiter,
    check_and_fix_df,
    coerce_majority_numeric_cols,
    drop_unnamed_columns,
    get_features,
    suffix_of,
)


class FakeUpload(io.BytesIO):
    """Stands in for a Streamlit UploadedFile: a named, single-read byte buffer."""

    def __init__(self, data, name):
        super().__init__(data)
        self.name = name


def _frame():
    return pd.DataFrame({
        "cell_id": ["fov1_1", "fov1_2", "fov2_1"],
        "image_name": ["fov1", "fov1", "fov2"],
        "treatment": ["control", "drug", "drug"],
        "Lifetime fit_ch1: T1": [0.4, 0.55, 0.61],
    })


def _upload(df, suffix, **to_kwargs):
    buf = io.BytesIO()
    if suffix in SPREADSHEET_SUFFIXES:
        engine = "odf" if suffix == ".ods" else "openpyxl"
        df.to_excel(buf, index=False, engine=engine, **to_kwargs)
    else:
        sep = {".tsv": "\t", ".txt": ";"}.get(suffix, ",")
        buf.write(df.to_csv(index=False, sep=sep, **to_kwargs).encode())
    return FakeUpload(buf.getvalue(), f"table{suffix}")


# --------------------------------------------------------------------------- #
# 1. Every supported format reaches the same frame
# --------------------------------------------------------------------------- #

def _read(upload):
    # Call the cached function's undecorated body: st.cache_data needs a script
    # run context, and the caching itself is not what these tests are about.
    return _read_table_cached.__wrapped__(upload, suffix_of(upload))


@pytest.mark.parametrize("suffix", SUPPORTED_SUFFIXES)
def test_every_supported_suffix_reads_identically_to_csv(suffix):
    expected = _frame()
    df, _meta = _read(_upload(expected, suffix))
    pd.testing.assert_frame_equal(df, expected)


def test_suffix_of_is_case_insensitive_and_defaults_to_csv():
    assert suffix_of(FakeUpload(b"", "TABLE.XLSX")) == ".xlsx"
    assert suffix_of(FakeUpload(b"", "no_extension")) == ".csv"
    # A dotfile has no extension either; ".csv" keeps it on the default branch
    # rather than inventing a suffix out of the name.
    assert suffix_of(FakeUpload(b"", ".hidden")) == ".csv"


@pytest.mark.parametrize("sep,suffix", [("\t", ".tsv"), (";", ".txt"), ("|", ".txt")])
def test_delimited_text_separator_is_detected_not_assumed(sep, suffix):
    expected = _frame()
    raw = expected.to_csv(index=False, sep=sep).encode()
    df, _meta = _read(FakeUpload(raw, f"table{suffix}"))
    pd.testing.assert_frame_equal(df, expected)


def test_spreadsheet_header_cells_are_stringified():
    """A numeric header cell arrives as an int; everything downstream assumes str."""
    df, _meta = _read(_upload(pd.DataFrame({1: [1, 2], "cell_id": ["a", "b"]}), ".xlsx"))
    assert [type(col) for col in df.columns] == [str, str]
    assert list(df.columns) == ["1", "cell_id"]


def test_excel_ids_that_collide_as_strings_are_rejected_before_analysis():
    frame = pd.DataFrame({"cell_id": [1, "1", "a"], "Area": [0.4, 0.5, 0.6]})
    df, meta = _read(_upload(frame, ".xlsx"))
    assert _diagnose_table(df, meta, "table.xlsx")[1] == ""
    assert df.cell_id.tolist() == [1, "1", "a"]

    reason = dataset_io.review_blocking_reason(
        df, {"cell_id": "row_id", "Area": "numerical"})
    fixed, _warning, error = check_and_fix_df(df, [], "cell_id", None)

    assert "appears 2 times" in reason, reason
    assert "appears 2 times" in error, error
    assert fixed is None


def test_only_the_first_sheet_is_read_and_the_rest_are_reported():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _frame().to_excel(writer, sheet_name="Data", index=False)
        pd.DataFrame({"junk": [1]}).to_excel(writer, sheet_name="Notes", index=False)
    df, meta = _read(FakeUpload(buf.getvalue(), "book.xlsx"))
    pd.testing.assert_frame_equal(df, _frame())
    assert meta["sheets"] == ["Data", "Notes"]


def test_read_rewinds_so_a_second_read_is_not_an_empty_parse():
    """An UploadedFile is a single-read buffer; a second parse must still work."""
    upload = _upload(_frame(), ".csv")
    first, _ = _read(upload)
    second, _ = _read(upload)
    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
# 2. Diagnostics name the rule that was broken
# --------------------------------------------------------------------------- #

def test_multi_sheet_warns_naming_both_the_sheet_read_and_the_ones_skipped():
    warning, error = _diagnose_table(_frame(), {"sheets": ["Data", "Notes", "QC"]}, "book.xlsx")
    assert error == ""
    assert "'Data'" in warning and "'Notes'" in warning and "'QC'" in warning
    assert "3 sheets" in warning
    assert "first sheet" in warning


def test_single_sheet_workbook_is_not_warned_about():
    warning, error = _diagnose_table(_frame(), {"sheets": ["Sheet1"]}, "book.xlsx")
    assert warning == "" and error == ""


@pytest.mark.parametrize("delimiter", [";", "\t", "|"])
def test_a_parseable_separator_loads_instead_of_being_diagnosed(delimiter):
    """Detection splits on it, so the file loads instead of being diagnosed."""
    expected = _frame()
    raw = expected.to_csv(index=False, sep=delimiter).encode()
    df, meta = _read(FakeUpload(raw, "plate.csv"))
    _warning, error = _diagnose_table(df, meta, "plate.csv")
    assert error == ""
    pd.testing.assert_frame_equal(df, expected)


def test_a_single_column_table_is_not_rejected_by_the_reader():
    """How many columns a table needs is get_features' question, not the reader's."""
    _warning, error = _diagnose_table(pd.DataFrame({"cell_id": ["a"]}), {}, "one.csv")
    assert error == ""


def test_a_colon_separated_file_passes_the_reader_and_is_judged_downstream():
    """Reading it is not the reader's failure; having no usable columns is
    check_and_fix_df's finding, and it reports that itself."""
    df = pd.DataFrame({"cell_id:image_name:treatment": ["a:f1:ctrl"]})
    _warning, error = _diagnose_table(df, {}, "plate.csv")
    assert error == ""
    _fixed, _w, downstream = check_and_fix_df(df, [], "cell_id", "image_name")
    assert "cell_id" in downstream


def test_unnamed_columns_are_skipped_with_a_warning_not_a_rejection():
    """A spacer column must not cost the user the whole file."""
    kept = pd.DataFrame({"cell_id": ["a"], "treatment": ["ctrl"]})
    meta = {"unnamed_count": 5, "all_unnamed": False, "sheets": ["Summary"]}
    warning, error = _diagnose_table(kept, meta, "plate.xlsx")
    assert error == ""
    assert "5 columns" in warning and "skipped" in warning
    assert "'Summary'" in warning


def test_a_file_where_nothing_has_a_name_is_rejected():
    """The one case that really is a missing header rather than stray columns."""
    blank = pd.DataFrame(columns=["Unnamed: 0", "Unnamed: 1"])
    blank.loc[0] = ["cell_id", "treatment"]
    meta = {"unnamed_count": 2, "all_unnamed": True, "sheets": ["Summary"]}
    _warning, error = _diagnose_table(blank, meta, "plate.xlsx")
    assert "no column" in error and "name" in error
    assert "'Summary'" in error


def test_drop_unnamed_columns_keeps_the_named_ones():
    df = pd.DataFrame({"cell_id": ["a"], "Unnamed: 1": ["x"],
                       "treatment": ["ctrl"], "Unnamed: 3": ["y"]})
    assert list(drop_unnamed_columns(df).columns) == ["cell_id", "treatment"]


def test_drop_unnamed_columns_leaves_an_entirely_unnamed_frame_intact():
    """Keep an all-unnamed frame intact so diagnostics can report missing headers."""
    df = pd.DataFrame({"Unnamed: 0": ["a"], "Unnamed: 1": ["b"]})
    assert list(drop_unnamed_columns(df).columns) == ["Unnamed: 0", "Unnamed: 1"]


def test_drop_unnamed_columns_is_a_no_op_on_a_clean_table():
    df = _frame()
    pd.testing.assert_frame_equal(drop_unnamed_columns(df), df)


def test_a_majority_of_unnamed_columns_is_not_a_rejection():
    """Three spacer columns out of five: dropped, not a rejection."""
    upload = FakeUpload(b"cell_id,,,treatment,\na,,,ctrl,\n", "plate.csv")
    df, meta = _read(upload)
    assert list(df.columns) == ["cell_id", "treatment"]
    assert meta["unnamed_count"] == 3 and meta["all_unnamed"] is False
    _warning, error = _diagnose_table(df, meta, "plate.csv")
    assert error == ""


def test_empty_sheet_is_named_rather_than_reported_as_empty():
    _warning, error = _diagnose_table(pd.DataFrame(), {"sheets": ["Summary"]}, "book.xlsx")
    assert "'Summary'" in error and "book.xlsx" in error


def test_headers_with_no_rows_beneath_them_are_an_error_not_a_silent_empty_plot():
    header_only = pd.DataFrame(columns=["cell_id", "image_name", "treatment"])
    _warning, error = _diagnose_table(header_only, {}, "plate.csv")
    assert "no rows beneath them" in error and "plate.csv" in error


def test_a_table_of_nothing_but_empty_columns_is_the_only_shape_rejected():
    """Those columns are dropped downstream, so nothing would survive."""
    all_blank = pd.DataFrame({"a": [None, None], "b": [None, None]})
    _warning, error = _diagnose_table(all_blank, {}, "plate.csv")
    assert "no rows beneath them" in error


def test_one_surviving_column_among_empty_ones_is_enough():
    partly = pd.DataFrame({"cell_id": ["a", "b"], "blank": [None, None]})
    _warning, error = _diagnose_table(partly, {}, "plate.csv")
    assert error == ""


def test_two_columns_of_one_name_are_rejected_rather_than_left_to_crash():
    """Reject numeric and text headers that collide after stringification, before Series
    lookups see duplicate columns.
    """
    upload = _upload(pd.DataFrame({"cell_id": ["a", "b"], 1: [0.4, 0.5], "1": [9.1, 9.2]}),
                     ".xlsx")
    df, meta = _read(upload)
    assert list(df.columns) == ["cell_id", "1", "1"]
    assert meta["duplicate_names"] == {"1": [2, 3]}

    _warning, error = _diagnose_table(df, meta, "plate.xlsx")
    assert "columns 2 and 3" in error and "'1'" in error and "Rename" in error
    # The advice that makes it findable: on screen the two header cells are identical.
    assert "cell formatting" in error


def test_the_positions_named_are_the_files_own_columns_not_the_surviving_ones():
    """Counted before the blank columns go, so column 4 of the message is column D."""
    blank_then_pair = pd.DataFrame([["a", None, 0.4, 9.1]], columns=["cell_id", "", 1, "1"])
    df, meta = _read(_upload(blank_then_pair, ".xlsx"))
    assert list(df.columns) == ["cell_id", "1", "1"]     # the blank one was dropped
    assert meta["duplicate_names"] == {"1": [3, 4]}
    _warning, error = _diagnose_table(df, meta, "plate.xlsx")
    assert "columns 3 and 4" in error


def test_a_header_repeated_in_the_file_is_renamed_by_pandas_and_left_alone():
    """Repeated source headers remain usable after pandas gives them distinct names."""
    text = FakeUpload(b"cell_id,Area,Area\na,1.0,2.0\n", "plate.csv")
    df, meta = _read(text)
    assert list(df.columns) == ["cell_id", "Area", "Area.1"]
    assert meta["duplicate_names"] == {}
    assert _diagnose_table(df, meta, "plate.csv") == ("", "")

    book = _upload(pd.DataFrame([["a", 1.0, 2.0]], columns=["cell_id", "Area", "Area"]),
                   ".xlsx")
    df, meta = _read(book)
    assert list(df.columns) == ["cell_id", "Area", "Area.1"]
    assert meta["duplicate_names"] == {}


def test_the_header_cell_advice_rides_only_on_the_files_that_can_cause_it():
    """Text readers deduplicate headers, so spreadsheet collision advice is omitted."""
    df = pd.DataFrame([[1.0, 2.0]], columns=["x", "x"])
    _warning, error = _diagnose_table(df, {"duplicate_names": {"x": [1, 2]}}, "plate.csv")
    assert "both named 'x'" in error
    assert "cell formatting" not in error


# --------------------------------------------------------------------------- #
# 3. Delimiter detection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("delimiter", ["\t", ";", "|", ","])
def test_delimiter_survives_feature_names_containing_spaces_and_colons(delimiter):
    """Spaces and colons inside feature names must not become delimiters."""
    expected = _frame()
    raw = expected.to_csv(index=False, sep=delimiter).encode()
    assert _resolve_delimiter(FakeUpload(raw, "table.txt"))[0] == delimiter
    df, _meta = _read(FakeUpload(raw, "table.txt"))
    pd.testing.assert_frame_equal(df, expected)


def test_csv_is_detected_exactly_like_any_other_text_table():
    """No suffix is privileged: .csv, .tsv and .txt share one rule."""
    raw = b"a;b;c\n1;2;3\n"
    for name in ("table.csv", "table.tsv", "table.txt"):
        assert _resolve_delimiter(FakeUpload(raw, name))[0] == ";"


def test_a_real_comma_table_still_parses_as_comma():
    """A comma per column boundary; nothing else comes close on a real header."""
    expected = _frame()
    raw = expected.to_csv(index=False).encode()
    assert _resolve_delimiter(FakeUpload(raw, "table.csv"))[0] == ","
    df, meta = _read(FakeUpload(raw, "table.csv"))
    pd.testing.assert_frame_equal(df, expected)
    assert meta["delimiter"] == ","


def test_exactly_one_candidate_must_split_the_file_rectangularly():
    """The whole rule: same field count on every row, and more than one field."""
    raw = b"a;b;c,d\n1;2;3,4\n5;6;7,8\n"
    _delimiter, viable, _pr, _un = _resolve_delimiter(FakeUpload(raw, "t.csv"))
    # Both carve this into a rectangle, so neither is "the" separator.
    assert set(viable) == {";", ","}


def test_a_wrong_separator_is_excluded_by_collapsing_the_table():
    """A separator that yields one column is not viable, however often it occurs."""
    raw = _frame().to_csv(index=False).encode()
    _delimiter, viable, _pr, _un = _resolve_delimiter(FakeUpload(raw, "t.csv"))
    assert viable == (",",), "a comma table admits no other separator"


def test_a_separator_inside_quotes_is_content_not_structure():
    """csv.reader is used rather than str.split, so quoting is honoured."""
    raw = b'a\tb\tc\n"x,y"\tz\tw\n"p,q"\tr\ts\n'
    delimiter, viable, _pr, _un = _resolve_delimiter(FakeUpload(raw, "t.csv"))
    assert delimiter == "\t" and viable == ("\t",)


def test_european_decimals_are_explained_rather_than_rejected():
    """Semicolon-separated comma decimals parse as text; the no-features error explains
    why.
    """
    raw = b"cell_id;a;b\nc1;1,5;2,3\nc2;2,5;3,3\n"
    delimiter, viable, _pr, _un = _resolve_delimiter(FakeUpload(raw, "t.csv"))
    assert delimiter == ";" and viable == (";",)

    df, meta = _read(FakeUpload(raw, "plate.csv"))
    _warning, error = _diagnose_table(df, meta, "plate.csv")
    assert error == ""
    hint = _comma_decimal_hint(df)
    assert "1,5" in hint and "full stop" in hint


def test_a_good_table_is_not_lost_to_one_label_column():
    """A comma-separated coordinate label must not reject valid numeric measurements."""
    raw = (b"cell_id;Lifetime fit_ch1: T1;Lifetime fit_ch1: T2;position\n"
           b"c1;1.5;2.5;3,4\nc2;1.6;2.6;5,7\n")
    df, meta = _read(FakeUpload(raw, "plate.csv"))
    _warning, error = _diagnose_table(df, meta, "plate.csv")
    assert error == ""
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    assert numeric == ["Lifetime fit_ch1: T1", "Lifetime fit_ch1: T2"]
    # `position` is text, so it is never a feature — but it does not fire the hint
    # either, because the table it sits in has real numbers in it.
    assert _comma_decimal_hint(df[numeric + ["cell_id"]]) == ""


def test_a_spreadsheet_gets_the_decimal_advice_too():
    """The comma-decimal hint also applies to text numbers stored in spreadsheets."""
    frame = pd.DataFrame({"cell_id": ["c1", "c2"], "a": ["1,5", "1,7"]})
    df, _meta = _read(_upload(frame, ".xlsx"))
    hint = _comma_decimal_hint(df)
    assert "1,5" in hint and "full stop" in hint


def test_a_comma_separated_table_is_never_called_comma_decimal():
    """Under a comma separator "1,5" is two fields, not one European number."""
    raw = b"a,b,c\n1,5,3\n2,5,3\n"
    df, _meta = _read(FakeUpload(raw, "t.csv"))
    assert _comma_decimal_hint(df) == ""


def test_ordinary_full_stop_decimals_pass_under_any_separator():
    for sep in (b";", b"\t", b"|"):
        raw = b"a" + sep + b"b\n1.5" + sep + b"2.5\n3.5" + sep + b"4.5\n"
        df, _meta = _read(FakeUpload(raw, "t.csv"))
        assert _comma_decimal_hint(df) == ""


def test_a_thousands_separator_still_matches_but_only_costs_advice():
    """Ambiguous comma-separated text adds advice only when the table already has no
    features.
    """
    raw = b"a;b\n1,234;5\n2,345;6\n"
    df, _meta = _read(FakeUpload(raw, "t.csv"))
    assert "full stop" in _comma_decimal_hint(df)


def test_a_column_of_mixed_text_and_commas_is_not_called_decimal():
    """Every value must match, so a stray "3,4" among words says nothing."""
    raw = b"cell_id;note\nc1;alpha\nc2;3,4\n"
    df, _meta = _read(FakeUpload(raw, "t.csv"))
    assert _comma_decimal_hint(df) == ""


def test_the_suffix_never_influences_the_answer():
    """One rule for every text table — the name is not evidence."""
    raw = b"a;b;c\n1;2;3\n4;5;6\n"
    for name in ("t.csv", "t.tsv", "t.txt"):
        assert _resolve_delimiter(FakeUpload(raw, name))[0] == ";"


def test_an_ambiguous_file_is_rejected_rather_than_guessed_at():
    """Choosing one would look right and bury the other in a column name."""
    raw = b"dose;units,value\n10;mg,5\n20;mg,7\n"
    df, meta = _read(FakeUpload(raw, "plate.csv"))
    assert len(meta["viable_delimiters"]) == 2
    _warning, error = _diagnose_table(df, meta, "plate.csv")
    assert "semicolon and comma" in error
    assert "one of them only" in error


def test_an_ordinary_table_is_never_called_ambiguous():
    raw = _frame().to_csv(index=False).encode()
    df, meta = _read(FakeUpload(raw, "plate.csv"))
    assert meta["viable_delimiters"] == (",",)
    warning, error = _diagnose_table(df, meta, "plate.csv")
    assert error == "" and "read as" not in warning


def test_delimiter_detection_rewinds_the_buffer_it_peeked_at():
    upload = _upload(_frame(), ".tsv")
    _resolve_delimiter(upload)
    df, _meta = _read(upload)
    pd.testing.assert_frame_equal(df, _frame())


def test_single_column_file_with_no_candidate_falls_back_to_comma():
    assert _resolve_delimiter(FakeUpload(b"cell_id\na\n", "table.txt"))[0] == ","


# --------------------------------------------------------------------------- #
# 4. The suffix lists stay coherent with each other
# --------------------------------------------------------------------------- #

def test_every_advertised_suffix_reaches_a_read_branch():
    assert set(SUPPORTED_SUFFIXES) <= {*DELIMITED_SUFFIXES, *SPREADSHEET_SUFFIXES}
    assert all(suffix.startswith(".") and suffix.islower() for suffix in SUPPORTED_SUFFIXES)
    # .csv is not privileged: it shares the detection rule with .tsv and .txt.
    assert ".csv" in DELIMITED_SUFFIXES


def test_untested_spreadsheet_formats_are_readable_but_not_advertised():
    """The reader routes .xlsb and .xls, but they remain unadvertised until genuine
    fixtures cover them.
    """
    for suffix in (".xlsb", ".xls"):
        assert suffix in SPREADSHEET_SUFFIXES
        assert suffix not in SUPPORTED_SUFFIXES


# --------------------------------------------------------------------------- #
# 5. Row terminators — pandas' job, except for the sample the probe normalises
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("eol,eol_name", [("\n", "unix"), ("\r\n", "windows"), ("\r", "classic-mac")])
@pytest.mark.parametrize("sep", ["\t", ";", "|", ","])
def test_every_row_terminator_parses_with_every_separator(eol, eol_name, sep):
    """A bare \\r is what Excel's "CSV (Macintosh)" export still writes."""
    expected = _frame()
    lines = [sep.join(str(c) for c in expected.columns)]
    lines += [sep.join(str(v) for v in row) for row in expected.itertuples(index=False)]
    raw = (eol.join(lines) + eol).encode()

    assert _resolve_delimiter(FakeUpload(raw, "table.txt"))[0] == sep
    df, _meta = _read(FakeUpload(raw, "table.txt"))
    pd.testing.assert_frame_equal(df, expected)


def test_a_bare_cr_is_normalised_before_the_rows_are_split():
    """Normalize bare carriage returns before sampling rows so delimiters are counted per
    row.
    """
    header = "cell_id\timage_name\tvalue"
    rows = ["a\tf1\t" + ";".join("0" for _ in range(50)) for _ in range(20)]
    raw = ("\r".join([header, *rows]) + "\r").encode()
    assert _resolve_delimiter(FakeUpload(raw, "table.txt"))[0] == "\t"


def test_a_quoted_header_cell_is_content_not_structure():
    raw = b'"a;b;c;d",cell_id,image_name\n"x;y;z;w",c1,f1\n'
    assert _resolve_delimiter(FakeUpload(raw, "table.txt"))[0] == ","
    df, _meta = _read(FakeUpload(raw, "table.txt"))
    assert list(df.columns) == ["a;b;c;d", "cell_id", "image_name"]


# --------------------------------------------------------------------------- #
# 6. Filename vs contents — the branch is chosen by suffix, so they must agree
# --------------------------------------------------------------------------- #

def _xlsx_bytes():
    buf = io.BytesIO()
    _frame().to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def test_a_workbook_named_csv_is_named_not_left_to_a_unicode_error():
    """Binary workbook content under a text suffix reports the filename/content mismatch."""
    error = _name_content_mismatch(FakeUpload(_xlsx_bytes(), "export.csv"), "export.csv")
    assert "named like a text file" in error and "spreadsheet" in error
    assert ".xlsx" in error


def test_a_text_table_named_xlsx_is_named_not_left_to_a_calamine_error():
    """Text content under a spreadsheet suffix reports the filename/content mismatch."""
    raw = _frame().to_csv(index=False).encode()
    error = _name_content_mismatch(FakeUpload(raw, "export.xlsx"), "export.xlsx")
    assert "named like a spreadsheet" in error and "plain text" in error
    assert ".csv" in error


@pytest.mark.parametrize("suffix", SUPPORTED_SUFFIXES)
def test_correctly_named_files_report_no_mismatch(suffix):
    upload = _upload(_frame(), suffix)
    assert _name_content_mismatch(upload, upload.name) == ""


def test_legacy_xls_is_recognised_by_its_ole2_signature():
    # .xls is an OLE2 compound file, not a zip like the modern formats.
    ole2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 32
    assert _content_is_spreadsheet(ole2)
    assert _name_content_mismatch(FakeUpload(ole2, "old.csv"), "old.csv") != ""


def test_the_mismatch_probe_rewinds_what_it_peeked_at():
    upload = _upload(_frame(), ".xlsx")
    _name_content_mismatch(upload, upload.name)
    df, _meta = _read(upload)
    pd.testing.assert_frame_equal(df, _frame())


# --------------------------------------------------------------------------- #
# 7. Spreadsheet-native types that CSV never produces
# --------------------------------------------------------------------------- #

def test_a_date_column_never_becomes_a_nanosecond_feature():
    """Spreadsheet dates must not turn into numeric nanosecond measurements."""
    df = _frame()
    df["acquired"] = pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"])
    out, _warning = coerce_majority_numeric_cols(df.copy(), {"cell_id", "image_name", "treatment"})
    assert not pd.api.types.is_numeric_dtype(out["acquired"])
    assert pd.api.types.is_datetime64_any_dtype(out["acquired"])


def test_a_duration_column_is_left_alone_too():
    df = _frame()
    df["exposure"] = pd.to_timedelta([1, 2, 3], unit="s")
    out, _warning = coerce_majority_numeric_cols(df.copy(), {"cell_id", "image_name", "treatment"})
    assert not pd.api.types.is_numeric_dtype(out["exposure"])


def test_a_workbook_and_its_csv_export_agree_on_which_columns_are_features():
    """The equivalence that the datetime guard exists to protect."""
    df = _frame()
    df["acquired"] = pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"])
    skip = {"cell_id", "image_name", "treatment"}

    book, _ = _read(_upload(df, ".xlsx"))
    text, _ = _read(_upload(df, ".csv"))
    book, _ = coerce_majority_numeric_cols(book, skip)
    text, _ = coerce_majority_numeric_cols(text, skip)

    features = lambda frame: [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    assert features(book) == features(text)
    assert "acquired" not in features(book)


def test_real_numeric_columns_are_still_coerced_through_stray_text():
    """The datetime guard must not disturb the existing <=1%-stray-text rule."""
    values = [str(i / 10) for i in range(200)]
    values[7] = "N/A"                      # 0.5% non-numeric, inside the threshold
    df = pd.DataFrame({"cell_id": [f"c{i}" for i in range(200)], "mostly_numeric": values})
    out, warning = coerce_majority_numeric_cols(df.copy(), {"cell_id"})
    assert pd.api.types.is_numeric_dtype(out["mostly_numeric"])
    assert "1 non-numeric value" in warning


def test_a_genuinely_textual_column_is_still_left_alone():
    df = pd.DataFrame({"cell_id": ["a", "b", "c"], "notes": ["ok", "bad", "ok"]})
    out, _warning = coerce_majority_numeric_cols(df.copy(), {"cell_id"})
    assert not pd.api.types.is_numeric_dtype(out["notes"])


# --------------------------------------------------------------------------- #
# 8. Advice must never send the user in a circle
# --------------------------------------------------------------------------- #

def test_a_ragged_candidate_is_rejected_even_when_it_is_frequent():
    """A tab that appears in the header and never in the data is not structural."""
    # Two tabs in the header and none in the data: frequent, but not structural.
    raw = b"a\tb;c;d\n1;2;3\n4;5;6\n"
    delimiter, viable, _pr, _un = _resolve_delimiter(FakeUpload(raw, "t.csv"))
    assert "\t" not in viable, "a tab absent from the data rows cannot be the separator"
    assert delimiter == ";" and viable == (";",)


def test_an_empty_file_is_named_rather_than_raising_pandas_jargon():
    """An empty text file reports missing content."""
    assert "empty" in _name_content_mismatch(FakeUpload(b"", "e.csv"), "e.csv")


def test_an_empty_workbook_is_not_misreported_as_plain_text():
    """An empty workbook reports missing content before format probing."""
    error = _name_content_mismatch(FakeUpload(b"", "e.xlsx"), "e.xlsx")
    assert "empty" in error
    assert "plain text" not in error


# --------------------------------------------------------------------------- #
# 9. Parse caching
# --------------------------------------------------------------------------- #

def _uploaded_file(raw, name):
    """A real UploadedFile — st.cache_data has a dedicated hasher branch for it."""
    from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec

    return UploadedFile(UploadedFileRec("id", name, "text/csv", raw), None)


def _count_parses(monkeypatch):
    # The cache outlives a single test, so a later test would otherwise "hit" on
    # an entry an earlier one created and count zero parses.
    _read_table_cached.clear()
    calls = {"n": 0}
    for entry_point in ("read_csv", "ExcelFile"):
        original = getattr(pd, entry_point)

        def counted(*args, _original=original, **kwargs):
            calls["n"] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(pd, entry_point, counted)
    return calls


def test_repeated_reruns_of_one_upload_parse_the_file_once(monkeypatch):
    """Streamlit reruns the whole script on every widget click; only the first
    should pay for the parse. Caching is parse-only on purpose — see load_table.
    """
    calls = _count_parses(monkeypatch)
    raw = _frame().to_csv(index=False).encode()
    for _ in range(3):
        upload = _uploaded_file(raw, "d.csv")
        _name_content_mismatch(upload, "d.csv")     # uncached probe, every rerun
        _read_table_cached(upload, suffix_of(upload))
    assert calls["n"] == 1


def test_a_probe_that_forgets_to_rewind_would_cost_a_reparse(monkeypatch):
    """The seek position is part of the cache key, not just the buffer's state."""
    calls = _count_parses(monkeypatch)
    raw = _frame().to_csv(index=False).encode()
    for offset in (0, 0):
        upload = _uploaded_file(raw, "d.csv")
        upload.seek(offset)
        _read_table_cached(upload, suffix_of(upload))
    assert calls["n"] == 1, "identical position should hit"

    moved = _uploaded_file(raw, "d.csv")
    moved.seek(4)                                    # what a non-rewinding probe leaves
    _read_table_cached(moved, suffix_of(moved))
    assert calls["n"] == 2, "a moved position must miss — hence every probe rewinds"


def test_a_spreadsheet_and_a_text_file_of_the_same_name_do_not_collide(monkeypatch):
    """Suffix remains part of the cache key because text and spreadsheet readers differ."""
    calls = _count_parses(monkeypatch)
    raw = _frame().to_csv(index=False).encode()
    _read_table_cached(_uploaded_file(raw, "d.csv"), ".csv")
    _read_table_cached(_uploaded_file(raw, "d.tsv"), ".tsv")
    assert calls["n"] == 2, "different names must not share an entry"


def test_load_table_itself_is_not_cached():
    """It reads analysis-profile config that is not in its arguments, so caching
    it wholesale would serve a stale frame after a profile switch.
    """
    from src.dataset_io import load_table

    assert not hasattr(load_table, "clear")
    assert hasattr(_read_table_cached, "clear")


# --------------------------------------------------------------------------- #
# 10. Zero viable separators: absent is fine, present-but-ragged is not
# --------------------------------------------------------------------------- #

def test_a_ragged_file_is_named_rather_than_read_as_one_column():
    """Reject inconsistent field counts with a row-format error."""
    raw = b"cell_id\ttrt\tv\nA_1\tdrug\t1\nA_2\tctrl\nA_3\tctrl\t3\n"
    df, meta = _read(FakeUpload(raw, "plate.tsv"))
    assert meta["viable_delimiters"] == () and meta["present_delimiters"] == ("\t",)
    _warning, error = _diagnose_table(df, meta, "plate.tsv")
    # The offending line is named by its position in the file, header included.
    assert "with tab, row 3 has 2 fields where row 1 has 3" in error


def test_a_blank_value_is_not_a_ragged_row():
    """An empty field preserves the column count and loads as NaN."""
    raw = b"cell_id\ta\tb\nc1\t1\t2\nc2\t\t4\n"
    df, meta = _read(FakeUpload(raw, "plate.tsv"))
    assert meta["viable_delimiters"] == ("\t",)
    _warning, error = _diagnose_table(df, meta, "plate.tsv")
    assert error == ""
    assert len(df) == 2 and df["a"].isna().sum() == 1


def test_a_ragged_row_is_numbered_by_its_line_in_the_file(): 
    """Blank lines shift nothing: the count is physical lines, as an editor shows."""
    raw = b"cell_id\ta\tb\nc1\t1\t2\n\nc2\t3\n"
    df, meta = _read(FakeUpload(raw, "plate.tsv"))
    _warning, error = _diagnose_table(df, meta, "plate.tsv")
    assert "row 4 has 2 fields" in error


def test_a_comment_line_above_the_header_is_named():
    raw = b"# exported by SPCImage v8\ncell_id\ttrt\tv\nA_1\tdrug\t1\n"
    df, meta = _read(FakeUpload(raw, "plate.tsv"))
    _warning, error = _diagnose_table(df, meta, "plate.tsv")
    assert "title or comment line" in error
    # Physical line numbering includes the comment, so the misplaced header is row 2.
    assert "row 2 has 3 fields where row 1 has 1" in error


def test_a_table_with_no_separator_at_all_is_left_to_the_semantic_layer():
    """One column is not the reader's business — get_features decides."""
    raw = b"cell_id\na\nb\n"
    df, meta = _read(FakeUpload(raw, "plate.csv"))
    assert meta["present_delimiters"] == ()
    _warning, error = _diagnose_table(df, meta, "plate.csv")
    assert error == ""


def test_a_colon_separated_file_is_named_by_the_same_consistency_rule():
    """A consistent colon separator is recognized as unsupported and reported clearly."""
    raw = b"cell_id:treatment:value\na:ctrl:1\nb:drug:2\n"
    df, meta = _read(FakeUpload(raw, "plate.csv"))
    assert meta["unusable_delimiter"] == "colon"
    _warning, error = _diagnose_table(df, meta, "plate.csv")
    assert "colon-separated" in error
    assert "comma, tab, semicolon or pipe" in error


def test_a_lone_colon_in_a_feature_name_is_not_a_separator():
    """"Lifetime fit_ch1: T1" is a real column name; its rows hold no colon, so
    colon does not divide the file consistently and nothing is claimed."""
    raw = b"Lifetime fit_ch1: T1\n0.4\n0.55\n"
    df, meta = _read(FakeUpload(raw, "plate.csv"))
    assert meta["unusable_delimiter"] is None
    _warning, error = _diagnose_table(df, meta, "plate.csv")
    assert error == ""


def test_a_usable_separator_is_never_overridden_by_the_unusable_check():
    """The colon probe only runs when nothing usable divided the file."""
    raw = b"a,b\nx:1,y:2\nx:3,y:4\n"          # colons in every cell, comma divides
    df, meta = _read(FakeUpload(raw, "plate.csv"))
    assert meta["viable_delimiters"] == (",",) and meta["unusable_delimiter"] is None
    _warning, error = _diagnose_table(df, meta, "plate.csv")
    assert error == ""


# ---------------------------------------------------------------------------
# 11. readable errors for encoding and row-format failures
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("offset", range(8))
def test_a_valid_utf8_file_larger_than_the_sample_is_not_called_cp1252(offset):
    """A multibyte character crossing the sample boundary does not invalidate UTF-8."""
    raw = b"a" * (_SAMPLE_BYTES - 4 - offset) + "\U0001F600".encode() + b"b" * 200
    raw.decode("utf-8")  # the fixture really is valid UTF-8
    assert _name_content_mismatch(FakeUpload(raw, "big.csv"), "big.csv") == ""


def test_a_genuinely_mis_encoded_file_is_still_caught():
    """The boundary forgiveness must not swallow a real cp1252 file."""
    raw = "cell_id,note\nc1,caf\xe9\n".encode("cp1252")
    error = _name_content_mismatch(FakeUpload(raw, "t.csv"), "t.csv")
    assert "UTF-8" in error


@pytest.mark.parametrize("raw,label", [
    (b"\n\n\n", "blank lines"),
    (b"   \n", "whitespace"),
    (b"\xef\xbb\xbf", "a lone BOM"),
])
def test_a_file_with_no_columns_is_named_rather_than_raising(raw, label):
    """Whitespace and BOM-only files report missing content instead of raising
    EmptyDataError.
    """
    assert _name_content_mismatch(FakeUpload(raw, "t.csv"), "t.csv") == ""
    df, meta = _read(FakeUpload(raw, "t.csv"))
    assert meta["parse_error"] == "empty"
    _warning, error = _diagnose_table(df, meta, "t.csv")
    assert "no table could be read" in error and "t.csv" in error
    assert "No columns to parse" not in error


def test_an_unclosed_quote_is_named_rather_than_raising():
    """pandas raises ParserError: "EOF inside string starting at row 1"."""
    raw = b'cell_id,a\n"x,1\nc2,2\n'
    df, meta = _read(FakeUpload(raw, "t.csv"))
    assert meta["parse_error"] == "unbalanced_quote"
    _warning, error = _diagnose_table(df, meta, "t.csv")
    assert "quotation mark" in error and "EOF inside string" not in error


def test_a_row_with_too_many_fields_is_not_blamed_on_quotes():
    """pandas raises the same ParserError for both; quote parity separates them."""
    raw = b"cell_id,a,b\nc1,1,2\nc2,1,2,9\n"
    df, meta = _read(FakeUpload(raw, "t.csv"))
    assert meta["parse_error"] == "unreadable"
    _warning, error = _diagnose_table(df, meta, "t.csv")
    assert "row 3 has 4 fields where row 1 has 3" in error
    assert "quotation mark" not in error


def test_a_short_row_past_the_detection_sample_is_still_rejected():
    """Validate field counts across the whole file after the sample selects a delimiter.
    """
    body = b"cell_id,a,b\n" + b"".join(b"c%d,1.5,2.5\n" % i for i in range(6000))
    assert len(body) > _SAMPLE_BYTES
    df, meta = _read(FakeUpload(body + b"c9999,1.5\n", "big.csv"))
    _warning, error = _diagnose_table(df, meta, "big.csv")
    assert "row 6002 has 2 fields where row 1 has 3" in error


def test_a_clean_file_larger_than_the_sample_still_loads():
    body = b"cell_id,a,b\n" + b"".join(b"c%d,1.5,2.5\n" % i for i in range(6000))
    df, meta = _read(FakeUpload(body, "big.csv"))
    _warning, error = _diagnose_table(df, meta, "big.csv")
    assert error == "" and len(df) == 6000


@pytest.mark.parametrize("name", ["a<b", "<LOD>", "<i>note</i>", "T1 <2ns", "R&D"])
def test_a_column_name_cannot_truncate_its_own_warning(name):
    """Column names remain literal when a plain-text warning is rendered as HTML."""
    df = pd.DataFrame({"cell_id": ["c1", "c2"], name: [None, None], "a": [1.5, 1.6]})
    _fixed, warning, _error = check_and_fix_df(df, [], "cell_id", "image_name")
    assert "<" not in warning or name in warning       # built as plain text
    rendered = _as_html(warning)
    assert html.escape(name) in rendered              # survives escaping
    assert "<br>" in rendered or "\n" not in warning.strip()


def test_an_ordinary_table_records_no_parse_error():
    _df, meta = _read(_upload(_frame(), ".csv"))
    assert meta["parse_error"] is None


# ---------------------------------------------------------------------------
# 12. layer-2 conventions
# ---------------------------------------------------------------------------

def test_pandas_de_duplicates_headers_before_anything_can_warn_about_them():
    """Both readers rename repeated source headers before downstream validation."""
    csv_df, _meta = _read(FakeUpload(b"cell_id,T1,T1\nc1,1,9\n", "t.csv"))
    assert list(csv_df.columns) == ["cell_id", "T1", "T1.1"]

    frame = pd.DataFrame([["c1", 1, 9]], columns=["cell_id", "T1", "T1"])
    book, _meta = _read(_upload(frame, ".xlsx"))
    assert not book.columns.duplicated().any()


def test_get_features_returns_an_empty_warning_not_none_on_its_error_path():
    """The error path returns a string warning that callers can concatenate."""
    df = pd.DataFrame({"cell_id": ["c1"], "image_name": ["f"], "label": ["alpha"]})
    _df, _groups, warning, error = get_features(df, [], use_data_extraction=True)
    assert error != "" and warning == ""


def test_the_decimal_hint_survives_a_missing_row_id_column(monkeypatch):
    """A missing identifier must not suppress the comma-decimal advice."""
    rendered = []
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    monkeypatch.setattr(dataset_io, "get_fov_name_col_analysis", lambda *a, **k: "image_name")
    monkeypatch.setattr(dataset_io.st, "markdown", lambda msg, **k: rendered.append(msg))
    monkeypatch.setattr(dataset_io.st, "write", lambda msg, **k: rendered.append(msg))

    # A real UploadedFile: st.cache_data's hasher rejects a BytesIO carrying a
    # .name, and load_table goes through the cache rather than around it.
    upload = _uploaded_file(b"a;b\n1,5;2,3\n2,5;3,3\n", "euro.csv")
    df, groups, complete, delimiter, _row_id = dataset_io.load_table(upload, [])

    assert (df, groups, complete, delimiter) == (None, None, False, ";")
    shown = " ".join(rendered)
    assert "cell_id column is missing" in shown          # Name the missing identifier.
    assert "1,5" in shown and "full stop" in shown       # Also explain the decimal format.


def test_load_table_hands_back_the_separator_it_actually_read(monkeypatch):
    """The exported script bakes in this answer rather than re-detecting one."""
    monkeypatch.setattr(dataset_io, "get_unique_row_id_col", lambda *a, **k: "cell_id")
    monkeypatch.setattr(dataset_io, "get_fov_name_col_analysis", lambda *a, **k: "image_name")
    monkeypatch.setattr(dataset_io.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(dataset_io.st, "write", lambda *a, **k: None)

    frame = _frame()
    for suffix, expected in ((".tsv", "\t"), (".csv", ","), (".xlsx", ",")):
        raw = _upload(frame, suffix).getvalue()
        _df, _g, complete, delimiter, _row_id = dataset_io.load_table(
            _uploaded_file(raw, f"table{suffix}"), ["treatment"])
        assert complete is True, suffix
        assert delimiter == expected, suffix


# ---------------------------------------------------------------------------
# 13. the page that calls the reader
# ---------------------------------------------------------------------------

def test_the_analysis_page_still_runs_and_offers_every_supported_suffix():
    """The rendered analysis uploader offers every supported suffix."""
    from streamlit.testing.v1 import AppTest

    page = Path(__file__).resolve().parents[1] / "pages" / "data_analysis.py"
    at = AppTest.from_file(str(page), default_timeout=90)
    at.run()

    assert not at.exception, [e.value for e in at.exception]
    uploaders = at.get("file_uploader")
    assert uploaders, "the analysis page renders no file uploader"
    assert list(uploaders[0].proto.type) == list(SUPPORTED_SUFFIXES)
