"""An upload is refused before it can cost more memory than this machine can give.

Delimited text is counted (rows x columns) before the parser runs. Spreadsheets are
parsed in a capped child process, never in the server. A refusal reaches the page as
read_table's error text.
"""
import io
import threading
import time
from pathlib import Path

import pandas as pd
import psutil
import pytest
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec

from src import dataset_io, spreadsheet_child, table_budget
from tests.hostile_workbooks import (
    far_cell_xlsx,
    formatting_only_far_cell_xlsx,
    libreoffice_filler_ods,
    long_shared_text_xlsx,
    mixed_workbook,
    repeat_ods,
    zip_without_a_workbook,
)

ROOT = Path(__file__).resolve().parents[1]


def _upload(payload: bytes, name="t.csv"):
    return UploadedFile(UploadedFileRec("id", name, "text/csv", payload), None)


GRID_CASES = {
    "lf": b"a,b\n1,2\n3,4\n",
    "crlf": b"a,b\r\n1,2\r\n3,4\r\n",
    "cr_only": b"a,b\r1,2\r3,4\r",
    "no_final_newline": b"a,b\n1,2\n3,4",
    "bom": b"\xef\xbb\xbfa,b\n1,2\n",
    "blank_lines": b"a,b\n\n1,2\n\n3,4\n",
    "quoted_delimiter_in_header": b'"a,x",b\n1,2\n',
    "quoted_newline_in_cell": b'a,b\n"1\n1",2\n3,4\n',
    "header_only": b"a,b\n",
    "trailing_delimiter": b"a,b,\n1,2,\n",
}


@pytest.mark.parametrize("name", GRID_CASES)
def test_the_counted_grid_is_never_smaller_than_the_parsed_frame(name):
    payload = GRID_CASES[name]
    frame = pd.read_csv(io.BytesIO(payload), index_col=False, sep=",", low_memory=False)
    rows, columns = dataset_io.text_grid(_upload(payload), ",")
    assert len(frame) <= rows <= len(frame) + 4 and columns >= frame.shape[1]


def test_a_header_wider_than_the_sample_is_counted_in_full():
    payload = (",".join(f"c{i}" for i in range(20_000)) + "\n").encode()
    assert len(payload) > dataset_io._SAMPLE_BYTES
    assert dataset_io.text_grid(_upload(payload), ",")[1] == 20_000


def test_a_csv_over_the_budget_is_refused_before_parsing(monkeypatch):
    monkeypatch.setattr(table_budget, "memory_budget_bytes", lambda hosted: 5_000)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("the parser ran"))
    df, *_rest, error = dataset_io.read_table(_upload(b"a,b,c\n" + b"1,2,3\n" * 30))
    assert df is None and "is too large to open on this computer" in error and "× 3 columns" in error


def test_the_same_csv_opens_when_memory_allows():
    df, *_rest, error = dataset_io.read_table(_upload(b"a,b,c\n" + b"1,2,3\n" * 30))
    assert error == "" and df.shape == (30, 3)


def test_short_rows_are_judged_by_what_pandas_would_allocate(monkeypatch):
    # 10,000 header columns and 1,000 one-field rows pad to 10M cells.
    payload = (",".join(f"c{i}" for i in range(10_000)) + "\n" + "1\n" * 1_000).encode()
    monkeypatch.setattr(table_budget, "memory_budget_bytes", lambda hosted: 100_000_000)
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("the parser ran"))
    assert "is too large" in dataset_io.read_table(_upload(payload))[4]


# Shapes where pandas would build far more cells than the first record suggests. Each is
# under 250 KB and cost the server 0.5 to 1 GB before the parser was gated on a
# rectangular file.
UNDERCOUNTED = {
    "first_data_row_wider_than_the_header": b"a,b\n" + b"1," * 1999 + b"1\n" + b"1,2\n" * 20_000,
    "header_cell_past_the_csv_field_limit": (b"A" * 140_000 + b"," + ",".join(f"c{i}" for i in range(1000)).encode()
                                             + b"\n" + b"1\n" * 30_000),
    "whitespace_only_first_line": b" \n" + ",".join(f"c{i}" for i in range(1000)).encode() + b"\n" + b"1\n" * 30_000,
    "no_separator_in_the_sample": b"a\n" + b"A" * 70_000 + b"," * 999 + b"\n" + b"1\n" * 30_000,
}


@pytest.mark.parametrize("name", UNDERCOUNTED)
def test_pandas_never_parses_a_file_it_would_pad_past_the_count(monkeypatch, name):
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: pytest.fail("the parser ran"))
    df, *_rest, error = dataset_io.read_table(_upload(UNDERCOUNTED[name]))
    assert df is None and error.startswith("Error:")


def test_an_unclosed_quote_is_still_named_as_such():
    df, *_rest, error = dataset_io.read_table(_upload(b'a,b\n1,"2\n3,4\n5,6\n'))
    assert df is None and "unclosed quotation mark" in error


def test_two_uploads_are_admitted_one_after_the_other(monkeypatch):
    """The second upload's budget is read only after the first parse has finished."""
    events, real = [], pd.read_csv

    def slow_parse(*a, **k):
        events.append("parse starts")
        time.sleep(0.3)
        frame = real(*a, **k)
        events.append("parse ends")
        return frame

    monkeypatch.setattr(pd, "read_csv", slow_parse)
    monkeypatch.setattr(table_budget, "memory_budget_bytes", lambda hosted: (events.append("budget"), 10**12)[1])
    threads = [threading.Thread(target=dataset_io.read_table,
                                args=(_upload(f"a,b\n{i},2\n".encode(), f"concurrent{i}.csv"),)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert events == ["budget", "parse starts", "parse ends"] * 2


def test_the_online_demo_refuses_more_than_five_million_cells(monkeypatch):
    monkeypatch.setattr(dataset_io, "_is_hosted", lambda: True)
    monkeypatch.setattr(dataset_io, "text_grid", lambda upload, delimiter: (2_500_001, 2))
    assert "is too large for the online demo" in dataset_io.read_table(_upload(b"a,b\n1,2\n"))[4]


def test_the_online_demo_says_busy_when_its_memory_is_used(monkeypatch):
    monkeypatch.setattr(dataset_io, "_is_hosted", lambda: True)
    monkeypatch.setattr(table_budget, "memory_budget_bytes", lambda hosted: 0)
    assert "holding other visitors' tables" in dataset_io.read_table(_upload(b"a,b\n1,2\n"))[4]


@pytest.mark.parametrize("path", sorted((ROOT / "example_data" / "Data_Analysis").glob("*.csv")),
                         ids=lambda path: path.name)
def test_the_example_tables_still_open(path):
    assert dataset_io.read_table(_upload(path.read_bytes(), path.name))[4] == ""


HOSTILE = {
    "far_valued_cell.xlsx": lambda: far_cell_xlsx("CV", 1_048_576),
    "ods_repeat_counts.ods": lambda: repeat_ods(rows=1_000_000, columns=50),
    "long_shared_text.xlsx": lambda: long_shared_text_xlsx(3_000, 100_000),
    # The reader goes by content: xlsx parts under an .ods name are read as xlsx.
    "xlsx_parts_named.ods": lambda: far_cell_xlsx("CV", 1_048_576),
}


@pytest.mark.capped_child
@pytest.mark.parametrize("name", HOSTILE)
def test_a_hostile_workbook_is_refused_without_the_server_parsing_it(monkeypatch, name):
    monkeypatch.setattr(table_budget, "memory_budget_bytes", lambda hosted: 300_000_000)
    monkeypatch.setattr(spreadsheet_child, "parse_workbook", lambda source: pytest.fail("parsed in the server"))
    before = psutil.Process().memory_info().rss
    df, *_rest, error = dataset_io.read_table(_upload(HOSTILE[name](), name))
    assert df is None and "needs more memory than can be used here (300 MB)" in error
    assert psutil.Process().memory_info().rss - before < 200_000_000


@pytest.mark.capped_child
@pytest.mark.parametrize("build, name, shape", [(formatting_only_far_cell_xlsx, "t.xlsx", (1, 2)),
                                                (libreoffice_filler_ods, "t.ods", (3, 2))])
def test_small_tables_that_only_look_large_still_open(build, name, shape):
    df, *_rest, error = dataset_io.read_table(_upload(build(), name))
    assert error == "" and df.shape == shape


@pytest.mark.capped_child
def test_a_refusal_is_not_remembered(monkeypatch):
    upload = _upload(long_shared_text_xlsx(3_000, 100_000), "t.xlsx")
    monkeypatch.setattr(table_budget, "memory_budget_bytes", lambda hosted: 300_000_000)
    assert dataset_io.read_table(upload)[4] != ""
    monkeypatch.setattr(table_budget, "memory_budget_bytes", lambda hosted: 4_000_000_000)
    assert dataset_io.read_table(upload)[4] == ""


@pytest.mark.capped_child
def test_the_online_demo_applies_its_cell_ceiling_to_workbooks(monkeypatch):
    monkeypatch.setattr(dataset_io, "_is_hosted", lambda: True)
    monkeypatch.setattr(table_budget, "memory_budget_bytes", lambda hosted: 1_000_000_000)
    monkeypatch.setattr(table_budget, "HOSTED_MAX_CELLS", 4)
    assert "is too large for the online demo" in dataset_io.read_table(_upload(mixed_workbook(3), "t.xlsx"))[4]


def test_a_spent_budget_refuses_before_starting_a_child(monkeypatch):
    monkeypatch.setattr(dataset_io, "_read_workbook", dataset_io._REAL_READ_WORKBOOK)
    monkeypatch.setattr(dataset_io, "_is_hosted", lambda: True)
    monkeypatch.setattr(table_budget, "memory_budget_bytes", lambda hosted: 50_000_000)
    monkeypatch.setattr(spreadsheet_child, "read_workbook_capped", lambda *a, **k: pytest.fail("a child was started"))
    assert "holding other visitors' tables" in dataset_io.read_table(_upload(mixed_workbook(3), "t.xlsx"))[4]


@pytest.mark.capped_child
def test_a_workbook_the_reader_rejects_still_raises():
    with pytest.raises(ValueError):
        dataset_io.read_table(_upload(zip_without_a_workbook(), "broken.xlsx"))


def test_unmarked_tests_read_workbooks_in_process():
    assert dataset_io._read_workbook is dataset_io._read_workbook_in_process


def test_the_read_cache_keeps_three_tables(monkeypatch):
    dataset_io._read_table_cached.clear()
    parses, real = [], pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: (parses.append(1), real(*a, **k))[1])
    uploads = [_upload(f"a,b\n{i},2\n".encode(), f"t{i}.csv") for i in range(4)]
    for upload in uploads:
        dataset_io.read_table(upload)
    dataset_io.read_table(uploads[3])
    assert len(parses) == 4          # still held
    dataset_io.read_table(uploads[0])
    assert len(parses) == 5          # the oldest was evicted
