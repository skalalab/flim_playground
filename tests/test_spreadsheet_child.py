"""A workbook is parsed in a child process that is stopped at a memory or time cap.

These tests start real child processes, so they stay in this file.
"""
import io
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import psutil
import pytest

from src import spreadsheet_child
from tests.abnormal_workbooks import (
    far_cell_xlsx,
    long_shared_text_xlsx,
    mixed_workbook,
    zip_without_a_workbook,
)

ROOT = Path(__file__).resolve().parents[1]


def test_an_ordinary_workbook_comes_back_as_the_in_process_frame():
    data = mixed_workbook()
    status, (frame, names) = spreadsheet_child.read_workbook_capped(
        data, memory_limit=2_000_000_000, time_limit=120)
    expected, expected_names = spreadsheet_child.parse_workbook(io.BytesIO(data))
    assert status == "ok" and names == expected_names
    assert list(frame.columns) == ["id", "7", "x", "when", "note"]
    pd.testing.assert_frame_equal(frame, expected)


def test_a_far_away_cell_is_stopped_at_the_memory_cap():
    before, start = psutil.Process().memory_info().rss, time.perf_counter()
    outcome = spreadsheet_child.read_workbook_capped(
        far_cell_xlsx("CV", 1_048_576), memory_limit=300_000_000, time_limit=120)
    assert outcome == ("memory", None) and time.perf_counter() - start < 10
    assert psutil.Process().memory_info().rss - before < 200_000_000


def test_one_long_shared_text_is_stopped_at_the_memory_cap():
    assert spreadsheet_child.read_workbook_capped(
        long_shared_text_xlsx(3_000, 100_000), memory_limit=300_000_000, time_limit=120)[0] == "memory"


def test_a_slow_read_is_stopped_at_the_time_cap(monkeypatch):
    monkeypatch.setattr(spreadsheet_child, "child_command",
                        lambda: [sys.executable, "-c", "import time; time.sleep(30)"])
    start = time.perf_counter()
    assert spreadsheet_child.read_workbook_capped(b"x", memory_limit=10**9, time_limit=0.5)[0] == "time"
    assert time.perf_counter() - start < 5


@pytest.mark.parametrize("command", [
    ["/nonexistent/python"],                                              # cannot start
    [sys.executable, "-c", "raise SystemExit(3)"],                        # dies without a result
    [sys.executable, "-c", "import sys; open(sys.argv[2], 'wb').write(b'not a pickle')"],
    [sys.executable, "-c", "import sys; open(sys.argv[2], 'wb').write(b'\\x80\\x04')"],   # cut off mid-write
])
def test_a_child_without_a_usable_result_is_a_failure(monkeypatch, command):
    monkeypatch.setattr(spreadsheet_child, "child_command", lambda: command)
    assert spreadsheet_child.read_workbook_capped(b"x", memory_limit=10**9, time_limit=30) == ("failed", None)


def test_a_file_the_reader_rejects_comes_back_as_its_message():
    status, message = spreadsheet_child.read_workbook_capped(
        zip_without_a_workbook(), memory_limit=10**9, time_limit=60)
    assert status == "error" and isinstance(message, str) and message


def test_no_copy_of_the_upload_stays_on_disk(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    assert spreadsheet_child.read_workbook_capped(
        far_cell_xlsx("CV", 1_048_576), memory_limit=300_000_000, time_limit=120)[0] == "memory"
    assert spreadsheet_child.read_workbook_capped(
        mixed_workbook(50), memory_limit=2_000_000_000, time_limit=120)[0] == "ok"
    assert list(tmp_path.iterdir()) == []


def test_linux_children_get_a_hard_address_space_limit(monkeypatch):
    resource = pytest.importorskip("resource")
    from types import SimpleNamespace

    limits = []
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(spreadsheet_child.psutil, "Process",
                        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(vms=1_000_000_000)))
    monkeypatch.setattr(resource, "setrlimit", lambda kind, limit: limits.append((kind, limit)))
    spreadsheet_child._cap_address_space(300_000_000)
    assert limits == [(resource.RLIMIT_AS, (1_300_000_000, 1_300_000_000))]


def test_other_platforms_rely_on_the_watchdog(monkeypatch):
    resource = pytest.importorskip("resource")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(resource, "setrlimit", lambda *a: pytest.fail("a limit was set"))
    spreadsheet_child._cap_address_space(300_000_000)


def test_a_child_that_ran_out_of_memory_says_so(monkeypatch):
    monkeypatch.setattr(spreadsheet_child, "child_command", lambda: [
        sys.executable, "-c", "import pickle, sys; pickle.dump(('memory',), open(sys.argv[2], 'wb'))"])
    assert spreadsheet_child.read_workbook_capped(b"x", memory_limit=10**9, time_limit=30) == ("memory", None)


def test_the_child_is_started_without_the_servers_standard_input(monkeypatch):
    import subprocess

    seen = {}
    real = subprocess.Popen

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(spreadsheet_child.subprocess, "Popen", spy)
    spreadsheet_child.read_workbook_capped(mixed_workbook(20), memory_limit=2_000_000_000, time_limit=120)
    assert seen["stdin"] is subprocess.DEVNULL


@pytest.mark.skipif(sys.platform == "win32", reason="Windows processes are not ended by signals")
def test_a_child_killed_by_a_signal_under_the_cap_ran_out_of_memory(monkeypatch):
    """Under a hard limit the reader aborts when an allocation is refused, before it can report."""
    monkeypatch.setattr(spreadsheet_child, "child_command", lambda: [sys.executable, "-c", "import os; os.abort()"])
    assert spreadsheet_child.read_workbook_capped(b"x", memory_limit=10**9, time_limit=30) == ("memory", None)


def test_the_self_test_passes_when_a_table_opens_and_an_abnormal_one_is_refused(tmp_path):
    good, abnormal, report = tmp_path / "good.xlsx", tmp_path / "abnormal.xlsx", tmp_path / "report.txt"
    good.write_bytes(mixed_workbook(50))
    abnormal.write_bytes(far_cell_xlsx("CV", 1_048_576))
    assert spreadsheet_child.self_test(str(good), str(abnormal), str(report)) == 0
    lines = report.read_text().splitlines()
    assert lines[0] == "ordinary workbook under a 300 MB cap: ok, 50 rows x 5 columns"
    assert lines[1] == "abnormal workbook under a 300 MB cap: memory"
    assert lines[-1] == "PASS"


def test_the_self_test_fails_when_the_ordinary_workbook_does_not_open(tmp_path):
    good, abnormal, report = tmp_path / "good.xlsx", tmp_path / "abnormal.xlsx", tmp_path / "report.txt"
    good.write_bytes(zip_without_a_workbook())
    abnormal.write_bytes(far_cell_xlsx("CV", 1_048_576))
    assert spreadsheet_child.self_test(str(good), str(abnormal), str(report)) == 1
    assert report.read_text().splitlines()[-1] == "FAIL"


def test_the_self_test_reports_an_error_instead_of_raising(tmp_path):
    """A windowed app that raises shows a dialog nobody can dismiss on a build machine."""
    report = tmp_path / "report.txt"
    assert spreadsheet_child.self_test(str(tmp_path / "missing.xlsx"), str(tmp_path / "missing.xlsx"), str(report)) == 1
    lines = report.read_text().splitlines()
    assert lines[0].startswith("error: FileNotFoundError") and lines[-1] == "FAIL"


def test_the_launcher_answers_the_self_test_flag():
    assert '"--self-test"' in (ROOT / "launcher.py").read_text()


def test_the_bundle_starts_the_child_through_the_launcher(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert spreadsheet_child.child_command() == [sys.executable, "--read-workbook"]
    assert '"--read-workbook"' in (ROOT / "launcher.py").read_text()
