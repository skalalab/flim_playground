"""The memory budget decides which uploads may be opened, and words every refusal.

Desktop follows free memory (half of it, never below a floor that keeps small tables
opening). The online demo follows a fixed figure minus what the server already holds,
so its budget shrinks as other visitors' tables fill it.
"""
from types import SimpleNamespace

from src import table_budget


def test_desktop_budget_is_half_of_free_memory(monkeypatch):
    monkeypatch.setattr(table_budget.psutil, "virtual_memory", lambda: SimpleNamespace(available=8_000_000_000))
    assert table_budget.memory_budget_bytes(hosted=False) == 4_000_000_000


def test_a_small_table_opens_when_the_os_reports_nothing_free(monkeypatch):
    monkeypatch.setattr(table_budget.psutil, "virtual_memory", lambda: SimpleNamespace(available=0))
    budget = table_budget.memory_budget_bytes(hosted=False)
    assert budget == table_budget.DESKTOP_MIN_BUDGET_BYTES
    assert table_budget.text_refusal("iris.csv", 4_000, 151, 5, hosted=False, budget=budget) == ""


def test_hosted_budget_shrinks_as_the_server_fills(monkeypatch):
    rss = iter([400_000_000, 2_000_000_000, 3_000_000_000])
    monkeypatch.setattr(table_budget.psutil, "Process",
                        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=next(rss))))
    assert [table_budget.memory_budget_bytes(hosted=True) for _ in range(3)] == [1_000_000_000, 200_000_000, 0]


def test_the_estimate_tracks_the_three_measured_tables():
    """Measured peaks before any plot: 1M x 13 mixed 1.05 GB, 1M x 57 floats 2.6 GB, 30M x 2 6.6 GB."""
    def need(rows, columns):
        return rows * columns * table_budget.TEXT_BYTES_PER_CELL + rows * table_budget.TEXT_BYTES_PER_ROW
    assert 0.6e9 < need(1_000_000, 13) < 1.3e9
    assert 2.2e9 < need(1_000_000, 57) < 3.2e9
    assert 5.9e9 < need(30_000_000, 2) < 7.9e9


def test_desktop_refusal_states_the_numbers():
    text = table_budget.text_refusal("big.csv", 177_000_000, 30_000_000, 2, hosted=False, budget=1_000_000_000)
    assert text == ("Error: 'big.csv' is too large to open on this computer. It has about 30,000,000 rows × "
                    "2 columns (60,000,000 cells), which needs about 6.9 GB of memory; 1.0 GB can be used "
                    "right now. Filter or split the table and upload it again.")


def test_hosted_refusals():
    assert table_budget.text_refusal("t.csv", 10, 2_500_001, 2, hosted=True, budget=10**12) == (
        "Error: 't.csv' is too large for the online demo. It has about 2,500,001 rows × 2 columns "
        "(5,000,002 cells), and the demo opens tables of up to 5,000,000 cells. Filter or split the table, "
        "or open it in the desktop app.")
    assert table_budget.text_refusal("t.csv", 1_000_000, 100, 2, hosted=True, budget=0) == (
        "Error: the online demo is holding other visitors' tables and cannot open 't.csv' right now. "
        "Try again in a few minutes, or open it in the desktop app.")
    assert table_budget.text_refusal("t.csv", 1_000, 100, 2, hosted=True, budget=10**9) == ""


def test_workbook_refusals():
    memory = table_budget.workbook_refusal("w.xlsx", "memory", hosted=False, budget=300_000_000, seconds=600)
    assert memory == ("Error: the first sheet of 'w.xlsx' needs more memory than can be used here (300 MB), "
                      "so it was not opened. If the table itself is small, the sheet holds content far outside "
                      "it or very long repeated text: copy the table into a new workbook, or save it as CSV.")
    assert table_budget.workbook_refusal("w.xlsx", "memory", hosted=True, budget=300_000_000, seconds=60) == (
        memory + " Larger tables open in the desktop app.")
    assert table_budget.workbook_refusal("w.xlsx", "time", hosted=False, budget=1, seconds=600) == (
        "Error: 'w.xlsx' was still being read after 600 seconds, so reading was stopped. "
        "Save the table as CSV and upload that.")
    assert table_budget.workbook_refusal("w.xlsx", "failed", hosted=False, budget=1, seconds=600) == (
        "Error: 'w.xlsx' could not be opened safely, so it was not read. Save the table as CSV and upload that.")
