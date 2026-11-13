"""The uploaded table is read and interpreted once per upload and decision, not on every rerun."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import dataset_io, table_session
from src.analysis_config import working_copy_arguments

ARGS = {"categorical_cols": ["g"], "unique_row_id_col": "id", "ignored_cols": [], "feature_groups": {}}
UPLOAD = SimpleNamespace(name="t.csv", size=10)
_PAGE = str(ROOT / "pages" / "data_analysis.py")


@pytest.fixture
def reads(monkeypatch):
    calls = []
    frame = pd.DataFrame({"id": ["a", "b"], "g": ["x", "y"], "v": [1.0, 2.0]})
    monkeypatch.setattr(dataset_io, "read_table",
                        lambda upload: (calls.append(upload), (frame.copy(), {}, ",", "", ""))[1])
    monkeypatch.setattr(dataset_io, "render_interpretation", lambda *a: None)
    return calls


def test_a_rerun_reads_nothing(reads):
    state = {}
    first, second = table_session.current_read(state, UPLOAD, 1), table_session.current_read(state, UPLOAD, 1)
    assert len(reads) == 1 and second[0] is first[0]


def test_a_replaced_upload_is_read_again(reads):
    state = {}
    table_session.current_read(state, UPLOAD, 1)
    table_session.current_read(state, UPLOAD, 2)                                 # same name and size
    table_session.current_read(state, SimpleNamespace(name="t.csv", size=10, file_id="A"), 2)
    table_session.current_read(state, SimpleNamespace(name="t.csv", size=10, file_id="B"), 2)
    assert len(reads) == 4


def test_a_real_upload_is_identified_by_its_file_id_alone(reads):
    state, upload = {}, SimpleNamespace(name="t.csv", size=10, file_id="A")
    table_session.current_read(state, upload, 1)
    table_session.current_read(state, upload, 2)
    assert len(reads) == 1


def test_a_rerun_interprets_nothing_and_a_changed_decision_does(reads, monkeypatch):
    calls, real = [], dataset_io.interpret_frame
    monkeypatch.setattr(dataset_io, "interpret_frame", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    state = {}
    first = table_session.current_interpretation(state, UPLOAD, 1, ARGS, "")
    again = table_session.current_interpretation(state, UPLOAD, 1, ARGS, "")
    assert len(calls) == 1 and again[0] is first[0] and first[2] is True
    table_session.current_interpretation(state, UPLOAD, 1, {**ARGS, "ignored_cols": ["v"]}, "")
    assert len(calls) == 2


def test_messages_are_drawn_on_every_run(reads, monkeypatch):
    drawn = []
    monkeypatch.setattr(dataset_io, "render_interpretation", lambda *a: drawn.append(a))
    state = {}
    for _ in range(2):
        table_session.current_interpretation(state, UPLOAD, 1, ARGS, "Warning: w\n")
    assert len(drawn) == 2 and drawn[0] == drawn[1] and "Warning: w" in drawn[0][1]


def test_a_failed_interpretation_is_replayed_but_a_corrected_decision_interprets_again(reads):
    state = {}
    bad = {**ARGS, "unique_row_id_col": "missing"}
    assert table_session.current_interpretation(state, UPLOAD, 1, bad, "")[2] is False
    assert table_session.current_interpretation(state, UPLOAD, 1, ARGS, "")[2] is True


def test_an_interpretation_that_raises_is_not_remembered(reads, monkeypatch):
    attempts, real = [], dataset_io.interpret_frame

    def flaky(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("boom")
        return real(*a, **k)

    monkeypatch.setattr(dataset_io, "interpret_frame", flaky)
    state = {}
    with pytest.raises(RuntimeError):
        table_session.current_interpretation(state, UPLOAD, 1, ARGS, "")
    assert table_session.current_interpretation(state, UPLOAD, 1, ARGS, "")[2] is True and len(attempts) == 2


def test_clearing_forgets_the_upload(reads):
    state = {}
    table_session.current_read(state, UPLOAD, 1)
    table_session.clear_table_session(state)
    table_session.current_read(state, UPLOAD, 1)
    assert len(reads) == 2


def test_interpret_table_draws_what_it_always_drew(monkeypatch):
    drawn = []
    monkeypatch.setattr(dataset_io, "_render_warning", lambda message: drawn.append(("warning", message)))
    monkeypatch.setattr(dataset_io.st, "write", lambda text: drawn.append(("write", text)))
    frame = pd.DataFrame({"id": ["a", "b"], "g": ["x", "y"], "v": [1.0, None]})
    out = dataset_io.interpret_table(frame, ["g"], "id", None, use_data_extraction=False)
    assert out[2] is True and drawn[0][0] == "warning" and "NaN" in drawn[0][1] and "successfully" in drawn[1][1]


# ---------------------------------------------------------------- the page

def _frame():
    rows = []
    for i, (dish, treatment) in enumerate([("D1", "DMSO"), ("D2", "PD-L1"), ("D3", "DMSO"), ("D4", "PD-L1")]):
        for j in range(6):
            rows.append({"cell_id": f"{dish}_c{j}", "dish": dish, "treatment": treatment,
                         "nadh_tm_mean": 1200.0 + 10 * i + j + 0.37})
    return pd.DataFrame(rows)


@pytest.fixture
def page(monkeypatch, isolated_config_paths):
    """The real page and gate over a fake uploader; counts reads and interpretations."""
    counts = {"read": 0, "interpret": 0}
    fired = [False]

    def uploader(*args, **kwargs):
        if not fired[0]:
            fired[0] = True
            if kwargs.get("on_change"):
                kwargs["on_change"]()
        return SimpleNamespace(name="pdl1.csv")

    def read(_upload):
        counts["read"] += 1
        return _frame(), {}, ",", "", ""

    real = dataset_io.interpret_frame

    def interpret(*a, **k):
        counts["interpret"] += 1
        return real(*a, **k)

    monkeypatch.setattr(st, "file_uploader", uploader)
    monkeypatch.setattr(dataset_io, "read_table", read)
    monkeypatch.setattr(dataset_io, "interpret_frame", interpret)
    return counts


def _shown():
    """Use the guess, then pick the feature so a plot renders."""
    from streamlit.testing.v1 import AppTest

    from tests.test_review_page import _fresh_render

    at = AppTest.from_file(_PAGE)
    at.run(timeout=90)
    assert not at.exception, at.exception
    at = _fresh_render(next(b for b in at.button if b.key == "review_use").click().run(timeout=90))
    box = next(b for b in at.selectbox if "nadh_tm_mean" in b.options)
    box.set_value("nadh_tm_mean").run(timeout=90)
    assert not at.exception, at.exception
    return at


def test_the_page_interprets_once_across_reruns(page):
    at = _shown()
    at.run(timeout=90)
    at.run(timeout=90)
    assert not at.exception and len(at.get("plotly_chart")) == 1
    assert page["interpret"] == 1 and page["read"] <= 2


def test_unfiltered_filters_return_the_frame_itself():
    from src.widgets import filter_widgets

    frame = pd.DataFrame({"id": ["a", "b"], "g": ["x", "y"], "v": [1.0, 2.0]})
    assert filter_widgets.filters_widget(frame, ["g"]) is frame


def test_the_review_editor_checks_the_table_once_until_a_role_changes(page, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.widgets import review_table_widget as rtw

    checks, previews = [], []
    real_check, real_preview = rtw.review_blocking_reason, rtw.column_preview
    monkeypatch.setattr(rtw, "review_blocking_reason", lambda *a, **k: (checks.append(1), real_check(*a, **k))[1])
    monkeypatch.setattr(rtw, "column_preview", lambda *a, **k: (previews.append(1), real_preview(*a, **k))[1])
    at = AppTest.from_file(_PAGE)
    at.run(timeout=90)                     # the editor opens on the guess
    columns = len(_frame().columns)
    assert not at.exception and len(checks) == 1 and len(previews) == columns
    at.run(timeout=90)
    at.run(timeout=90)
    assert len(checks) == 1 and len(previews) == columns
    next(s for s in at.selectbox if s.label == "Role of treatment").set_value("Ignore").run(timeout=90)
    assert not at.exception and len(checks) == 2 and len(previews) == columns


def test_the_stored_table_is_not_modified_by_the_page(page):
    at = _shown()
    at.run(timeout=90)
    args = working_copy_arguments(at.session_state._review_roles, at.session_state._review_groups,
                                  at.session_state._review_group_names)
    fresh = dataset_io.interpret_frame(
        _frame(), args["categorical_cols"], args["unique_row_id_col"], None,
        ignored_cols=args["ignored_cols"], feature_groups=args["feature_groups"], use_data_extraction=False)[0]
    session = at.session_state["_table_session"]
    pd.testing.assert_frame_equal(session.interpretation[0], fresh)
    pd.testing.assert_frame_equal(session.read[0], _frame())
    assert at.session_state.analysis_columns == list(fresh.columns)
