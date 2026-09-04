"""The review gate feeding Collapse by: two features that meet only on this path.

The gap these close: the collapse suites drive the *extraction* branch, where
categorical_cols comes from config.toml and a FOV column is designated; the review
suites never build a plot. So the user-table path into Collapse by was untested,
and it is the path where the three things the gate owns all differ -- the
categoricals come from the review table's working copy, no role names a FOV column,
and the row id may have been invented rather than read.

The page has no file_uploader accessor, so the gate is driven the way
test_review_page.py drives it: the chooser by prefix, then `_review_confirmed`.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.dataset_io as dataset_io
from src.widgets import analysis_config_widgets as acw
from src.widgets import visualization_widgets as vw
from src.widgets.review_table_widget import AUTO_DETECT

_PAGE = str(ROOT / "pages" / "data_analysis.py")


class _Upload:
    def __init__(self, name="pdl1.csv"):
        self.name = name


def _frame():
    """Two treatments x three days, three dishes per cell, several cells each."""
    rows = []
    for i, (dish, treatment, day) in enumerate([
        ("D1", "DMSO", "Day 2"), ("D2", "PD-L1", "Day 2"),
        ("D3", "DMSO", "Day 5"), ("D4", "PD-L1", "Day 5"),
        ("D5", "DMSO", "Day 10"), ("D6", "PD-L1", "Day 10"),
    ]):
        for j in range(6):
            rows.append({
                "cell_id": f"{dish}_c{j}",
                "image_name": f"{dish}_f{j % 2}",
                "dish": dish,
                "treatment": treatment,
                "day": day,
                # Fractional, like a real lifetime. Round numbers here made this
                # column an all-distinct whole-numbered bijection in the no-cell_id
                # case below, so auto-detect claimed it as the identifier and the
                # table had no measurement left -- an artefact of the fixture, not
                # of the data this stands in for.
                "nadh_tm_mean": 1200.0 + 10 * i + j + 0.37,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def page(tmp_path, monkeypatch):
    monkeypatch.setattr(acw, "_ANALYSIS_CONFIG_PATH", tmp_path / "analysis_config.toml")
    frame = _frame()
    monkeypatch.setattr(st, "file_uploader", lambda *a, **k: _Upload())
    monkeypatch.setattr(dataset_io, "read_table",
                        lambda _u: (frame.copy(), {}, ",", "", ""))
    return tmp_path


def _gated(page, **session):
    """The page past the gate, on the user-table branch, roles auto-detected."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(_PAGE)
    for key, value in session.items():
        at.session_state[key] = value
    at.run(timeout=90)
    assert not at.exception, at.exception
    at.checkbox[0].uncheck().run(timeout=90)
    assert not at.exception, at.exception
    for widget in at.button:
        if str(widget.label).startswith(AUTO_DETECT):
            widget.click().run(timeout=90)
            break
    else:
        raise AssertionError("no auto-detect row in the chooser")
    assert not at.exception, at.exception
    at.session_state._review_confirmed = True
    at.run(timeout=90)
    assert not at.exception, at.exception
    return at


def _pick_feature(at, group="Uncategorized Features", feature="nadh_tm_mean"):
    """Select the feature in its *group* selectbox.

    selected_var is read off the group menus, not off the filter's "Select Feature 1"
    box -- and it defaults to "Select", which skips the whole plot branch, so without
    this no figure is built and the export button never renders.
    """
    box = next(b for b in at.selectbox if b.label == group)
    assert feature in box.options, box.options
    box.set_value(feature).run(timeout=90)
    assert not at.exception, at.exception
    return at


def _labels(at):
    return [box.label for box in at.selectbox]


def _options(at, label):
    return next(box.options for box in at.selectbox if box.label == label)


def test_the_gate_hands_its_categoricals_to_collapse_by(page):
    """The gate's working copy -- not config.toml -- is what fills the picker.

    Collapse by offers every categorical the gate found except the ones already
    fixing a point's x slot: the first categorical seeds Color by, so `image_name`
    is spoken for and the remaining three are the offer.
    """
    at = _gated(page)
    assert at.session_state.vis_df is not None, "the frame never loaded past the gate"
    assert "Collapse by" in _labels(at), _labels(at)
    colour = set(at.session_state[vw.COLOR_BY_KEY])
    offered = set(_options(at, "Collapse by"))
    assert colour == {"image_name"}, colour
    assert offered == {"dish", "treatment", "day"}, offered
    # Freeing the slot must hand the column back, which is the direction of the chain.
    next(b for b in at.multiselect if b.label.startswith("Color by")).set_value(
        ["treatment"]).run(timeout=90)
    assert not at.exception, at.exception
    assert set(_options(at, "Collapse by")) == {"dish", "day", "image_name"}


def test_a_user_table_collapses_without_a_designated_fov_column(page):
    """The FOV role is gone on this branch, so plot_fov_name_col resolves from ""
    -- the collapse must not demand a column no role names."""
    at = _pick_feature(_gated(page))
    box = next(b for b in at.selectbox if b.label == "Collapse by")
    box.set_value("dish").run(timeout=90)
    assert not at.exception, at.exception
    assert at.session_state.effective_fov_name_col is None
    # The plot actually built on the collapsed frame.
    assert "📊 Plot Styling" in [h.value for h in at.subheader]


def test_collapsing_by_an_invented_row_number_table(page, monkeypatch):
    """No cell_id at all: the gate invents "Row number", and the collapse replaces
    the row id with the replicate column. The invented column must not also show up
    as a numeric feature."""
    bare = _frame().drop(columns=["cell_id"])
    monkeypatch.setattr(dataset_io, "read_table",
                        lambda _u: (bare.copy(), {}, ",", "", ""))
    at = _gated(page)
    df = at.session_state.vis_df
    assert df is not None
    assert any(c.startswith("Row number") for c in df.columns), list(df.columns)
    at = _pick_feature(at)
    box = next(b for b in at.selectbox if b.label == "Collapse by")
    box.set_value("dish").run(timeout=90)
    assert not at.exception, at.exception
    assert "📊 Plot Styling" in [h.value for h in at.subheader]


def test_the_exported_script_of_a_collapsed_user_table_compiles(page, monkeypatch):
    """The two halves of the merge meet in the export: the branch supplies the
    *configured* row id (blank included) and main supplies the collapse block.

    AppTest exposes no download_button, so the state the page hands to
    generate_script is captured at the seam and the real generator run on it.
    """
    import src.export_script as es

    seen = []
    real = es.generate_script
    monkeypatch.setattr(es, "generate_script",
                        lambda state: seen.append(dict(state)) or real(state))

    at = _pick_feature(_gated(page))
    box = next(b for b in at.selectbox if b.label == "Collapse by")
    box.set_value("dish").run(timeout=90)
    assert not at.exception, at.exception
    assert seen, "the page never built an export state"
    state = seen[-1]
    # Feature-Comparison-only, so it rides method_params rather than a top-level key.
    assert state["method_params"]["collapse_by"] == "dish", state["method_params"]
    # The *configured* row id, which this file does have: cell_id.
    assert state["unique_row_id_col"] == "cell_id", state["unique_row_id_col"]
    # No role names a FOV column on this branch, so none may be baked in.
    assert state["fov_name_col"] is None, state["fov_name_col"]
    script = real(state)
    compile(script, "analysis.py", "exec")
    assert "collapse_rows" in script, "the collapse never reached the script"


def test_a_blank_row_id_reaches_the_script_blank(page, monkeypatch):
    """The invented "Row number" must never be baked in: the script re-invents it."""
    import src.export_script as es

    bare = _frame().drop(columns=["cell_id"])
    monkeypatch.setattr(dataset_io, "read_table",
                        lambda _u: (bare.copy(), {}, ",", "", ""))
    seen = []
    real = es.generate_script
    monkeypatch.setattr(es, "generate_script",
                        lambda state: seen.append(dict(state)) or real(state))

    at = _pick_feature(_gated(page))
    assert seen, "the page never built an export state"
    assert seen[-1]["unique_row_id_col"] == "", seen[-1]["unique_row_id_col"]
    compile(real(seen[-1]), "analysis.py", "exec")
