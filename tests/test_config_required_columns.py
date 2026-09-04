"""The two column names Data Extraction cannot run without.

Extraction is FOV-based and id-based. `fov_name_col` is not a label but the column
extraction *reads* -- `metadata_df[fov_name_col]` in `file_io.py`, the required-column
check in `parse_metadata_file` -- and `unique_cell_id_col` names the identifier it
composes per cell as `{fov}_{mask label}`. Neither may be blank.

Blank does not fail loudly, which is why this is checked where the name is typed. An
empty string is a *name*: `prepare_fov_dataframe` sets `index.name = ""`, `reset_index`
makes a column literally called `""`, and every in-session lookup then succeeds. The run
completes and writes a CSV whose first header cell is empty -- which Data Analysis reads
back as `Unnamed: 0` and `drop_unnamed_columns` deletes, taking the identifier with it.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

_PAGE = str(Path(__file__).resolve().parents[1] / "main.py")
_SAVE = "Update Configuration"


@pytest.fixture
def page(tmp_path, monkeypatch):
    """A Configuration page on a private config, valid enough to offer its save button.

    A fresh install has no feature extractor selected, which is itself a blocking error,
    so one is picked here -- otherwise every assertion below would pass against a page
    that was refusing to save for an entirely different reason.
    """
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(config, "_CONFIG_PATH", tmp_path / "config.toml")
    at = AppTest.from_file(_PAGE).run(timeout=90)
    at.multiselect[0].set_value(["Intensity morphology"]).run(timeout=90)
    assert [b for b in at.button if b.label == _SAVE], [b.label for b in at.button]
    return at


def _field(at, label):
    return next(box for box in at.text_input if box.label == label)


def _saveable(at):
    return bool([b for b in at.button if b.label == _SAVE])


@pytest.mark.parametrize(("label", "restored", "named"), [
    ("Unique cell identifier column name", "cell_id", "holding cells"),
    ("FOV column name", "image_name", "holding fields of view"),
])
def test_a_blank_required_column_name_cannot_be_saved(page, label, restored, named):
    """Spaces count as blank: a name of whitespace is one no column will ever match.

    Restoring it has to clear the block, or the test would pass against a page that
    simply never offers the button.
    """
    _field(page, label).set_value("   ").run(timeout=90)

    assert not _saveable(page), "a blank name is still saveable"
    assert any(named in error.value.lower() for error in page.error), \
        [error.value for error in page.error]

    _field(page, label).set_value(restored).run(timeout=90)
    assert _saveable(page), [error.value for error in page.error]
