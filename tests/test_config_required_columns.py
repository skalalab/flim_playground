"""Extraction requires nonblank FOV and cell-identifier column names.
An empty identifier header is dropped as unnamed when the exported CSV is read
back, so the Configuration page blocks saving it.
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
    """Use a private config with a selected extractor so column-name validation is
    the only possible reason for disabling Save.
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
    """Whitespace names disable Save; restoring a valid name enables it again."""
    _field(page, label).set_value("   ").run(timeout=90)

    assert not _saveable(page), "a blank name is still saveable"
    assert any(named in error.value.lower() for error in page.error), \
        [error.value for error in page.error]

    _field(page, label).set_value(restored).run(timeout=90)
    assert _saveable(page), [error.value for error in page.error]
