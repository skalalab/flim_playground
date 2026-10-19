"""The Categorical step's folder scan says why no csv file can be used."""
import os
import sys
from pathlib import Path

import pytest
import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

_PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "data_extraction.py")
_CATEGORICAL = "**Categorical** (e.g. treatment, day)"


def _scan_errors(tmp_path, monkeypatch, folder):
    from streamlit.testing.v1 import AppTest

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(toml.dumps({
        "current_profile": "exp",
        "profiles": {"exp": {
            "num_channels": 1,
            "flim_decay_input_type": "Decay (3/4D)",
            "ch1": {"channel_name": "NADH", "input_type": "Decay (3/4D)"},
        }},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "_CONFIG_PATH", cfg_path)

    at = AppTest.from_file(_PAGE).run(timeout=60)
    at.radio[0].set_value(_CATEGORICAL).run(timeout=60)
    at.text_input[0].set_value(str(folder)).run(timeout=60)

    assert not at.exception, [e.value for e in at.exception]
    return [e.value for e in at.error]


@pytest.mark.parametrize(("files", "expected"), [
    ({}, "No csv files found"),
    ({"fad_irf.csv": "time,counts\n0,1\n"}, "has both the `cell_id` and `image_name` columns"),
    ({"features.csv": "cell_id,image_name\nA_1,A\n"}, "can be assigned categories"),
])
def test_scan_error_names_the_reason(tmp_path, monkeypatch, files, expected):
    folder = tmp_path / "tables"
    folder.mkdir()
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")

    errors = _scan_errors(tmp_path, monkeypatch, folder)
    assert len(errors) == 1 and expected in errors[0], errors
    assert "No available csv" not in errors[0]


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="needs POSIX permissions as a non-root user")
def test_unreadable_folder_reports_permission_denied(tmp_path, monkeypatch):
    # rglob yields nothing for a folder it cannot list, which used to read as "no csv files".
    folder = tmp_path / "tables"
    folder.mkdir()
    (folder / "features.csv").write_text("cell_id,image_name\nA_1_1,A_1\n", encoding="utf-8")
    folder.chmod(0)
    try:
        errors = _scan_errors(tmp_path, monkeypatch, folder)
    finally:
        folder.chmod(0o755)
    assert len(errors) == 1 and "Permission denied" in errors[0], errors
