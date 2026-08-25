"""The config-change notifier lets an already-open Data Extraction tab tell the
user their view is stale (another tab clicked "Update Configuration") so they can
refresh when ready — instead of silently reloading the page.

It records the config mtime the page rendered with; a periodic poll then renders a
"refresh to apply" banner when the on-disk mtime has moved ahead. It must NOT
auto-adopt the new mtime (the user, not the app, decides when to refresh).
"""
import os
import sys
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config

_NOTIFY_HARNESS = str(Path(__file__).resolve().parents[1] / "tests" / "harness_config_watch.py")
_BANNER_HARNESS = str(Path(__file__).resolve().parents[1] / "tests" / "harness_config_banner.py")


def _point_config_at(tmp_path, monkeypatch, mtime=None):
    p = tmp_path / "config.toml"
    p.write_text(toml.dumps({"num_channels": 1}))
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    monkeypatch.setattr(config, "_CONFIG_PATH", p)
    return p


def test_fresh_page_records_baseline_and_shows_no_banner(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    p = _point_config_at(tmp_path, monkeypatch)
    at = AppTest.from_file(_NOTIFY_HARNESS).run(timeout=60)

    assert not at.exception, f"harness raised: {[e.value for e in at.exception]}"
    # The page records the mtime it was built from...
    assert at.session_state["_config_mtime_seen"] == p.stat().st_mtime
    # ...and a freshly-loaded page is not stale, so no banner.
    warnings = " ".join(w.value.lower() for w in at.warning)
    assert "configuration was updated" not in warnings


def test_stale_config_shows_refresh_banner_without_adopting(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    _point_config_at(tmp_path, monkeypatch)
    at = AppTest.from_file(_BANNER_HARNESS)
    # A tab opened *before* another tab's save: its remembered mtime is stale.
    at.session_state["_config_mtime_seen"] = 1.0
    at.run(timeout=60)

    assert not at.exception, f"harness raised: {[e.value for e in at.exception]}"
    # It tells the user to refresh...
    warnings = " ".join(w.value.lower() for w in at.warning)
    assert "configuration was updated" in warnings
    # ...but must NOT silently adopt the new mtime — the user decides when to apply.
    assert at.session_state["_config_mtime_seen"] == 1.0


def test_up_to_date_config_shows_no_banner(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    p = _point_config_at(tmp_path, monkeypatch)
    at = AppTest.from_file(_BANNER_HARNESS)
    at.session_state["_config_mtime_seen"] = p.stat().st_mtime  # matches disk
    at.run(timeout=60)

    assert not at.exception, f"harness raised: {[e.value for e in at.exception]}"
    warnings = " ".join(w.value.lower() for w in at.warning)
    assert "configuration was updated" not in warnings
