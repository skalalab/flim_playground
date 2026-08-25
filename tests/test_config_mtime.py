"""``get_config_mtime`` — the shared signal that lets an already-open tab notice
that ``config.toml`` changed on disk (e.g. after another tab clicks "Update
Configuration"). It must read the file's modification time, report a sentinel
when the file is missing, and change when the file is rewritten.
"""
import os
import sys
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config


def _write(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(toml.dumps({"num_channels": 1}))
    return p


def test_mtime_zero_when_missing(tmp_path):
    missing = tmp_path / "does_not_exist.toml"
    assert config.get_config_mtime(config_path=missing) == 0.0


def test_mtime_matches_stat_when_present(tmp_path):
    p = _write(tmp_path)
    assert config.get_config_mtime(config_path=p) == p.stat().st_mtime


def test_mtime_increases_when_rewritten(tmp_path):
    p = _write(tmp_path)
    before = config.get_config_mtime(config_path=p)
    # Force a strictly-later mtime deterministically (coarse fs clocks make a
    # bare re-write flaky, so stamp it explicitly instead of sleeping).
    os.utime(p, (before + 10, before + 10))
    assert config.get_config_mtime(config_path=p) > before


def test_mtime_uses_config_path_global_by_default(tmp_path, monkeypatch):
    p = _write(tmp_path)
    monkeypatch.setattr(config, "_CONFIG_PATH", p)
    assert config.get_config_mtime() == p.stat().st_mtime
