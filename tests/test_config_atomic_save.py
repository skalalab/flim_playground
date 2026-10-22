"""save_config replaces the previous file only once the new one is complete on disk."""
import sys
from pathlib import Path

import pytest
import toml

from src.config import load_config, save_config

ORIGINAL = {"current_profile": "a", "profiles": {"a": {"laser_rate": 0.08}}}


def _other_files(path):
    """Anything beside the config, such as a temporary file a save left behind."""
    return sorted(p.name for p in path.parent.iterdir() if p != path)


def test_a_write_that_dies_midway_leaves_the_previous_config_intact(tmp_path, monkeypatch):
    """A kill, a crash, or a full disk during the write must not truncate the old file."""
    path = tmp_path / "config.toml"
    save_config(ORIGINAL, path)

    def dump_then_die(cfg, fh):
        fh.write("current_profile = 'b'\n")   # part of the new content reaches the disk...
        raise OSError("disk full")            # ...and then the write stops

    monkeypatch.setattr(toml, "dump", dump_then_die)
    with pytest.raises(OSError):
        save_config({"current_profile": "b"}, path)

    assert load_config(path) == ORIGINAL
    assert _other_files(path) == []


def test_a_completed_save_replaces_the_file_and_leaves_nothing_else(tmp_path):
    path = tmp_path / "config.toml"
    save_config(ORIGINAL, path)
    replacement = {"current_profile": "b", "profiles": {"b": {}}}

    save_config(replacement, path)

    assert load_config(path) == replacement
    assert _other_files(path) == []


def test_both_config_files_sit_in_the_project_root_during_development(monkeypatch):
    """Analysis profiles live beside config.toml, not in the repository's parent."""
    from src import analysis_config, config

    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    root = Path(__file__).resolve().parents[1]

    assert config._get_config_path() == root / "config.toml"
    assert analysis_config._get_analysis_config_path() == root / "analysis_config.toml"
