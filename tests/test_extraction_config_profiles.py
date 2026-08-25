"""Multi-profile support for the extraction config (config.toml).

The extraction config gained named profiles mirroring the analysis config: a
legacy flat config migrates to ``{current_profile, profiles.default}``, and the
``src/config.py`` accessor layer resolves the active profile transparently so
that every downstream consumer keeps working unchanged.
"""
import sys
from pathlib import Path

import toml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.config as config
from src.config import (
    _migrate_extraction_config_to_profiles,
    _load_active_profile_cfg,
    get_current_profile_name,
    list_profiles,
    set_current_profile,
    create_profile,
    delete_profile,
)


# ---- migration ---------------------------------------------------------

def test_migrate_flat_config_wraps_under_default():
    flat = {
        "num_channels": 2,
        "flim_decay_input_type": "Decay (3/4D)",
        "ch1": {"channel_name": "FAD"},
    }
    migrated = _migrate_extraction_config_to_profiles(flat)
    assert migrated["current_profile"] == "default"
    assert migrated["profiles"]["default"]["num_channels"] == 2
    assert migrated["profiles"]["default"]["ch1"]["channel_name"] == "FAD"
    # legacy keys must not leak to the top level
    assert "num_channels" not in migrated
    assert "ch1" not in migrated


def test_migrate_is_idempotent():
    already = {"current_profile": "exp", "profiles": {"exp": {"num_channels": 1}}}
    assert _migrate_extraction_config_to_profiles(already) is already


def test_migrate_empty_config_left_untouched():
    assert _migrate_extraction_config_to_profiles({}) == {}


# ---- management helpers (round-trip through a temp config.toml) ---------

def _write(tmp_path, cfg):
    p = tmp_path / "config.toml"
    p.write_text(toml.dumps(cfg))
    return p


def test_create_and_list_profiles(tmp_path):
    p = _write(tmp_path, {"num_channels": 1})  # legacy flat -> migrates to default
    create_profile("experiment-B", config_path=p)
    profiles = list_profiles(config_path=p)
    assert "default" in profiles
    assert "experiment-B" in profiles
    # creating a profile makes it the current one
    assert get_current_profile_name(config_path=p) == "experiment-B"
    on_disk = toml.load(p)
    assert on_disk["current_profile"] == "experiment-B"
    assert "experiment-B" in on_disk["profiles"]


def test_create_profile_is_blank(tmp_path):
    p = _write(tmp_path, {"num_channels": 3})
    create_profile("blank", config_path=p)
    on_disk = toml.load(p)
    # blank with app defaults -> seeded later by the UI, empty on disk for now
    assert on_disk["profiles"]["blank"] == {}


def test_set_current_profile_switches(tmp_path):
    p = _write(tmp_path, {"num_channels": 1})
    create_profile("B", config_path=p)
    set_current_profile("default", config_path=p)
    assert get_current_profile_name(config_path=p) == "default"


def test_delete_profile_switches_when_current(tmp_path):
    p = _write(tmp_path, {"num_channels": 1})
    create_profile("B", config_path=p)  # current = B
    delete_profile("B", config_path=p)
    assert "B" not in list_profiles(config_path=p)
    assert get_current_profile_name(config_path=p) == "default"


def test_delete_last_profile_recreates_default(tmp_path):
    p = _write(tmp_path, {"current_profile": "only", "profiles": {"only": {"num_channels": 2}}})
    delete_profile("only", config_path=p)
    assert list_profiles(config_path=p) == ["default"]
    assert get_current_profile_name(config_path=p) == "default"


def test_load_active_profile_cfg_returns_active(tmp_path):
    p = _write(tmp_path, {
        "current_profile": "B",
        "profiles": {
            "default": {"num_channels": 1},
            "B": {"num_channels": 4},
        },
    })
    active = _load_active_profile_cfg(config_path=p)
    assert active["num_channels"] == 4


# ---- accessor rewire (end-to-end through the default path) --------------

def test_accessor_reads_active_profile(tmp_path, monkeypatch):
    p = _write(tmp_path, {
        "current_profile": "B",
        "profiles": {
            "default": {"flim_decay_input_type": "Decay (3/4D)", "num_channels": 1},
            "B": {
                "flim_decay_input_type": "Decay (2D)",
                "num_channels": 2,
                "ch1": {"channel_name": "NADH"},
                "ch2": {"channel_name": "FAD"},
            },
        },
    })
    monkeypatch.setattr(config, "_CONFIG_PATH", p)
    assert config.get_decay_input_type() == "Decay (2D)"
    assert config.get_channel_names() == {"ch1": "NADH", "ch2": "FAD"}
    # switching the active profile makes the accessors follow
    set_current_profile("default", config_path=p)
    assert config.get_decay_input_type() == "Decay (3/4D)"
    assert config.get_channel_names() == {"ch1": "ch1"}


def test_legacy_flat_config_read_through_accessor(tmp_path, monkeypatch):
    # a pre-migration flat config must still be readable via accessors
    p = _write(tmp_path, {"flim_decay_input_type": "Decay (2D)", "num_channels": 1})
    monkeypatch.setattr(config, "_CONFIG_PATH", p)
    assert config.get_decay_input_type() == "Decay (2D)"
