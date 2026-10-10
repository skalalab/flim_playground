"""Extraction config supports named profiles and reads legacy flat configs.
Migration wraps flat settings under profiles.default; accessors resolve the
active profile.
"""
import sys
from pathlib import Path

import toml
import pytest

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

pytestmark = pytest.mark.usefixtures("isolated_config_paths")

BUILTIN_EXTRACTORS = [
    "Lifetime fit", "Lifetime fit free", "Intensity morphology",
    "Intensity texture", "Dry-mass statistics", "Spatial texture",
]


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
    p.write_text(toml.dumps(cfg), encoding="utf-8")
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
    # Empty profiles are seeded with defaults when the UI opens them.
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


# Active-profile accessors

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


# Analysis reads hints from every profile without changing extraction settings.

@pytest.fixture
def read_only_hints(monkeypatch):
    def unexpected_save(*args, **kwargs):
        pytest.fail("Collecting extraction hints must not save config")
    monkeypatch.setattr(config, "save_config", unexpected_save)


@pytest.mark.parametrize("stored", [None, "", {}, {"profiles": {}},
                                   {"profiles": {"empty": {}}}])
def test_missing_or_empty_extraction_hints_use_defaults(
        isolated_config_paths, read_only_hints, stored):
    path, analysis_path = isolated_config_paths
    if stored is not None:
        path.write_text(stored if isinstance(stored, str) else toml.dumps(stored),
                        encoding="utf-8")
    before = path.read_bytes() if path.exists() else None

    assert config.get_extraction_hints() == {
        "id_hints": ["cell_id"],
        "categorical_hints": ["image_name"],
        "extractor_hints": BUILTIN_EXTRACTORS,
        "channel_hints": [],
    }
    assert path.exists() == (stored is not None)
    assert (path.read_bytes() if path.exists() else None) == before
    assert not analysis_path.exists()


def test_extraction_hints_include_inactive_profiles_and_exact_names(
        tmp_path, read_only_hints):
    path = _write(tmp_path, {
        "current_profile": "active",
        "profiles": {
            "inactive": {
                "unique_cell_id_col": " Cell ID ", "fov_name_col": " Field ",
                "categorical_cols": ["Treatment", "treatment", "Treatment"],
                "all_feature_extractors": ["Custom extractor", "Lifetime fit"],
                "num_channels": 2,
                "ch1": {"channel_name": "NADH",
                        "Decay (2D)": {"selected_feature_extractors": ["Channel only"]},
                        "Decay (3/4D)": {"selected_feature_extractors": ["Other input"]}},
                "ch2": {"channel_name": ""},
            },
            "active": {
                "unique_cell_id_col": "cell_id", "fov_name_col": "image_name",
                "categorical_cols": ["treatment", "Batch", " Cell ID ", ""],
                "all_feature_extractors": ["Custom extractor", "custom extractor"],
                "num_channels": 2,
                "ch1": {"channel_name": "nadh",
                        "Decay (2D)": {"selected_feature_extractors": ["Channel only"]}},
            },
        },
    })
    before = path.read_bytes()

    assert config.get_extraction_hints() == {
        "id_hints": [" Cell ID ", "cell_id"],
        "categorical_hints": ["Treatment", "treatment", " Field ", "Batch",
                              " Cell ID ", "image_name"],
        "extractor_hints": BUILTIN_EXTRACTORS + ["Custom extractor", "Channel only",
                                                  "Other input", "custom extractor"],
        "channel_hints": ["NADH", "nadh", "ch2"],
    }
    assert path.read_bytes() == before
    # Existing extraction accessors still read the active profile only.
    assert config.get_unique_cell_id_col() == "cell_id"
    assert config.get_fov_name_col() == "image_name"
    assert config.get_categorical_cols() == ["treatment", "Batch", " Cell ID ", ""]
    assert config.get_all_feature_extractors() == ["Custom extractor", "custom extractor"]


@pytest.mark.parametrize(("settings", "ids", "categories"), [
    ({}, ["cell_id"], ["image_name"]),
    ({"unique_cell_id_col": "", "fov_name_col": ""}, [], []),
    ({"unique_cell_id_col": "renamed", "fov_name_col": "fov"}, ["renamed"], ["fov"]),
])
def test_extraction_hints_distinguish_missing_and_blank_names(
        tmp_path, read_only_hints, settings, ids, categories):
    _write(tmp_path, {"profiles": {"one": settings, "duplicate": dict(settings)}})
    hints = config.get_extraction_hints()
    assert hints["id_hints"] == ids
    assert hints["categorical_hints"] == categories


def test_legacy_extraction_hints_normalize_only_in_memory(
        isolated_config_paths, read_only_hints):
    path, _ = isolated_config_paths
    path.write_text('# Keep this legacy file exactly as stored.\n'
                    'unique_cell_id_col = "object"\n'
                    'categorical_cols = ["Condition", "Condition"]\n'
                    'all_feature_extractors = ["Legacy extractor"]\n', encoding="utf-8")
    before = path.read_bytes()

    assert config.get_extraction_hints() == {
        "id_hints": ["object"], "categorical_hints": ["Condition", "image_name"],
        "extractor_hints": BUILTIN_EXTRACTORS + ["Legacy extractor"],
        "channel_hints": [],
    }
    assert path.read_bytes() == before


def test_builtin_extractor_names_keep_the_configuration_order():
    assert list(config.BUILTIN_FEATURE_EXTRACTORS) == BUILTIN_EXTRACTORS
