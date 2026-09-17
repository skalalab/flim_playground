"""Preparation owns one durable metadata record and explicit extraction decisions."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def prepared(tmp_path):
    from src.extraction_session import ExtractionSession

    settings = {
        "channel_names": ["NADH", "FAD"],
        "channels_shift": {"NADH": "fit", "FAD": "fit free"},
        "fov_name_col": "image_name", "unique_cell_id_col": "cell_id",
        "fitting_algo": "WLS", "fitting_mode": "Hybrid", "fix_shift": True,
        "derived_features": [{"name": "ratio", "expression": "A / B", "operands": ["a", "b"]}],
        "NADH": {"input_type": "Decay (2D)", "imaging_modality": "FLIM",
                 "selected_feature_extractors": ["Lifetime fit"],
                 "num_components": 2, "start": 1, "end": 12,
                 "fixed_lifetimes": {"t1": 0.4, "t2": None}},
        "FAD": {"input_type": "Decay (2D)", "imaging_modality": "FLIM",
                "selected_feature_extractors": ["Lifetime fit free"]},
    }
    rows = pd.DataFrame({"image_name": ["fov1", "fov2"]})
    return ExtractionSession.create(rows, settings, tmp_path)


def test_preparation_automatically_saves_one_record(prepared):
    assert prepared.metadata_path.is_file()
    assert prepared.metadata_path.name.startswith("fov_metadata_")
    assert not prepared.can_extract
    rows = pd.read_csv(prepared.metadata_path)
    assert rows["image_name"].tolist() == ["fov1", "fov2"]
    assert rows["fitting_mode"].eq("Hybrid").all()
    assert rows["NADH_fixed_t1"].eq(0.4).all()


def test_recalibration_keeps_last_record_until_every_shift_is_confirmed(prepared):
    assert prepared.confirm_calibration({"NADH": 1.0, "FAD": [2.0, 3.0]}) == ""
    path = prepared.metadata_path
    confirmed = path.read_bytes()
    prepared.features = pd.DataFrame({"feature": [1.0]})
    defaults = deepcopy(prepared.settings)
    prepared.begin_recalibration()
    assert not prepared.can_extract
    assert prepared.features is None
    assert prepared.settings == defaults
    assert "NADH_shift" not in prepared.metadata_df
    assert path.read_bytes() == confirmed
    assert prepared.confirm_calibration({"NADH": 4.0})
    assert not prepared.can_extract
    assert path.read_bytes() == confirmed
    assert prepared.confirm_calibration({"NADH": 4.0, "FAD": [5.0, 6.0]}) == ""
    assert prepared.can_extract
    assert prepared.metadata_path == path
    assert len(list(path.parent.glob("fov_metadata_*.csv"))) == 1
    assert pd.read_csv(path)["NADH_shift"].eq(4.0).all()


@pytest.mark.parametrize("shift", [np.nan, np.inf, True, "bad", [1.0], [1.0, np.nan]])
def test_invalid_channel_shift_cannot_confirm_or_save(prepared, shift):
    before = prepared.metadata_path.read_bytes()
    assert prepared.confirm_calibration({"NADH": 1.0, "FAD": shift})
    assert not prepared.can_extract
    assert prepared.metadata_path.read_bytes() == before
    assert "NADH_shift" not in prepared.metadata_df


def test_mode_changes_preserve_shifts_and_defaults_and_save_same_record(prepared):
    prepared.confirm_calibration({"NADH": 1.0, "FAD": 2.0})
    prepared.features = pd.DataFrame({"feature": [1.0]})
    assert prepared.change_mode("Local") == ""
    assert prepared.can_extract
    assert prepared.features is None
    assert prepared.settings["fitting_mode"] == "Local"
    assert prepared.settings["NADH"]["start"] == 1
    saved = pd.read_csv(prepared.metadata_path)
    assert saved["fitting_mode"].eq("Local").all()
    assert saved["NADH_shift"].eq(1.0).all()
    prepared.begin_recalibration()
    assert prepared.settings["fitting_mode"] == "Local"
    prepared.confirm_calibration({"NADH": 3.0, "FAD": 4.0})
    assert pd.read_csv(prepared.metadata_path)["fitting_mode"].eq("Local").all()


@pytest.mark.parametrize("failure_stage", ["write", "replace"])
def test_failed_metadata_save_preserves_file_and_edits_and_blocks_extraction(prepared, monkeypatch, failure_stage):
    import src.extraction_session as session_module

    prepared.confirm_calibration({"NADH": 1.0, "FAD": 2.0})
    before = prepared.metadata_path.read_bytes()

    def fail(*args, **kwargs):
        raise PermissionError("locked metadata")

    with monkeypatch.context() as patch:
        if failure_stage == "write":
            patch.setattr(pd.DataFrame, "to_csv", fail)
        else:
            patch.setattr(session_module.os, "replace", fail)
        error = prepared.change_mode("Local")
        assert "locked metadata" in error
        assert not prepared.can_extract
        assert prepared.before_extraction()
        assert prepared.settings["fitting_mode"] == "Local"
        assert prepared.metadata_path.read_bytes() == before
    assert prepared.save_metadata() == ""
    assert prepared.can_extract
    assert pd.read_csv(prepared.metadata_path)["fitting_mode"].eq("Local").all()
    assert list(prepared.metadata_path.parent.iterdir()) == [prepared.metadata_path]


def test_before_extraction_saves_final_edits_and_leaves_prior_feature_export_alone(prepared):
    prepared.confirm_calibration({"NADH": 1.0, "FAD": 2.0})
    features = pd.DataFrame({"image_name": ["fov1"], "feature": [42.0]}, index=pd.Index(["fov1_1"], name="cell_id"))
    assert prepared.export_features(features) == ""
    feature_path = prepared.features_path
    exported = feature_path.read_bytes()
    prepared.settings["NADH"]["start"] = 2
    assert prepared.before_extraction() == ""
    assert pd.read_csv(prepared.metadata_path)["NADH_start"].eq(2).all()
    prepared.change_mode("Local")
    prepared.begin_recalibration()
    assert feature_path.read_bytes() == exported
    assert prepared.before_extraction()


def test_atomic_save_honours_umask_like_a_plain_write(tmp_path):
    """Records must stay readable to lab colleagues; a 0600 temp file must not leak."""
    import stat
    from src.extraction_session import atomic_save_csv
    probe = tmp_path / "probe.csv"
    probe.write_text("a\n1\n")
    target = tmp_path / "record.csv"
    assert atomic_save_csv(pd.DataFrame({"a": [1]}), target) == ""
    assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(probe.stat().st_mode)
    assert not list(tmp_path.glob(".*.tmp"))
