"""Preparation uses explicit settings and validates the in-memory source table."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from ptufile import PtuFile

from src import config, metadata
from test_ptu_references import write_reference


STANDARD = "Fluorescence Lifetime Standard"


def _channel(input_type="Decay (3/4D)", extractors=("Lifetime fit",), **extra):
    return {
        "input_type": input_type,
        "imaging_modality": "Intensity-only" if input_type == "Intensity (2D)" else "FLIM",
        "selected_feature_extractors": list(extractors),
        "num_components": 2,
        "fixed_lifetimes": {"t1": 0.4, "t2": None},
        **extra,
    }


def _fovs(tmp_path, channel="ch1", kinds=("Decay", "Mask", "IRF"), count=2):
    data = {"image_name": [f"fov{i}" for i in range(count)]}
    for kind in kinds:
        paths = []
        for i in range(count):
            suffix = "shared" if kind in ("IRF", STANDARD) else str(i)
            path = tmp_path / f"{channel}_{kind}_{suffix}.tif"
            path.touch()
            paths.append(str(path))
        data[f"{channel}_{kind}"] = paths
    data.update(time_bins=[32] * count, duration=[25.0] * count)
    data[f"{channel}_channel"] = [-1] * count
    if STANDARD in kinds:
        data[f"{channel}_fluorescence_lifetime_standard_time_axis"] = [2] * count
    return pd.DataFrame(data)


def _prepare(fovs, channels=None, **kwargs):
    return metadata.prepare_extraction(
        fovs, {"ch1": _channel()} if channels is None else channels,
        fov_name_col="image_name", unique_cell_id_col="cell_id", **kwargs,
    )


def test_explicit_definitions_do_not_read_config_or_discover_metadata_columns(tmp_path, monkeypatch):
    name = "marker_input_type"
    fovs = _fovs(tmp_path, name, ("Intensity (2D)", "Mask"))
    fovs["ghost_input_type"] = "Decay (2D)"
    fovs[f"{name}_Lifetime fit"] = True
    fovs["derived_features"] = '[{"name": "stale"}]'
    channels = {name: _channel("Intensity (2D)", ("Intensity texture",))}
    definitions = [{"name": "scaled", "expression": "A / 2", "operands": ["signal"]}]
    before = fovs.copy(deep=True)

    def unexpected_config_read():
        raise AssertionError("Preparation must use the supplied settings")

    monkeypatch.setattr(config, "_load_active_profile_cfg", unexpected_config_read)
    err, settings = _prepare(fovs, channels, derived_features=definitions)

    assert err == ""
    assert settings["channel_names"] == [name]
    assert settings["Intensity texture"] == [name]
    assert "Lifetime fit" not in settings
    assert settings["fov_name_col"] == "image_name"
    assert settings["unique_cell_id_col"] == "cell_id"
    assert settings["derived_features"] == definitions
    assert settings["channels_shift"] == {}
    assert "time_bins" not in settings
    pd.testing.assert_frame_equal(fovs, before)
    settings["derived_features"][0]["operands"].append("other")
    settings[name]["selected_feature_extractors"].append("Intensity morphology")
    assert definitions[0]["operands"] == ["signal"]
    assert channels[name]["selected_feature_extractors"] == ["Intensity texture"]


def test_raw_fitting_uses_direct_component_settings_and_defaults(tmp_path):
    fovs = _fovs(tmp_path)
    fovs["ch1_num_components"] = 3
    fovs["ch1_fixed_t1"] = 100
    channels = {"ch1": _channel()}
    original = deepcopy(channels)

    err, settings = _prepare(fovs, channels)

    assert err == ""
    assert settings["Lifetime fit"] == ["ch1"]
    assert settings["channels_shift"] == {"ch1": "fit"}
    assert settings["decay_input_type"] == "Decay (3/4D)"
    assert settings["time_bins"] == 32
    assert settings["duration"] == 25.0
    assert settings["fitting_algo"] == "MLE"
    assert settings["fitting_mode"] == "Local"
    assert settings["fix_shift"] is True
    assert settings["ch1"]["num_components"] == 2
    assert settings["ch1"]["fixed_lifetimes"] == {"t1": 0.4, "t2": None}
    assert settings["ch1"]["start"] == 0
    assert settings["ch1"]["end"] == 32
    assert settings["ch1"]["channel_no"] == -1
    settings["ch1"]["fixed_lifetimes"]["t1"] = 2
    assert channels == original


@pytest.mark.parametrize("fit, method, expected_shift", [
    (False, "IRF", {"ch1": "fit free"}),
    (False, STANDARD, {}),
    (True, STANDARD, {"ch1": "fit"}),
])
def test_calibration_method_determines_required_reference_and_shift(tmp_path, fit, method, expected_shift):
    kinds = ["Decay", "Mask"]
    if fit or method == "IRF":
        kinds.append("IRF")
    if method == STANDARD:
        kinds.append(STANDARD)
    fovs = _fovs(tmp_path, kinds=kinds)
    extractors = ["Lifetime fit free"] + (["Lifetime fit"] if fit else [])

    err, settings = _prepare(
        fovs, {"ch1": _channel(extractors=extractors)}, laser_rate=0.04,
        fit_free_calibration_method=method,
        fluorescence_lifetime_standard_lifetime=4.0,
    )

    assert err == ""
    assert settings["channels_shift"] == expected_shift
    assert settings["laser_rate"] == 0.04
    assert settings["fit_free_calibration_method"] == method
    if method == STANDARD:
        assert settings["ch1"]["fluorescence_lifetime_standard_time_axis"] == 2
        assert settings["fluorescence_lifetime_standard_lifetime"] == 4.0


def test_prefitted_lifetime_only_needs_no_decay_timing_channel_or_shift(tmp_path):
    fovs = _fovs(tmp_path, kinds=("Mask", "SPCImage t1", "a1", "t2"))
    fovs = fovs.drop(columns=["time_bins", "duration", "ch1_channel"])

    err, settings = _prepare(fovs, {"ch1": _channel("Decay (3/4D) pixel-prefitted")})

    assert err == ""
    assert settings["channels_shift"] == {}
    assert settings["ch1"]["num_components"] == 2
    assert "channel_no" not in settings["ch1"]
    assert "time_bins" not in settings


def test_2d_decay_needs_no_mask_or_channel(tmp_path):
    fovs = _fovs(tmp_path, kinds=("Decay", "IRF")).drop(columns="ch1_channel")
    err, settings = _prepare(fovs, {"ch1": _channel("Decay (2D)")})
    assert err == ""
    assert "channel_no" not in settings["ch1"]


@pytest.mark.parametrize("input_type, extractors, kinds", [
    ("Intensity (2D)", ("Intensity morphology",), ("Intensity (2D)", "Mask")),
    ("Decay (3/4D)", ("Lifetime fit free",), ("Decay", "Mask", "IRF")),
    ("Decay (3/4D) pixel-prefitted", ("Lifetime fit",), ("Mask", "SPCImage t1", "a1", "t2")),
])
def test_non_fitting_workflows_have_no_fitting_algorithm_or_mode(tmp_path, input_type, extractors, kinds):
    err, settings = _prepare(
        _fovs(tmp_path, kinds=kinds), {"ch1": _channel(input_type, extractors)},
        laser_rate=0.04, fit_free_calibration_method="IRF",
    )
    assert err == ""
    assert "fitting_algo" not in settings
    assert "fitting_mode" not in settings


@pytest.mark.parametrize("value", [None, 0, np.nan])
def test_unset_fixed_lifetimes_are_normalized_to_free_parameters(tmp_path, value):
    err, settings = _prepare(
        _fovs(tmp_path), {"ch1": _channel(fixed_lifetimes={"t1": value})},
    )
    assert err == ""
    assert settings["ch1"]["fixed_lifetimes"]["t1"] is None


@pytest.mark.parametrize("value", [-0.4, np.inf, -np.inf, "invalid"])
def test_invalid_fixed_lifetimes_are_rejected(tmp_path, value):
    err, settings = _prepare(
        _fovs(tmp_path), {"ch1": _channel(fixed_lifetimes={"t1": value})},
    )
    assert settings is None
    assert "ch1" in err and "t1" in err and "lifetime" in err.lower()


@pytest.mark.parametrize("column", ["image_name", "ch1_Decay", "ch1_Mask"])
def test_duplicate_fovs_or_source_paths_are_rejected(tmp_path, column):
    fovs = _fovs(tmp_path)
    fovs.loc[1, column] = fovs.loc[0, column]
    err, settings = _prepare(fovs)
    assert settings is None
    assert "not unique" in err and column in err


@pytest.mark.parametrize("column", ["image_name", "ch1_Decay", "ch1_Mask", "ch1_IRF", "ch1_channel", "time_bins", "duration"])
def test_required_source_columns_are_reported(tmp_path, column):
    err, settings = _prepare(_fovs(tmp_path).drop(columns=column))
    assert settings is None
    assert column in err


@pytest.mark.parametrize("column", ["ch1_Decay", "ch1_Mask", "ch1_IRF"])
@pytest.mark.parametrize("bad_value", [None, "", 123, "missing.tif"])
def test_invalid_file_paths_are_reported_without_raising(tmp_path, column, bad_value):
    fovs = _fovs(tmp_path)
    fovs.loc[0, column] = bad_value
    err, settings = _prepare(fovs)
    assert settings is None
    assert column in err and "valid" in err


@pytest.mark.parametrize("column", ["time_bins", "duration", "ch1_channel"])
def test_acquisition_values_must_be_consistent(tmp_path, column):
    fovs = _fovs(tmp_path)
    fovs.loc[1, column] += 1
    err, settings = _prepare(fovs)
    assert settings is None
    assert column in err and "consistent" in err


@pytest.mark.parametrize("column, value", [
    ("time_bins", 0), ("time_bins", 3.5), ("time_bins", np.nan),
    ("duration", -1), ("duration", np.inf),
    ("ch1_channel", -2), ("ch1_channel", 0.5),
])
def test_invalid_acquisition_values_are_rejected(tmp_path, column, value):
    fovs = _fovs(tmp_path)
    fovs[column] = value
    err, settings = _prepare(fovs)
    assert settings is None
    assert column in err


@pytest.mark.parametrize("method", ["IRF", STANDARD])
def test_ptu_reference_checks_headers_with_explicit_laser_rate(tmp_path, monkeypatch, method):
    kinds = ("Decay", "Mask", method)
    fovs = _fovs(tmp_path, kinds=kinds)
    reference_path = tmp_path / "reference.ptu"
    write_reference(reference_path)
    fovs[f"ch1_{method}"] = str(reference_path)
    fovs["laser_rate"] = 123  # Repeated settings columns are not the definition source.
    before = fovs.copy(deep=True)

    def unexpected_decode(*args, **kwargs):
        raise AssertionError("Preparation must only read PTU headers")

    monkeypatch.setattr(PtuFile, "read_records", unexpected_decode)
    kwargs = dict(laser_rate=0.04, fit_free_calibration_method=method,
                  fluorescence_lifetime_standard_lifetime=4.0)
    channels = {"ch1": _channel(extractors=("Lifetime fit free",))}
    err, settings = _prepare(fovs, channels, **kwargs)
    assert err == ""
    assert settings["laser_rate"] == 0.04
    pd.testing.assert_frame_equal(fovs, before)

    kwargs["laser_rate"] = 0.08
    err, settings = _prepare(fovs, channels, **kwargs)
    assert settings is None
    assert "frequency" in err and "40" in err and "80" in err


def test_ptu_validation_checks_every_distinct_irf(tmp_path):
    fovs = _fovs(tmp_path)
    for i, rate in enumerate((40_000_000, 80_000_000)):
        path = tmp_path / f"irf{i}.ptu"
        write_reference(path, frequency=rate)
        fovs.loc[i, "ch1_IRF"] = str(path)
    err, settings = _prepare(fovs, laser_rate=0.04)
    assert settings is None
    assert "IRF for ch1" in err and "reference=80" in err
