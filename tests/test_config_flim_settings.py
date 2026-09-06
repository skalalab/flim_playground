"""Shared FLIM controls follow the channel modalities without losing settings."""

from pathlib import Path

import pytest
import toml
from streamlit.testing.v1 import AppTest

import src.config as config


_PAGE = str(Path(__file__).resolve().parents[1] / "main.py")
_RAW = "Decay (3/4D)"
_PREFITTED = "Decay (3/4D) pixel-prefitted"
_TABULAR = "Decay (2D)"
_INTENSITY = "Intensity (2D)"


def _profile(modalities=("FLIM",), input_type=_RAW, extractors=None):
    cfg = {"num_channels": len(modalities), "flim_decay_input_type": input_type}
    for i, modality in enumerate(modalities, 1):
        cfg[f"ch{i}"] = {
            "channel_name": f"Channel {i}",
            "imaging_modality": modality,
            "input_type": input_type if modality == "FLIM" else _INTENSITY,
            **{
                kind: {"selected_feature_extractors": (
                    list(extractors) if extractors is not None and kind != _INTENSITY
                    else ["Intensity texture"]
                )}
                for kind in (_RAW, _PREFITTED, _TABULAR, _INTENSITY)
            },
        }
    return cfg


def _open(tmp_path, monkeypatch, profiles=None):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({
        "current_profile": "default",
        "profiles": profiles or {"default": _profile()},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    at = AppTest.from_file(_PAGE).run(timeout=30)
    _healthy(at)
    return at, path


def _healthy(at):
    assert not at.exception, [e.value for e in at.exception]
    assert not at.warning, [w.value for w in at.warning]


def _save_button(at):
    return next((b for b in at.button if b.label == "Update Configuration"), None)


@pytest.mark.parametrize(("modalities", "input_type"), [
    (("FLIM",), _RAW),
    (("FLIM",), _PREFITTED),
    (("FLIM",), _TABULAR),
    (("FLIM", "Intensity-only"), _RAW),
    (("FLIM", "Intensity-only"), _PREFITTED),
    (("Intensity-only",), _RAW),
    (("Intensity-only",), _TABULAR),
])
def test_shared_settings_only_appear_for_flim_channels(
    tmp_path, monkeypatch, modalities, input_type,
):
    at, _ = _open(tmp_path, monkeypatch, {"default": _profile(modalities, input_type)})
    has_flim = "FLIM" in modalities
    assert bool([h for h in at.subheader if h.value == "Shared FLIM settings"]) == has_flim
    assert bool([s for s in at.selectbox if s.label == "FLIM input format"]) == has_flim
    assert not at.radio  # These profiles select intensity texture only.
    assert bool(at.number_input) == (has_flim and input_type == _TABULAR)
    assert _save_button(at) is not None
    for i, modality in enumerate(modalities, 1):
        selector = at.selectbox(key=f"imaging_modality_ch{i}_default")
        assert selector.options == ["FLIM", "Intensity-only"]
        assert selector.value == modality


@pytest.mark.parametrize("input_type", [_RAW, _TABULAR])
def test_hidden_shared_values_survive_saving_and_reopening(
    tmp_path, monkeypatch, input_type,
):
    at, path = _open(tmp_path, monkeypatch, {
        "default": _profile(input_type=input_type, extractors=["Lifetime fit free"]),
    })
    at.number_input(key=f"laser_rate_{input_type}_default").set_value(0.12).run()
    at.radio(key=f"fit_free_calibration_{input_type}_default").set_value(
        "Fluorescence Lifetime Standard").run()
    at.number_input(key=f"fluorescence_lifetime_standard_lifetime_{input_type}_default").set_value(3.5).run()
    expected = {
        "laser_rate": 0.12,
        "fit_free_calibration": "Fluorescence Lifetime Standard",
        "fluorescence_lifetime_standard_lifetime": 3.5,
    }
    if input_type == _TABULAR:
        at.number_input(key=f"{_TABULAR}_duration_default").set_value(24.0).run()
        at.number_input(key=f"{_TABULAR}_time_bins_default").set_value(512).run()
        expected.update(duration=24.0, time_bins=512)

    selector = at.selectbox(key="imaging_modality_ch1_default")
    assert "Intensity-only" in selector.options
    selector.set_value("Intensity-only").run()
    at.run()  # Let Streamlit clean up widgets absent from the previous run.
    _healthy(at)
    assert not at.number_input
    assert not at.radio
    assert not [s for s in at.selectbox if s.key == "flim_decay_input_type_default"]
    _save_button(at).click().run()
    saved = toml.load(path)["profiles"]["default"]
    assert saved["flim_decay_input_type"] == input_type
    assert {key: saved[input_type][key] for key in expected} == expected

    # Values must also survive a new browser session, not just live widget state.
    at = AppTest.from_file(_PAGE).run(timeout=30)
    at.selectbox(key="imaging_modality_ch1_default").set_value("FLIM").run()
    _healthy(at)
    assert at.selectbox(key="flim_decay_input_type_default").value == input_type
    assert at.number_input(key=f"laser_rate_{input_type}_default").value == 0.12
    assert at.radio(key=f"fit_free_calibration_{input_type}_default").value == expected["fit_free_calibration"]
    assert at.number_input(key=f"fluorescence_lifetime_standard_lifetime_{input_type}_default").value == 3.5
    if input_type == _TABULAR:
        assert at.number_input(key=f"{_TABULAR}_duration_default").value == 24.0
        assert at.number_input(key=f"{_TABULAR}_time_bins_default").value == 512


def test_unsaved_values_are_independent_per_format_and_profile(tmp_path, monkeypatch):
    at, _ = _open(tmp_path, monkeypatch, {
        "default": _profile(extractors=["Lifetime fit free"]),
        "second": _profile(input_type=_TABULAR, extractors=["Lifetime fit free"]),
    })
    at.number_input(key=f"laser_rate_{_RAW}_default").set_value(0.06).run()
    at.selectbox(key="flim_decay_input_type_default").set_value(_TABULAR).run()
    at.number_input(key=f"laser_rate_{_TABULAR}_default").set_value(0.09).run()
    at.number_input(key=f"{_TABULAR}_duration_default").set_value(26.0).run()
    at.selectbox(key="flim_decay_input_type_default").set_value(_RAW).run()
    assert at.number_input(key=f"laser_rate_{_RAW}_default").value == 0.06

    at.selectbox(key="extraction_profile_selector").set_value("second").run()
    at.number_input(key=f"laser_rate_{_TABULAR}_second").set_value(0.11).run()
    at.selectbox(key="extraction_profile_selector").set_value("default").run()
    assert at.number_input(key=f"laser_rate_{_RAW}_default").value == 0.06
    at.selectbox(key="flim_decay_input_type_default").set_value(_TABULAR).run()
    assert at.number_input(key=f"laser_rate_{_TABULAR}_default").value == 0.09
    assert at.number_input(key=f"{_TABULAR}_duration_default").value == 26.0
    at.selectbox(key="extraction_profile_selector").set_value("second").run()
    assert at.number_input(key=f"laser_rate_{_TABULAR}_second").value == 0.11
    _healthy(at)


def test_deleted_profile_does_not_restore_its_shared_settings(tmp_path, monkeypatch):
    at, _ = _open(tmp_path, monkeypatch, {
        "default": _profile(extractors=["Lifetime fit free"]), "second": _profile(),
    })
    at.number_input(key=f"laser_rate_{_RAW}_default").set_value(0.13).run()
    at.button(key="delete_extraction_profile").click().run()
    at.text_input(key="new_extraction_profile_name").set_value("default")
    next(b for b in at.button if b.label == "➕ Create").click().run()
    at.multiselect(key=f"{_RAW}_ch1_feature_extractors_default").set_value(["Lifetime fit free"]).run()
    assert at.number_input(key=f"laser_rate_{_RAW}_default").value == 0.08
    _healthy(at)


@pytest.mark.parametrize("correction", ["format", "modality"])
def test_mixed_2d_inputs_require_explicit_correction(tmp_path, monkeypatch, correction):
    at, _ = _open(tmp_path, monkeypatch, {
        "default": _profile(("FLIM", "Intensity-only"), _TABULAR),
    })
    assert at.selectbox(key="imaging_modality_ch2_default").value == "Intensity-only"
    assert any("2D" in e.value and "intensity-only" in e.value for e in at.error)
    assert _save_button(at) is None
    if correction == "format":
        at.selectbox(key="flim_decay_input_type_default").set_value(_RAW).run()
    else:
        at.selectbox(key="imaging_modality_ch2_default").set_value("FLIM").run()
    _healthy(at)
    assert not at.error, [e.value for e in at.error]
    assert _save_button(at) is not None


@pytest.mark.parametrize("count", range(1, 9))
def test_channel_controls_precede_shared_settings_and_extraction_details(
    tmp_path, monkeypatch, count,
):
    modalities = tuple("FLIM" if i % 2 == 0 else "Intensity-only" for i in range(count))
    at, path = _open(tmp_path, monkeypatch, {"default": _profile(modalities)})
    nodes = list(at.main)

    def position(key):
        return next(i for i, node in enumerate(nodes) if getattr(node, "key", None) == key)

    shared_position = position("flim_decay_input_type_default")
    for i, modality in enumerate(modalities, 1):
        assert position(f"channel_name_ch{i}_default") < shared_position
        assert position(f"imaging_modality_ch{i}_default") < shared_position
        kind = _RAW if modality == "FLIM" else _INTENSITY
        assert position(f"{kind}_ch{i}_feature_extractors_default") > shared_position
        file_type = "Decay" if modality == "FLIM" else _INTENSITY
        at.text_input(key=f"ch{i}_{kind}_{file_type}_default").set_value(f"_ch{i}.tiff")

    if count > 4:
        collapsed = [e for e in at.expander if e.label == "Channels 1–4"]
        assert len(collapsed) == 2  # Channel setup and extraction details.
        assert all(not e.proto.expanded for e in collapsed)
        second_label = "Channel 5" if count == 5 else f"Channels 5–{count}"
        visible = [e for e in at.expander if e.label == second_label]
        assert len(visible) == 2
        assert all(e.proto.expanded for e in visible)

    _save_button(at).click().run()
    _healthy(at)
    saved = toml.load(path)["profiles"]["default"]
    for i, modality in enumerate(modalities, 1):
        channel = saved[f"ch{i}"]
        kind = _RAW if modality == "FLIM" else _INTENSITY
        assert channel["input_type"] == kind
        file_type = "Decay" if modality == "FLIM" else _INTENSITY
        assert channel[kind]["input_suffixes"][file_type] == f"_ch{i}.tiff"


@pytest.mark.parametrize("input_type", [_RAW, _PREFITTED, _TABULAR])
def test_fit_free_controls_follow_live_selection_and_retain_hidden_values(
    tmp_path, monkeypatch, input_type,
):
    at, path = _open(tmp_path, monkeypatch, {
        "default": _profile(input_type=input_type, extractors=["Lifetime fit"]),
    })
    feature_key = f"{input_type}_ch1_feature_extractors_default"
    laser_key = f"laser_rate_{input_type}_default"
    method_key = f"fit_free_calibration_{input_type}_default"
    lifetime_key = f"fluorescence_lifetime_standard_lifetime_{input_type}_default"
    irf_key = f"ch1_{input_type}_IRF_default"

    assert not [w for w in at.number_input if w.key == laser_key]
    assert not at.radio
    assert bool([w for w in at.text_input if w.key == irf_key]) == (input_type != _PREFITTED)
    if input_type == _TABULAR:
        assert at.number_input(key=f"{input_type}_duration_default").value == 20.0
        assert at.number_input(key=f"{input_type}_time_bins_default").value == 1024

    # A single rerun must reflect the new extractor choice above the picker.
    at.multiselect(key=feature_key).set_value(["Lifetime fit free"]).run()
    at.number_input(key=laser_key).set_value(0.12).run()
    at.radio(key=method_key).set_value("Fluorescence Lifetime Standard").run()
    at.number_input(key=lifetime_key).set_value(3.5).run()

    at.multiselect(key=feature_key).set_value(["Lifetime fit"]).run()
    assert not [w for w in at.number_input if w.key in (laser_key, lifetime_key)]
    assert not at.radio
    assert bool([w for w in at.text_input if w.key == irf_key]) == (input_type != _PREFITTED)
    at.run()  # Hidden widgets are cleaned up before the save.
    _save_button(at).click().run()
    saved = toml.load(path)["profiles"]["default"][input_type]
    assert saved["laser_rate"] == 0.12
    assert saved["fit_free_calibration"] == "Fluorescence Lifetime Standard"
    assert saved["fluorescence_lifetime_standard_lifetime"] == 3.5

    at = AppTest.from_file(_PAGE).run(timeout=30)
    assert not at.radio
    at.multiselect(key=feature_key).set_value(["Lifetime fit", "Lifetime fit free"]).run()
    assert at.number_input(key=laser_key).value == 0.12
    assert at.radio(key=method_key).value == "Fluorescence Lifetime Standard"
    assert at.number_input(key=lifetime_key).value == 3.5
    # Regular fitting still needs the IRF even when fit-free uses a standard.
    assert bool([w for w in at.text_input if w.key == irf_key]) == (input_type != _PREFITTED)
    at.multiselect(key=feature_key).set_value([]).run()
    assert not at.radio  # Deliberately clearing the picker must stay cleared.
    assert not [w for w in at.number_input if w.key == laser_key]
    _healthy(at)


def test_fit_free_controls_only_consider_active_flim_channels_and_format(tmp_path, monkeypatch):
    profile = _profile(("FLIM", "FLIM", "Intensity-only"))
    profile["ch2"][_RAW]["selected_feature_extractors"] = ["Lifetime fit free"]
    # An intensity-only channel can retain unused FLIM settings from earlier work.
    profile["ch3"][_RAW]["selected_feature_extractors"] = ["Lifetime fit free"]
    at, _ = _open(tmp_path, monkeypatch, {"default": profile})
    assert at.radio
    at.multiselect(key=f"{_RAW}_ch2_feature_extractors_default").set_value(["Intensity texture"]).run()
    assert not at.radio
    at.multiselect(key=f"{_RAW}_ch2_feature_extractors_default").set_value(["Lifetime fit free"]).run()
    assert at.radio

    at.selectbox(key="flim_decay_input_type_default").set_value(_PREFITTED).run()
    assert not at.radio  # The other format has no fit-free extractors selected.
    at.selectbox(key="flim_decay_input_type_default").set_value(_RAW).run()
    assert at.radio
    at.selectbox(key="num_channels_default").set_value(1).run()
    assert not at.radio  # Saved channels outside the active count do not apply.
    _healthy(at)
