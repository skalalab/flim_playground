"""The Configuration page offers QPI as a modality with three required constants.

Modelled on test_config_flim_settings.py. Saving is refused until a positive pixel size and
an OPD unit are given (no defaults: a wrong unit is a million-fold mass error).
"""
from pathlib import Path

import pytest
import toml
from streamlit.testing.v1 import AppTest

from src import config, qpi
from src.config import get_qpi_constants

_PAGE = str(Path(__file__).resolve().parents[1] / "main.py")
_RAW = "Decay (3/4D)"
_TABULAR = "Decay (2D)"
_QPI = "QPI (2D)"
_OLD_EXTRACTORS = ["Lifetime fit", "Lifetime fit free", "Intensity morphology", "Intensity texture"]
_NEW_EXTRACTORS = _OLD_EXTRACTORS + ["Dry-mass statistics", "Spatial texture"]
_QPI_EXTRACTORS = ["Intensity morphology", "Dry-mass statistics", "Spatial texture"]
_PIXEL = f"ch1_{_QPI}_pixel_size_um_default"
_UNIT = f"ch1_{_QPI}_opd_unit_default"
_ALPHA = f"ch1_{_QPI}_alpha_um3_per_pg_default"


def _profile(modalities=("QPI",), constants=None, flim_input_type=_RAW):
    """A saved profile predating QPI (four-name extractor list) with the given channels."""
    cfg = {"num_channels": len(modalities), "flim_decay_input_type": flim_input_type,
           "all_feature_extractors": list(_OLD_EXTRACTORS)}
    for i, modality in enumerate(modalities, 1):
        if modality == "QPI":
            settings = {"selected_feature_extractors": ["Intensity morphology", "Dry-mass statistics"]}
            settings.update(constants or {})
            cfg[f"ch{i}"] = {"channel_name": f"Channel {i}", "imaging_modality": "QPI",
                             "input_type": _QPI, _QPI: settings}
        else:
            kind = flim_input_type if modality == "FLIM" else "Intensity (2D)"
            cfg[f"ch{i}"] = {"channel_name": f"Channel {i}", "imaging_modality": modality, "input_type": kind,
                             **{k: {"selected_feature_extractors": ["Intensity texture"]}
                                for k in (_RAW, _TABULAR, "Intensity (2D)")}}
    return cfg


def _open(tmp_path, monkeypatch, profile):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"current_profile": "default", "profiles": {"default": profile}}), encoding="utf-8")
    monkeypatch.setattr(config, "_CONFIG_PATH", path)
    at = AppTest.from_file(_PAGE).run(timeout=30)
    assert not at.exception, [e.value for e in at.exception]
    return at, path


def _save_button(at):
    return next((b for b in at.button if b.label == "Update Configuration"), None)


def test_qpi_channel_offers_its_extractors_and_constants(tmp_path, monkeypatch):
    at, _ = _open(tmp_path, monkeypatch, _profile())
    selector = at.selectbox(key="imaging_modality_ch1_default")
    assert selector.options == ["FLIM", "Intensity-only", "QPI"]
    assert selector.value == "QPI"
    assert at.multiselect(key=f"{_QPI}_ch1_feature_extractors_default").options == _QPI_EXTRACTORS
    assert at.number_input(key=_PIXEL).value is None
    assert at.selectbox(key=_UNIT).value is None
    assert at.selectbox(key=_UNIT).options == ["m", "um", "nm"]
    assert at.number_input(key=_ALPHA).value == pytest.approx(0.181818)
    assert not [h for h in at.subheader if h.value == "Shared FLIM settings"]   # QPI alone needs no FLIM block
    assert [t for t in at.text_input if t.key == f"ch1_{_QPI}_{_QPI}_default"]   # wavefront suffix
    assert [t for t in at.text_input if t.key == f"ch1_{_QPI}_Mask_default"]


@pytest.mark.parametrize("fill", ["pixel", "unit", "both"])
def test_update_configuration_needs_pixel_size_and_unit(tmp_path, monkeypatch, fill):
    at, path = _open(tmp_path, monkeypatch, _profile())
    assert _save_button(at) is None
    assert any("Channel 1" in e.value and "pixel size" in e.value for e in at.error), [e.value for e in at.error]
    if fill in ("pixel", "both"):
        at.number_input(key=_PIXEL).set_value(0.275).run()
    if fill in ("unit", "both"):
        at.selectbox(key=_UNIT).set_value("m").run()
    if fill != "both":
        assert _save_button(at) is None
        return
    assert not at.error, [e.value for e in at.error]
    _save_button(at).click().run()
    saved = toml.load(path)["profiles"]["default"]
    assert saved["ch1"]["input_type"] == _QPI
    assert saved["ch1"][_QPI]["pixel_size_um"] == 0.275
    assert saved["ch1"][_QPI]["opd_unit"] == "m"
    assert saved["ch1"][_QPI]["alpha_um3_per_pg"] == pytest.approx(0.181818)
    assert saved[_QPI]["available_feature_extractors"] == _QPI_EXTRACTORS
    assert saved[_QPI]["file_types"] == [_QPI, "Mask"]
    assert saved["qpi_input_type"] == _QPI
    assert get_qpi_constants("ch1", _QPI) == {
        "pixel_size_um": 0.275, "opd_unit": "m", "alpha_um3_per_pg": pytest.approx(0.181818)}


def test_nonpositive_pixel_size_is_refused(tmp_path, monkeypatch):
    at, _ = _open(tmp_path, monkeypatch, _profile(constants={"pixel_size_um": 0.0, "opd_unit": "m"}))
    assert _save_button(at) is None
    assert get_qpi_constants("ch1", _QPI)["pixel_size_um"] is None


def test_saved_constants_reopen(tmp_path, monkeypatch):
    at, _ = _open(tmp_path, monkeypatch, _profile(
        constants={"pixel_size_um": 0.39, "opd_unit": "nm", "alpha_um3_per_pg": 0.2}))
    assert at.number_input(key=_PIXEL).value == 0.39
    assert at.selectbox(key=_UNIT).value == "nm"
    assert at.number_input(key=_ALPHA).value == 0.2
    assert _save_button(at) is not None


def test_saved_profiles_learn_the_new_extractor_names_on_render(tmp_path, monkeypatch):
    """A profile saved before QPI existed keeps its four-name list; the page appends the
    missing names so Data Analysis groups QPI columns instead of leaving them Uncategorized."""
    at, path = _open(tmp_path, monkeypatch, _profile(("FLIM",)))
    assert toml.load(path)["profiles"]["default"]["all_feature_extractors"] == _OLD_EXTRACTORS
    _save_button(at).click().run()
    saved = toml.load(path)["profiles"]["default"]
    assert saved["all_feature_extractors"] == _NEW_EXTRACTORS
    assert saved[_QPI]["available_feature_extractors"] == _QPI_EXTRACTORS   # seeded for every profile
    assert config.get_all_feature_extractors() == _NEW_EXTRACTORS


def test_two_d_decays_cannot_mix_with_qpi(tmp_path, monkeypatch):
    at, _ = _open(tmp_path, monkeypatch, _profile(
        ("FLIM", "QPI"), constants={"pixel_size_um": 0.275, "opd_unit": "m"}, flim_input_type=_TABULAR))
    assert any("2D" in e.value and "QPI" in e.value for e in at.error), [e.value for e in at.error]
    assert _save_button(at) is None
    at.selectbox(key="flim_decay_input_type_default").set_value(_RAW).run()
    assert not at.error, [e.value for e in at.error]
    assert _save_button(at) is not None


def test_tracked_config_lists_qpi_extractors_for_every_profile():
    """The online Data Analysis deployment reads the tracked config.toml without rendering main.py."""
    tracked = toml.load(Path(__file__).resolve().parents[1] / "config.toml")
    for name, profile in tracked["profiles"].items():
        assert {"Dry-mass statistics", "Spatial texture"} <= set(profile.get("all_feature_extractors", [])), name


def test_unit_lists_agree_between_config_and_module():
    assert tuple(config.QPI_OPD_UNITS) == qpi.OPD_UNITS
    assert set(config.QPI_OPD_UNITS) == set(qpi.OPD_TO_UM)
    assert config.QPI_ALPHA_DEFAULT == 0.181818
