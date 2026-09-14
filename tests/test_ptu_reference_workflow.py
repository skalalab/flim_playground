"""PTU references work through discovery, metadata CSVs, shifts, and extraction."""

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from ptufile import PtuFile
from streamlit.testing.v1 import AppTest
import tifffile
import toml

from src import config, metadata
from src.choose_shift import choose_shift_fit_free
from src.fov_extraction import fov_extraction
from src.widgets import metadata_widgets as mw
from test_ptu_references import reference_metadata, write_reference


INPUT_TYPE = "Decay (3/4D)"
STANDARD = "Fluorescence Lifetime Standard"
SUFFIXES = {"Decay": ".ptu", "Mask": "_mask.tif", "IRF": "quenched.ptu",
            STANDARD: "Atto488.ptu"}


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    curves = {}
    for name in ("fov1", "fov2", "Atto488", "quenched"):
        data = write_reference(folder / f"{name}.ptu")
        tifffile.imwrite(folder / f"{name}_mask.tif", np.ones((2, 3), dtype=np.uint8))
        curves[name] = data.sum(axis=(0, 1, 2, 3))
    cfg = {
        "num_channels": 1, "flim_decay_input_type": INPUT_TYPE,
        "fov_name_col": "image_name", "unique_cell_id_col": "cell_id",
        INPUT_TYPE: {"available_feature_extractors": ["Lifetime fit free"],
                     "file_types": list(SUFFIXES), "laser_rate": 0.04,
                     "fit_free_calibration": STANDARD,
                     "fluorescence_lifetime_standard_lifetime": 4.0},
        "ch1": {"channel_name": "ch1", "input_type": INPUT_TYPE,
                "imaging_modality": "FLIM",
                INPUT_TYPE: {"selected_feature_extractors": ["Lifetime fit free"],
                             "input_suffixes": SUFFIXES}},
    }
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(toml.dumps(cfg))
    monkeypatch.setattr(config, "_CONFIG_PATH", cfg_path)
    mw.clear_folder_scan_caches()
    fov_extraction.clear()
    choose_shift_fit_free.clear()
    return folder, curves


def sample_metadata(folder, method=STANDARD):
    rows = []
    for name in ("fov1", "fov2"):
        row = reference_metadata(folder / "Atto488.ptu")
        row = pd.concat([row, pd.Series({
            "image_name": name, "ch1_input_type": INPUT_TYPE,
            "ch1_imaging_modality": "FLIM", "ch1_Lifetime fit free": True,
            "ch1_Decay": str(folder / f"{name}.ptu"),
            "ch1_Mask": str(folder / f"{name}_mask.tif"),
            "ch1_channel": -1, "fit_free_calibration_method": method,
            "fluorescence_lifetime_standard_lifetime": 4.0,
            "ch1_fluorescence_lifetime_standard_time_axis": 2,
        })])
        row["ch1_IRF"] = str(folder / "quenched.ptu")
        if method != STANDARD:
            row = row.drop([f"ch1_{STANDARD}", "ch1_fluorescence_lifetime_standard_time_axis"])
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.parametrize("reference_first", [False, True])
def test_discovery_excludes_reference_files_even_if_reference_suffix_comes_first(dataset, reference_first):
    folder, _ = dataset
    suffixes = {f"ch1_{key}": value for key, value in SUFFIXES.items()}
    if reference_first:
        suffixes = dict(reversed(list(suffixes.items())))
    found = mw.load_list_data_from_folder_widget(str(folder), suffixes)
    assert set(found) == {"fov1", "fov2"}
    assert found["fov1"][f"ch1_{STANDARD}"] == found["fov2"][f"ch1_{STANDARD}"]
    assert found["fov1"]["ch1_IRF"] == found["fov2"]["ch1_IRF"]


@pytest.mark.parametrize("inactive_irf_present", [True, False])
def test_metadata_page_creates_and_exports_shared_ptu_standard(dataset, inactive_irf_present):
    folder, _ = dataset
    if not inactive_irf_present:
        (folder / "quenched.ptu").unlink()
        (folder / "quenched_mask.tif").unlink()
    page = Path(__file__).resolve().parents[1] / "pages/data_extraction.py"
    app = AppTest.from_file(str(page)).run(timeout=30)
    assert not app.exception
    app.text_input(key="fov_metadata_folder_path").set_value(str(folder)).run(timeout=30)
    assert not app.exception
    assert not app.error, [e.value for e in app.error]
    app.button(key="export_metadata_button").click().run(timeout=30)
    assert not app.exception
    csv_path = app.session_state["last_extracted_metadata_filepath"]
    rows = pd.read_csv(csv_path)
    assert rows["image_name"].tolist() == ["fov1", "fov2"]
    assert rows["ch1_fluorescence_lifetime_standard_time_axis"].tolist() == [2, 2]
    assert rows["fluorescence_lifetime_standard_lifetime"].tolist() == [4.0, 4.0]
    err, info = metadata.parse_metadata_file(rows, "image_name")
    assert err == ""
    assert info["channels_shift"] == {}


@pytest.mark.parametrize("method", [STANDARD, "IRF"])
def test_csv_replay_shift_and_single_object_extraction_match_existing_formats(dataset, method):
    folder, curves = dataset
    rows = pd.read_csv(StringIO(sample_metadata(folder, method).to_csv(index=False)))
    err, info = metadata.parse_metadata_file(rows, "image_name")
    assert err == ""
    if method != STANDARD:
        err, shift = choose_shift_fit_free(rows, 32, INPUT_TYPE, "ch1")
        assert err == ""
        assert shift["decay_id"] == ["fov1", "fov2"]
        assert shift["shift"] == [0, 0]
        rows["ch1_shift"] = shift["shift"]
    actual = []
    for _, row in rows.iterrows():
        err, features = fov_extraction(row, info)
        assert err == ""
        assert features.index.tolist() == [f"{row['image_name']}_1"]
        actual.append(features)
    if method == STANDARD:
        path = folder / "standard.tiff"
        tifffile.imwrite(path, curves["Atto488"].reshape(1, 1, -1), photometric="minisblack")
        rows[f"ch1_{STANDARD}"] = str(path)
    else:
        path = folder / "irf.csv"
        np.savetxt(path, curves["quenched"], delimiter=",")
        rows["ch1_IRF"] = str(path)
        err, old_shift = choose_shift_fit_free(rows, 32, INPUT_TYPE, "ch1")
        assert err == "" and old_shift == shift
    err, info = metadata.parse_metadata_file(rows, "image_name")
    assert err == ""
    for (_, row), expected in zip(rows.iterrows(), actual):
        err, features = fov_extraction(row, info)
        assert err == ""
        pd.testing.assert_frame_equal(features, expected)


@pytest.mark.parametrize("method", [STANDARD, "IRF"])
def test_csv_replay_rejects_incompatible_reference_frequency(dataset, method):
    folder, _ = dataset
    rows = sample_metadata(folder, method)
    rows["laser_rate"] = 0.08
    err, info = metadata.parse_metadata_file(rows, "image_name")
    assert info is None
    assert "frequency" in err and "40" in err and "80" in err


@pytest.mark.parametrize("method", [STANDARD, "IRF"])
def test_numeric_step_checks_reference_timing_without_reading_photons(dataset, monkeypatch, method):
    folder, _ = dataset
    rows = sample_metadata(folder, method)
    reads = []
    original = PtuFile.read_records

    def tracked_read(self, *args, **kwargs):
        reads.append(self.filename)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PtuFile, "read_records", tracked_read)
    page = Path(__file__).resolve().parents[1] / "pages/data_extraction.py"
    app = AppTest.from_file(str(page)).run(timeout=30)
    app.session_state["last_extracted_metadata"] = rows
    next(r for r in app.radio if r.label == "Select a step to perform").set_value(
        "Numeric Feature Extraction (fitting, phasor, etc.)"
    ).run(timeout=30)
    assert not app.exception
    assert not app.error, [e.value for e in app.error]
    action = "Optimize for Shifts" if method == "IRF" else "Confirm and Start"
    assert any(button.label == action for button in app.button)
    assert reads == []


def test_metadata_timing_checks_every_distinct_irf_header(dataset):
    folder, _ = dataset
    rows = sample_metadata(folder, "IRF")
    replacement = folder / "second_irf.ptu"
    write_reference(replacement, frequency=80_000_000)
    rows.loc[1, "ch1_IRF"] = str(replacement)
    err, info = metadata.parse_metadata_file(rows, "image_name")
    assert info is None
    assert "frequency" in err and "reference=80" in err and "sample=40" in err


@pytest.mark.parametrize("method", [STANDARD, "IRF"])
def test_metadata_timing_rereads_replaced_reference_header(dataset, method):
    folder, _ = dataset
    rows = sample_metadata(folder, method)
    assert metadata.parse_metadata_file(rows, "image_name")[0] == ""
    path = folder / ("Atto488.ptu" if method == STANDARD else "quenched.ptu")
    write_reference(path, frequency=80_000_000)
    err, info = metadata.parse_metadata_file(rows, "image_name")
    assert info is None
    assert "frequency" in err and "reference=80" in err and "sample=40" in err


@pytest.mark.parametrize("method", [STANDARD, "IRF"])
def test_rescan_refreshes_calculations_after_reference_is_replaced(dataset, method):
    folder, _ = dataset
    rows = sample_metadata(folder, method)
    rows["ch1_shift"] = 0.0
    err, info = metadata.parse_metadata_file(rows, "image_name")
    assert err == ""
    err, before = fov_extraction(rows.iloc[0], info)
    assert err == ""
    if method != STANDARD:
        err, original_shift = choose_shift_fit_free(rows, 32, INPUT_TYPE, "ch1")
        assert err == ""
    path = folder / ("Atto488.ptu" if method == STANDARD else "quenched.ptu")
    data = write_reference(folder / "replacement.ptu")
    write_reference(path, np.roll(data, 2, axis=-1))
    mw.clear_folder_scan_caches()
    err, after = fov_extraction(rows.iloc[0], info)
    assert err == ""
    assert not np.allclose(before.select_dtypes("number"), after.select_dtypes("number"), equal_nan=True)
    if method != STANDARD:
        err, shifted = choose_shift_fit_free(rows, 32, INPUT_TYPE, "ch1")
        assert err == ""
        assert original_shift["shift"] == [0, 0]
        assert shifted["shift"] == [-2, -2]
