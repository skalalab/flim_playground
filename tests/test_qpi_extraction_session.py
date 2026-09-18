"""QPI calibration participates in the folder-based extraction session."""

from copy import deepcopy

import pandas as pd
import pytest

from qpi_fixtures import CONSTANTS, QPI_EXTRACTORS, QPI_INPUT, qpi_metadata_rows, write_qpi_fov
from src.extraction_session import ExtractionSession
from src.metadata import prepare_extraction
from src.qpi import DEFAULT_BACKGROUND


def prepare_qpi(tmp_path, *, extractors=QPI_EXTRACTORS, constants=None):
    write_qpi_fov(tmp_path, "fov1")
    rows = qpi_metadata_rows(tmp_path, ["fov1"])
    definition = {
        "input_type": QPI_INPUT, "imaging_modality": "QPI",
        "selected_feature_extractors": list(extractors),
        "qpi": deepcopy(CONSTANTS if constants is None else constants),
    }
    error, settings = prepare_extraction(
        rows, {"QPI": definition}, fov_name_col="image_name", unique_cell_id_col="cell_id",
    )
    return error, rows, settings


def test_preparation_takes_qpi_constants_from_explicit_definitions(tmp_path):
    constants = {**CONSTANTS, "pixel_size_um": 0.5}
    error, rows, settings = prepare_qpi(tmp_path, constants=constants)
    assert error == "", error
    assert settings["QPI"]["qpi"] == constants
    assert settings["channels_background"] == ["QPI"]
    assert not settings["channels_shift"]
    session = ExtractionSession.create(rows, settings, tmp_path)
    assert not session.can_extract
    assert pd.read_csv(session.metadata_path)["QPI_pixel_size_um"].eq(0.5).all()


@pytest.mark.parametrize(("key", "value"), [
    ("pixel_size_um", 0), ("pixel_size_um", float("inf")),
    ("alpha_um3_per_pg", -1), ("alpha_um3_per_pg", float("nan")),
    ("opd_unit", "px"), ("opd_unit", None),
])
def test_preparation_rejects_invalid_qpi_constants(tmp_path, key, value):
    error, _, settings = prepare_qpi(tmp_path, constants={**CONSTANTS, key: value})
    assert settings is None
    assert key in error, error


def test_qpi_recipe_is_saved_with_constants_and_recalibrated_in_same_record(tmp_path):
    error, rows, settings = prepare_qpi(tmp_path)
    assert error == "", error
    session = ExtractionSession.create(rows, settings, tmp_path)
    before = session.metadata_path.read_bytes()
    assert session.confirm_calibration({}, {})
    assert not session.can_extract
    assert session.metadata_path.read_bytes() == before
    assert session.confirm_calibration({}, {"QPI": DEFAULT_BACKGROUND}) == ""
    assert session.can_extract
    saved = pd.read_csv(session.metadata_path)
    assert saved["QPI_bg_method"].eq("polynomial").all()
    assert saved["QPI_bg_degree"].eq(4).all()
    assert session.settings["QPI"]["background"] == DEFAULT_BACKGROUND
    confirmed = session.metadata_path.read_bytes()
    session.begin_recalibration()
    assert not session.can_extract
    assert "QPI_bg_method" not in session.metadata_df
    assert "background" not in session.settings["QPI"]
    assert session.metadata_path.read_bytes() == confirmed
    recipe = {"method": "none", "degree": None, "expand_pct": 0.0}
    assert session.confirm_calibration({}, {"QPI": recipe}) == ""
    assert session.can_extract
    assert pd.read_csv(session.metadata_path)["QPI_bg_degree"].isna().all()
    assert len(list(tmp_path.glob("fov_metadata_*.csv"))) == 1


@pytest.mark.parametrize("recipe", [
    {}, {**DEFAULT_BACKGROUND, "degree": None},
    {**DEFAULT_BACKGROUND, "method": "wavelet"},
    {**DEFAULT_BACKGROUND, "expand_pct": float("nan")},
    {**DEFAULT_BACKGROUND, "expand_pct": 150.0},
])
def test_invalid_background_cannot_confirm_or_change_record(tmp_path, recipe):
    error, rows, settings = prepare_qpi(tmp_path)
    assert error == "", error
    session = ExtractionSession.create(rows, settings, tmp_path)
    before = session.metadata_path.read_bytes()
    assert session.confirm_calibration({}, {"QPI": recipe})
    assert not session.can_extract
    assert "background" not in session.settings["QPI"]
    assert session.metadata_path.read_bytes() == before


def test_morphology_only_qpi_needs_no_background_recipe(tmp_path):
    error, rows, settings = prepare_qpi(tmp_path, extractors=["Intensity morphology"])
    assert error == "", error
    assert settings["channels_background"] == []
    assert ExtractionSession.create(rows, settings, tmp_path).can_extract


def test_mixed_calibration_preserves_completed_flim_shift(tmp_path):
    error, rows, settings = prepare_qpi(tmp_path)
    assert error == "", error
    settings["channels_shift"] = {"FLIM": "fit free"}
    rows["FLIM_shift"] = 2.0
    session = ExtractionSession.create(rows, settings, tmp_path)
    assert session.confirm_calibration({}, {"QPI": DEFAULT_BACKGROUND}) == ""
    assert session.can_extract
    assert session.metadata_df["FLIM_shift"].eq(2.0).all()


@pytest.mark.parametrize("missing", ["shift", "background"])
def test_incomplete_mixed_calibration_never_partially_confirms(tmp_path, missing):
    error, rows, settings = prepare_qpi(tmp_path)
    assert error == "", error
    settings["channels_shift"] = {"FLIM": "fit free"}
    session = ExtractionSession.create(rows, settings, tmp_path)
    before = session.metadata_path.read_bytes()
    shifts = {} if missing == "shift" else {"FLIM": 2.0}
    backgrounds = {} if missing == "background" else {"QPI": DEFAULT_BACKGROUND}
    assert session.confirm_calibration(shifts, backgrounds)
    assert not session.can_extract
    assert "FLIM_shift" not in session.metadata_df
    assert "background" not in session.settings["QPI"]
    assert session.metadata_path.read_bytes() == before


def test_background_save_failure_blocks_extraction_until_retry(tmp_path, monkeypatch):
    from src import extraction_session

    error, rows, settings = prepare_qpi(tmp_path)
    assert error == "", error
    session = ExtractionSession.create(rows, settings, tmp_path)
    before = session.metadata_path.read_bytes()
    with monkeypatch.context() as patch:
        def fail(*args):
            raise PermissionError("locked metadata")
        patch.setattr(extraction_session.os, "replace", fail)
        assert "locked metadata" in session.confirm_calibration({}, {"QPI": DEFAULT_BACKGROUND})
        assert not session.can_extract
        assert session.metadata_path.read_bytes() == before
    assert session.save_metadata() == ""
    assert session.can_extract
    assert pd.read_csv(session.metadata_path)["QPI_bg_method"].eq("polynomial").all()
