"""QPI through folder preparation, calibration, and automatic feature export.

Synthetic data only: PTU decays on a 2x3 grid stand in for FLIM, 64x64 wavefront TIFFs for
QPI. The real fixture is exercised by test_qpi_fixture_acceptance.py.
"""
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from qpi_fixtures import (
    CELL1_MASS_PG,
    CONSTANTS,
    QPI_INPUT,
    flim_channel,
    qpi_channel,
    qpi_metadata_rows,
    write_config,
    write_flim_fov,
    write_irf,
    write_qpi_fov,
)
from streamlit.testing.v1 import AppTest
from test_data_extraction_real_dataset import (
    _button,
    _refresh_after_rerun,
)

from src.fov_extraction import fov_extraction
from src.metadata import background_columns, pending_calibration, prepare_extraction
from src.qpi import DEFAULT_BACKGROUND
from src.widgets import metadata_widgets as mw

PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "data_extraction.py")


def _open_step1(folder):
    app = AppTest.from_file(PAGE).run(timeout=60)
    assert not app.exception, [e.value for e in app.exception]
    app.text_input(key="fov_metadata_folder_path").set_value(str(folder)).run(timeout=60)
    assert not app.exception, [e.value for e in app.exception]
    return app


def _export_metadata(app):
    app.button(key="prepare_extraction_button").click().run(timeout=60)
    assert not app.exception, [e.value for e in app.exception]
    _refresh_after_rerun(app)
    return pd.read_csv(app.session_state["prepared_extraction"].metadata_path)


# ---- Folder preparation: grids are compared per modality --------------------------

def test_step1_accepts_flim_and_qpi_channels_on_different_grids(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    write_irf(folder)
    for name in ("fov1", "fov2"):
        write_flim_fov(folder, name, grid=(2, 3))
        write_qpi_fov(folder, name, shape=(64, 64))
    write_config(tmp_path, monkeypatch, [flim_channel("ch1"), qpi_channel("QPI")])
    app = _open_step1(folder)
    assert not app.error, [e.value for e in app.error]
    rows = _export_metadata(app)
    assert rows["image_name"].tolist() == ["fov1", "fov2"]
    assert rows["fov_dimensions"].tolist() == ["(2, 3)", "(2, 3)"]
    assert rows["fov_dimensions_QPI"].tolist() == ["(64, 64)", "(64, 64)"]
    assert rows["QPI_imaging_modality"].eq("QPI").all()
    assert rows["QPI_input_type"].eq(QPI_INPUT).all()
    assert rows["QPI_pixel_size_um"].eq(0.275).all()
    assert rows["QPI_opd_unit"].eq("m").all()
    assert rows["QPI_alpha_um3_per_pg"].eq(0.181818).all()
    assert rows["QPI_Dry-mass statistics"].all() and rows["QPI_Spatial texture"].all()
    assert rows[f"QPI_{QPI_INPUT}"].str.endswith("_WaveFront.tiff").all()
    assert not [c for c in rows if c.startswith("QPI_bg_")]        # no recipe before Step 2


def test_step1_rejects_two_qpi_channels_on_different_grids(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    for name in ("fov1", "fov2"):
        write_qpi_fov(folder, name, shape=(64, 64))
        write_qpi_fov(folder, name, shape=(48, 48), image_suffix="_Phase.tiff", mask_suffix="_Phase_cellpose.tiff")
    write_config(tmp_path, monkeypatch, [
        qpi_channel("QPI"), qpi_channel("Phase", image_suffix="_Phase.tiff", mask_suffix="_Phase_cellpose.tiff")])
    app = _open_step1(folder)
    errors = [e.value for e in app.error]
    assert any("QPI channels" in e and "Phase (48, 48)" in e and "QPI (64, 64)" in e for e in errors), errors


def test_step1_rejects_two_flim_channels_on_different_grids(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    write_irf(folder)
    for name in ("fov1", "fov2"):
        write_flim_fov(folder, name, grid=(2, 3), decay_suffix="a.ptu", mask_suffix="a_mask.tif")
        write_flim_fov(folder, name, grid=(2, 4), decay_suffix="b.ptu", mask_suffix="b_mask.tif")
    write_config(tmp_path, monkeypatch, [
        flim_channel("A", decay_suffix="a.ptu", mask_suffix="a_mask.tif"),
        flim_channel("B", decay_suffix="b.ptu", mask_suffix="b_mask.tif")])
    app = _open_step1(folder)
    errors = [e.value for e in app.error]
    assert any("FLIM channels" in e and "A (2, 3)" in e and "B (2, 4)" in e for e in errors), errors


def test_check_raw_intensity_data_reads_the_named_image_column(tmp_path):
    wavefront, mask_path, _ = write_qpi_fov(tmp_path, "fov1")
    fov_df = pd.DataFrame({"image_name": ["fov1"], f"QPI_{QPI_INPUT}": [str(wavefront)], "QPI_Mask": [str(mask_path)]})
    mw._scan_intensity_images.clear()
    err, dims = mw.check_raw_intensity_data(fov_df, "QPI", image_file_type=QPI_INPUT)
    assert err == "" and dims == (64, 64)
    err, _ = mw.check_raw_intensity_data(fov_df, "QPI")          # the default column is the intensity one
    assert "Intensity (2D) image path not found" in err


# ---- pending calibration -----------------------------------------------------

def test_background_columns_and_pending_with_a_flim_channel():
    assert background_columns("QPI") == ["QPI_bg_method", "QPI_bg_degree", "QPI_bg_expand_pct"]
    rows = pd.DataFrame({"image_name": ["fov1"]})
    settings = {"channels_shift": {"ch1": "fit free"}, "channels_background": ["QPI"], "QPI": {}}
    assert pending_calibration(rows, settings) == (["ch1"], ["QPI"])
    rows["ch1_shift"] = 0.0
    assert pending_calibration(rows, settings) == ([], ["QPI"])
    for col in background_columns("QPI"):
        rows[col] = DEFAULT_BACKGROUND[col.removeprefix("QPI_bg_")]
    # Output columns alone cannot replace the explicit session decision.
    assert pending_calibration(rows, settings) == ([], ["QPI"])
    settings["QPI"]["background"] = dict(DEFAULT_BACKGROUND)
    assert pending_calibration(rows, settings) == ([], [])
    rows = rows.drop(columns=["QPI_bg_expand_pct"])
    assert pending_calibration(rows, settings) == ([], ["QPI"])


# ---- extraction --------------------------------------------------------------

def _extraction_inputs(tmp_path, monkeypatch, names=("fov1",), recipe=DEFAULT_BACKGROUND, extractors=None):
    folder = tmp_path / "data"
    folder.mkdir(exist_ok=True)
    for name in names:
        write_qpi_fov(folder, name)
    extractors = extractors or ["Intensity morphology", "Dry-mass statistics", "Spatial texture"]
    rows = qpi_metadata_rows(folder, list(names), recipe=recipe, extractors=extractors)
    err, info = prepare_extraction(
        rows, {"QPI": {"input_type": QPI_INPUT, "imaging_modality": "QPI",
                       "selected_feature_extractors": extractors, "qpi": dict(CONSTANTS)}},
        fov_name_col="image_name", unique_cell_id_col="cell_id",
    )
    if not err and recipe is not None:
        info["QPI"]["background"] = dict(recipe)
    assert err == "", err
    from src import fov_extraction as fe
    fe.fov_extraction.clear()
    fe.corrected_qpi_image.clear()
    fe.qpi_fov_diagnostics.clear()
    return folder, rows, info


def test_extraction_emits_all_three_qpi_groups(tmp_path, monkeypatch):
    _, rows, info = _extraction_inputs(tmp_path, monkeypatch)
    err, features = fov_extraction(rows.iloc[0], info)
    assert err == "", err
    assert features.index.tolist() == ["fov1_1", "fov1_2"]
    assert features.index.name == "cell_id"
    dry = [c for c in features if c.startswith("Dry-mass statistics_QPI: ")]
    texture = [c for c in features if c.startswith("Spatial texture_QPI: ")]
    morph = [c for c in features if c.startswith("Intensity morphology_QPI: ")]
    assert (len(dry), len(texture), len(morph)) == (7, 3, 7)
    assert {"QPI_centroid_x", "QPI_centroid_y", "image_name"} <= set(features)
    assert features.loc["fov1_1", "Intensity morphology_QPI: area"] == 256
    assert features.loc["fov1_1", "Dry-mass statistics_QPI: dry_mass_pg"] == pytest.approx(CELL1_MASS_PG, rel=0.02)
    # 200 nm of OPD over one pixel: 0.2 um / 0.181818 um^3/pg = 1.1 pg/um^2
    assert features.loc["fov1_1", "Dry-mass statistics_QPI: mass_density_pg_per_um2"] == pytest.approx(0.2 / 0.181818, rel=0.02)
    assert features["Dry-mass statistics_QPI: dry_mass_evenness"].between(0.99, 1.0).all()
    # The diagnostics wrapper sees the same correction: the -40 nm plane is gone afterwards.
    from src.fov_extraction import qpi_fov_diagnostics
    err, diag = qpi_fov_diagnostics(rows.iloc[0]["QPI_QPI (2D)"], rows.iloc[0]["QPI_Mask"], "m",
                                    "polynomial", 4, 15.0)
    assert err == "", err
    assert abs(diag["bg_percentiles_after_nm"][2]) < 1.0


def test_extraction_without_a_recipe_reports_and_skips_the_channel(tmp_path, monkeypatch):
    _, rows, info = _extraction_inputs(tmp_path, monkeypatch, recipe=None)
    shown = []
    monkeypatch.setattr("src.fov_extraction.st.error", lambda msg: shown.append(msg))
    err, features = fov_extraction(rows.iloc[0], info)
    assert err == "", err                                   # morphology still extracted
    assert not [c for c in features if c.startswith("Dry-mass statistics_QPI: ")]
    assert len([c for c in features if c.startswith("Intensity morphology_QPI: ")]) == 7
    assert shown and "Background correction settings for QPI are missing" in shown[0]


def test_extraction_rejects_an_unknown_modality(tmp_path, monkeypatch):
    _, rows, info = _extraction_inputs(tmp_path, monkeypatch)
    info["QPI"]["imaging_modality"] = "Holography"
    err, features = fov_extraction(rows.iloc[0], info)
    assert "Unknown imaging modality 'Holography'" in err and features.empty


@pytest.mark.parametrize("shared_mask", [True, False])
def test_extraction_dedups_morphology_across_channels_sharing_a_mask(tmp_path, monkeypatch, shared_mask):
    folder, rows, info = _extraction_inputs(tmp_path, monkeypatch)
    # A second, intensity-only channel over the same wavefront file.
    other_mask = folder / "fov1_other_cellpose.tiff"
    tifffile.imwrite(other_mask, tifffile.imread(folder / "fov1_WaveFront_cellpose.tiff"))
    row = rows.iloc[0].copy()
    row["red_Intensity (2D)"] = str(folder / "fov1_WaveFront.tiff")
    row["red_Mask"] = row["QPI_Mask"] if shared_mask else str(other_mask)
    row["red_input_type"] = "Intensity (2D)"
    row["red_imaging_modality"] = "Intensity-only"
    row["red_Intensity morphology"] = True
    info = dict(info)
    info["channel_names"] = ["QPI", "red"]
    info["red"] = {"input_type": "Intensity (2D)", "imaging_modality": "Intensity-only",
                   "selected_feature_extractors": ["Intensity morphology"], "input_suffixes": {}}
    err, features = fov_extraction(row, info)
    assert err == "", err
    red_morph = [c for c in features if c.startswith("Intensity morphology_red: ")]
    assert len(red_morph) == (0 if shared_mask else 7)
    assert len([c for c in features if c.startswith("Intensity morphology_QPI: ")]) == 7


def test_rescan_clears_the_correction_cache(tmp_path, monkeypatch):
    folder, rows, info = _extraction_inputs(tmp_path, monkeypatch)
    err, before = fov_extraction(rows.iloc[0], info)
    assert err == "", err
    # Raise only the cell pixels (a whole-image offset would be removed by the fit).
    wavefront = folder / "fov1_WaveFront.tiff"
    raised = tifffile.imread(wavefront).astype(np.float64)
    mask = tifffile.imread(folder / "fov1_WaveFront_cellpose.tiff")
    raised[mask > 0] += 0.1e-6
    tifffile.imwrite(wavefront, raised.astype(np.float32))
    from src import fov_extraction as fe
    fe.fov_extraction.clear()                                # the per-FOV cache alone is not enough
    err, stale = fov_extraction(rows.iloc[0], info)
    col = "Dry-mass statistics_QPI: dry_mass_pg"
    assert stale[col].equals(before[col])                    # correction cache still serves the old image
    mw.clear_folder_scan_caches()
    err, fresh = fov_extraction(rows.iloc[0], info)
    assert err == "", err
    assert (fresh[col] / before[col]).gt(1.3).all()          # 200 nm -> 300 nm cells


# ---- Calibration ------------------------------------------------------------------

def _open_calibration(folder):
    app = _open_step1(folder)
    _export_metadata(app)
    assert not app.exception, [e.value for e in app.exception]
    return app


def _qpi_only_dataset(tmp_path, monkeypatch, names=("fov1", "fov2")):
    folder = tmp_path / "data"
    folder.mkdir()
    for name in names:
        write_qpi_fov(folder, name)
    write_config(tmp_path, monkeypatch, [qpi_channel("QPI")])
    return folder


def test_step2_morphology_only_skips_background_calibration(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    write_qpi_fov(folder, "fov1")
    extractors = ["Intensity morphology"]
    write_config(tmp_path, monkeypatch, [qpi_channel("QPI", extractors=extractors)])
    app = _open_calibration(folder)
    assert not app.error, [e.value for e in app.error]
    assert not [r for r in app.radio if r.key == "qpi_method_QPI"]
    assert not app.get("plotly_chart")
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert not set(background_columns("QPI")) & set(app.session_state["prepared_extraction"].metadata_df.columns)
    features = pd.read_csv(next(folder.glob("single_cell_features_*.csv")), index_col=0)
    assert len(features) == 2
    assert features.loc["fov1_1", "Intensity morphology_QPI: area"] == 256
    assert features.loc["fov1_2", "Intensity morphology_QPI: area"] == 400
    assert not any(c.startswith(("Dry-mass statistics_", "Spatial texture_")) for c in features)


def _trace_values(values):
    """Plotly writes numpy arrays into the chart spec as ``{"dtype", "bdata"}`` typed arrays."""
    if isinstance(values, dict):
        return np.frombuffer(base64.b64decode(values["bdata"]), dtype=np.dtype(values["dtype"]))
    return np.asarray(values, dtype=float)


def test_step2_invalid_background_pixels_preserve_preview_and_cell_masses(tmp_path, monkeypatch):
    folder = _qpi_only_dataset(tmp_path, monkeypatch, names=("fov1",))
    wavefront = folder / "fov1_WaveFront.tiff"
    image = tifffile.imread(wavefront)
    image[0, :3] = [np.nan, np.inf, -np.inf]
    tifffile.imwrite(wavefront, image)

    app = _open_calibration(folder)
    assert not app.error, [e.value for e in app.error]
    charts = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    assert np.isfinite(np.asarray(charts[0]["data"][0]["median"], dtype=float)).all()
    assert [t["name"] for t in charts[1]["data"]] == ["raw", "surface", "corrected"]  # one axis, no mass rescale
    for trace in charts[1]["data"]:                                            # the line scan stays finite
        assert np.isfinite(_trace_values(trace["y"])).all(), trace["name"]
    _button(app, "Confirm calibration for each channel").click().run(timeout=120)
    _refresh_after_rerun(app)
    _button(app, "Start extraction").click().run(timeout=60)
    _refresh_after_rerun(app)
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    features = pd.read_csv(next(folder.glob("single_cell_features_*.csv")), index_col=0)
    masses = features["Dry-mass statistics_QPI: dry_mass_pg"]
    assert len(features) == 2 and np.isfinite(masses).all()
    assert masses.loc["fov1_1"] == pytest.approx(CELL1_MASS_PG, rel=0.02)
    assert masses.loc["fov1_2"] == pytest.approx(CELL1_MASS_PG * 400 / 256, rel=0.02)


def test_step2_qpi_only_corrects_background_and_extracts(tmp_path, monkeypatch):
    folder = _qpi_only_dataset(tmp_path, monkeypatch)
    app = _open_calibration(folder)
    assert not app.error, [e.value for e in app.error]
    assert not [b for b in app.button if b.label in ("Optimize for Shifts", "Calibrate channels", "Correct background")]
    assert not [c for c in app.checkbox if c.label == "Fix the Shift"]           # no FLIM pending
    assert app.session_state["prepared_extraction"].choosing_shift is True                          # QPI-only opens the block without a gate button
    assert app.radio(key="qpi_method_QPI").value == "polynomial"
    assert app.selectbox(key="qpi_degree_QPI").value == 4
    assert app.number_input(key="qpi_expand_QPI").value == 15.0
    assert app.selectbox(key="qpi_cell_QPI_0").value == 2                       # largest cell preselected
    assert not [w for w in app.warning if "kept" in w.value]                    # kept fraction is fine on synthetic data
    assert [m for m in app.get("metric") if m.proto.label == "Fit kept"]        # the polynomial method fits something
    mass = [m for m in app.get("metric") if m.proto.label == "Cell dry mass"]   # what the QPI constants add up to
    assert len(mass) == 1
    assert float(mass[0].proto.body.removesuffix(" pg")) == pytest.approx(CELL1_MASS_PG * 400 / 256, rel=0.05)
    _button(app, "Confirm calibration for each channel").click().run(timeout=120)
    _refresh_after_rerun(app)
    saved = app.session_state["prepared_extraction"].metadata_df
    for col, expected in zip(background_columns("QPI"), ("polynomial", 4, 15.0, 0)):
        assert saved[col].eq(expected).all(), col
    assert [b for b in app.button if b.label == "Start extraction"]
    _button(app, "Go back and correct background").click().run(timeout=120)
    _refresh_after_rerun(app)
    assert not set(background_columns("QPI")) & set(app.session_state["prepared_extraction"].metadata_df.columns)
    assert app.radio(key="qpi_method_QPI").value == "polynomial"                  # the block is open again
    app.radio(key="qpi_method_QPI").set_value("none").run(timeout=120)
    assert not [s for s in app.selectbox if s.key == "qpi_degree_QPI"]           # degree only for polynomial
    # "none" and inpainting report a hard-coded kept fraction, so a confident 100 % would
    # claim a check that never ran; only the negative-interior readout stays.
    assert not [m for m in app.get("metric") if m.proto.label == "Fit kept"]
    assert [m for m in app.get("metric") if m.proto.label == "Negative pixels inside cells"]
    _button(app, "Confirm calibration for each channel").click().run(timeout=120)
    _refresh_after_rerun(app)
    saved = app.session_state["prepared_extraction"].metadata_df
    assert saved["QPI_bg_method"].eq("none").all() and saved["QPI_bg_degree"].isna().all()
    _button(app, "Start extraction").click().run(timeout=300)
    _refresh_after_rerun(app)
    assert not app.error, [e.value for e in app.error]
    features = pd.read_csv(next((folder).glob("single_cell_features_*.csv")))
    assert len(features) == 4                                                   # 2 FOVs x 2 cells
    assert "Dry-mass statistics_QPI: dry_mass_pg" in features
    assert "Spatial texture_QPI: radial_mass_index" in features
    assert "Intensity morphology_QPI: area" in features
    # "none" leaves the -40 nm plane in place, so the masses come out low but finite.
    assert np.isfinite(features["Dry-mass statistics_QPI: dry_mass_pg"]).all()


@pytest.mark.parametrize("method", ["polynomial", "none"])
def test_recalibration_preserves_nondefault_qpi_controls(tmp_path, monkeypatch, method):
    folder = _qpi_only_dataset(tmp_path, monkeypatch, names=("fov1",))
    app = _open_calibration(folder)
    if method == "none":
        app.radio(key="qpi_method_QPI").set_value(method).run(timeout=120)
    else:
        app.selectbox(key="qpi_degree_QPI").set_value(6)
        app.number_input(key="qpi_expand_QPI").set_value(30.0).run(timeout=120)
    _button(app, "Confirm calibration for each channel").click().run(timeout=120)
    _refresh_after_rerun(app)
    prepared = app.session_state["prepared_extraction"]
    saved = prepared.metadata_path.read_bytes()
    _button(app, "Go back and correct background").click().run(timeout=120)
    _refresh_after_rerun(app)
    assert app.radio(key="qpi_method_QPI").value == method
    if method == "polynomial":
        assert app.selectbox(key="qpi_degree_QPI").value == 6
        assert app.number_input(key="qpi_expand_QPI").value == 30.0
    assert not prepared.can_extract
    assert "background" not in prepared.settings["QPI"]
    assert prepared.metadata_path.read_bytes() == saved


def test_unconfirmed_qpi_controls_survive_switching_steps(tmp_path, monkeypatch):
    folder = _qpi_only_dataset(tmp_path, monkeypatch, names=("fov1",))
    app = _open_calibration(folder)
    app.radio(key="qpi_method_QPI").set_value("inpainting").run(timeout=120)
    app.number_input(key="qpi_expand_QPI").set_value(30.0).run(timeout=120)
    prepared = app.session_state["prepared_extraction"]
    saved = prepared.metadata_path.read_bytes()
    app.radio[0].set_value("**Categorical** (e.g. treatment, day)").run(timeout=120)
    app.radio[0].set_value("**Numerical** (e.g. lifetime, morphology)").run(timeout=120)
    assert not app.exception, [e.value for e in app.exception]
    assert app.radio(key="qpi_method_QPI").value == "inpainting"
    assert app.number_input(key="qpi_expand_QPI").value == 30.0
    assert not prepared.can_extract
    assert "background" not in prepared.settings["QPI"]
    assert prepared.metadata_path.read_bytes() == saved


def test_step2_renders_only_the_pending_channels(tmp_path, monkeypatch):
    """A CSV with the FLIM shift present and no QPI recipe must not rerun any shift fit."""
    folder = tmp_path / "data"
    folder.mkdir()
    write_irf(folder)
    for name in ("fov1", "fov2"):
        write_flim_fov(folder, name)
        write_qpi_fov(folder, name)
    write_config(tmp_path, monkeypatch, [flim_channel("ch1"), qpi_channel("QPI")])
    app = _open_step1(folder)
    _export_metadata(app)
    app.session_state["prepared_extraction"].metadata_df["ch1_shift"] = 0.0

    def _boom(*args, **kwargs):
        raise AssertionError("shift optimization must not run for a calibrated channel")

    monkeypatch.setattr("src.widgets.lifetime_widgets.choose_shift_fit_free", _boom)
    monkeypatch.setattr("src.widgets.lifetime_widgets.choose_shift_fit", _boom)
    app.run(timeout=120)
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert not [c for c in app.checkbox if c.label == "Fix the Shift"]
    assert not [b for b in app.button if b.label in ("Optimize for Shifts", "Calibrate channels", "Correct background")]
    assert app.session_state["prepared_extraction"].choosing_shift is True                          # opened without a gate button
    assert [r for r in app.radio if r.key == "qpi_method_QPI"]
    assert len(app.get("plotly_chart")) == 2                  # box plot, line scan; no shift plot
    # Axis titles and ticks follow the theme on every axis of both charts. AppTest reports
    # no browser theme, so the helper returns the dark colour.
    for chart in app.get("plotly_chart"):
        layout = json.loads(chart.proto.spec)["layout"]
        axes = [key for key in layout if key.startswith(("xaxis", "yaxis"))]
        assert len(axes) >= 2, axes
        for key in axes:
            assert layout[key]["title"]["font"]["color"] == "white", (key, layout[key])
            assert layout[key]["tickfont"]["color"] == "white", (key, layout[key])
    _button(app, "Confirm calibration for each channel").click().run(timeout=120)
    _refresh_after_rerun(app)
    saved = app.session_state["prepared_extraction"].metadata_df
    assert saved["ch1_shift"].eq(0.0).all()                                     # untouched
    assert set(background_columns("QPI")) <= set(saved.columns)
    assert [b for b in app.button if b.label == "Go back and recalibrate"]
    _button(app, "Start extraction").click().run(timeout=300)
    _refresh_after_rerun(app)
    assert not app.error, [e.value for e in app.error]
    features = pd.read_csv(next(folder.glob("single_cell_features_*.csv")))
    assert "Lifetime fit free_ch1: G(1st)" in features
    assert "Dry-mass statistics_QPI: dry_mass_pg" in features


def test_step2_labels_when_both_are_pending(tmp_path, monkeypatch):
    folder = tmp_path / "data"
    folder.mkdir()
    write_irf(folder)
    for name in ("fov1", "fov2"):
        write_flim_fov(folder, name)
        write_qpi_fov(folder, name)
    write_config(tmp_path, monkeypatch, [flim_channel("ch1"), qpi_channel("QPI")])
    app = _open_calibration(folder)
    assert [c for c in app.checkbox if c.label == "Fix the Shift"]
    _button(app, "Calibrate channels").click().run(timeout=300)
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]                          # the fit-free shift on this PTU is known to succeed (test_ptu_reference_workflow)
    assert [r for r in app.radio if r.key == "qpi_method_QPI"]
    assert [b for b in app.button if b.label == "Confirm calibration for each channel"]
    assert len(app.get("plotly_chart")) >= 3                  # the two QPI charts plus the ch1 shift plot
    # Each channel's block sits in its own open expander so a finished channel can be
    # collapsed to make room for the others; the single Confirm stays outside them.
    settings = next(e for e in app.expander if e.label == "Metadata settings")
    nested = {e.label for e in settings.get("expander")}
    blocks = {e.label: e for e in app.expander if e.label != settings.label and e.label not in nested}
    assert set(blocks) == {"ch1: shift calibration", "QPI: background correction"}
    assert all(e.proto.expanded for e in blocks.values())
    assert [r for r in blocks["QPI: background correction"].radio if r.key == "qpi_method_QPI"]
    assert blocks["ch1: shift calibration"].get("plotly_chart")
    assert not [b for e in blocks.values() for b in e.button if b.label == "Confirm calibration for each channel"]
    assert not [m for m in app.markdown if m.value == "**QPI**: background correction"]     # the label replaces the heading


def test_step2_keeps_a_channel_pending_when_its_image_cannot_be_read(tmp_path, monkeypatch):
    """An unreadable wavefront shows an error; Confirm then writes no recipe and the block stays open."""
    folder = _qpi_only_dataset(tmp_path, monkeypatch)
    app = _open_calibration(folder)
    (folder / "fov2_WaveFront.tiff").write_bytes(b"not a tiff")
    mw.clear_folder_scan_caches()
    app.run(timeout=120)
    assert any("fov2_WaveFront.tiff" in e.value for e in app.error), [e.value for e in app.error]
    _button(app, "Confirm calibration for each channel").click().run(timeout=120)
    _refresh_after_rerun(app)
    assert not set(background_columns("QPI")) & set(app.session_state["prepared_extraction"].metadata_df.columns)
    assert app.session_state["prepared_extraction"].choosing_shift is True                          # still pending, block reopened
    assert not [b for b in app.button if b.label == "Start extraction"]
    assert any("fov2_WaveFront.tiff" in e.value for e in app.error)
