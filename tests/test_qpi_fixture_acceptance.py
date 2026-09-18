"""QPI acceptance against the neutrophil fixture and the lab's cleaned modules.

Set FLIM_REAL_DATA_ROOT to the flim_playground_example_data folder (the fixture is its
QPI_neutrophil subfolder; its README is authoritative). Set QPI_LAB_MODULES to a folder holding
the lab's cleaned qpi_calibration.py and qpi_single_cell_measurements.py to run the oracle tests
too. Thresholds are the values measured on 2026-09-15 (plan, "Measured facts"), not the spec's.
"""
import importlib.util
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import toml
from scipy.ndimage import binary_dilation
from skimage.measure import profile_line
from streamlit.testing.v1 import AppTest
from test_data_extraction_real_dataset import (
    _button,
    _refresh_after_rerun,
)

from src import config, qpi

_ROOT = Path(__file__).resolve().parents[1]
_PIXEL = 0.275
_ALPHA = 0.181818
_LAB_ALPHA = 1 / 5.5          # the lab's hard-coded 5_500_000 multiplier implies this alpha
_FOVS = ("2026-08-13_Neut_Control_spor_105m_1", "2026-08-13_Neut_Control_spor_105m_2",
         "2026-08-13_Neut_Crypto_spor_105m_1", "2026-08-13_Neut_Crypto_spor_105m_2")
_CELL_COUNTS = {"2026-08-13_Neut_Control_spor_105m_1": 71, "2026-08-13_Neut_Control_spor_105m_2": 73,
                "2026-08-13_Neut_Crypto_spor_105m_1": 79, "2026-08-13_Neut_Crypto_spor_105m_2": 59}
_SPOT = "2026-08-13_Neut_Crypto_spor_105m_1"
# Our suffix -> the lab CSV column. The gradient is compared separately (the lab took abs()
# of the bounding-box crop, so neighbouring negative pixels change it slightly).
_LAB_COLUMNS = {"dry_mass_pg": "dry_mass_pg", "dry_mass_variance": "dry_mass_variance",
                "dry_mass_skewness": "dry_mass_skewness", "dry_mass_kurtosis": "dry_mass_kurtosis",
                "dry_mass_entropy": "dry_mass_entropy", "dry_mass_evenness": "dry_mass_entropy_normalizedperpixel",
                "radial_mass_index": "radial_mass_index", "polar_gradient_index": "polar_gradient_index"}


@pytest.fixture(scope="module")
def fixture_root():
    configured = os.environ.get("FLIM_REAL_DATA_ROOT")
    if not configured:
        pytest.skip("Set FLIM_REAL_DATA_ROOT to the external example-data folder.")
    root = Path(configured).expanduser().resolve() / "QPI_neutrophil"
    assert root.is_dir(), root
    return root


@pytest.fixture(scope="module")
def fovs(fixture_root):
    """{stem: (raw_um, mask, reference_um)} for the four FOVs; files hold metres."""
    out = {}
    for stem in _FOVS:
        raw = qpi.to_um(tifffile.imread(fixture_root / f"{stem}_WaveFront.tiff"), "m")
        mask = tifffile.imread(fixture_root / f"{stem}_WaveFront_cellpose.tiff")
        ref = qpi.to_um(tifffile.imread(fixture_root / "reference" / "bg_corrected" / f"{stem}_WaveFront.tiff"), "m")
        assert raw.shape == mask.shape == ref.shape == (552, 552)
        out[stem] = (raw, mask, ref)
    return out


@pytest.fixture(scope="module")
def lab_csv(fixture_root):
    frames = [pd.read_csv(p) for p in sorted((fixture_root / "reference").glob("all_cell_stats_*.csv"))]
    csv = pd.concat(frames, ignore_index=True)
    csv["stem"] = csv["image_name"].str.replace(r"_\d{8}_WaveFront$", "", regex=True)
    return csv


@pytest.fixture(scope="module")
def lab():
    """The lab's cleaned modules, imported by path (qpi_single_cell_measurements imports qpi_calibration by name)."""
    folder = os.environ.get("QPI_LAB_MODULES")
    if not folder:
        pytest.skip("Set QPI_LAB_MODULES to the folder holding qpi_calibration.py and qpi_single_cell_measurements.py.")
    folder = Path(folder).expanduser().resolve()
    modules = {}
    sys.path.insert(0, str(folder))
    try:
        for name in ("qpi_calibration", "qpi_single_cell_measurements"):
            spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            modules[name] = module
    finally:
        sys.path.remove(str(folder))
    return modules


def _masses(image_um, mask, alpha=_ALPHA):
    return {label: f["dry_mass_pg"] for label, f in qpi.cell_features(image_um, mask, _PIXEL, alpha).items()}


def test_exclusion_at_15_percent_is_two_dilations_on_every_fov(fovs):
    for stem, (_, mask, _) in fovs.items():
        expected = binary_dilation(mask > 0, iterations=2)
        assert np.array_equal(qpi.exclusion_region(mask, 15.0), expected), stem
        assert not np.array_equal(qpi.exclusion_region(mask, 5.0), expected), stem   # one dilation stays under 15 %


@pytest.mark.parametrize(("degree", "median_limit", "q25_floor", "q75_ceiling"), [
    (6, 1.5, -1.0, 2.5),      # the reference's own polynomial order
    (4, 2.0, -2.5, 3.5),      # the app default
])
def test_masses_track_the_reference_correction(fovs, degree, median_limit, q25_floor, q75_ceiling):
    settings = {"method": "polynomial", "degree": degree, "expand_pct": 15.0}
    for stem, (raw, mask, ref) in fovs.items():
        corrected, _, _, info = qpi.correct_background(raw, mask, settings)
        ours, theirs = _masses(corrected, mask), _masses(ref, mask)
        pct = np.array([(ours[label] - theirs[label]) / theirs[label] * 100 for label in theirs])
        assert len(pct) == _CELL_COUNTS[stem]
        assert abs(np.median(pct)) < median_limit, (stem, np.median(pct))
        assert np.percentile(pct, 25) > q25_floor, (stem, np.percentile(pct, 25))
        assert np.percentile(pct, 75) < q75_ceiling, (stem, np.percentile(pct, 75))
        assert np.abs(pct).max() < 15.0, (stem, np.abs(pct).max())
        assert info["kept_fraction"] > qpi.KEPT_FRACTION_WARNING, (stem, info)
        assert 1 <= info["rounds"] <= 10


def test_label_5_matches_the_lab_csv_on_the_reference_image(fovs, lab_csv):
    """Our signed feature math on the lab's own corrected TIFF reproduces the lab's CSV row."""
    _, mask, ref = fovs[_SPOT]
    row = lab_csv[(lab_csv["stem"] == _SPOT) & (lab_csv["cell_id"] == 5)].iloc[0]
    assert row["area_pixels"] == 961
    assert not (ref[mask == 5] < 0).any()                    # so abs() and signed agree on this cell
    ours = qpi.cell_features(ref, mask, _PIXEL, _LAB_ALPHA)[5]
    assert ours["dry_mass_pg"] == pytest.approx(row["dry_mass_pg"], rel=1e-6)
    for suffix, column in _LAB_COLUMNS.items():
        assert ours[suffix] == pytest.approx(row[column], rel=1e-5), suffix
    assert ours["avg_dm_gradient_mag"] == pytest.approx(row["avg_dm_gradient_mag"], rel=1e-2)
    # Lab density is pg per pixel; ours is pg per um^2.
    assert ours["mass_density_pg_per_um2"] == pytest.approx(row["size_norm_dry_mass"] / _PIXEL**2, rel=1e-5)
    # The default alpha moves the same cell by one part per million, hence 1/5.5 above.
    assert qpi.cell_features(ref, mask, _PIXEL, _ALPHA)[5]["dry_mass_pg"] == pytest.approx(row["dry_mass_pg"], rel=2e-6)


def test_lab_oracle_exclusion_and_inpainting(fovs, lab):
    calibration = lab["qpi_calibration"]
    raw_um, mask, _ = fovs[_SPOT]
    raw_m = raw_um * 1e-6
    for pct in (0, 15, 30):
        assert np.array_equal(qpi.exclusion_region(mask, pct), calibration.expand_mask_by_area(mask, pct)), pct
    theirs_corr, theirs_surface, theirs_excl = calibration.fit_inpainted_background(raw_m, mask, dilate_area_pct=15)
    ours_corr, ours_surface, ours_excl, _ = qpi.correct_background(
        raw_um, mask, {"method": "inpainting", "degree": None, "expand_pct": 15.0})
    assert np.array_equal(ours_excl, theirs_excl)
    np.testing.assert_allclose(ours_surface * 1e-6, theirs_surface, rtol=1e-6, atol=1e-15)
    np.testing.assert_allclose(ours_corr * 1e-6, theirs_corr, rtol=1e-6, atol=1e-15)


def test_real_cell_line_scan_follows_its_principal_axis(fovs):
    """Derive the axis from coordinate covariance; the lab scan mirrors rotated cells."""
    raw_um, mask, corrected_um = fovs[_SPOT]
    coords = np.column_stack(np.nonzero(mask == 5))
    centre = coords.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(coords, rowvar=False, bias=True))
    direction = eigenvectors[:, -1]
    if direction[0] < 0:
        direction = -direction
    half_axis = 2 * np.sqrt(eigenvalues[-1]) * direction
    start, end = centre - half_axis, centre + half_axis
    raw_profile = profile_line(raw_um, start, end, order=1, mode="constant", cval=0)
    corrected_profile = profile_line(corrected_um, start, end, order=1, mode="constant", cval=0)
    inside = profile_line((mask == 5).astype(float), start, end, order=0, mode="constant", cval=0) > 0.5

    scan = qpi.line_scan(raw_um, corrected_um, raw_um - corrected_um, mask, 5, _PIXEL, _ALPHA)
    np.testing.assert_allclose(scan["raw_nm"], raw_profile * 1e3, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(scan["corrected_nm"], corrected_profile * 1e3, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(scan["surface_nm"], (raw_profile - corrected_profile) * 1e3, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(scan["mass_pg"], corrected_profile * _PIXEL**2 / _ALPHA, rtol=1e-10, atol=1e-10)
    np.testing.assert_array_equal(scan["inside"], inside)


def test_lab_oracle_cell_stats_agree_on_cells_without_negative_pixels(fovs, lab):
    """On cells with no negative pixel, signed and abs() math coincide, so every statistic must match."""
    measurements = lab["qpi_single_cell_measurements"]
    _, mask, ref = fovs[_SPOT]
    theirs = {r["cell_id"]: r for r in measurements.compute_cell_stats(
        ref * 1e-6, mask, _SPOT, "reference", pixel_size_um=_PIXEL, alpha_um3_per_pg=_ALPHA)}
    ours = qpi.cell_features(ref, mask, _PIXEL, _ALPHA)
    compared = 0
    for label, f in ours.items():
        pixels = ref[mask == label]
        if (pixels < 0).any() or pixels.size < 4:
            continue
        compared += 1
        for suffix, column in _LAB_COLUMNS.items():
            assert f[suffix] == pytest.approx(theirs[label][column], rel=1e-9, abs=1e-12), (label, suffix)
    assert compared >= 1, "no cell without negative pixels on the reference image"


def test_full_workflow_flim_plus_qpi_on_the_fixture(fixture_root, tmp_path, monkeypatch):
    """Configuration page -> folder preparation -> mixed calibration -> extraction, on the real fixture."""
    data_dir = tmp_path / "data"
    shutil.copytree(fixture_root, data_dir, ignore=shutil.ignore_patterns("reference", "README.md", "*f.sdt", "Ch1_*", "Ch2_*"))
    config_path = tmp_path / "config.toml"
    monkeypatch.setattr(config, "_CONFIG_PATH", config_path)

    def run(app, timeout=600):
        app.run(timeout=timeout)
        assert not app.exception, [e.value for e in app.exception]
        return app

    raw = "Decay (3/4D)"
    app = run(AppTest.from_file(str(_ROOT / "main.py")))
    app.selectbox(key="num_channels_default").set_value(2)
    run(app)
    app.text_input(key="channel_name_ch1_default").set_value("NADH")
    app.text_input(key="channel_name_ch2_default").set_value("QPI")
    app.selectbox(key="imaging_modality_ch2_default").set_value("QPI")
    run(app)
    app.multiselect(key=f"{raw}_ch1_feature_extractors_default").set_value(["Lifetime fit free"])
    app.multiselect(key="QPI (2D)_ch2_feature_extractors_default").set_value(
        ["Intensity morphology", "Dry-mass statistics", "Spatial texture"])
    run(app)
    app.number_input(key=f"laser_rate_{raw}_default_mhz").set_value(80.0)
    app.radio(key=f"fit_free_calibration_{raw}_default").set_value("IRF")
    for file_type, suffix in {"Decay": "n.sdt", "IRF": "Ch3_IRF_750__2024.txt", "Mask": "n_cellpose.tiff"}.items():
        app.text_input(key=f"ch1_{raw}_{file_type}_default").set_value(suffix)
    app.text_input(key="ch2_QPI (2D)_QPI (2D)_default").set_value("_WaveFront.tiff")
    app.text_input(key="ch2_QPI (2D)_Mask_default").set_value("_WaveFront_cellpose.tiff")
    app.number_input(key="ch2_QPI (2D)_pixel_size_um_default").set_value(_PIXEL)
    app.selectbox(key="ch2_QPI (2D)_opd_unit_default").set_value("m")
    run(app)
    assert not app.error, [e.value for e in app.error]
    _button(app, "Update Configuration").click()
    run(app)
    saved = toml.load(config_path)["profiles"]["default"]
    assert saved["ch2"]["QPI (2D)"]["pixel_size_um"] == _PIXEL
    assert saved["ch2"]["QPI (2D)"]["opd_unit"] == "m"
    assert saved["ch2"]["QPI (2D)"]["alpha_um3_per_pg"] == pytest.approx(_ALPHA)

    app.switch_page("pages/data_extraction.py")
    run(app)
    app.text_input(key="fov_metadata_folder_path").set_value(str(data_dir))
    run(app)
    app.selectbox(key="NADH_channel_selectbox").set_value(3)      # NADH is the third acquisition channel of n.sdt
    run(app)
    assert not app.error, [e.value for e in app.error]
    app.button(key="prepare_extraction_button").click()
    run(app)
    _refresh_after_rerun(app)
    metadata_path = app.session_state["prepared_extraction"].metadata_path
    metadata = pd.read_csv(metadata_path)
    assert set(metadata["image_name"]) == set(_FOVS)
    assert metadata["fov_dimensions"].eq("(256, 256)").all()
    assert metadata["fov_dimensions_QPI"].eq("(552, 552)").all()
    assert metadata["QPI_pixel_size_um"].eq(_PIXEL).all() and metadata["QPI_opd_unit"].eq("m").all()
    assert metadata["laser_rate"].eq(0.08).all() and metadata["time_bins"].eq(256).all()

    _button(app, "Calibrate channels").click()
    run(app)
    assert not app.error, [e.value for e in app.error]
    assert not [w for w in app.warning if "kept only" in w.value]
    first_mask = tifffile.imread(data_dir / f"{metadata['image_name'].iloc[0]}_WaveFront_cellpose.tiff")
    labels, counts = np.unique(first_mask[first_mask > 0], return_counts=True)
    assert app.selectbox(key="qpi_cell_QPI_0").value == int(labels[np.argmax(counts)])
    _button(app, "Confirm calibration for each channel").click()
    run(app)
    _refresh_after_rerun(app)
    assert not app.error, [e.value for e in app.error]
    metadata = pd.read_csv(metadata_path)
    assert np.isfinite(metadata["NADH_shift"]).all()
    assert metadata["QPI_bg_method"].eq("polynomial").all() and metadata["QPI_bg_degree"].eq(4).all()
    assert metadata["QPI_bg_expand_pct"].eq(15.0).all()

    _button(app, "Start extraction").click()
    run(app)
    _refresh_after_rerun(app)
    assert not app.error, [e.value for e in app.error]
    features = pd.read_csv(next(data_dir.glob("single_cell_features_*.csv")), index_col=0)
    assert features.groupby("image_name").size().to_dict() == _CELL_COUNTS
    spot = features.loc[f"{_SPOT}_5"]
    assert spot["Intensity morphology_QPI: area"] == 961
    assert spot["Dry-mass statistics_QPI: dry_mass_pg"] == pytest.approx(101.684, rel=0.03)   # default recipe: measured -0.44 %
    assert np.isfinite(spot["Spatial texture_QPI: radial_mass_index"])
    assert np.isfinite(spot["Lifetime fit free_NADH: G(1st)"])
    assert "Intensity morphology_NADH: area" not in features       # NADH selected no morphology
    assert features["Dry-mass statistics_QPI: dry_mass_pg"].gt(0).all()
