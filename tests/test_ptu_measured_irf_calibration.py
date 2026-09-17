"""Synthetic PTU and extraction records for the actual quenched-Atto IRF.

Opt in with FLIM_PTU_REFERENCE_ROOT pointing to the supplied Atto folder.
Keep pytest's --basetemp directory to inspect or copy the generated dataset.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from ptufile import imwrite
import pytest
from streamlit.testing.v1 import AppTest
import tifffile
import toml

from src import config
from src.choose_shift import choose_shift_fit_free
from src.decay_io import read_decay_with_frames
from src.file_io import get_decay_curves, get_irf
from src.fit_helper import irf_shift
from src.fov_extraction import fov_extraction
from src.metadata import prepare_extraction
from test_data_extraction_real_dataset import _refresh_after_rerun
from test_ptu_measured_standard_calibration import _digest, _read_measured_reference


IRF_NAME = "Quenched Atto 488 500 nM PI 5 uM high repetition.ptu"
INPUT_TYPE = "Decay (3/4D)"
PREFIX = "Lifetime fit free_dye: "


def _bin_integrated_exponential(times, bin_width, tau):
    """Periodic exponential delays assigned to their nearest bin center.

    Bin zero contains the first half-bin and the last half-bin of the period.
    This keeps quantization centered on the phase calculation's n * dt grid.
    """
    period = len(times) * bin_width
    denominator = -np.expm1(-period / tau)
    weights = (
        np.exp(-(times - bin_width / 2) / tau)
        - np.exp(-(times + bin_width / 2) / tau)
    ) / denominator
    weights[0] = (
        -np.expm1(-bin_width / 2 / tau)
        + np.exp(-(period - bin_width / 2) / tau)
        - np.exp(-period / tau)
    ) / denominator
    assert weights.min() >= 0
    np.testing.assert_allclose(weights.sum(), 1, rtol=1e-12)
    return weights


def _phasor(curve, basis):
    counts = np.asarray(curve, dtype=np.float64)
    return counts @ basis / counts.sum()


def _lifetimes(phasor, omega):
    first = phasor[0]
    return float(first.imag / first.real / omega), float(np.sqrt(1 / abs(first)**2 - 1) / omega)


def _click(app, label):
    next(button for button in app.button if button.label == label).click().run(timeout=45)
    assert not app.exception
    assert not app.error, [error.value for error in app.error]


def test_synthetic_ptu_calibrates_against_measured_irf(tmp_path, monkeypatch):
    root = os.environ.get("FLIM_PTU_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set FLIM_PTU_REFERENCE_ROOT to the supplied Atto folder.")
    irf_path = Path(root).resolve() / IRF_NAME
    before_digest = _digest(irf_path)
    independent_irf, frequency_hz, resolution_s = _read_measured_reference(irf_path)
    bins = len(independent_irf)
    bin_width = resolution_s * 1e9
    times = np.arange(bins) * bin_width
    duration = bins * bin_width
    omega = 2 * np.pi * frequency_hz * 1e-9
    basis = np.exp(1j * times[:, None] * omega * np.array([1, 2]))
    normalized_irf = independent_irf / independent_irf.sum()
    nominal_lifetimes = np.array([4.16, 6.24, 8.32])
    profiles = []
    for tau in nominal_lifetimes:
        kernel = _bin_integrated_exponential(times, bin_width, tau)
        # Direct circular convolution, independent of the application's FFT
        # fitting model and its phasor calibration implementation.
        profile = sum(weight * np.roll(normalized_irf, index)
                      for index, weight in enumerate(kernel))
        assert profile.min() >= 0
        profiles.append(profile / profile.sum())

    mask = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8)
    seed = 20260912
    rng = np.random.default_rng(seed)
    counts = np.empty((3, 2, 3, 1, bins), dtype=np.uint32)
    for frame, y, x in np.ndindex(counts.shape[:3]):
        photons = 100_000 * (1 + 0.3 * frame + 0.2 * y + 0.1 * x)
        counts[frame, y, x, 0] = rng.poisson(photons * profiles[mask[y, x] - 1])
    sample_path = tmp_path / "synthetic_irf_sample.ptu"
    mask_path = tmp_path / "synthetic_irf_sample_mask.tif"
    imwrite(sample_path, counts, global_resolution=1 / frequency_hz,
            tcspc_resolution=resolution_s)
    tifffile.imwrite(mask_path, mask)
    expected_image = counts.sum(axis=0, dtype=np.uint64)[:, :, 0, :]
    error, decoded = read_decay_with_frames(str(sample_path))
    assert error == ""
    assert decoded[1] == 3
    np.testing.assert_array_equal(decoded[0], expected_image)

    cfg_path = tmp_path / "test_config.toml"
    cfg_path.write_text(toml.dumps({
        "num_channels": 1, "flim_decay_input_type": INPUT_TYPE,
        "fov_name_col": "image_name", "unique_cell_id_col": "cell_id",
        INPUT_TYPE: {
            "file_types": ["Decay", "Mask", "IRF"],
            "available_feature_extractors": ["Lifetime fit free"],
            "laser_rate": frequency_hz * 1e-9, "fit_free_calibration": "IRF",
        },
        "ch1": {
            "channel_name": "dye", "input_type": INPUT_TYPE,
            "imaging_modality": "FLIM",
            INPUT_TYPE: {
                "selected_feature_extractors": ["Lifetime fit free"],
                "input_suffixes": {"Decay": ".ptu", "Mask": "_mask.tif", "IRF": IRF_NAME},
            },
        },
    }))
    monkeypatch.setattr(config, "_CONFIG_PATH", cfg_path)
    rows = pd.DataFrame([{
        "image_name": "synthetic_irf_sample", "dye_input_type": INPUT_TYPE,
        "dye_imaging_modality": "FLIM", "dye_Lifetime fit free": True,
        "dye_Decay": str(sample_path), "dye_Mask": str(mask_path),
        "dye_channel": -1, "time_bins": bins, "duration": duration,
        "laser_rate": frequency_hz * 1e-9,
        "fit_free_calibration_method": "IRF", "dye_IRF": str(irf_path),
    }])
    def prepare(rows):
        return prepare_extraction(
            rows, {"dye": {"input_type": INPUT_TYPE, "imaging_modality": "FLIM",
                           "selected_feature_extractors": ["Lifetime fit free"]}},
            fov_name_col="image_name", unique_cell_id_col="cell_id",
            laser_rate=frequency_hz * 1e-9, fit_free_calibration_method="IRF",
        )

    error, metadata = prepare(rows)
    assert error == ""
    assert metadata["channels_shift"] == {"dye": "fit free"}
    assert metadata["fit_free_calibration_method"] == "IRF"
    error, loaded_irf = get_irf(rows, "dye", bins)
    assert error == ""
    assert loaded_irf.shape == (bins,)
    assert loaded_irf.dtype == np.uint64
    np.testing.assert_array_equal(loaded_irf, independent_irf)
    irf_csv = tmp_path / "synthetic_irf_curve.csv"
    np.savetxt(irf_csv, independent_irf[None, :], delimiter=",", fmt="%d")

    error, curves = get_decay_curves(rows, INPUT_TYPE, "dye", bins, shift=False)
    assert error == ""
    choose_shift_fit_free.clear()
    error, shifts = choose_shift_fit_free(rows, bins, INPUT_TYPE, "dye")
    assert error == ""
    assert shifts["decay_id"] == ["synthetic_irf_sample"]
    # Independent overlap sums verify the estimator, not its interpretation
    # as physical timing: fluorescence broadening can bias this estimate.
    fov_curve = expected_image.sum(axis=(0, 1), dtype=np.uint64).astype(float)
    lags = np.arange(-bins + 1, bins)
    scores = [np.dot(
        normalized_irf[max(0, -lag):min(bins, bins - lag)],
        fov_curve[max(0, lag):min(bins, bins + lag)],
    ) for lag in lags]
    estimated_shift = int(shifts["shift"][0])
    assert estimated_shift == lags[np.argmax(scores)]

    comparisons = []
    feature_tables = {}
    known_rows = rows.assign(dye_shift=0.0)
    known_rows.to_csv(tmp_path / "synthetic_irf_metadata.csv", index=False)
    fov_extraction.clear()
    for case, shift in [("known_shift_zero", 0.0), ("estimated_shift", estimated_shift)]:
        shifted_rows = rows.assign(dye_shift=float(shift))
        error, info = prepare(shifted_rows)
        assert error == ""
        error, features = fov_extraction(shifted_rows.iloc[0], info)
        assert error == ""
        assert features.index.tolist() == [f"synthetic_irf_sample_{n}" for n in (1, 2, 3)]
        feature_tables[case] = features
        name = ("synthetic_irf_expected_features.csv" if shift == 0
                else "synthetic_irf_expected_features_auto_shift.csv")
        features.to_csv(tmp_path / name)

        # The oracle independently calculates complex sums and IRF division.
        # Shift interpolation remains the application's existing preprocessing.
        shifted_reference = irf_shift(loaded_irf, shift)
        reference_phasor = _phasor(shifted_reference, basis)
        for label, (nominal, profile) in enumerate(zip(nominal_lifetimes, profiles), start=1):
            cell_id = f"synthetic_irf_sample_{label}"
            curve = expected_image[mask == label].sum(axis=0, dtype=np.uint64)
            np.testing.assert_array_equal(curves[cell_id], curve)
            corrected = np.maximum(curve.astype(float) - curve[int(bins * 0.9):].mean(), 0)
            uncalibrated = _phasor(corrected, basis)
            expected = uncalibrated / reference_phasor
            actual = np.array([
                complex(features.loc[cell_id, PREFIX + f"G({suffix})"],
                        features.loc[cell_id, PREFIX + f"S({suffix})"])
                for suffix in ("1st", "2nd")
            ])
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
            phase, modulation = _lifetimes(expected, omega)
            np.testing.assert_allclose(
                features.loc[cell_id, [PREFIX + "Tau_phase", PREFIX + "Tau_mod"]].to_numpy(float),
                [phase, modulation], rtol=1e-12, atol=1e-12,
            )
            raw_phase, raw_mod = _lifetimes(uncalibrated, omega)
            model_phasor = _phasor(profile, basis) / _phasor(independent_irf, basis)
            control_phasor = _phasor(curve, basis) / _phasor(independent_irf, basis)
            model_lifetimes = _lifetimes(model_phasor, omega)
            control_lifetimes = _lifetimes(control_phasor, omega)
            np.testing.assert_allclose(model_lifetimes, nominal, rtol=0.003, atol=0)
            np.testing.assert_allclose(control_lifetimes, nominal, rtol=0, atol=0.15)
            comparisons.append({
                "case": case, "cell_id": cell_id, "shift_bins": float(shift),
                "model_lifetime_ns": float(nominal), "photons": int(curve.sum()),
                "uncalibrated_phase_ns": raw_phase, "uncalibrated_modulation_ns": raw_mod,
                "app_phase_ns": phase, "app_modulation_ns": modulation,
                "model_phase_ns": model_lifetimes[0], "model_modulation_ns": model_lifetimes[1],
                "without_tail_or_shift_processing_phase_ns": control_lifetimes[0],
                "without_tail_or_shift_processing_modulation_ns": control_lifetimes[1],
                "max_phasor_error": float(np.abs(actual - expected).max()),
            })

    # The same measured histogram loaded from one CSV row must calibrate identically.
    csv_rows = known_rows.assign(dye_IRF=str(irf_csv))
    error, csv_irf = get_irf(csv_rows, "dye", bins)
    assert error == ""
    np.testing.assert_array_equal(csv_irf, loaded_irf)
    error, csv_info = prepare(csv_rows)
    assert error == ""
    error, csv_features = fov_extraction(csv_rows.iloc[0], csv_info)
    assert error == ""
    pd.testing.assert_frame_equal(csv_features, feature_tables["known_shift_zero"])

    # Exercise source-folder preparation, shift selection, and automatic exports.
    page = Path(__file__).resolve().parents[1] / "pages/data_extraction.py"
    folder = tmp_path / "ui_data"
    folder.mkdir()
    shutil.copy2(sample_path, folder / sample_path.name)
    shutil.copy2(mask_path, folder / mask_path.name)
    (folder / IRF_NAME).symlink_to(irf_path)
    for case in ("known_shift_zero", "estimated_shift"):
        app = AppTest.from_file(str(page)).run(timeout=45)
        assert not app.exception
        app.text_input(key="fov_metadata_folder_path").set_value(str(folder)).run(timeout=45)
        _click(app, "Start calibration")
        _click(app, "Optimize for Shifts")
        shift_input = next(widget for widget in app.number_input if widget.label == "dye Shift")
        assert shift_input.value == estimated_shift
        if case == "known_shift_zero":
            shift_input.set_value(0.0).run(timeout=45)
        _click(app, "Confirm Time Gates (if applicable) and Shift for each channel")
        _refresh_after_rerun(app)
        _click(app, "Start extraction")
        displayed = [table.value for table in app.dataframe if PREFIX + "Tau_phase" in table.value.columns]
        assert len(displayed) == 1
        pd.testing.assert_frame_equal(displayed[0], feature_tables[case])

    assert _digest(irf_path) == before_digest
    pd.DataFrame(comparisons).to_csv(tmp_path / "synthetic_irf_comparison.csv", index=False)
    summary = {
        "irf_path": str(irf_path), "irf_sha256": before_digest,
        "irf_photons": int(independent_irf.sum()), "time_bins": bins,
        "frequency_hz": frequency_hz, "bin_width_ns": bin_width, "duration_ns": duration,
        "synthetic_shape_TYXCH": list(counts.shape), "poisson_seed": seed,
        "total_synthetic_photons": int(counts.sum()), "known_shift_bins": 0,
        "estimated_shift_bins": estimated_shift,
        "model": "Circular convolution of the measured IRF with bin-integrated periodic monoexponential delays, followed by Poisson counting.",
        "background": "Existing application subtracts the mean of the last 10% of sample bins and clips at zero; IRF remains uncorrected.",
        "shift_note": "No timing shift was added. Cross-correlation can interpret fluorescence broadening as an additional shift; the known-zero and automatically estimated cases are reported separately.",
        "comparisons": comparisons,
    }
    (tmp_path / "synthetic_irf_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
