"""Synthetic sample PTU calibrated against the original measured Atto488 PTU.

Opt in with FLIM_PTU_REFERENCE_ROOT. FLIM_STANDARD_LIFETIME_NS defaults to
4.16 ns, the known reference lifetime supplied by the user.
Keep pytest's --basetemp directory to inspect the generated PTU, mask,
metadata CSV, extracted features, and independent numerical comparison.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from ptufile import PtuFile, imwrite
import pytest
import tifffile
import toml

from src import config
from src.decay_io import read_decay_with_frames
from src.file_io import get_decay_curves
from src.fov_extraction import fov_extraction
from src.metadata import parse_metadata_file


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_measured_reference(path):
    """Independent decoder call: retain scan lines, then sum them in NumPy."""
    with PtuFile(path) as ptu:
        records = ptu.read_records(memmap=True, cache=True)
        frequency_hz = int(ptu.tags["TTResult_SyncRate"])
        resolution_s = float(ptu.tcspc_resolution)
        bins = math.floor(1 / frequency_hz / resolution_s)
        lines = ptu.decode_image(
            (slice(None), slice(None), slice(None, None, -1), slice(None), slice(None)),
            records=records, frame=-1, dtime=bins, dtype=np.uint64,
        )
    return lines.sum(axis=(0, 1, 2, 3), dtype=np.uint64), frequency_hz, resolution_s


def _sample_profiles(reference, times, standard_ns, sample_lifetimes):
    """Apply a lifetime transfer ratio while retaining the measured response.

    With R = IRF * E_standard in the discrete periodic model, a sample is
    R * E_sample / E_standard in Fourier space. Lifetimes >= the standard's
    yield nonnegative relative kernels. This avoids inventing a Gaussian IRF
    or silently clipping negative deconvolution artifacts.
    """
    measured = reference.astype(np.float64) / reference.sum()
    reference_decay = np.exp(-times / standard_ns)
    reference_decay /= reference_decay.sum()
    profiles = []
    for tau in sample_lifetimes:
        decay = np.exp(-times / tau)
        decay /= decay.sum()
        profile = np.fft.ifft(
            np.fft.fft(measured) * np.fft.fft(decay) / np.fft.fft(reference_decay)
        ).real
        assert profile.min() >= 0, "Synthetic model produced negative photon probabilities"
        profiles.append(profile / profile.sum())
    return np.asarray(profiles)


def _independent_calibration(curve, reference_phasor, ideal_standard, basis, omega, *, subtract_background):
    """Complex sums and ratios; does not call PhasorPy or application helpers."""
    counts = np.asarray(curve, dtype=np.float64)
    if subtract_background:
        counts = np.maximum(counts - counts[int(len(counts) * 0.9):].mean(), 0)
    measured = counts @ basis / counts.sum()
    calibrated = measured / reference_phasor * ideal_standard
    first = calibrated[0]
    tau_phase = first.imag / first.real / omega
    tau_mod = np.sqrt(1 / abs(first)**2 - 1) / omega
    return calibrated, float(tau_phase), float(tau_mod)


def test_synthetic_ptu_calibrates_against_measured_atto488(tmp_path, monkeypatch):
    root = os.environ.get("FLIM_PTU_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set FLIM_PTU_REFERENCE_ROOT to the supplied Atto folder.")
    reference_path = Path(root).resolve() / "Atto488.ptu"
    original_digest = _digest(reference_path)
    standard_ns = float(os.environ.get("FLIM_STANDARD_LIFETIME_NS", "4.16"))
    assert np.isfinite(standard_ns) and standard_ns > 0
    reference, frequency_hz, resolution_s = _read_measured_reference(reference_path)
    bins = len(reference)
    times = np.arange(bins) * resolution_s * 1e9
    duration_ns = bins * resolution_s * 1e9
    omega = 2 * np.pi * frequency_hz * 1e-9
    harmonics = np.array([1, 2])
    basis = np.exp(1j * times[:, None] * omega * harmonics)
    measured_reference = reference @ basis / reference.sum()
    ideal_standard = 1 / (1 - 1j * omega * harmonics * standard_ns)
    sample_lifetimes = standard_ns * np.array([1.0, 1.5, 2.0])
    profiles = _sample_profiles(reference, times, standard_ns, sample_lifetimes)

    # Three ROIs, each with two pixels, repeated over three frames. Vary the
    # brightness spatially and by frame, with reproducible Poisson counting.
    mask = np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8)
    seed = 20260911
    rng = np.random.default_rng(seed)
    counts = np.empty((3, 2, 3, 1, bins), dtype=np.uint32)
    for frame, y, x in np.ndindex(counts.shape[:3]):
        photons = 100_000 * (1 + 0.3 * frame + 0.2 * y + 0.1 * x)
        counts[frame, y, x, 0] = rng.poisson(photons * profiles[mask[y, x] - 1])
    sample_path = tmp_path / "synthetic_sample.ptu"
    mask_path = tmp_path / "synthetic_sample_mask.tif"
    # Automatic pixel dwell time is long enough to encode every generated
    # photon. A short fixed dwell time would silently discard records.
    imwrite(sample_path, counts, global_resolution=1 / frequency_hz,
            tcspc_resolution=resolution_s)
    tifffile.imwrite(mask_path, mask)
    expected_image = counts.sum(axis=0, dtype=np.uint64)[:, :, 0, :]
    error, decoded = read_decay_with_frames(str(sample_path))
    assert error == ""
    assert decoded[1] == 3
    np.testing.assert_array_equal(decoded[0], expected_image)

    # Redirect only configuration; all file readers, background processing,
    # phasor calibration, and extraction run as they do in the application.
    config_path = tmp_path / "test_config.toml"
    config_path.write_text(toml.dumps({
        "fov_name_col": "image_name", "unique_cell_id_col": "cell_id",
        "Decay (3/4D)": {
            "file_types": ["Decay", "Mask", "Fluorescence Lifetime Standard"],
            "available_feature_extractors": ["Lifetime fit free"],
        },
    }))
    monkeypatch.setattr(config, "_CONFIG_PATH", config_path)
    rows = pd.DataFrame([{
        "image_name": "synthetic_sample", "dye_input_type": "Decay (3/4D)",
        "dye_imaging_modality": "FLIM", "dye_Lifetime fit free": True,
        "dye_Decay": str(sample_path), "dye_Mask": str(mask_path),
        "dye_channel": -1, "time_bins": bins, "duration": duration_ns,
        "laser_rate": frequency_hz * 1e-9,
        "fit_free_calibration_method": "Fluorescence Lifetime Standard",
        "dye_Fluorescence Lifetime Standard": str(reference_path),
        "dye_fluorescence_lifetime_standard_time_axis": 2,
        "fluorescence_lifetime_standard_lifetime": standard_ns,
    }])
    metadata_path = tmp_path / "fov_metadata.csv"
    rows.to_csv(metadata_path, index=False)
    replayed = pd.read_csv(metadata_path)
    error, metadata = parse_metadata_file(replayed, "image_name")
    assert error == ""
    assert metadata["channels_shift"] == {}
    error, curves = get_decay_curves(replayed.iloc[0], "Decay (3/4D)", "dye", bins, shift=False)
    assert error == ""
    fov_extraction.clear()
    error, features = fov_extraction(replayed.iloc[0], metadata)
    assert error == ""
    assert features.index.tolist() == [f"synthetic_sample_{label}" for label in (1, 2, 3)]
    features.to_csv(tmp_path / "extracted_features.csv")

    comparisons = []
    prefix = "Lifetime fit free_dye: "
    for label, (nominal_tau, profile) in enumerate(zip(sample_lifetimes, profiles), start=1):
        cell_id = f"synthetic_sample_{label}"
        curve = expected_image[mask == label].sum(axis=0, dtype=np.uint64)
        np.testing.assert_array_equal(curves[cell_id], curve)
        expected, expected_phase, expected_mod = _independent_calibration(
            curve, measured_reference, ideal_standard, basis, omega, subtract_background=True,
        )
        actual = np.array([
            complex(features.loc[cell_id, prefix + f"G({suffix})"],
                    features.loc[cell_id, prefix + f"S({suffix})"])
            for suffix in ("1st", "2nd")
        ])
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        phase = features.loc[cell_id, prefix + "Tau_phase"]
        modulation = features.loc[cell_id, prefix + "Tau_mod"]
        assert phase == pytest.approx(expected_phase, rel=1e-12, abs=1e-12)
        assert modulation == pytest.approx(expected_mod, rel=1e-12, abs=1e-12)

        # Separately quantify model/discretization and Poisson error before
        # background subtraction. Do not label the app's biased output as an
        # exact recovery of the nominal synthetic lifetime.
        _, model_phase, model_mod = _independent_calibration(
            profile, measured_reference, ideal_standard, basis, omega, subtract_background=False,
        )
        _, raw_phase, raw_mod = _independent_calibration(
            curve, measured_reference, ideal_standard, basis, omega, subtract_background=False,
        )
        np.testing.assert_allclose([model_phase, model_mod], nominal_tau, rtol=0.003, atol=0)
        np.testing.assert_allclose([raw_phase, raw_mod], nominal_tau, rtol=0, atol=0.15)
        comparisons.append({
            "cell_id": cell_id, "model_lifetime_ns": float(nominal_tau),
            "photons": int(curve.sum()),
            "model_phase_ns": model_phase, "model_modulation_ns": model_mod,
            "without_tail_subtraction_phase_ns": raw_phase,
            "without_tail_subtraction_modulation_ns": raw_mod,
            "app_phase_ns": float(phase), "app_modulation_ns": float(modulation),
            "independent_pipeline_phase_ns": expected_phase,
            "independent_pipeline_modulation_ns": expected_mod,
            "max_phasor_error": float(np.abs(actual - expected).max()),
        })

    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(tmp_path / "calibration_comparison.csv", index=False)
    assert _digest(reference_path) == original_digest
    summary = {
        "reference_path": str(reference_path), "reference_sha256": original_digest,
        "reference_lifetime_ns": standard_ns,
        "frequency_hz": frequency_hz, "time_bins": bins,
        "bin_width_ns": resolution_s * 1e9, "duration_ns": duration_ns,
        "synthetic_shape_TYXCH": list(counts.shape), "poisson_seed": seed,
        "total_synthetic_photons": int(counts.sum()),
        "model": "FFT(measured dye) * FFT(periodic sample exponential) / FFT(periodic standard exponential)",
        "background": "Unmodified application: subtract mean of last 10% of sample bins and clip at zero; reference remains uncorrected.",
        "comparisons": comparisons,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(comparison[["model_lifetime_ns", "app_phase_ns", "app_modulation_ns", "max_phasor_error"]].to_string(index=False))
