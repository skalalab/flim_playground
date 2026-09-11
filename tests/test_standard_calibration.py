"""Standard calibration uses MHz and a measured reference at each harmonic."""

import numpy as np
import pytest
import tifffile

from src import fov_extraction
from src.file_io import load_image


LASER_RATE = 0.08  # GHz: 80 MHz, with time axes and lifetimes in ns.
STANDARD_LIFETIME = 4.0
PREFIX = "Lifetime fit free_ch1: "


def _complex_phasor(decay, time, harmonic):
    """Independent discrete Fourier sum, using the acquisition's time bins."""
    phase = harmonic * 2 * np.pi * LASER_RATE * time
    return np.sum(decay * np.exp(1j * phase)) / np.sum(decay)


def _known_standard_phasor(harmonic):
    return 1 / (1 - 1j * harmonic * 2 * np.pi * LASER_RATE * STANDARD_LIFETIME)


def _coordinates(features, harmonic):
    suffix = "1st" if harmonic == 1 else "2nd"
    return complex(features[f"{PREFIX}G({suffix})"], features[f"{PREFIX}S({suffix})"])


@pytest.mark.parametrize("duration", [12.5, 8.0], ids=["full_period", "truncated"])
def test_four_ns_standard_referenced_to_itself_recovers_lifetime(monkeypatch, duration):
    time = np.arange(128) * duration / 128
    reference = np.exp(-time / STANDARD_LIFETIME)
    # The tail still contains fluorescence. Isolate calibration from the
    # existing sample-only tail estimate so sample and reference are identical.
    monkeypatch.setattr(fov_extraction, "get_offset", lambda _decay: 0.0)

    error, results = fov_extraction.extract_fit_free_results(
        "ch1", {"cell": reference}, LASER_RATE, duration,
        "Fluorescence Lifetime Standard",
        fluorescence_lifetime_standard_image=reference,
        fluorescence_lifetime_standard_lifetime=STANDARD_LIFETIME,
        fluorescence_lifetime_standard_time_axis=0,
    )

    assert error == ""
    features = results["cell"]
    assert features[f"{PREFIX}Tau_phase"] == pytest.approx(4.0)
    assert features[f"{PREFIX}Tau_mod"] == pytest.approx(4.0)
    for harmonic in (1, 2):
        assert _coordinates(features, harmonic) == pytest.approx(
            _known_standard_phasor(harmonic), abs=1e-12
        )


@pytest.mark.parametrize("duration", [12.5, 8.0], ids=["full_period", "truncated"])
@pytest.mark.parametrize("harmonic", [1, 2])
@pytest.mark.parametrize("time_axis", [0, 1, 2])
def test_standard_tiff_calibration_matches_complex_ratio(
    tmp_path, duration, harmonic, time_axis
):
    time = np.arange(96) * duration / 96
    sample = 140 * np.exp(-time / 1.7) + 45 * np.exp(-time / 5.3) + 17
    reference_decay = np.exp(-time / STANDARD_LIFETIME)
    scales = np.array([[1.0, 3.0, 10.0], [0.25, 2.0, 7.0]])
    # Different intensities and delays make an unweighted average of pixel
    # phasors differ from the required intensity-weighted reference center.
    reference = np.stack([
        scale * np.roll(reference_decay, shift) + 0.02
        for shift, scale in enumerate(scales.flat)
    ]).reshape(2, 3, -1)
    reference_path = tmp_path / "standard.tiff"
    tifffile.imwrite(
        reference_path, np.moveaxis(reference, -1, time_axis),
        photometric="minisblack",
    )

    error, results = fov_extraction.extract_fit_free_results(
        "ch1", {"cell": sample}, LASER_RATE, duration,
        "Fluorescence Lifetime Standard",
        fluorescence_lifetime_standard_image=load_image(reference_path),
        fluorescence_lifetime_standard_lifetime=STANDARD_LIFETIME,
        fluorescence_lifetime_standard_time_axis=time_axis,
    )

    assert error == ""
    # Keep the established sample-only background subtraction and clipping.
    background = np.mean(sample[int(len(sample) * 0.9):])
    corrected_sample = np.maximum(sample - background, 0)
    measured_sample = _complex_phasor(corrected_sample, time, harmonic)
    measured_reference = _complex_phasor(reference.sum(axis=(0, 1)), time, harmonic)
    expected = measured_sample * _known_standard_phasor(harmonic) / measured_reference
    assert _coordinates(results["cell"], harmonic) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("duration", [12.5, 8.0], ids=["full_period", "truncated"])
def test_irf_calibration_keeps_complex_division(duration):
    time = np.arange(96) * duration / 96
    sample = 100 * np.exp(-time / 2.1) + 8
    irf = np.exp(-0.5 * ((time - 0.8) / 0.2) ** 2)

    error, results = fov_extraction.extract_fit_free_results(
        "ch1", {"cell": sample}, LASER_RATE, duration, "IRF", shifted_irf=irf
    )

    assert error == ""
    corrected_sample = np.maximum(sample - np.mean(sample[int(len(sample) * 0.9):]), 0)
    for harmonic in (1, 2):
        expected = _complex_phasor(corrected_sample, time, harmonic) / _complex_phasor(
            irf, time, harmonic
        )
        assert _coordinates(results["cell"], harmonic) == pytest.approx(expected, abs=1e-12)
