"""phasor_params_widget detects which harmonics a fit-free channel supports.
A channel only supports a harmonic when BOTH its G and S phasor coordinates are
present; otherwise the harmonic list is empty. The empty-list case previously
made the harmonic selectbox empty, leaving the laser-rate `f` unassigned and
crashing the widget with UnboundLocalError.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.widgets.visualization_widgets import _compute_channel_harmonics


def test_first_harmonic_detected_when_both_g_and_s_present():
    fgd = {
        "Lifetime fit free_ch1": [
            "Lifetime fit free_ch1: G(1st)",
            "Lifetime fit free_ch1: S(1st)",
        ]
    }
    assert _compute_channel_harmonics(fgd) == {"ch1": [1]}


def test_harmonic_omitted_when_s_coordinate_missing():
    # Only G(1st), no S(1st) -> no usable harmonic -> empty list.
    fgd = {"Lifetime fit free_ch1": ["Lifetime fit free_ch1: G(1st)"]}
    assert _compute_channel_harmonics(fgd) == {"ch1": []}


def test_both_harmonics_detected():
    fgd = {
        "Lifetime fit free_ch1": [
            "Lifetime fit free_ch1: G(1st)",
            "Lifetime fit free_ch1: S(1st)",
            "Lifetime fit free_ch1: G(2nd)",
            "Lifetime fit free_ch1: S(2nd)",
        ]
    }
    assert _compute_channel_harmonics(fgd) == {"ch1": [1, 2]}


def test_non_fit_free_extractors_are_ignored():
    fgd = {"Lifetime fit_ch1": ["Lifetime fit_ch1: t1"]}
    assert _compute_channel_harmonics(fgd) == {}
