"""A fit-free channel supports a harmonic only when both G and S are present.
The parameter widget handles an empty harmonic list without raising.
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
