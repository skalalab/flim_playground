"""Phasor column names and the category fallback, shared by page, plot and export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vis.phasor import phasor_columns, resolve_phasor_category


def test_phasor_columns_name_the_channel_and_harmonic():
    assert phasor_columns("nadh", 1) == (
        "Lifetime fit free_nadh: G(1st)", "Lifetime fit free_nadh: S(1st)", "1st")
    assert phasor_columns("fad", 2) == (
        "Lifetime fit free_fad: G(2nd)", "Lifetime fit free_fad: S(2nd)", "2nd")


def test_resolve_phasor_category_prefers_the_request_then_the_first_level():
    assert resolve_phasor_category(["Day 2", "Day 10"], "Day 10") == "Day 10"
    assert resolve_phasor_category(["Day 2", "Day 10"], "Day 99") == "Day 2"
    assert resolve_phasor_category(["Day 2", "Day 10"], None) == "Day 2"
    assert resolve_phasor_category([], "Day 2") is None
