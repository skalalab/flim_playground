"""Feature Comparison geometry and labels shared by the app's figure and exported scripts."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import pytest
from scipy.stats import ttest_ind

from src.vis.helpers import (
    _add_effect_size_annotations,
    _density_at_points,
    box_statistics,
    bracket_positioning_metrics,
    cohens_d,
    p_value_stars,
    plan_effect_size_brackets,
    sina_offsets,
)
from src.vis.univar import comparison_labels, comparison_layout


def test_comparison_layout_orders_groups_and_spaces_sections():
    layout = comparison_layout(["a", "b"], ["s1", "s2"], {("s1", "a"), ("s1", "b"), ("s2", "b")})
    assert layout["compare_groups"] == ["a", "b"] and layout["separate_groups"] == ["s1", "s2"]
    assert layout["compare_pairs"] == [("a", "b")]
    assert layout["x_positions"] == {("s1", "a"): 0, ("s1", "b"): 1, ("s2", "b"): 2.5}
    assert layout["tick_positions"] == [0, 1, 2.5] and layout["tick_labels"] == ["a", "b", "b"]
    assert [(s["group"], s["center"]) for s in layout["sections"]] == [("s1", 0.5), ("s2", 2.5)]
    assert layout["sections"][1]["combinations"] == [("s2", "b")]
    assert layout["dividers"] == [1.75]


def test_comparison_layout_applies_custom_order_and_blanks_the_all_data_label():
    layout = comparison_layout(["b", "a", "c"], [], set(), custom_order={"compare_groups": ["c", "a"]})
    assert layout["compare_groups"] == ["c", "a", "b"]
    assert layout["x_positions"] == {"c": 0, "a": 1, "b": 2}
    assert layout["sections"] == [] and layout["dividers"] == []
    assert comparison_layout(["all_data"], [], set())["tick_labels"] == [""]


def test_box_statistics_cap_the_fences_at_the_observed_extrema():
    stats = box_statistics(np.array([1.0, 2.0, 3.0, 4.0, 100.0]))
    assert (stats["q1"], stats["median"], stats["q3"]) == (2.0, 3.0, 4.0)
    assert stats["lower_fence"] == 1.0 and stats["upper_fence"] == 7.0
    assert stats["mean"] == 22.0


def test_sina_offsets_follow_local_density_and_the_fixed_seed():
    values = np.array([0.0, 0.1, 0.2, 5.0])
    densities = _density_at_points(values)
    # Same operation order as the app's original jitter block, so offsets stay bit-identical.
    expected = np.random.default_rng(seed=42).uniform(-1, 1, size=4) * (densities / densities.max()) * 0.35
    np.testing.assert_array_equal(sina_offsets(values, 0.35), expected)
    constant = sina_offsets(np.array([1.0, 1.0, 1.0]), 0.1)
    assert np.all(np.abs(constant) <= 0.1) and np.any(constant != 0)


def test_comparison_labels_name_the_feature_and_every_active_encoding():
    labels = comparison_labels("F1", ["treatment"], log_y=True, separate_by="day",
                               opacity_by="", shape_by="dish", subcolor_by=None, engine="mpl")
    assert labels["title"] == "Distribution of F1 by treatment (separated by: day, shape: dish)"
    assert labels["y"] == "log₁₀(F1)"
    plain = comparison_labels("F1", ["treatment"], log_y=False, separate_by=None,
                              opacity_by=None, shape_by=None, subcolor_by=None, engine="mpl")
    assert plain == {"title": "Distribution of F1 by treatment", "y": "F1"}


def _bracket_frame():
    return pd.DataFrame({"grp": ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
                         "val": [1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 5.0, 5.1, 5.2, 5.3]})


def test_p_value_stars_use_the_four_conventional_cutoffs():
    stars = [p_value_stars(p) for p in (0.00005, 0.0005, 0.005, 0.05, 0.051, float("nan"))]
    assert stars == ["****", "***", "**", "*", "", ""]


def test_plan_effect_size_brackets_orders_pairs_and_stacks_overlapping_spans():
    brackets, skipped = plan_effect_size_brackets(
        _bracket_frame(), "val", "grp", [("b", "c"), ("a", "c"), ("a", "b")],
        {"a": 0, "b": 1, "c": 2}, effect_size_method="Absolute Cohen's d",
        mean_or_median="Mean", threshold=0.0, statistical_test="Welch's t-test", data_range=4.3)
    assert skipped == []
    # Left to right, shorter span first; the two brackets reaching c stack above each other.
    assert [b["pair"] for b in brackets] == [("a", "b"), ("a", "c"), ("b", "c")]
    assert [round(b["y_top"], 3) for b in brackets] == [2.515, 5.515, 6.16]
    assert [round(b["y_text"], 3) for b in brackets] == [2.73, 5.73, 6.375]
    assert brackets[0]["height"] == pytest.approx(
        bracket_positioning_metrics(4.3)["bracket_vertical_length_abs"])
    assert [b["text"] for b in brackets] == ["7.75****", "30.98****", "23.24****"]
    frame = _bracket_frame()
    assert brackets[0]["effect_size"] == pytest.approx(cohens_d(
        frame.loc[frame["grp"] == "a", "val"], frame.loc[frame["grp"] == "b", "val"], "Mean"))


def test_plan_effect_size_brackets_skips_empty_groups_and_sub_threshold_effects():
    frame = _bracket_frame()
    frame.loc[frame["grp"] == "c", "val"] = np.nan
    brackets, skipped = plan_effect_size_brackets(
        frame, "val", "grp", [("a", "b"), ("b", "c")], {"a": 0, "b": 1, "c": 2},
        effect_size_method="Glass's Delta", mean_or_median="Mean", threshold=100.0,
        statistical_test="None", data_range=1.3)
    assert skipped == [("b", "c")] and brackets == []
    brackets, _ = plan_effect_size_brackets(
        frame, "val", "grp", [("a", "b")], {"a": 0, "b": 1}, effect_size_method="Glass's Delta",
        mean_or_median="Mean", threshold=0.0, statistical_test="None", data_range=1.3)
    assert [b["text"] for b in brackets] == [f"\u0394={brackets[0]['effect_size']:.2f}"]


def test_plan_effect_size_brackets_star_only_drops_insignificant_pairs():
    frame = pd.DataFrame({"grp": ["a"] * 3 + ["b"] * 3 + ["c"] * 3,
                          "val": [1.0, 2.0, 3.0, 1.1, 2.1, 2.9, 9.0, 9.1, 9.2]})
    brackets, _ = plan_effect_size_brackets(
        frame, "val", "grp", [("a", "b"), ("b", "c")], {"a": 0, "b": 1, "c": 2},
        effect_size_method="None", mean_or_median=None, threshold=0.0,
        statistical_test="Independent t-test", data_range=8.2)
    assert [b["pair"] for b in brackets] == [("b", "c")]
    assert brackets[0]["p_value"] == pytest.approx(
        ttest_ind([1.1, 2.1, 2.9], [9.0, 9.1, 9.2], equal_var=True).pvalue)
    assert brackets[0]["text"] == p_value_stars(brackets[0]["p_value"]) != ""
    assert brackets[0]["effect_size"] is None


def test_app_brackets_follow_the_shared_plan(monkeypatch):
    """The Plotly figure draws exactly the planned brackets (geometry pinned before the extraction)."""
    from src.vis import helpers

    monkeypatch.setattr(helpers.st, "warning", lambda *args, **kwargs: None)
    fig = go.Figure()
    _add_effect_size_annotations(
        fig, _bracket_frame(), "val", ["a", "b", "c"], "grp", [("a", "b"), ("a", "c"), ("b", "c")],
        "black", effect_size_method="Absolute Cohen's d", mean_or_median="Mean",
        position_map={"a": 0, "b": 1, "c": 2}, selected_pairs=[("a", "c"), ("a", "b"), ("b", "c")],
        threshold=0.0, statistical_test="Welch's t-test")
    shapes = [(s.x0, round(float(s.y0), 3), s.x1, round(float(s.y1), 3)) for s in fig.layout.shapes]
    assert shapes == [(0, 2.515, 1, 2.515), (0, 2.515, 0, 2.386), (1, 2.515, 1, 2.386),
                      (0, 5.515, 2, 5.515), (0, 5.515, 0, 5.386), (2, 5.515, 2, 5.386),
                      (1, 6.16, 2, 6.16), (1, 6.16, 1, 6.031), (2, 6.16, 2, 6.031)]
    assert [(a.x, round(float(a.y), 3), a.text) for a in fig.layout.annotations] == [
        (0.5, 2.73, "7.75****"), (1.0, 5.73, "30.98****"), (1.5, 6.375, "23.24****")]


def test_app_brackets_in_separated_sections_follow_the_shared_plan(monkeypatch):
    """Per-section calls with the global range (geometry pinned before the extraction)."""
    from itertools import combinations

    from src.vis import helpers

    monkeypatch.setattr(helpers.st, "warning", lambda *args, **kwargs: None)
    frame = pd.DataFrame({
        "sec": ["s1"] * 12 + ["s2"] * 8,
        "grp": ["a"] * 4 + ["b"] * 4 + ["c"] * 4 + ["a"] * 4 + ["b"] * 4,
        "val": [1.0, 1.1, 1.2, 1.3, 2.0, 2.1, 2.2, 2.3, 5.0, 5.1, 5.2, 5.3,
                1.5, 1.6, 1.7, 1.8, 4.0, 4.1, 4.2, 4.3]})
    positions = {("s1", "a"): 0, ("s1", "b"): 1, ("s1", "c"): 2, ("s2", "a"): 3.5, ("s2", "b"): 4.5}
    sections = [("s1", ["a", "b", "c"]), ("s2", ["a", "b"])]
    selected = [("a", "b"), ("a", "c"), ("b", "c")]
    fig = go.Figure()
    for section, groups in sections:
        pairs = list(combinations(groups, 2))
        section_frame = frame[frame["sec"] == section]
        _add_effect_size_annotations(
            fig, section_frame.copy(), "val", groups, "grp", pairs, "black",
            effect_size_method="Absolute Cohen's d", mean_or_median="Mean",
            position_map={group: positions[(section, group)] for group in groups},
            selected_pairs=[p for p in pairs if p in selected], threshold=0.0,
            statistical_test="Welch's t-test", global_data_range=(frame["val"].min(), frame["val"].max()),
            section_label=section, display_df=section_frame)
    shapes = [(s.x0, round(float(s.y0), 3), s.x1, round(float(s.y1), 3)) for s in fig.layout.shapes]
    assert shapes == [(0, 2.515, 1, 2.515), (0, 2.515, 0, 2.386), (1, 2.515, 1, 2.386),
                      (0, 5.515, 2, 5.515), (0, 5.515, 0, 5.386), (2, 5.515, 2, 5.386),
                      (1, 6.16, 2, 6.16), (1, 6.16, 1, 6.031), (2, 6.16, 2, 6.031),
                      (3.5, 4.515, 4.5, 4.515), (3.5, 4.515, 3.5, 4.386), (4.5, 4.515, 4.5, 4.386)]
    assert [(a.x, round(float(a.y), 3), a.text) for a in fig.layout.annotations] == [
        (0.5, 2.73, "7.75****"), (1.0, 5.73, "30.98****"), (1.5, 6.375, "23.24****"),
        (4.0, 4.73, "19.36****")]
