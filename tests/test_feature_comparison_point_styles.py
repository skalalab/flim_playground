"""Feature Comparison sends per-point symbols only while Shape by is active.

Plotly validates a per-point symbol list one entry at a time (0.7 s at 200k points,
again for every restyle copy), so an inactive Shape channel is one symbol per trace.
Opacity stays per point even when Opacity by is off: a WebGL trace whose symbol and
opacity are both single values blends its colours by a different path (up to 2/255
per channel), and a numeric list costs little to validate. The group labels behind
every point method must stay the row-wise ``"::"`` join the exported scripts compute.
"""
import contextlib

import numpy as np
import pandas as pd
import pytest

from src.collapse import collapse_rows
from src.vis import feature_comparison, helpers


@pytest.fixture
def controls(monkeypatch):
    settings = {"overlay": "None"}
    monkeypatch.setattr(feature_comparison.st, "session_state", {"plot_show_group_counts": False})
    monkeypatch.setattr(feature_comparison.st, "container", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(feature_comparison.st, "checkbox", lambda label, value=False, **kwargs: value)
    monkeypatch.setattr(feature_comparison.st, "caption", lambda *args, **kwargs: None)
    monkeypatch.setattr(feature_comparison, "get_context_theme_color", lambda: "black")
    monkeypatch.setattr(feature_comparison, "comparison_overlay_widget", lambda *args: settings["overlay"])
    return settings


def _cells():
    """Batch is constant within each dish, so it survives Collapse by as a decoration."""
    rng = np.random.default_rng(0)
    rows = []
    for treatment in ("ctrl", "drug"):
        for dish, batch in (("D1", "B1"), ("D2", "B1"), ("D3", "B2")):
            for i in range(12):
                rows.append({"cell_id": f"{treatment}_{dish}_{i}", "treatment": treatment,
                             "dish": dish, "batch": batch, "value": rng.normal(10, 2)})
    return pd.DataFrame(rows)


def _render(controls, overlay, **channels):
    """Return the figure and each hover label's batch."""
    controls["overlay"] = overlay
    cells = _cells()
    batch_of = dict(zip(cells["cell_id"], cells["batch"]))
    if overlay == "SuperPlot":
        primary, row_id, _ = collapse_rows(cells, "dish", ["treatment"], "cell_id")
        batch_of.update(zip(primary[row_id].astype(str), primary["batch"]))
        fig = feature_comparison.feature_comparison_plot(
            primary, row_id, None, "value", ["treatment"], collapse_by="dish",
            source_df=cells, source_row_id_col="cell_id", **channels)
    else:
        fig = feature_comparison.feature_comparison_plot(
            cells, "cell_id", None, "value", ["treatment"], **channels)
    return fig, batch_of


def _point_traces(fig):
    return [trace for trace in fig.data if trace.text is not None and len(trace.text)]


def _layer(trace):
    return (trace.meta or {}).get("superplot_role", "points")


# Each layer's marker opacity without an Opacity channel.
DEFAULT_OPACITY = {"points": 0.7, "replicate": 1.0, "observation": 0.3}


def _assert_per_point(values, trace):
    assert np.ndim(values) == 1 and len(values) == len(trace.text)


@pytest.mark.parametrize("overlay", ["None", "SuperPlot"])
@pytest.mark.parametrize("subcolor_by", [None, "batch"])
def test_inactive_shape_is_one_symbol_while_opacity_stays_per_point(controls, overlay, subcolor_by):
    fig, _ = _render(controls, overlay, subcolor_by=subcolor_by)
    traces = _point_traces(fig)
    assert {_layer(trace) for trace in traces} == (
        {"replicate", "observation"} if overlay == "SuperPlot" else {"points"})
    for trace in traces:
        assert trace.marker.symbol == "circle"
        _assert_per_point(trace.marker.opacity, trace)
        assert trace.marker.opacity == pytest.approx(
            [DEFAULT_OPACITY[_layer(trace)]] * len(trace.text))


@pytest.mark.parametrize("overlay", ["None", "SuperPlot"])
@pytest.mark.parametrize("channel", ["shape", "opacity"])
def test_active_channel_varies_by_point(controls, overlay, channel):
    fig, batch_of = _render(controls, overlay, **{f"{channel}_by": "batch"})
    active = "symbol" if channel == "shape" else "opacity"
    styles = {}
    for trace in _point_traces(fig):
        values = getattr(trace.marker, active)
        _assert_per_point(values, trace)
        for label, value in zip(trace.text, values):
            styles.setdefault((_layer(trace), batch_of[str(label)]), set()).add(value)
        _assert_per_point(trace.marker.opacity, trace)
        if channel == "opacity":
            assert trace.marker.symbol == "circle"
    # Within each layer, one style per batch, and the two batches differ.
    assert all(len(values) == 1 for values in styles.values())
    for layer in {layer for layer, _ in styles}:
        assert styles[(layer, "B1")] != styles[(layer, "B2")]


@pytest.mark.parametrize("columns", [
    ["text"], ["number"], ["category"], ["mixed"], ["text", "number", "integer"],
    ["flag", "category"], ["category", "text", "number", "integer", "flag", "mixed"],
    ["text", "text"],
])
def test_group_labels_match_the_exported_row_wise_join(columns):
    # Missing values, categoricals, booleans, mixed objects and a repeated index label.
    frame = pd.DataFrame({
        "text": ["a", None, "c", "a"], "number": [1.0, np.nan, 2.5, 1.0],
        "integer": [1, 2, 3, 1], "flag": [True, False, True, True],
        "category": pd.Categorical(["x", "y", None, "x"]), "mixed": [1, "1", None, 2.0],
    }, index=[3, 3, 7, 1])
    expected = frame[columns].astype(str).agg("::".join, axis=1)
    work = frame.copy()
    groups, _ = helpers._prepare_group_data(work, columns, "grp")
    assert work["grp"].tolist() == expected.tolist()
    assert list(work.index) == [3, 3, 7, 1]
    assert sorted(groups) == sorted(set(expected))


@pytest.mark.parametrize("columns", [["text"], ["text", "number"]])
def test_group_labels_of_an_empty_frame_are_an_empty_column(columns):
    """The row-wise join returned a frame here and could not be assigned for two columns."""
    frame = pd.DataFrame({"text": pd.Series([], dtype=object), "number": pd.Series([], dtype=float)})
    groups, _ = helpers._prepare_group_data(frame, columns, "grp")
    assert frame["grp"].tolist() == []
    assert len(groups) == 0
