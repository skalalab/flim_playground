"""A chart too large for the browser shows a message instead of crashing.

Plotly keeps each base64 number array and attaches its decoded copy, and
Streamlit's chart serializes the whole figure into one string to apply its theme.
Past Chrome's string limit that throws "RangeError: Invalid string length".
"""
import numpy as np
import plotly.graph_objects as go

from src.widgets import plot_layout


def figure(points, panels=0, ids=True):
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for slot in range(panels + 1):
        axis = "x" if slot == 0 else f"x{slot + 1}"
        fig.add_trace(go.Scattergl(
            x=rng.normal(size=points), y=rng.normal(size=points), xaxis=axis,
            text=np.array([f"cell{i}" for i in range(points)], dtype=object) if ids else None))
    fig.add_trace(go.Scatter(x=[None], y=[None]))  # A legend swatch draws no points.
    return fig


def test_a_small_chart_draws_normally():
    assert plot_layout.figure_size_message(figure(1000)) is None


def test_the_browser_copy_counts_each_number_twice():
    # About 49 characters per coordinate in the browser: base64 plus "index":value.
    estimate = plot_layout.browser_figure_chars(figure(10000, ids=False))
    assert 2 * 10000 * 40 < estimate < 2 * 10000 * 60


def test_a_chart_over_the_browser_limit_gets_a_message(monkeypatch):
    fig = figure(1000, panels=2)
    monkeypatch.setattr(plot_layout, "BROWSER_STRING_LIMIT", plot_layout.browser_figure_chars(fig) // 2)
    message = plot_layout.figure_size_message(fig, separated=True)
    assert "3,000 points" in message
    assert "Separate by" in message


def test_without_separate_by_the_message_only_suggests_filtering(monkeypatch):
    fig = figure(1000)
    monkeypatch.setattr(plot_layout, "BROWSER_STRING_LIMIT", plot_layout.browser_figure_chars(fig) // 2)
    message = plot_layout.figure_size_message(fig, separated=False)
    assert "Separate by" not in message
    assert "filter" in message.lower()


def test_a_chart_over_the_server_message_limit_gets_a_message(monkeypatch):
    fig = figure(1000)
    monkeypatch.setattr(plot_layout, "_max_message_chars", lambda: plot_layout.sent_figure_chars(fig) // 2)
    assert plot_layout.figure_size_message(fig) is not None
