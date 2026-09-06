"""SuperPlot adds observations without changing replicate calculations."""
import contextlib

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.collapse import collapse_rows
from src.vis import univar
from src.vis.helpers import apply_plot_styling


def sample_frame():
    rows = []
    for treatment, shift in (("Control", 0), ("Drug", -4)):
        for dish, count, mean in (("D1", 4, 10), ("D2", 10, 20), ("D3", 2, 40)):
            for i, value in enumerate(np.linspace(mean + shift - 2, mean + shift + 2, count)):
                rows.append(dict(cell_id=f"{treatment}_{dish}_{i}", treatment=treatment,
                                 dish=dish, image_name=f"{dish}_f{i % 2}", value=value))
    return pd.DataFrame(rows)


@pytest.fixture
def controls(monkeypatch):
    settings = {"overlay": "SuperPlot", "log_y": False, "connect_means": False}
    monkeypatch.setattr(st, "session_state", {})
    monkeypatch.setattr(st, "columns", lambda widths, **kw: [contextlib.nullcontext() for _ in widths])
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kw:
                        settings["log_y"] if label == "Log Y" else
                        settings["connect_means"] if label == "Connect means" else value)
    monkeypatch.setattr(st, "selectbox", lambda label, options, **kw:
                        settings["overlay"] if label == "Overlay" else options[0])
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "black")
    return settings


def render(source, **kwargs):
    primary, label, _ = collapse_rows(source.dropna(subset=["value"]), "dish",
                                     ["treatment", kwargs.get("separate_by")], "cell_id")
    return univar.feature_comparison_plot(
        primary, label, None, "value", ["treatment"], row_id_label="dish",
        collapse_by="dish", source_df=source, source_row_id_col="cell_id",
        source_fov_name_col="image_name", source_row_id_label="cell_id", **kwargs)


def layer(fig, role):
    return [t for t in fig.data if isinstance(t.meta, dict)
            and t.meta.get("superplot_role") == role]


def positions(traces):
    return {str(label): (float(x), float(y)) for t in traces if t.text is not None
            for label, x, y in zip(t.text, t.x, t.y)}


def test_overlay_keeps_each_original_cell_and_each_dish_mean(controls):
    source = sample_frame()
    fig = render(source, subcolor_by="dish")
    assert sum(len(t.y) for t in layer(fig, "observation")) == len(source)
    assert sum(len(t.y) for t in layer(fig, "replicate")) == 6
    assert sorted(y for t in layer(fig, "replicate") for y in t.y) == [6, 10, 16, 20, 36, 40]
    assert set(positions(layer(fig, "observation"))) == set(source.cell_id)
    assert all("cell_id" in t.hovertemplate and "image_name" in t.hovertemplate
               for t in layer(fig, "observation"))


def test_summary_weights_dishes_equally_instead_of_cells(controls):
    fig = render(sample_frame())
    summaries = layer(fig, "summary")
    assert len(summaries) == 2
    expected_sem = np.std([10, 20, 40], ddof=1) / np.sqrt(3)
    assert sorted(t.meta["mean"] for t in summaries) == pytest.approx([70 / 3 - 4, 70 / 3])
    for trace in summaries:
        assert trace.meta["count"] == 3
        assert trace.meta["sem"] == pytest.approx(expected_sem)
        ys = [y for y in trace.y if y is not None]
        assert min(ys) == pytest.approx(trace.meta["mean"] - expected_sem)
        assert max(ys) == pytest.approx(trace.meta["mean"] + expected_sem)


def test_superplot_centers_primary_points_without_recalculating_their_values(controls):
    source = sample_frame()
    controls["overlay"] = "None"
    plain = render(source)
    controls["overlay"] = "SuperPlot"
    superplot = render(source)
    colored = render(source, subcolor_by="dish")
    assert {label: y for label, (_, y) in positions(plain.data).items()} == {
        label: y for label, (_, y) in positions(layer(superplot, "replicate")).items()}
    for x, _ in positions(layer(superplot, "replicate")).values():
        assert min(abs(x - center) for center in superplot.layout.xaxis.tickvals) <= 0.1
    assert positions(layer(colored, "replicate")) == positions(layer(superplot, "replicate"))
    assert [t.meta for t in layer(colored, "summary")] == [t.meta for t in layer(superplot, "summary")]


def test_styling_keeps_smaller_fainter_cells_and_shared_counted_legends(controls):
    st.session_state["plot_show_group_counts"] = True
    fig = apply_plot_styling(render(sample_frame(), subcolor_by="dish"), 10, 14, 12)
    assert {t.marker.size for t in layer(fig, "replicate")} == {15}
    assert {t.marker.size for t in layer(fig, "observation")} == {7.5}
    assert all(np.allclose(t.marker.opacity, 0.3) for t in layer(fig, "observation"))
    assert all(np.allclose(t.marker.opacity, 1) for t in layer(fig, "replicate"))
    assert {t.marker.line.width for t in layer(fig, "observation")} == {0}
    assert {t.marker.line.width for t in layer(fig, "replicate")} == {1.5}
    assert len([t for t in fig.data if t.showlegend]) == 3
    assert all("n=2" in t.name for t in fig.data if t.showlegend)
    assert {t.legendgroup for t in layer(fig, "observation")} == {
        t.legendgroup for t in layer(fig, "replicate")}


def test_log_applies_after_collapse_and_to_each_source_cell(controls):
    source = sample_frame()
    controls["log_y"] = True
    fig = render(source)
    assert sorted(y for t in layer(fig, "replicate") for y in t.y) == pytest.approx(
        np.log10(np.array([6, 10, 16, 20, 36, 40]) + 1e-6))
    assert sorted(y for t in layer(fig, "observation") for y in t.y) == pytest.approx(
        sorted(np.log10(source.value + 1e-6)))
    assert "log₁₀" in fig.layout.yaxis.title.text


def test_negative_source_rejects_log_even_when_dish_mean_is_positive(controls, monkeypatch):
    source = sample_frame()
    source.loc[0, "value"] = -1
    controls["log_y"] = True
    errors = []
    monkeypatch.setattr(st, "error", errors.append)
    fig = render(source)
    assert errors and "negative" in errors[0]
    assert min(y for t in layer(fig, "observation") for y in t.y) == -1
    assert "log₁₀" not in fig.layout.yaxis.title.text


def test_single_replicate_keeps_mean_without_inventing_sem(controls, monkeypatch):
    source = sample_frame().query("dish == 'D1'")
    notices = []
    monkeypatch.setattr(st, "caption", lambda text, **kwargs: notices.append(text))
    fig = render(source)
    assert len(layer(fig, "summary")) == 2
    assert all(t.meta["sem"] is None and len(t.y) == 2 for t in layer(fig, "summary"))
    assert any("SEM" in notice for notice in notices)


@pytest.mark.parametrize("copies", [1, 160])
def test_renderer_keeps_replicate_dots_above_sem_bars(controls, copies):
    source = pd.concat([sample_frame()] * copies, ignore_index=True)
    source["cell_id"] = np.arange(len(source)).astype(str)
    fig = render(source)
    assert all(t.type == ("scattergl" if copies == 160 else "scatter")
               for role in ("observation", "replicate", "summary")
               for t in layer(fig, role))
    roles = [t.meta["superplot_role"] for t in fig.data if isinstance(t.meta, dict)]
    assert roles.index("summary") > max(i for i, role in enumerate(roles) if role == "observation")
    assert roles.index("replicate") > max(i for i, role in enumerate(roles) if role == "summary")
    if copies == 1:
        assert max(t.zorder for t in layer(fig, "observation")) < min(t.zorder for t in layer(fig, "summary"))
        assert max(t.zorder for t in layer(fig, "summary")) < min(t.zorder for t in layer(fig, "replicate"))


@pytest.mark.parametrize("statistical_test", ["Independent t-test", "Welch's t-test"])
@pytest.mark.parametrize("significant", [False, True])
def test_statistical_inputs_stay_collapsed_and_brackets_require_significance(
    controls, monkeypatch, statistical_test, significant
):
    source = sample_frame()
    source.loc[0, "value"] = 200
    if significant:
        source.loc[source["treatment"] == "Drug", "value"] -= 200
    import src.vis.helpers as helpers
    real_test = helpers.ttest_ind
    sizes = []

    def spy(a, b, **kwargs):
        sizes.append((len(a), len(b)))
        return real_test(a, b, **kwargs)

    monkeypatch.setattr(helpers, "ttest_ind", spy)
    monkeypatch.setattr(helpers, "comparison_pair_widget", lambda pairs: pairs)
    fig = render(source, statistical_test=statistical_test)
    assert sizes == [(3, 3)]
    assert len(fig.layout.shapes) == (3 if significant else 0)
    assert len(fig.layout.annotations) == (1 if significant else 0)
    assert all(a.text and set(a.text) == {"*"} for a in fig.layout.annotations)
    assert all(a.y > 200 for a in fig.layout.annotations)


@pytest.mark.parametrize("section_column", ["count", "mean", "sem"])
def test_section_names_cannot_collide_with_summary_statistic_names(controls, section_column):
    source = sample_frame()
    source[section_column] = "Day 1"
    fig = render(source, separate_by=section_column)
    assert sum(len(t.y) for t in layer(fig, "replicate")) == 6
    assert sorted(t.meta["mean"] for t in layer(fig, "summary")) == pytest.approx([70 / 3 - 4, 70 / 3])
