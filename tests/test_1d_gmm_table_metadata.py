"""Editable 1D GMM tables retain fitted identities and category-local details."""

import numpy as np
import pandas as pd
import pytest

from src.vis import histogram, univar


class StubGMM:
    """Deterministic fit with unsorted means and an unassigned middle component."""

    def __init__(self, values, n_components=3):
        self.n_components = n_components
        center = np.mean(values)
        self.means_ = (center + np.array([2., -2., 0.])[:n_components]).reshape(-1, 1)
        scale = max(1., np.ptp(values) / 6)
        self.covariances_ = (np.array([.25, .0625, .5625])[:n_components]
                             * scale ** 2).reshape(-1, 1, 1)
        self.weights_ = np.array([.25, .65, .1])[:n_components]
        self.weights_ /= self.weights_.sum()

    def _densities(self, values):
        variance = self.covariances_.ravel()
        return (np.exp(-.5 * (values - self.means_.ravel()) ** 2 / variance)
                * self.weights_ / np.sqrt(2 * np.pi * variance))

    def score_samples(self, values):
        return np.log(self._densities(values).sum(axis=1))

    def predict_proba(self, values):
        density = self._densities(values)
        return density / density.sum(axis=1, keepdims=True)

    def predict(self, values):
        if self.n_components == 1:
            return np.zeros(len(values), dtype=int)
        # Only the lowest and highest means get hard assignments.
        return np.where(np.arange(len(values)) % 2, 0, 1)


def frame():
    rows = []
    for day, day_offset in [("Day 10", 100.), ("Day 2", 0.)]:
        for treatment, treatment_offset in [("drug", 20.), ("ctrl", 0.)]:
            for index in range(6):
                rows.append(dict(day=day, treatment=treatment,
                                 value=day_offset + treatment_offset + index))
    return pd.DataFrame(rows, index=["duplicate"] * len(rows))


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    monkeypatch.setattr(univar.st, "session_state", {})
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "black")


@pytest.mark.parametrize("intersection", [False, True])
@pytest.mark.parametrize("color_by,separate_by,expected_keys", [
    (["treatment"], "day", [("Day 2", "ctrl"), ("Day 2", "drug"),
                           ("Day 10", "ctrl"), ("Day 10", "drug")]),
    (None, "day", [("Day 2", "all_data"), ("Day 10", "all_data")]),
    (["treatment"], None, [(None, "ctrl"), (None, "drug")]),
    (None, None, [(None, "all_data")]),
])
def test_metadata_matches_prepared_fit_order_stats_and_assignments(
        monkeypatch, intersection, color_by, separate_by, expected_keys):
    monkeypatch.setattr(histogram, "_find_best_gmm", lambda values, **_kwargs: StubGMM(values))
    monkeypatch.setattr(histogram, "find_intersection",
                        lambda _w1, m1, _s1, _w2, m2, _s2: (m1 + m2) / 2)
    data = frame()
    before = data.copy(deep=True)
    prepared = histogram.prepare_histogram(
        data, "value", color_by, separate_by, apply_gmm=True,
        intersection_threshold=intersection, label_column="Assigned population")
    fig = univar._histogram_figure(prepared, "tab10", False)

    tables = fig.layout.meta["gmm_component_tables"]
    assert [(table["category"], table["group"]) for table in tables] == expected_keys
    groups = [group for panel in prepared["panels"] for group in panel["groups"]]
    summaries = [group for panel in fig.layout.meta["histogram_summaries"]
                 for group in panel["groups"]]
    assert [group["color_group"] for group in summaries] == [key[1] for key in expected_keys]
    for table, group, summary in zip(tables, groups, summaries):
        assert table["h_index"] == group["h_index"]
        model = group["gmm"]
        order = np.argsort(model.means_.ravel())
        prefix = (f'{table["category"]}::{table["group"]}' if color_by else table["category"]
                  ) if separate_by else table["group"]
        assert table["features"] == ["value"]
        assert len(table["rows"]) == 3
        for rank, (row, component_index) in enumerate(zip(table["rows"], order), 1):
            mean = model.means_.ravel()[component_index]
            std = np.sqrt(model.covariances_.ravel()[component_index])
            assert row == {
                "source_label": f"{prefix}_group{rank}",
                "component": rank,
                "x_mean_sd": f"{mean:.2f} ± {std:.2f}",
                "weight": float(model.weights_[component_index]),
            }
            assert list(summary["components"][rank - 1]) == [
                rank, row["x_mean_sd"], f'{row["weight"]:.2f}']
        assigned = prepared["df"].iloc[group["positions"]]["Assigned population"].tolist()
        if intersection:
            ranks = np.digitize(group["values"], group["thresholds"])
        else:
            rank_by_index = np.argsort(order)
            ranks = rank_by_index[model.predict(group["values"].reshape(-1, 1))]
            assert table["rows"][1]["source_label"] not in assigned
        assert assigned == [table["rows"][rank]["source_label"] for rank in ranks]
    pd.testing.assert_frame_equal(data, before)
    assert len(prepared["df"]) == len(data)


@pytest.mark.parametrize("mode", ["single", "invalid", "failed", "sparse", "count"])
def test_uneditable_fits_and_count_mode_have_no_component_tables(monkeypatch, mode):
    def fit(values, **_kwargs):
        if mode == "invalid":
            return None
        if mode == "failed":
            raise ValueError("singular fit")
        return StubGMM(values, n_components=1 if mode == "single" else 3)

    monkeypatch.setattr(histogram, "_find_best_gmm", fit)
    data = frame()
    if mode == "sparse":
        data["value"] = 1.
    prepared = histogram.prepare_histogram(data, "value", apply_gmm=mode != "count")
    fig = univar._histogram_figure(prepared, "tab10", False)

    assert fig.layout.meta["gmm_component_tables"] == []
    summary = fig.layout.meta["histogram_summaries"][0]["groups"][0]
    if mode == "single":
        assert len(summary["components"]) == 1
        assert summary["h_index"] == 0.
        assert prepared["df"]["GMM_group"].isna().all()
    elif mode != "count":
        assert summary["notices"]
        assert summary["components"] == []
        assert prepared["df"]["GMM_group"].isna().all()


def rendering_events(monkeypatch):
    """Record visible category headings and the details that follow them."""
    active = [None]
    events = []

    def record(kind, value, **_kwargs):
        if kind == "markdown" and value.startswith("**`"):
            active[0] = value
        events.append((kind, value, active[0]))

    monkeypatch.setattr(univar.st, "expander", lambda *a, **k: pytest.fail("GMM details must stay visible"))
    monkeypatch.setattr(univar.st, "info", lambda value: record("info", value))
    monkeypatch.setattr(univar.st, "markdown", lambda value, **kwargs: record("markdown", value, **kwargs))
    return events, record


def mixed_figure(monkeypatch):
    def fit(values, **_kwargs):
        # One category has a single-component group alongside its editable fit.
        return StubGMM(values, n_components=1 if 20 < np.mean(values) < 30 else 3)

    monkeypatch.setattr(histogram, "_find_best_gmm", fit)
    monkeypatch.setattr(histogram, "find_intersection",
                        lambda _w1, m1, _s1, _w2, m2, _s2: (m1 + m2) / 2)
    data = frame()
    # A failed/sparse group keeps its notice inside the second category.
    data.loc[(data["day"] == "Day 10") & (data["treatment"] == "drug"), "value"] = 120.
    prepared = histogram.prepare_histogram(
        data, "value", ["treatment"], "day", apply_gmm=True,
        intersection_threshold=True)
    return univar._histogram_figure(prepared, "tab10", False)


def test_callback_renders_matching_tables_in_each_category_and_keeps_other_details(monkeypatch):
    fig = mixed_figure(monkeypatch)
    events, record = rendering_events(monkeypatch)

    def editor(tables):
        record("editor", tables)
        return {row["source_label"]: f'Population {row["component"]}'
                for table in tables for row in table["rows"]}

    result = univar.render_histogram_summaries(fig, component_editor=editor)

    callbacks = [(tables, section) for kind, tables, section in events if kind == "editor"]
    assert len(callbacks) == 2
    assert [tables[0]["category"] for tables, _section in callbacks] == ["Day 2", "Day 10"]
    for tables, section in callbacks:
        assert all(table["group"] == "ctrl" for table in tables)
        assert all(f'day={table["category"]}' in section for table in tables)
        assert all(table["h_index"] is not None for table in tables)
    assert result == {row["source_label"]: f'Population {row["component"]}'
                      for table in fig.layout.meta["gmm_component_tables"] for row in table["rows"]}
    html_events = [(value, section) for kind, value, section in events
                   if kind == "markdown" and "flim-gmm-table" in value]
    assert len(html_events) == 1
    assert "day=Day 2 | drug" in html_events[0][0]
    assert "day=Day 2" in html_events[0][1]
    assert "ctrl" not in html_events[0][0]
    assert "H-index" in html_events[0][0] and "0.000" in html_events[0][0]
    assert any(kind == "info" and "distinct observations" in value and "Day 10" in section
               for kind, value, section in events)
    assert not any(kind == "markdown" and value.startswith("H-index for") for kind, value, _ in events)
    assert sum(kind == "markdown" and "Threshold" in value for kind, value, _ in events) == 4


def test_without_callback_keeps_all_static_tables_and_returns_empty_names(monkeypatch):
    fig = mixed_figure(monkeypatch)
    events, _record = rendering_events(monkeypatch)

    assert univar.render_histogram_summaries(fig) == {}

    html_events = [(value, section) for kind, value, section in events
                   if kind == "markdown" and "flim-gmm-table" in value]
    assert len(html_events) == 2
    assert "day=Day 2 | ctrl" in html_events[0][0]
    assert "day=Day 2 | drug" in html_events[0][0]
    assert "day=Day 10 | ctrl" in html_events[1][0]
    assert any(kind == "info" for kind, _value, _section in events)


def test_callback_falls_back_to_static_tables_when_descriptors_are_absent(monkeypatch):
    fig = mixed_figure(monkeypatch)
    fig.layout.meta.pop("gmm_component_tables", None)
    events, _record = rendering_events(monkeypatch)

    def editor(_tables):
        pytest.fail("Legacy figures have no editable component tables")

    assert univar.render_histogram_summaries(fig, component_editor=editor) == {}
    assert sum(kind == "markdown" and "flim-gmm-table" in value
               for kind, value, _section in events) == 2


def test_count_summaries_return_empty_mapping_without_rendering(monkeypatch):
    prepared = histogram.prepare_histogram(frame(), "value")
    fig = univar._histogram_figure(prepared, "tab10", False)
    events, _record = rendering_events(monkeypatch)

    assert univar.render_histogram_summaries(fig, component_editor=lambda tables: {}) == {}
    assert events == []
