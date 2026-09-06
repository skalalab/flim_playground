"""Structured 2D GMM tables expose fitted values without parsing legacy HTML."""

import numpy as np
import pandas as pd
import pytest
import streamlit as st

from src.vis import bivar


class StubGMM:
    """Small deterministic fitted-model surface used by the plot."""

    def __init__(self, values, n_components=3):
        values = np.asarray(values)
        center = values.mean(axis=0)
        self.n_components = n_components
        self.means_ = np.array([center + [index * .25, index * .5]
                                for index in range(n_components)])
        self.covariances_ = np.array([
            [[(index + 1) ** 2, 0.], [0., (index + 2) ** 2]]
            for index in range(n_components)
        ])
        self.weights_ = np.array([.6, .3, .1])[:n_components]
        self.weights_ /= self.weights_.sum()

    def predict(self, values):
        # Component three deliberately receives no hard-assigned observations.
        return np.arange(len(values)) % min(self.n_components, 2)


def frame():
    rows = []
    for day, day_offset in [("Day 2", 0.), ("Day 10", 100.)]:
        for treatment, treatment_offset in [("ctrl", 0.), ("drug", 20.)]:
            for index in range(6):
                rows.append({
                    "id": f"{day}-{treatment}-{index}",
                    "day": day,
                    "treatment": treatment,
                    "shape": f"s{index % 2}",
                    "x": day_offset + treatment_offset + index,
                    "y": day_offset + treatment_offset + 2 * index + index % 2,
                })
    return pd.DataFrame(rows)


def options(**overrides):
    values = dict(log_x=False, log_y=False, marginal_plot_type="boxplot",
                  fit_regression=True, fit_gmm=True,
                  max_components=3, min_weight_threshold=.1)
    values.update(overrides)
    return values


@pytest.fixture(autouse=True)
def settings():
    st.session_state.clear()
    yield
    st.session_state.clear()


def test_separated_metadata_uses_fitted_stats_and_canonical_source_labels(monkeypatch):
    fitted = []

    def fit(values, **_kwargs):
        model = StubGMM(values)
        fitted.append(model)
        return model

    monkeypatch.setattr(bivar, "_find_best_gmm", fit)
    data = frame()
    fig, legacy_summary, result = bivar.feature_2d_distribution_plot(
        data, "id", None, "x", "y", color_by=["treatment"], shape_by="shape",
        separate_by="day", analysis_options=options())

    tables = fig.layout.meta["gmm_component_tables"]
    assert [(table["category"], table["group"]) for table in tables] == [
        ("Day 2", "ctrl"), ("Day 2", "drug"),
        ("Day 10", "ctrl"), ("Day 10", "drug"),
    ]
    assert len(fitted) == len(tables) == 4
    for table, model in zip(tables, fitted):
        assert table["features"] == ["x", "y"]
        assert len(table["rows"]) == 3
        for component, (row, mean, covariance, weight) in enumerate(zip(
                table["rows"], model.means_, model.covariances_, model.weights_), 1):
            assert row == {
                "source_label": f'{table["category"]}::{table["group"]}_group{component}',
                "component": component,
                "x_mean_sd": f"{mean[0]:.2f} ± {np.sqrt(covariance[0][0]):.2f}",
                "y_mean_sd": f"{mean[1]:.2f} ± {np.sqrt(covariance[1][1]):.2f}",
                "weight": float(weight),
            }

    # Every fitted component is described even though prediction never emits group3.
    assert not result["2D_GMM_group"].str.endswith("group3").any()
    assert all(table["rows"][2]["source_label"].endswith("group3") for table in tables)
    assert len(result) == len(data)
    assert set(result.id) == set(data.id)
    assert "flim-gmm-table" in legacy_summary


def test_category_switch_updates_stats_only_text_and_keeps_legacy_summary(monkeypatch):
    monkeypatch.setattr(bivar, "_find_best_gmm", lambda values, **_kwargs: StubGMM(values, 2))
    fig, _, _ = bivar.feature_2d_distribution_plot(
        frame(), "id", None, "x", "y", color_by=["treatment"], separate_by="day",
        analysis_options=options())

    meta = fig.layout.meta
    assert set(meta["distribution_statistics_summaries"]) == {"Day 2", "Day 10"}
    for category in meta["distribution_categories"]:
        bivar.select_distribution_category(fig, category)
        meta = fig.layout.meta
        assert meta["distribution_statistics"] == meta["distribution_statistics_summaries"][category]
        assert "Pearson r" in meta["distribution_statistics"]
        assert "Regression R²" in meta["distribution_statistics"]
        assert "flim-gmm-table" not in meta["distribution_statistics"]
        assert "flim-gmm-table" in meta["distribution_summary"]


def test_combined_metadata_has_none_category_and_stats_without_gmm_html(monkeypatch):
    monkeypatch.setattr(bivar, "_find_best_gmm", lambda values, **_kwargs: StubGMM(values, 2))
    fig, legacy_summary, result = bivar.feature_2d_distribution_plot(
        frame(), "id", None, "x", "y", analysis_options=options())

    assert fig.layout.meta["distribution_statistics"] == legacy_summary.split("<style>", 1)[0].rstrip()
    assert "flim-gmm-table" not in fig.layout.meta["distribution_statistics"]
    assert fig.layout.meta["gmm_component_tables"][0]["category"] is None
    assert fig.layout.meta["gmm_component_tables"][0]["group"] == "all_data"
    assert [row["source_label"] for row in fig.layout.meta["gmm_component_tables"][0]["rows"]] == [
        "all_data_group1", "all_data_group2"]
    assert result["2D_GMM_group"].notna().all()


@pytest.mark.parametrize("color_by,separate_by,expected", [
    (["treatment"], None, (None, "ctrl", "ctrl_group1")),
    (None, "day", ("Day 2", "all_data", "Day 2_group1")),
])
def test_color_and_separation_each_qualify_source_labels_independently(
        monkeypatch, color_by, separate_by, expected):
    monkeypatch.setattr(bivar, "_find_best_gmm", lambda values, **_kwargs: StubGMM(values, 2))
    fig, _, _ = bivar.feature_2d_distribution_plot(
        frame(), "id", None, "x", "y", color_by=color_by, separate_by=separate_by,
        analysis_options=options())

    first = fig.layout.meta["gmm_component_tables"][0]
    assert (first["category"], first["group"], first["rows"][0]["source_label"]) == expected


@pytest.mark.parametrize("fit_result,expected_notice", [
    (None, "No suitable GMM"),
    ("single", "Only one GMM component"),
])
def test_zero_or_single_component_fit_has_no_structured_table(
        monkeypatch, fit_result, expected_notice):
    def fit(values, **_kwargs):
        return None if fit_result is None else StubGMM(values, 1)

    monkeypatch.setattr(bivar, "_find_best_gmm", fit)
    fig, legacy_summary, result = bivar.feature_2d_distribution_plot(
        frame(), "id", None, "x", "y", analysis_options=options())

    assert fig.layout.meta["gmm_component_tables"] == []
    assert expected_notice in fig.layout.meta["distribution_statistics"]
    assert expected_notice in legacy_summary
    assert "2D_GMM_group" not in result


def test_invalid_group_has_no_structured_table_and_retains_notices():
    data = frame().iloc[:3].copy()
    data["y"] = 1.
    fig, _, result = bivar.feature_2d_distribution_plot(
        data, "id", None, "x", "y", analysis_options=options())

    assert fig.layout.meta["gmm_component_tables"] == []
    assert "Pearson r and regression unavailable" in fig.layout.meta["distribution_statistics"]
    assert "Skipping GMM" in fig.layout.meta["distribution_statistics"]
    assert "2D_GMM_group" not in result
