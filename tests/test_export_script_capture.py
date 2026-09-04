"""Tests for export-script state capture and app/export parity.

Each test builds a `state` dict (as pages/data_analysis.py collects it), generates
a standalone script via src.export_script.generate_script, executes it with runpy
against a synthetic CSV, and asserts on the resulting namespace/figure.
"""
import runpy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.export_script import generate_script
from src.feature_labels import format_feature_label


def _base_state(method, **overrides):
    state = {
        "csv_filename": "synth.csv",
        "unique_row_id_col": "cell_id",
        "fov_name_col": "image_name",
        "method": method,
        "categorical_filters": {},
        "numerical_filters": [],
        "color_by": [],
        "opacity_by": None,
        "shape_by": None,
        "separate_by": None,
        "point_size": 5,
        "axis_label_size": 12,
        "legend_size": 10,
        "colormap": "tab10",
        "categorical_cols": [],
        "method_params": {},
    }
    state.update(overrides)
    return state


def _run_script(tmp_path, state, df, monkeypatch, script_transform=None):
    df.to_csv(tmp_path / state["csv_filename"], index=False)
    script = generate_script(state)
    if script_transform:
        script = script_transform(script)
    script_path = tmp_path / "analysis.py"
    script_path.write_text(script)
    monkeypatch.chdir(tmp_path)
    try:
        ns = runpy.run_path(str(script_path))
    finally:
        plt.close("all")
    return ns


def _enable_derived_data(script):
    """Simulate the user flipping the opt-in constant in the generated script."""
    assert "SAVE_DERIVED_DATA = False" in script
    return script.replace("SAVE_DERIVED_DATA = False", "SAVE_DERIVED_DATA = True")


def _feature_comparison_params(**overrides):
    params = {
        "selected_var": "feature_a",
        "effect_size_method": "None",
        "mean_or_median": None,
        "statistical_test": "None",
        "log_y": False,
        "add_boxplot": False,
        "connect_means": False,
        "effect_size_threshold": 0.0,
        "custom_order": None,
        "selected_pairs": None,
        "collapse_by": None,
    }
    params.update(overrides)
    return params


def _grouped_df(group_means, n_per_group=20, group_col="treatment", seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for i, (group, mean) in enumerate(group_means.items()):
        frames.append(
            pd.DataFrame(
                {
                    "cell_id": [f"img{i + 1}_{j}" for j in range(n_per_group)],
                    "image_name": [f"img{i + 1}"] * n_per_group,
                    group_col: [group] * n_per_group,
                    "feature_a": rng.normal(mean, 0.1, n_per_group),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Bug 1 — data loading must mirror the app's CSV normalization
# ---------------------------------------------------------------------------

def test_categorical_filter_matches_numeric_column(tmp_path, monkeypatch):
    """App coerces categorical cols to str; filters like ['1','2'] must match an int64 CSV column."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "cell_id": [f"img{i % 3 + 1}_{i}" for i in range(60)],
            "image_name": [f"img{i % 3 + 1}" for i in range(60)],
            "day": [1] * 20 + [2] * 20 + [3] * 20,
            "feature_a": rng.normal(1.0, 0.2, 60),
        }
    )
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["day"],
        categorical_filters={"day": ["1", "2"]},
        method_params={"selected_var": "feature_a"},
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert len(ns["df"]) == 40
    assert ns["df"]["day"].dtype == object


def test_categorical_column_keeps_its_own_spelling_for_grouping(tmp_path, monkeypatch):
    """A categorical is matched by exact name and never renamed; the script must agree."""
    df = _grouped_df({"ctrl": 1.0, "drug": 2.0}, group_col="Treatments")
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["Treatments"],
        color_by=["Treatments"],
        method_params={"selected_var": "feature_a"},
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert "Treatments" in ns["df"].columns
    assert "treatment" not in ns["df"].columns
    assert sorted(ns["color_groups"]) == ["ctrl", "drug"]


def test_duplicate_row_ids_stop_the_script(tmp_path, monkeypatch):
    """The app refuses the file; the script must refuse it too, and say the same thing.

    Both used to keep the first row per id -- consistent, and consistently wrong: the
    script would plot 10 of 11 cells and print a warning nobody reads. The generated
    loader already raises SystemExit on check_and_fix_df's error, so making it an error
    is the whole change on this side.
    """
    df = _grouped_df({"ctrl": 1.0}, n_per_group=10)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate one cell_id
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        method_params={"selected_var": "feature_a"},
    )
    with pytest.raises(SystemExit) as stop:
        _run_script(tmp_path, state, df, monkeypatch)
    assert "cell_id" in str(stop.value) and "identify a row" in str(stop.value)


def test_majority_numeric_object_feature_coerced(tmp_path, monkeypatch, capsys):
    """A feature column with <=1% stray strings is coerced to numeric like the app."""
    df = _grouped_df({"ctrl": 1.0, "drug": 2.0}, n_per_group=60)
    df["feature_a"] = df["feature_a"].astype(object)
    # 1/120 non-numeric; "n.d." is NOT in pandas' default na_values, so it
    # survives read_csv as a string and the column loads as object dtype
    df.loc[0, "feature_a"] = "n.d."
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params={"selected_var": "feature_a"},
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert pd.api.types.is_numeric_dtype(ns["df"]["feature_a"])
    # the app's coercion warning must surface in the script output too
    assert "converted to NaN" in capsys.readouterr().out


def test_an_ignored_column_is_left_alone_by_the_script_too(tmp_path, monkeypatch, capsys):
    """The other half of the app's skip set: the columns the review table marked Ignore.

    `plate_number` is the case the role exists for -- a label that reads as a number, with
    a stray value in it. The app skips coercing it because the user dismissed it, so the
    script must skip it as well: converted here it would report a conversion the app
    suppressed, and with no ANALYSIS_COLUMNS captured (as here) the script's frame would
    carry a numeric column the app's never held.
    """
    df = _grouped_df({"ctrl": 1.0, "drug": 2.0}, n_per_group=60)
    df["plate_number"] = [str(1 + i % 3) for i in range(len(df))]
    df.loc[0, "plate_number"] = "n.d."          # 1/120, under the 1% rule
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        ignored_cols=["plate_number"],
        method_params={"selected_var": "feature_a"},
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert not pd.api.types.is_numeric_dtype(ns["df"]["plate_number"])
    assert "plate_number" not in capsys.readouterr().out


def test_missing_unique_id_column_errors_like_the_app(tmp_path, monkeypatch):
    """The app refuses CSVs without the unique row id column; the script must too."""
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "image_name": ["img1"] * 20,
            "treatment": ["ctrl"] * 20,
            "feature_a": rng.normal(1.0, 0.2, 20),
        }
    )  # no cell_id column
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        method_params={
            "selected_var": "feature_a",
            "log_x": False,
            "apply_gmm": False,
            "intersection_threshold": False,
            "bin_width": None,
        },
    )
    with pytest.raises(SystemExit):
        _run_script(tmp_path, state, df, monkeypatch)


# ---------------------------------------------------------------------------
# Bug 2 — statistical-test-only annotations (no effect size selected)
# ---------------------------------------------------------------------------

def test_stat_test_only_draws_significance_stars(tmp_path, monkeypatch):
    df = _grouped_df({"ctrl": 1.0, "drug": 3.0})
    state = _base_state(
        "Feature Comparison",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params=_feature_comparison_params(statistical_test="Welch's t-test"),
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    texts = [t.get_text() for t in ns["ax"].texts]
    assert any("*" in t for t in texts), f"expected significance stars, got {texts}"


# ---------------------------------------------------------------------------
# Comparison-pair selection capture
# ---------------------------------------------------------------------------

def test_selected_pairs_limits_annotated_pairs(tmp_path, monkeypatch):
    df = _grouped_df({"A": 0.0, "B": 2.0, "C": 4.0}, n_per_group=15)
    base_params = _feature_comparison_params(
        effect_size_method="Absolute Cohen's d",
        mean_or_median="Mean",
    )

    state = _base_state(
        "Feature Comparison",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params={**base_params, "selected_pairs": ["A vs B"]},
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert len(ns["ax"].texts) == 1

    # Reversed labels must also match (app accepts either orientation)
    state["method_params"] = {**base_params, "selected_pairs": ["B vs A"]}
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert len(ns["ax"].texts) == 1

    # None -> annotate all pairs (app default)
    state["method_params"] = {**base_params, "selected_pairs": None}
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert len(ns["ax"].texts) == 3


def test_separate_by_sections_star_only_with_pair_filter(tmp_path, monkeypatch):
    """Star-only annotations and pair filtering must also work inside separate_by sections."""
    rng = np.random.default_rng(4)
    rows = []
    for day in ["1", "2"]:
        for treatment, mean in [("A", 1.0), ("B", 3.0)]:
            for j in range(15):
                rows.append(
                    {
                        "cell_id": f"img_{day}_{treatment}_{j}",
                        "image_name": f"img_{day}",
                        "treatment": treatment,
                        "day": day,
                        "feature_a": rng.normal(mean, 0.1),
                    }
                )
    df = pd.DataFrame(rows)
    state = _base_state(
        "Feature Comparison",
        categorical_cols=["treatment", "day"],
        color_by=["treatment"],
        separate_by="day",
        method_params=_feature_comparison_params(
            statistical_test="Welch's t-test",
            selected_pairs=["A vs B"],
        ),
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    # ax.texts also holds the per-section headers ("1", "2") the export draws below
    # the axis, so filter to the bracket annotations before counting them.
    texts = [t.get_text() for t in ns["ax"].texts if t.get_text() not in {"1", "2"}]
    assert len(texts) == 2  # one bracket per day section
    assert all("*" in t for t in texts)


# ---------------------------------------------------------------------------
# Effect-size threshold capture helper (page-side session read)
# ---------------------------------------------------------------------------

def test_effect_size_threshold_capture_reads_session_keys():
    from src.export_script import get_effect_size_threshold_capture

    session = {"cohens_d_thresh_feature_a": 1.3}
    assert get_effect_size_threshold_capture(session, "Absolute Cohen's d", "feature_a", None) == 1.3
    # Widget defaults are Cohen's d 0.5 and Glass's Delta 0.7 on BOTH paths --
    # src/vis/helpers.py:368-378 and src/vis/univar.py:841-851 are the same block,
    # so separate_by changes neither default. This previously asserted 0.7 for the
    # non-separate path, locking in a capture default the app never used.
    assert get_effect_size_threshold_capture({}, "Absolute Cohen's d", "feature_a", None) == 0.5
    assert get_effect_size_threshold_capture({}, "Absolute Cohen's d", "feature_a", "day") == 0.5
    session = {"glass_delta_thresh_feature_a": 0.9}
    assert get_effect_size_threshold_capture(session, "Glass's Delta", "feature_a", None) == 0.9
    assert get_effect_size_threshold_capture({}, "Glass's Delta", "feature_a", "day") == 0.7
    assert get_effect_size_threshold_capture({}, "None", "feature_a", None) == 0.0


# ---------------------------------------------------------------------------
# GMM hyperparameters capture (Feature Histogram + 2D Feature Distribution)
# ---------------------------------------------------------------------------

def _bimodal_df(n_per_mode=40, seed=1):
    rng = np.random.default_rng(seed)
    values = np.concatenate(
        [rng.normal(1.0, 0.15, n_per_mode), rng.normal(3.0, 0.15, n_per_mode)]
    )
    n = len(values)
    return pd.DataFrame(
        {
            "cell_id": [f"img1_{i}" for i in range(n)],
            "image_name": ["img1"] * n,
            "treatment": ["ctrl"] * n,
            "feature_a": values,
            "feature_b": values * 0.5 + rng.normal(0, 0.1, n),
        }
    )


def test_gmm_hyperparameters_flow_into_histogram_script(tmp_path, monkeypatch):
    df = _bimodal_df()
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        method_params={
            "selected_var": "feature_a",
            "log_x": False,
            "apply_gmm": True,
            "intersection_threshold": False,
            "bin_width": None,
            "gmm_max_components": 2,
            "gmm_min_weight_threshold": 0.2,
        },
    )
    script = generate_script(state)
    assert "GMM_MAX_COMPONENTS = 2" in script
    assert "GMM_MIN_WEIGHT_THRESHOLD = 0.2" in script
    assert "max_components=GMM_MAX_COMPONENTS" in script
    assert "min_weight_threshold=GMM_MIN_WEIGHT_THRESHOLD" in script

    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert ns["gmm"].n_components <= 2


def test_gmm_hyperparameters_flow_into_2d_script(tmp_path, monkeypatch):
    df = _bimodal_df()
    state = _base_state(
        "2D Feature Distribution",
        categorical_cols=["treatment"],
        method_params={
            "selected_x": "feature_a",
            "selected_y": "feature_b",
            "log_x": False,
            "log_y": False,
            "marginal_plot_type": "gaussian fit",
            "fit_regression": False,
            "fit_gmm_2d": True,
            "gmm_max_components": 2,
            "gmm_min_weight_threshold": 0.2,
        },
    )
    script = generate_script(state)
    assert "GMM_MAX_COMPONENTS = 2" in script
    assert "max_components=GMM_MAX_COMPONENTS" in script
    assert "min_weight_threshold=GMM_MIN_WEIGHT_THRESHOLD" in script

    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert ns["best_gmm"].n_components <= 2


# ---------------------------------------------------------------------------
# Histogram bin width capture
# ---------------------------------------------------------------------------

def test_bin_width_capture_used_for_histogram(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "cell_id": [f"img1_{i}" for i in range(60)],
            "image_name": ["img1"] * 60,
            "treatment": ["ctrl"] * 60,
            "feature_a": rng.uniform(0, 10, 60),
        }
    )
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        method_params={
            "selected_var": "feature_a",
            "log_x": False,
            "apply_gmm": False,
            "intersection_threshold": False,
            "bin_width": 0.5,
        },
    )
    script = generate_script(state)
    assert "BIN_WIDTH = 0.5" in script

    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert ns["bin_width"] == 0.5
    assert np.allclose(np.diff(ns["bin_edges"]), 0.5)


def test_bin_width_none_falls_back_to_auto(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "cell_id": [f"img1_{i}" for i in range(60)],
            "image_name": ["img1"] * 60,
            "treatment": ["ctrl"] * 60,
            "feature_a": rng.uniform(0, 10, 60),
        }
    )
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        method_params={
            "selected_var": "feature_a",
            "log_x": False,
            "apply_gmm": False,
            "intersection_threshold": False,
            "bin_width": None,
        },
    )
    script = generate_script(state)
    assert "BIN_WIDTH = None" in script

    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert ns["bin_width"] > 0


# ---------------------------------------------------------------------------
# Shape-by / opacity-by encodings must apply per point in exported scatters
# ---------------------------------------------------------------------------

def _encoding_df(n_per_combo=12, seed=7):
    """Full treatment x cell_line x day grid so every encoding combo is populated."""
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for treatment in ["A", "B"]:
        for cell_line in ["lineA", "lineB"]:
            for day in ["d1", "d2"]:
                for _ in range(n_per_combo):
                    rows.append(
                        {
                            "cell_id": f"img1_{i}",
                            "image_name": "img1",
                            "treatment": treatment,
                            "cell_line": cell_line,
                            "day": day,
                            "feature_a": rng.normal(1.0 if treatment == "A" else 2.0, 0.2),
                            "feature_b": rng.normal(1.0 if day == "d1" else 2.0, 0.2),
                        }
                    )
                    i += 1
    return pd.DataFrame(rows)


def _nonempty_collections(ax):
    return [c for c in ax.collections if len(c.get_offsets()) > 0]


def _marker_signatures(collections):
    """Distinct marker shapes drawn, identified by their path vertices."""
    return {tuple(np.round(c.get_paths()[0].vertices, 6).ravel()) for c in collections}


def test_2d_script_applies_shape_and_opacity_per_point(tmp_path, monkeypatch):
    """The app encodes shape_by/opacity_by per point; the exported 2D scatter must too."""
    df = _encoding_df()
    state = _base_state(
        "2D Feature Distribution",
        categorical_cols=["treatment", "cell_line", "day"],
        color_by=["treatment"],
        shape_by="cell_line",
        opacity_by="day",
        method_params={
            "selected_x": "feature_a",
            "selected_y": "feature_b",
            "log_x": False,
            "log_y": False,
            "marginal_plot_type": "none",
            "fit_regression": False,
            "fit_gmm_2d": False,
        },
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    points = _nonempty_collections(ns["ax_main"])
    # One scatter call per (colour, shape) -- 2 x 2. Opacity is deliberately NOT split
    # into its own calls: it goes in as a per-point alpha array, because splitting it
    # would paint one opacity group wholly over another and create_opacity_mapping
    # raises alpha with sort order, so the most opaque group would always land on top --
    # a paint order the screen does not have. See scatter_with_encodings.
    assert len(points) == 4, f"expected 4 (colour x shape) scatters, got {len(points)}"
    assert len(_marker_signatures(points)) == 2
    # Alpha is a per-point array on every call, carrying both opacity levels.
    for c in points:
        alpha = c.get_alpha()
        assert alpha is not None and not np.isscalar(alpha), (
            "opacity must be a per-point alpha array, not one alpha per sub-group")
        assert set(np.round(np.asarray(alpha), 6)) == {0.3, 1.0}
    # Per-point correspondence: _encoding_df separates the two days on feature_b
    # (d1 ~ N(1.0, 0.2), d2 ~ N(2.0, 0.2)), which is the y axis here -- so each point's
    # alpha must follow its own day, not its sub-group.
    for c in points:
        ys = np.asarray(c.get_offsets())[:, 1]
        alphas = np.asarray(c.get_alpha(), dtype=float)
        np.testing.assert_allclose(np.where(ys < 1.5, 0.3, 1.0), alphas)
    legend_texts = {t.get_text() for t in ns["ax_main"].get_legend().get_texts()}
    assert {"A", "B", "lineA", "lineB", "d1", "d2"} <= legend_texts


def _two_group_2d_df(seed=11):
    """Two color groups, each with a clean x<->y correlation."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, treatment in enumerate(["ctrl", "drug"]):
        for j in range(30):
            x = rng.normal(1.0 + i, 0.3)
            rows.append(
                {
                    "cell_id": f"img{i + 1}_{j}",
                    "image_name": f"img{i + 1}",
                    "treatment": treatment,
                    "feature_a": x,
                    "feature_b": 2.0 * x + rng.normal(0, 0.1),
                }
            )
    return pd.DataFrame(rows)


def _2d_corr_state(fit_regression):
    return _base_state(
        "2D Feature Distribution",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params={
            "selected_x": "feature_a",
            "selected_y": "feature_b",
            "log_x": False,
            "log_y": False,
            "marginal_plot_type": "none",
            "fit_regression": fit_regression,
            "fit_gmm_2d": False,
        },
    )


def test_2d_export_reports_correlation_without_regression(tmp_path, monkeypatch, capsys):
    """The app always prints Pearson r + p per color group; the export must too,
    even when the regression line is off (it was gated behind FIT_REGRESSION)."""
    df = _two_group_2d_df()
    _run_script(tmp_path, _2d_corr_state(fit_regression=False), df, monkeypatch)
    out = capsys.readouterr().out
    # One correlation line per color group (ctrl, drug), with regression OFF.
    assert out.count("Pearson r=") == 2
    # No regression R^2 readout when the line is off.
    assert "R²=" not in out


def test_2d_export_adds_r2_only_when_regression_on(tmp_path, monkeypatch, capsys):
    """Regression ON keeps the per-group correlation and adds R^2 (guards the refactor)."""
    df = _two_group_2d_df()
    _run_script(tmp_path, _2d_corr_state(fit_regression=True), df, monkeypatch)
    out = capsys.readouterr().out
    assert out.count("Pearson r=") == 2
    assert out.count("R²=") == 2


def test_phasor_script_applies_opacity_per_point(tmp_path, monkeypatch):
    rng = np.random.default_rng(8)
    df = _encoding_df()
    df["Lifetime fit free_Ch1: G(1st)"] = rng.uniform(0.2, 0.8, len(df))
    df["Lifetime fit free_Ch1: S(1st)"] = rng.uniform(0.1, 0.4, len(df))
    state = _base_state(
        "Phasor Plot",
        categorical_cols=["treatment", "cell_line", "day"],
        color_by=["treatment"],
        opacity_by="day",
        method_params={
            "selected_channel": "Ch1",
            "phasor_harmonic": 1,
            "phasor_f": 0.08,
            "k_means": False,
            "k_means_clusters": 2,
        },
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    points = _nonempty_collections(ns["ax"])
    # No shape_by, so one scatter call per colour group; opacity rides along as a
    # per-point alpha array rather than splitting the group in two (see the 2D test).
    assert len(points) == 2, f"expected 2 colour-group scatters, got {len(points)}"
    for c in points:
        alpha = c.get_alpha()
        assert alpha is not None and not np.isscalar(alpha), (
            "opacity must be a per-point alpha array, not one alpha per sub-group")
        alphas = np.asarray(alpha, dtype=float)
        assert set(np.round(alphas, 6)) == {0.3, 1.0}
        # _encoding_df is a full grid: each colour group holds 24 rows per day.
        assert sorted(np.bincount(np.searchsorted([0.65], alphas)).tolist()) == [24, 24]


def test_dimension_reduction_script_applies_shape_per_point(tmp_path, monkeypatch):
    df = _encoding_df()
    state = _base_state(
        "Dimension Reduction",
        categorical_cols=["treatment", "cell_line", "day"],
        color_by=["treatment"],
        shape_by="cell_line",
        method_params={
            "selected_features": ["feature_a", "feature_b"],
            "dr_method": "PCA",
            "hyperParam_dict": {},
        },
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    points = _nonempty_collections(ns["ax"])
    assert len(points) == 4, f"expected 4 sub-group scatters, got {len(points)}"
    assert len(_marker_signatures(points)) == 2


def test_feature_comparison_mixed_encoding_groups_split_points(tmp_path, monkeypatch):
    """Color groups containing multiple shape values must split per shape, not fall back to 'o'."""
    df = _encoding_df()
    state = _base_state(
        "Feature Comparison",
        categorical_cols=["treatment", "cell_line", "day"],
        color_by=["treatment"],
        shape_by="cell_line",
        method_params=_feature_comparison_params(),
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    points = _nonempty_collections(ns["ax"])
    assert len(points) == 4, f"expected 4 sub-group scatters, got {len(points)}"
    assert len(_marker_signatures(points)) == 2


# ---------------------------------------------------------------------------
# Phasor K-Means — app and export must run the identical clustering
# ---------------------------------------------------------------------------

def _kmeans_blobs(seed=2):
    """Three overlapping blobs where a single k-means++ init lands in a worse
    local optimum than best-of-10 restarts (seed chosen so the two disagree)."""
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.normal((0, 0), 1.2, (60, 2)),
            rng.normal((4, 4), 1.2, (60, 2)),
            rng.normal((0, 5), 1.2, (60, 2)),
        ]
    )


def test_phasor_kmeans_uses_ten_seeded_restarts():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    from src.vis.bivar import phasor_kmeans

    X = _kmeans_blobs()
    labels, centers = phasor_kmeans(X, 3)

    scaler = StandardScaler().fit(X)
    ten = KMeans(n_clusters=3, random_state=42, n_init=10).fit(scaler.transform(X))
    assert (labels == ten.labels_).all()
    assert np.allclose(centers, scaler.inverse_transform(ten.cluster_centers_))
    # the blobs are chosen so one init finds a worse optimum than ten — the
    # assertions above would fail if phasor_kmeans regressed to a single init
    one = KMeans(n_clusters=3, random_state=42, n_init=1).fit(scaler.transform(X))
    assert not (one.labels_ == ten.labels_).all()


def test_phasor_kmeans_parity_app_vs_export(tmp_path, monkeypatch):
    """The exported script must embed and run the app's clustering function."""
    from src.vis.bivar import phasor_kmeans

    X = _kmeans_blobs()
    n = len(X)
    df = pd.DataFrame(
        {
            "cell_id": [f"img1_{i}" for i in range(n)],
            "image_name": ["img1"] * n,
            "treatment": ["ctrl"] * n,
            "Lifetime fit free_Ch1: G(1st)": X[:, 0],
            "Lifetime fit free_Ch1: S(1st)": X[:, 1],
        }
    )
    state = _base_state(
        "Phasor Plot",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params={
            "selected_channel": "Ch1",
            "phasor_harmonic": 1,
            "phasor_f": 0.08,
            "k_means": True,
            "k_means_clusters": 3,
        },
    )
    assert "def phasor_kmeans(" in generate_script(state)

    ns = _run_script(tmp_path, state, df, monkeypatch)
    app_labels, app_centers = phasor_kmeans(X, 3)
    assert (ns["labels"] == app_labels).all()
    assert np.allclose(ns["centers"], app_centers)


# ---------------------------------------------------------------------------
# Derived-data CSVs — opt-in via SAVE_DERIVED_DATA, mirroring the app's
# download buttons (written only when the user flips the constant to True)
# ---------------------------------------------------------------------------

def test_histogram_gmm_derived_data_saved_behind_flag(tmp_path, monkeypatch):
    df = _bimodal_df()
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params={
            "selected_var": "feature_a",
            "log_x": False,
            "apply_gmm": True,
            "intersection_threshold": False,
            "bin_width": None,
            "gmm_max_components": 2,
            "gmm_min_weight_threshold": 0.2,
        },
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert not (tmp_path / "gmm_grouped_data.csv").exists()  # default off, like the app

    ns = _run_script(tmp_path, state, df, monkeypatch, script_transform=_enable_derived_data)
    saved = pd.read_csv(tmp_path / "gmm_grouped_data.csv")
    assert "_color_group" not in saved.columns  # internal helper, app drops its equivalent
    assert saved["GMM_group"].tolist() == ns["df"]["GMM_group"].tolist()


def test_exported_gmm_group_numbering_matches_app(tmp_path, monkeypatch):
    """`GMM_group` must be numbered by ascending-mean rank on BOTH paths.

    The app labels via `_assign_subpopulation_labels` (group1 == smallest-mean
    component, matching the component table it renders). The export used to label
    with sklearn's *internal* component index instead — `sorted_idx[digitize(...)]`
    on the intersection path and a raw `gmm.predict` on the hard-assignment path —
    so the exported CSV disagreed with the app, and with the export's own printed
    component table, whenever the fitted components were not already in
    ascending-mean order.
    """
    from src.vis.helpers import _find_best_gmm
    from src.vis.univar import _assign_subpopulation_labels

    # Three well-separated components: sklearn's internal component order is then
    # unlikely (1 in 6) to coincide with ascending-mean order by chance.
    rng = np.random.default_rng(7)
    vals = np.concatenate([
        rng.normal(1.0, 0.15, 60),
        rng.normal(5.0, 0.15, 60),
        rng.normal(9.0, 0.15, 60),
    ])
    df = pd.DataFrame({
        "cell_id": [f"img1_{i}" for i in range(vals.size)],
        "image_name": ["img1"] * vals.size,
        "treatment": ["ctrl"] * vals.size,
        "feature_a": vals,
    })
    state = _histogram_state(apply_gmm=True, gmm_max_components=3,
                             gmm_min_weight_threshold=0.05)
    _run_script(tmp_path, state, df, monkeypatch,
                script_transform=_enable_derived_data)
    saved = pd.read_csv(tmp_path / "gmm_grouped_data.csv")

    # App path, fitted independently from the same values with the same call.
    best = _find_best_gmm(vals, max_components=3, min_weight_threshold=0.05)
    expected = _assign_subpopulation_labels(vals, best, None, "ctrl")
    assert saved["GMM_group"].tolist() == expected

    # ...and the invariant those labels encode: group N has the Nth-smallest mean.
    means = saved.groupby("GMM_group")["feature_a"].mean()
    ordered = [means[k] for k in sorted(means.index)]
    assert ordered == sorted(ordered), f"labels must increase with mean: {means.to_dict()}"


def test_histogram_gmm_script_emits_no_future_warnings(tmp_path, monkeypatch):
    """String labels must not be assigned into a float column (pandas 3.x error)."""
    import warnings

    df = _bimodal_df()
    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params={
            "selected_var": "feature_a",
            "log_x": False,
            "apply_gmm": True,
            "intersection_threshold": False,
            "bin_width": None,
            "gmm_max_components": 2,
            "gmm_min_weight_threshold": 0.2,
        },
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ns = _run_script(tmp_path, state, df, monkeypatch)
    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert not future, [str(w.message) for w in future]
    assert set(ns["df"]["GMM_group"].dropna()) == {"ctrl_group1", "ctrl_group2"}


def test_2d_gmm_derived_data_saved_behind_flag(tmp_path, monkeypatch):
    df = _bimodal_df()
    state = _base_state(
        "2D Feature Distribution",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params={
            "selected_x": "feature_a",
            "selected_y": "feature_b",
            "log_x": False,
            "log_y": False,
            "marginal_plot_type": "none",
            "fit_regression": False,
            "fit_gmm_2d": True,
            "gmm_max_components": 2,
            "gmm_min_weight_threshold": 0.2,
        },
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert not (tmp_path / "2D_gmm_data.csv").exists()

    ns = _run_script(tmp_path, state, df, monkeypatch, script_transform=_enable_derived_data)
    saved = pd.read_csv(tmp_path / "2D_gmm_data.csv")
    assert "_color_group" not in saved.columns
    # the app assigns per-point labels via best_gmm.predict when n_components > 1
    assert saved["2D_GMM_group"].nunique() == 2
    assert saved["2D_GMM_group"].tolist() == ns["df"]["2D_GMM_group"].tolist()


def test_phasor_kmeans_derived_data_saved_behind_flag(tmp_path, monkeypatch):
    from src.vis.bivar import phasor_kmeans

    X = _kmeans_blobs()
    n = len(X)
    df = pd.DataFrame(
        {
            "cell_id": [f"img1_{i}" for i in range(n)],
            "image_name": ["img1"] * n,
            "treatment": ["ctrl"] * n,
            "Lifetime fit free_Ch1: G(1st)": X[:, 0],
            "Lifetime fit free_Ch1: S(1st)": X[:, 1],
        }
    )
    state = _base_state(
        "Phasor Plot",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params={
            "selected_channel": "Ch1",
            "phasor_harmonic": 1,
            "phasor_f": 0.08,
            "k_means": True,
            "k_means_clusters": 3,
        },
    )
    _run_script(tmp_path, state, df, monkeypatch)
    assert not (tmp_path / "kmeans_clustered_data.csv").exists()

    _run_script(tmp_path, state, df, monkeypatch, script_transform=_enable_derived_data)
    saved = pd.read_csv(tmp_path / "kmeans_clustered_data.csv")
    assert "_color_group" not in saved.columns
    app_labels, _ = phasor_kmeans(X, 3)
    # app label format: {color_group}_group{label + 1}
    assert saved["k_means_cluster"].tolist() == [f"ctrl_group{l + 1}" for l in app_labels]


# ---------------------------------------------------------------------------
# Phasor reference geometry — lifetime markers must follow the harmonic
# ---------------------------------------------------------------------------

def _phasor_df(seed=8):
    rng = np.random.default_rng(seed)
    n = 30
    df = pd.DataFrame(
        {
            "cell_id": [f"img1_{i}" for i in range(n)],
            "image_name": ["img1"] * n,
            "treatment": ["ctrl"] * n,
        }
    )
    for label in ("1st", "2nd"):
        df[f"Lifetime fit free_Ch1: G({label})"] = rng.uniform(0.2, 0.8, n)
        df[f"Lifetime fit free_Ch1: S({label})"] = rng.uniform(0.1, 0.4, n)
    return df


def _phasor_state(harmonic, f=0.08, **overrides):
    return _base_state(
        "Phasor Plot",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        **overrides,
        method_params={
            "selected_channel": "Ch1",
            "phasor_harmonic": harmonic,
            "phasor_f": f,
            "k_means": False,
            "k_means_clusters": 2,
        },
    )


def _exported_lifetime_markers(ax):
    """The single-point 'ko' markers the phasor template draws, in draw order."""
    return [
        (float(ln.get_xdata()[0]), float(ln.get_ydata()[0]))
        for ln in ax.lines
        if ln.get_marker() == "o" and len(ln.get_xdata()) == 1
    ]


def _app_lifetime_markers(f, harmonic):
    """The app's marker coordinates, read back off its Plotly figure."""
    import plotly.graph_objects as go

    from src.vis.bivar import _create_phasor_background

    fig = go.Figure()
    _create_phasor_background(fig, "black", f, harmonic)
    trace = next(t for t in fig.data if t.name == "Lifetime Markers")
    return list(zip([float(v) for v in trace.x], [float(v) for v in trace.y]))


@pytest.mark.parametrize("harmonic", [1, 2])
def test_phasor_lifetime_markers_match_the_app_at_each_harmonic(tmp_path, monkeypatch, harmonic):
    f = 0.08
    ns = _run_script(tmp_path, _phasor_state(harmonic, f), _phasor_df(), monkeypatch)
    exported = _exported_lifetime_markers(ns["ax"])
    expected = _app_lifetime_markers(f, harmonic)
    assert len(exported) == len(expected) == 11
    for (gx, sy), (ex, ey) in zip(exported, expected):
        assert gx == pytest.approx(ex, abs=1e-12)
        assert sy == pytest.approx(ey, abs=1e-12)


@pytest.mark.parametrize("harmonic", [1, 2])
def test_phasor_marker_labels_name_the_lifetime_they_mark(tmp_path, monkeypatch, harmonic):
    """A marker labelled "2 ns" must sit where tau=2 ns lands at this harmonic.

    G/S for harmonic n are computed at n*omega (fov_extraction.py passes
    harmonic=h to phasor_from_signal), so the reference point for tau is
    g = 1/(1+(n*2*pi*f*tau)^2). Ignoring n put every 2nd-harmonic label on the
    coordinate of a lifetime twice as long as the label claimed.
    """
    f = 0.08
    ns = _run_script(tmp_path, _phasor_state(harmonic, f), _phasor_df(), monkeypatch)
    labelled = {t.get_text(): t.xy for t in ns["ax"].texts if t.get_text().endswith(" ns")}
    assert set(labelled) == {"0.5 ns", "1 ns", "2 ns", "3 ns", "4 ns", "5 ns"}
    for text, (gx, sy) in labelled.items():
        tau = float(text.removesuffix(" ns"))
        wt = harmonic * 2 * np.pi * f * tau
        assert gx == pytest.approx(1.0 / (1.0 + wt**2), abs=1e-12)
        assert sy == pytest.approx(wt / (1.0 + wt**2), abs=1e-12)


def test_phasor_frequency_annotation_reports_the_harmonic_frequency(tmp_path, monkeypatch):
    ns = _run_script(tmp_path, _phasor_state(2, 0.08), _phasor_df(), monkeypatch)
    freq = next(t.get_text() for t in ns["ax"].texts if t.get_text().startswith("f = "))
    assert freq.splitlines() == ["f = 160.0 MHz", "(2 x 80.0 MHz)"]

    ns = _run_script(tmp_path, _phasor_state(1, 0.08), _phasor_df(), monkeypatch)
    freq = next(t.get_text() for t in ns["ax"].texts if t.get_text().startswith("f = "))
    assert freq == "f = 80.0 MHz"


# ---------------------------------------------------------------------------
# Column universe — the export must analyse the columns get_features() kept
# ---------------------------------------------------------------------------

def _column_universe_df():
    n = 12
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "cell_id": [f"img1_{i}" for i in range(n)],
            "image_name": ["img1"] * n,
            "treatment": ["ctrl", "drug"] * (n // 2),
            # unconfigured, non-numeric metadata: the app drops it, like `dish`
            # in example_data/Data_Analysis/inhibitors.csv
            "dish": [f"plate{i % 3}" for i in range(n)],
            "Lifetime fit_Ch1: feature_a": rng.normal(1.0, 0.1, n),
        }
    )


def test_exported_frame_keeps_the_same_columns_the_app_does(tmp_path, monkeypatch):
    from src.dataset_io import get_features

    df = _column_universe_df()
    categorical_cols = ["treatment"]
    app_df, _, _, error_msg = get_features(df.copy(), categorical_cols, use_data_extraction=True)
    assert error_msg == ""
    assert "dish" in df.columns and "dish" not in app_df.columns, "app should drop `dish`"

    state = _base_state(
        "Feature Comparison",
        categorical_cols=categorical_cols,
        color_by=["treatment"],
        analysis_columns=list(app_df.columns),
        method_params=_feature_comparison_params(selected_var="Lifetime fit_Ch1: feature_a"),
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    exported_cols = [c for c in ns["df"].columns if not c.startswith("_")]
    assert exported_cols == list(app_df.columns)


def test_derived_csv_excludes_columns_the_app_pruned(tmp_path, monkeypatch):
    from src.dataset_io import get_features

    df = _column_universe_df()
    df["Lifetime fit_Ch1: feature_a"] = np.r_[
        np.random.default_rng(4).normal(1.0, 0.05, 6),
        np.random.default_rng(5).normal(3.0, 0.05, 6),
    ]
    app_df, _, _, _ = get_features(df.copy(), ["treatment"], use_data_extraction=True)

    state = _base_state(
        "Feature Histogram",
        categorical_cols=["treatment"],
        color_by=[],
        analysis_columns=list(app_df.columns),
        method_params={
            "selected_var": "Lifetime fit_Ch1: feature_a",
            "log_x": False,
            "apply_gmm": True,
            "intersection_threshold": False,
            "bin_width": None,
            "gmm_max_components": 2,
            "gmm_min_weight_threshold": 0.1,
        },
    )
    _run_script(tmp_path, state, df, monkeypatch, script_transform=_enable_derived_data)
    saved = pd.read_csv(tmp_path / "gmm_grouped_data.csv")
    assert "dish" not in saved.columns
    assert set(app_df.columns).issubset(saved.columns)


def test_missing_analysed_column_warns_instead_of_raising(tmp_path, monkeypatch, capsys):
    df = _column_universe_df().drop(columns=["dish"])
    state = _base_state(
        "Feature Comparison",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        # names a column this CSV no longer has
        analysis_columns=["cell_id", "image_name", "treatment", "Lifetime fit_Ch1: feature_a", "gone"],
        method_params=_feature_comparison_params(selected_var="Lifetime fit_Ch1: feature_a"),
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert "gone" in capsys.readouterr().out
    assert "gone" not in ns["df"].columns


# ---------------------------------------------------------------------------
# Regression net — every method's script must compile with the shared sections
# ---------------------------------------------------------------------------

def test_all_methods_generate_compilable_scripts():
    method_params = {
        "Feature Histogram": {
            "selected_var": "feature_a",
            "log_x": False,
            "apply_gmm": True,
            "intersection_threshold": True,
            "bin_width": None,
            "gmm_max_components": 3,
            "gmm_min_weight_threshold": 0.1,
        },
        "Feature Comparison": _feature_comparison_params(
            effect_size_method="Glass's Delta",
            mean_or_median="Median",
            statistical_test="Independent t-test",
            selected_pairs=["A vs B"],
        ),
        "2D Feature Distribution": {
            "selected_x": "feature_a",
            "selected_y": "feature_b",
            "log_x": True,
            "log_y": False,
            "marginal_plot_type": "boxplot",
            "fit_regression": True,
            "fit_gmm_2d": True,
            "gmm_max_components": 3,
            "gmm_min_weight_threshold": 0.1,
        },
        "Phasor Plot": {
            "selected_channel": "Ch1",
            "phasor_harmonic": 1,
            "phasor_f": 0.08,
            "k_means": True,
            "k_means_clusters": 2,
        },
        "Dimension Reduction": {
            "selected_features": ["feature_a", "feature_b"],
            "dr_method": "PCA",
            "hyperParam_dict": {},
        },
        "Classification": {
            "selected_features": ["feature_a", "feature_b"],
            "classification_method": "Random Forest",
            "splits": 0.7,
            "sampling_method": "None",
            "class_weight": "None",
            "threshold_method": "None",
            "classifier_params": {},
            "classify_by": ["treatment"],
            "classify_classes": ["ctrl", "drug"],
        },
    }
    for method, mp in method_params.items():
        state = _base_state(method, categorical_cols=["treatment"], method_params=mp)
        script = generate_script(state)
        compile(script, f"<export:{method}>", "exec")


# ---------------------------------------------------------------------------
# Tier-1 app↔export parity fixes (2026-06-11 audit)
# ---------------------------------------------------------------------------

def _2d_gmm_state():
    return _base_state(
        "2D Feature Distribution",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params={
            "selected_x": "feature_a", "selected_y": "feature_b",
            "log_x": False, "log_y": False, "marginal_plot_type": "none",
            "fit_regression": False, "fit_gmm_2d": True,
            "gmm_max_components": 3, "gmm_min_weight_threshold": 0.1,
        },
    )


def test_2d_single_component_gmm_draws_no_ellipse(tmp_path, monkeypatch):
    """App draws 2D GMM ellipses only when n_components>1; a unimodal group gets none."""
    from matplotlib.patches import Ellipse
    rng = np.random.default_rng(20)
    n = 80
    df = pd.DataFrame({
        "cell_id": [f"img1_{i}" for i in range(n)],
        "image_name": ["img1"] * n,
        "treatment": ["ctrl"] * n,
        "feature_a": rng.normal(1.0, 0.2, n),
        "feature_b": rng.normal(1.0, 0.2, n),
    })
    ns = _run_script(tmp_path, _2d_gmm_state(), df, monkeypatch)
    ellipses = [p for p in ns["ax_main"].patches if isinstance(p, Ellipse)]
    assert ellipses == []


def test_2d_two_component_gmm_still_draws_ellipses(tmp_path, monkeypatch):
    """Guard: a genuinely bimodal group must still get its ellipses."""
    from matplotlib.patches import Ellipse
    rng = np.random.default_rng(21)
    pts = np.vstack([rng.normal([0.5, 0.5], 0.1, (50, 2)),
                     rng.normal([2.5, 2.5], 0.1, (50, 2))])
    n = len(pts)
    df = pd.DataFrame({
        "cell_id": [f"img1_{i}" for i in range(n)],
        "image_name": ["img1"] * n,
        "treatment": ["ctrl"] * n,
        "feature_a": pts[:, 0], "feature_b": pts[:, 1],
    })
    ns = _run_script(tmp_path, _2d_gmm_state(), df, monkeypatch)
    ellipses = [p for p in ns["ax_main"].patches if isinstance(p, Ellipse)]
    assert len(ellipses) >= 2


def test_phasor_kmeans_hulls_colored_by_group_not_cluster(tmp_path, monkeypatch):
    """App colors each group's k-means hulls/centroids by the GROUP color; the
    export must too — not a per-cluster tab10 ramp."""
    import matplotlib.colors as mcolors
    rng = np.random.default_rng(30)
    df = _encoding_df()
    df["Lifetime fit free_Ch1: G(1st)"] = rng.uniform(0.2, 0.8, len(df))
    df["Lifetime fit free_Ch1: S(1st)"] = rng.uniform(0.1, 0.4, len(df))
    state = _base_state(
        "Phasor Plot",
        categorical_cols=["treatment", "cell_line", "day"],
        color_by=["treatment"],
        method_params={
            "selected_channel": "Ch1", "phasor_harmonic": 1, "phasor_f": 0.08,
            "k_means": True, "k_means_clusters": 3,
        },
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    centroids = [ln for ln in ns["ax"].lines if ln.get_marker() == "x"]
    colors = {tuple(np.round(mcolors.to_rgba(ln.get_color()), 6)) for ln in centroids}
    # 2 color groups x 3 clusters = 6 centroids, but one color per GROUP, not per cluster.
    assert len(centroids) == 6
    expected = {tuple(np.round(mcolors.to_rgba(ns["color_map"][g][:3]), 6))
                for g in ns["color_groups"]}
    assert colors == expected


def _histogram_state(**mp_overrides):
    mp = {
        "selected_var": "feature_a", "log_x": False,
        "apply_gmm": False, "intersection_threshold": False, "bin_width": None,
    }
    mp.update(mp_overrides)
    return _base_state("Feature Histogram", categorical_cols=["treatment"],
                       color_by=["treatment"], method_params=mp)


def test_histogram_skewness_uses_app_seven_way_label(tmp_path, monkeypatch, capsys):
    """Near-symmetric data is 'almost symmetric' (app's 7-way ladder), not the
    export's old 3-way 'approximately symmetric'."""
    rng = np.random.default_rng(40)
    base = rng.uniform(0, 5, 150)
    vals = np.concatenate([5 - base, 5 + base])  # symmetric about 5 -> |skew| ~ 0
    df = pd.DataFrame({
        "cell_id": [f"img1_{i}" for i in range(len(vals))],
        "image_name": ["img1"] * len(vals),
        "treatment": ["ctrl"] * len(vals),
        "feature_a": vals,
    })
    _run_script(tmp_path, _histogram_state(), df, monkeypatch)
    out = capsys.readouterr().out
    assert "almost symmetric" in out


def test_histogram_skewness_value_is_bias_corrected(tmp_path, monkeypatch, capsys):
    """Printed skewness must equal pandas .skew() (bias-corrected), not scipy's
    uncorrected default — they differ at small n."""
    import re
    rng = np.random.default_rng(41)
    vals = rng.normal(0, 1, 15) ** 2  # right-skewed, small n
    df = pd.DataFrame({
        "cell_id": [f"img1_{i}" for i in range(len(vals))],
        "image_name": ["img1"] * len(vals),
        "treatment": ["ctrl"] * len(vals),
        "feature_a": vals,
    })
    _run_script(tmp_path, _histogram_state(), df, monkeypatch)
    out = capsys.readouterr().out
    m = re.search(r"skewness = (-?\d+\.\d+)", out)
    assert m is not None
    assert float(m.group(1)) == round(pd.Series(vals).skew(), 3)


def test_feature_gmm_single_component_group_left_unlabeled(tmp_path, monkeypatch):
    """App assigns GMM_group only when n_components>1; a unimodal group stays NaN."""
    rng = np.random.default_rng(42)
    uni = rng.normal(5.0, 0.5, 80)
    bi = np.concatenate([rng.normal(2.0, 0.3, 40), rng.normal(8.0, 0.3, 40)])
    df = pd.DataFrame({
        "cell_id": [f"img1_{i}" for i in range(160)],
        "image_name": ["img1"] * 160,
        "treatment": ["uni"] * 80 + ["bi"] * 80,
        "feature_a": np.concatenate([uni, bi]),
    })
    state = _histogram_state(apply_gmm=True, gmm_max_components=3,
                             gmm_min_weight_threshold=0.1)
    ns = _run_script(tmp_path, state, df, monkeypatch)
    out = ns["df"]
    # Unimodal group must NOT receive a fabricated subpopulation label — the app
    # leaves single-component groups unassigned (whatever sentinel pandas fills in).
    # The bimodal group must still be labeled.
    uni = out.loc[out["treatment"] == "uni", "GMM_group"].astype(str)
    bi = out.loc[out["treatment"] == "bi", "GMM_group"].astype(str)
    assert not uni.str.startswith("uni_group").any()
    assert bi.str.startswith("bi_group").any()


# ---------------------------------------------------------------------------
# Tier-2 app↔export parity fixes (edge cases)
# ---------------------------------------------------------------------------

def test_2d_skips_constant_column_group_like_app(tmp_path, monkeypatch, capsys):
    """App skips correlation for a group with a constant axis (nunique<2); export too."""
    rows = []
    for i in range(30):
        rows.append({"cell_id": f"img1_{i}", "image_name": "img1", "treatment": "var",
                     "feature_a": float(i), "feature_b": float(i) * 1.5})
    for i in range(30):
        rows.append({"cell_id": f"img2_{i}", "image_name": "img2", "treatment": "const",
                     "feature_a": float(i), "feature_b": 7.0})  # constant y
    df = pd.DataFrame(rows)
    state = _base_state(
        "2D Feature Distribution", categorical_cols=["treatment"], color_by=["treatment"],
        method_params={"selected_x": "feature_a", "selected_y": "feature_b",
                       "log_x": False, "log_y": False, "marginal_plot_type": "none",
                       "fit_regression": False, "fit_gmm_2d": False})
    _run_script(tmp_path, state, df, monkeypatch)
    out = capsys.readouterr().out
    assert "var: Pearson r" in out
    assert "const: Pearson r" not in out


def test_feature_comparison_logy_refuses_negative_like_app(tmp_path, monkeypatch, capsys):
    """App refuses Log Y on negative data (warns, plots raw); export must too — not silent NaN."""
    df = _grouped_df({"ctrl": 1.0, "drug": -2.0})  # drug group is negative
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        method_params=_feature_comparison_params(selected_var="feature_a", log_y=True))
    ns = _run_script(tmp_path, state, df, monkeypatch)
    out = capsys.readouterr().out
    assert "Cannot apply log" in out
    assert ns["df"]["feature_a"].notna().all()  # raw values preserved, no log-of-negative NaN


def test_classification_does_not_silently_drop_nan_feature_rows(tmp_path, monkeypatch):
    """The app passes feature rows straight to sklearn (errors on NaN); the export must
    not silently drop NaN-feature rows and train on a reduced subset."""
    state = _base_state(
        "Classification", categorical_cols=["treatment"],
        method_params={"selected_features": ["feature_a", "feature_b"],
                       "classification_method": "Random Forest", "splits": 0.7,
                       "sampling_method": "None", "class_weight": "None",
                       "threshold_method": "None", "classifier_params": {},
                       "classify_by": ["treatment"], "classify_classes": ["ctrl", "drug"]})
    script = generate_script(state)
    assert "notna().all(axis=1)" not in script


# ---------------------------------------------------------------------------
# Tier-3 app↔export parity fixes (figure faithfulness)
# ---------------------------------------------------------------------------

def test_2d_log_axis_labels_match_app(tmp_path, monkeypatch):
    """Logged 2D axes are relabeled log₁₀(col), matching the app."""
    rng = np.random.default_rng(60)
    n = 40
    df = pd.DataFrame({"cell_id": [f"img1_{i}" for i in range(n)], "image_name": ["img1"] * n,
                       "treatment": ["ctrl"] * n, "feature_a": rng.uniform(1, 10, n),
                       "feature_b": rng.uniform(1, 10, n)})
    state = _base_state(
        "2D Feature Distribution", categorical_cols=["treatment"], color_by=["treatment"],
        method_params={"selected_x": "feature_a", "selected_y": "feature_b",
                       "log_x": True, "log_y": False, "marginal_plot_type": "none",
                       "fit_regression": False, "fit_gmm_2d": False})
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert ns["ax_main"].get_xlabel() == "log₁₀(feature_a)"
    assert ns["ax_main"].get_ylabel() == "feature_b"


def test_a_blank_fov_cell_becomes_an_na_level_on_the_axis(tmp_path, monkeypatch):
    """A blank FOV cell is an ordinary N/A level -- check_and_fix_df fills it, and
    grouping by that column puts "N/A" on the axis. The export labels it and prints
    no warning. Carried by Feature Comparison; the subject is the fill, not the
    method."""
    rows = []
    for i in range(10):
        rows.append({"cell_id": f"a{i}", "image_name": "img1", "treatment": "A", "feature_a": float(i)})
    for i in range(10):
        rows.append({"cell_id": f"b{i}", "image_name": "img2", "treatment": "A", "feature_a": float(i)})
    for i in range(10):
        rows.append({"cell_id": f"c{i}", "image_name": None, "treatment": "A", "feature_a": float(i)})
    df = pd.DataFrame(rows)
    state = _base_state("Feature Comparison", categorical_cols=["treatment"],
                        color_by=["image_name"],
                        method_params=_feature_comparison_params())
    script = generate_script(state)
    assert "Could not find the FOV column" not in script
    assert "missing fov name" not in script
    ns = _run_script(tmp_path, state, df, monkeypatch)
    labels = [t.get_text() for t in ns["ax"].get_xticklabels()]
    assert any("N/A" in label for label in labels)


def test_phasor_annotates_six_lifetime_markers_like_app(tmp_path, monkeypatch):
    """App labels exactly 6 lifetime markers (0.5–5 ns); export must not label all 11."""
    rng = np.random.default_rng(61)
    df = _encoding_df()
    df["Lifetime fit free_Ch1: G(1st)"] = rng.uniform(0.2, 0.8, len(df))
    df["Lifetime fit free_Ch1: S(1st)"] = rng.uniform(0.1, 0.4, len(df))
    state = _base_state(
        "Phasor Plot", categorical_cols=["treatment", "cell_line", "day"], color_by=["treatment"],
        method_params={"selected_channel": "Ch1", "phasor_harmonic": 1, "phasor_f": 0.08,
                       "k_means": False, "k_means_clusters": 2})
    ns = _run_script(tmp_path, state, df, monkeypatch)
    annos = [a.get_text() for a in ns["ax"].texts if a.get_text().endswith("ns")]
    assert len(annos) == 6


def test_dimension_reduction_pca_label_format_matches_app(tmp_path, monkeypatch):
    """PCA axis labels: 'PC1(NN.NN%)' — no space, two decimals — matching the app."""
    import re
    df = _encoding_df()
    state = _base_state(
        "Dimension Reduction", categorical_cols=["treatment", "cell_line", "day"], color_by=["treatment"],
        method_params={"selected_features": ["feature_a", "feature_b"], "dr_method": "PCA",
                       "hyperParam_dict": {}})
    ns = _run_script(tmp_path, state, df, monkeypatch)
    xlabel = ns["ax"].get_xlabel()
    assert xlabel.startswith("PC1(")
    assert re.search(r"PC1\(\d+\.\d\d%\)", xlabel)


def test_feature_comparison_boxplot_shows_mean(tmp_path, monkeypatch):
    """App's boxplot shows the mean (boxmean=True); export must enable showmeans."""
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        method_params=_feature_comparison_params(selected_var="feature_a", add_boxplot=True))
    script = generate_script(state)
    assert "showmeans=True" in script


def test_feature_comparison_bracket_spacing_uses_global_range(tmp_path, monkeypatch):
    """Bracket spacing must use the global data range across sections, like the app —
    not a per-section range."""
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        method_params=_feature_comparison_params(selected_var="feature_a",
                                                 effect_size_method="Absolute Cohen's d",
                                                 mean_or_median="Mean"))
    script = generate_script(state)
    assert "all_y = df[SELECTED_VAR].dropna()" in script
    assert "all_y = sec_df[SELECTED_VAR].dropna()" not in script


def test_2d_point_alpha_matches_app_effective(tmp_path, monkeypatch):
    """2D points (no opacity encoding) use the app's effective alpha 0.8 (1.0 color × 0.8 marker),
    not the export's old flat 0.7."""
    rng = np.random.default_rng(62)
    n = 40
    df = pd.DataFrame({"cell_id": [f"img1_{i}" for i in range(n)], "image_name": ["img1"] * n,
                       "treatment": ["ctrl"] * n, "feature_a": rng.normal(1.0, 0.3, n),
                       "feature_b": rng.normal(1.0, 0.3, n)})
    state = _base_state(
        "2D Feature Distribution", categorical_cols=["treatment"], color_by=["treatment"],
        method_params={"selected_x": "feature_a", "selected_y": "feature_b",
                       "log_x": False, "log_y": False, "marginal_plot_type": "none",
                       "fit_regression": False, "fit_gmm_2d": False})
    ns = _run_script(tmp_path, state, df, monkeypatch)
    points = _nonempty_collections(ns["ax_main"])
    assert {round(c.get_alpha(), 6) for c in points} == {0.8}


# ---------------------------------------------------------------------------
# FLIM feature axis labels — GUI↔export parity via the SAME inlined helper
# ---------------------------------------------------------------------------

def _flim_df(seed=0, n_per_group=20):
    """Synthetic dataset using the real FP column naming convention."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, group in enumerate(["ctrl", "drug"]):
        frames.append(pd.DataFrame({
            "cell_id": [f"img{i + 1}_{j}" for j in range(n_per_group)],
            "image_name": [f"img{i + 1}"] * n_per_group,
            "treatment": [group] * n_per_group,
            "Lifetime fit_nadh: t1": rng.normal(390, 20, n_per_group),
            "Lifetime fit_nadh: a1": rng.normal(68, 5, n_per_group),
            "Lifetime fit_fad: t2": rng.normal(2000, 100, n_per_group),
        }))
    return pd.concat(frames, ignore_index=True)


def test_export_inlines_a_single_feature_label_helper():
    """The export must reuse the app's helper (inlined once), never re-implement it."""
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        method_params=_feature_comparison_params(selected_var="Lifetime fit_nadh: t1"))
    script = generate_script(state)
    assert script.count("def format_feature_label") == 1
    assert "format_feature_label(SELECTED_VAR" in script  # called with engine='mpl'


def test_export_modulation_label_matches_app_mpl(tmp_path, monkeypatch):
    """Modulation lifetime (Tau_mod) exports as matplotlib mathtext matching the helper."""
    col = "Lifetime fit free_nadh: Tau_mod"
    df = _flim_df()
    df[col] = 1.8  # ns-scale modulation lifetime
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        method_params=_feature_comparison_params(selected_var=col))
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert ns["ax"].get_ylabel() == format_feature_label(col, engine="mpl")
    assert ns["ax"].get_ylabel() == r"nadh $τ_{\mathrm{mod}}$ (ns)"


def test_export_axis_label_matches_app(tmp_path, monkeypatch):
    col = "Lifetime fit_nadh: t1"
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        method_params=_feature_comparison_params(selected_var=col))
    ns = _run_script(tmp_path, state, _flim_df(), monkeypatch)
    assert format_feature_label(col) == "nadh τ₁ (ps)"        # app notation
    assert ns["ax"].get_ylabel() == format_feature_label(col)  # export == app


def test_export_2d_axis_labels_match_app_including_log(tmp_path, monkeypatch):
    x, y = "Lifetime fit_nadh: a1", "Lifetime fit_fad: t2"
    state = _base_state(
        "2D Feature Distribution", categorical_cols=["treatment"], color_by=["treatment"],
        method_params={"selected_x": x, "selected_y": y, "log_x": True, "log_y": False,
                       "marginal_plot_type": "none", "fit_regression": False,
                       "fit_gmm_2d": False})
    ns = _run_script(tmp_path, state, _flim_df(), monkeypatch)
    # log axis wraps the pretty label; non-log axis is the pretty label itself
    assert ns["ax_main"].get_xlabel() == f"log₁₀({format_feature_label(x)})"
    assert ns["ax_main"].get_ylabel() == format_feature_label(y)
    assert format_feature_label(x) == "nadh α₁ (%)"
    assert format_feature_label(y) == "fad τ₂ (ps)"


def test_export_histogram_log_x_label(tmp_path, monkeypatch):
    """Histogram x-axis is relabeled log₁₀(feature) when LOG_X, matching the app."""
    col = "Lifetime fit_nadh: t1"
    state = _base_state(
        "Feature Histogram", categorical_cols=["treatment"], color_by=["treatment"],
        method_params={"selected_var": col, "log_x": True, "apply_gmm": False, "bin_width": None})
    ns = _run_script(tmp_path, state, _flim_df(), monkeypatch)
    assert ns["ax"].get_xlabel() == f"log₁₀({format_feature_label(col)})"  # nadh τ₁ (ps)


def test_export_gmm_log_x_label(tmp_path, monkeypatch):
    """GMM x-axis is relabeled log₁₀(feature) when LOG_X, matching the app and the histogram."""
    col = "Lifetime fit_nadh: t1"
    state = _base_state(
        "Feature Histogram", categorical_cols=["treatment"], color_by=["treatment"],
        method_params={"selected_var": col, "log_x": True, "apply_gmm": True,
                       "intersection_threshold": True, "bin_width": None,
                       "gmm_max_components": 3, "gmm_min_weight_threshold": 0.1})
    ns = _run_script(tmp_path, state, _flim_df(), monkeypatch)
    assert ns["ax"].get_xlabel() == f"log₁₀({format_feature_label(col)})"


def test_export_feature_comparison_log_y_label(tmp_path, monkeypatch):
    """Feature Comparison y-axis is relabeled log₁₀(feature) when LOG_Y, matching the app (univar.py:788)."""
    col = "Lifetime fit_nadh: t1"
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        method_params=_feature_comparison_params(selected_var=col, log_y=True))
    ns = _run_script(tmp_path, state, _flim_df(), monkeypatch)
    assert ns["ax"].get_ylabel() == f"log₁₀({format_feature_label(col)})"  # nadh τ₁ (ps)


def test_export_phasor_axes_lowercase_like_app(tmp_path, monkeypatch):
    """Phasor axes use lowercase g/s in both the app and the export."""
    df = _flim_df()
    df["Lifetime fit free_nadh: G(1st)"] = 0.6
    df["Lifetime fit free_nadh: S(1st)"] = 0.45
    state = _base_state(
        "Phasor Plot", categorical_cols=["treatment"], color_by=["treatment"],
        method_params={"phasor_x": "Lifetime fit free_nadh: G(1st)",
                       "phasor_y": "Lifetime fit free_nadh: S(1st)", "k_means": False})
    script = generate_script(state)
    assert 'ax.set_xlabel("g"' in script
    assert 'ax.set_ylabel("s"' in script


def test_feature_comparison_constant_group_keeps_uniform_jitter(tmp_path, monkeypatch):
    """A constant-valued group has no KDE.

    The app falls back to uniform jitter (`_estimate_density_1d` returns zero
    density, and `univar.py` then normalises to ones) so the points stay visible
    as a spread cloud. The export used to call `gaussian_kde` directly and fall
    back to *zero* jitter, collapsing the whole group onto a single x position.
    """
    n = 12
    df = pd.DataFrame({
        "cell_id": [f"img1_{i}" for i in range(2 * n)],
        "image_name": ["img1"] * (2 * n),
        "treatment": ["flat"] * n + ["varied"] * n,
        "feature_a": [2.5] * n + list(np.linspace(1.0, 4.0, n)),
    })
    state = _base_state(
        "Feature Comparison",
        categorical_cols=["treatment"],
        color_by=["treatment"],
        method_params=_feature_comparison_params(),
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)

    offsets = np.concatenate([c.get_offsets() for c in _nonempty_collections(ns["ax"])])
    flat_x = offsets[np.isclose(offsets[:, 1], 2.5), 0]
    assert flat_x.size == n, f"expected {n} points in the constant group, got {flat_x.size}"
    assert np.ptp(flat_x) > 0.05, (
        "constant-valued group collapsed onto one x position; the app spreads it "
        f"with uniform jitter (observed x spread {np.ptp(flat_x):.4f})"
    )


@pytest.mark.parametrize(
    "method, params, expected_title_fragment",
    [
        ("Feature Histogram",
         {"selected_var": "feature_a", "log_x": False, "apply_gmm": False,
          "intersection_threshold": False, "bin_width": None},
         "Frequency histogram of"),
        ("Feature Histogram",
         {"selected_var": "feature_a", "log_x": False, "apply_gmm": True,
          "intersection_threshold": False, "bin_width": None,
          "gmm_max_components": 3, "gmm_min_weight_threshold": 0.1},
         "Gaussian Mixture Model fit of"),
        ("Feature Comparison", _feature_comparison_params(), "Distribution of"),
        ("2D Feature Distribution",
         {"selected_x": "feature_a", "selected_y": "feature_b", "log_x": False,
          "log_y": False, "marginal_plot_type": "boxplot", "fit_regression": False,
          "fit_gmm_2d": False, "gmm_max_components": 3, "gmm_min_weight_threshold": 0.1},
         "2D Distribution of"),
        ("Phasor Plot",
         {"selected_channel": "Ch1", "phasor_harmonic": 1, "phasor_f": 0.08,
          "k_means": False, "k_means_clusters": 2},
         "Harmonic Phasor"),
    ],
)
def test_exported_figures_carry_the_app_title(method, params, expected_title_fragment):
    """Every figure the app titles must be titled in the export too.

    The app titles all figures except Dimension Reduction (src/vis/univar.py,
    src/vis/bivar.py); exported scripts previously emitted no `set_title` at all.
    """
    state = _base_state(method, categorical_cols=["treatment"], color_by=["treatment"],
                        method_params=params)
    script = generate_script(state)
    assert "set_title(" in script
    assert expected_title_fragment in script


def test_dimension_reduction_export_has_no_title_like_the_app():
    """Dimension Reduction is the one method the app deliberately leaves untitled."""
    state = _base_state(
        "Dimension Reduction", categorical_cols=["treatment"], color_by=["treatment"],
        method_params={"selected_features": ["feature_a", "feature_b"],
                       "dr_method": "PCA", "hyperParam_dict": {}})
    assert "set_title(" not in generate_script(state)


# ---------------------------------------------------------------------------
# separate_by section layout — app and export must place sections identically
# ---------------------------------------------------------------------------

def _sectioned_df(sections, groups, n=12, seed=5):
    """Rows for every (day, treatment) combination, so both sections are full."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "cell_id": f"img_{day}_{treatment}_{j}",
            "image_name": f"img_{day}",
            "treatment": treatment,
            "day": day,
            "feature_a": rng.normal(1.0, 0.1),
        }
        for day in sections
        for treatment in groups
        for j in range(n)
    ]
    return pd.DataFrame(rows)


def _app_feature_comparison_fig(df, monkeypatch, separate_by="day"):
    """Run the app's plot function outside a Streamlit runtime.

    Only `st.columns` and the three in-plot checkboxes are reached with effect sizes
    and stats off; each checkbox returns its own `value=` default, which is exactly
    the state the export capture assumes.
    """
    import contextlib

    import streamlit as st

    from src.vis import univar

    monkeypatch.setattr(st, "columns", lambda spec, **kw: tuple(
        contextlib.nullcontext() for _ in (spec if isinstance(spec, list) else range(spec))))
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kw: value)
    monkeypatch.setattr(st, "session_state", {}, raising=False)
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "black")

    return univar.feature_comparison_plot(
        df, "cell_id", "image_name", "feature_a",
        color_by=["treatment"], separate_by=separate_by,
    )


def _app_section_layout(fig):
    """Per-group slot positions and divider positions read off the Plotly figure.

    Point x values are jittered, so the slots come from the x-axis tickvals the app
    builds from `x_positions`; dividers are the dashed vlines it adds between sections.
    """
    slots = sorted(float(v) for v in fig.layout.xaxis.tickvals)
    dividers = sorted(float(s.x0) for s in fig.layout.shapes
                      if getattr(s.line, "dash", None) == "dash")
    return slots, dividers


@pytest.mark.parametrize("n_sections,n_groups", [(2, 3), (3, 2)])
def test_separate_by_section_spacing_matches_the_app(tmp_path, monkeypatch, n_sections, n_groups):
    """The gap between sections is 0.5 in the app (univar.py section_spacing).

    The export used a full 1.0, so section 2 started at 6.0 where the app put it at
    5.5 — and because positions accumulate, every later section drifted further.
    Group membership and statistics were unaffected; this is purely x-layout, but it
    is the layout a figure lifted into a paper carries.
    """
    sections = [str(i + 1) for i in range(n_sections)]
    groups = [chr(ord("A") + i) for i in range(n_groups)]
    df = _sectioned_df(sections, groups)

    state = _base_state(
        "Feature Comparison",
        categorical_cols=["treatment", "day"],
        color_by=["treatment"],
        separate_by="day",
        method_params=_feature_comparison_params(),
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    export_positions = sorted(float(v) for v in ns["x_positions"].values())
    export_dividers = sorted(float(b) for b in ns["section_boundaries"])

    app_positions, app_dividers = _app_section_layout(
        _app_feature_comparison_fig(df, monkeypatch))

    assert export_positions == pytest.approx(app_positions)
    assert export_dividers == pytest.approx(app_dividers)
    # Guard the fix against a later "tidy up to whole numbers": with a 0.5 gap the
    # second section must start half a slot after the first section's width.
    assert export_positions[n_groups] == pytest.approx(n_groups + 0.5 - 1 + 1)


def _app_section_annotations(df, monkeypatch, axis_label_size=24):
    """Section-header annotations from the app figure, at a given axis font size."""
    import contextlib

    import streamlit as st

    from src.vis import univar

    monkeypatch.setattr(st, "columns", lambda spec, **kw: tuple(
        contextlib.nullcontext() for _ in (spec if isinstance(spec, list) else range(spec))))
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kw: value)
    monkeypatch.setattr(st, "session_state", {"plot_axis_label_size": axis_label_size},
                        raising=False)
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "black")

    fig = univar.feature_comparison_plot(
        df, "cell_id", "image_name", "feature_a",
        color_by=["treatment"], separate_by="day",
    )
    headers = [a for a in fig.layout.annotations if a.yref == "paper"]
    return fig, headers


def test_app_section_header_offset_is_pixels_not_plot_fraction(tmp_path, monkeypatch):
    """The header used to sit at y=-0.20 of the plot height. Tick labels hang below the
    axis by a pixel amount that grows with the tick font, so above a certain font size
    the fraction was inside the labels and the two collided on screen. The offset is now
    a pixel yshift anchored at the axis, and it grows with the font.
    """
    df = _sectioned_df(["1", "2"], ["0-control", "Antimycin", "Cyanide"])

    _, small = _app_section_annotations(df, monkeypatch, axis_label_size=10)
    _, large = _app_section_annotations(df, monkeypatch, axis_label_size=34)

    assert [a.y for a in small] == [0, 0]           # anchored at the axis, not -0.20
    assert all(a.yshift < 0 for a in small)          # pushed below it
    assert all(a.yanchor == "top" for a in small)
    # Bigger font, longer drop — the whole point of the change.
    assert abs(large[0].yshift) > abs(small[0].yshift)


def test_app_pins_the_tick_angle_so_the_offset_is_exact(tmp_path, monkeypatch):
    """The header offset can only be right if the tick angle is known. Plotly otherwise
    chooses 0/45/90 from the container width, which the server cannot see, so the offset
    had to assume the worst angle and left a large gap whenever Plotly chose less.
    """
    long_df = _sectioned_df(["1", "2"], ["0-control", "Antimycin", "Cyanide"])
    fig_long, _ = _app_section_annotations(long_df, monkeypatch)
    assert fig_long.layout.xaxis.tickangle == -45

    short_df = _sectioned_df(["1", "2"], ["A", "B", "C"])
    fig_short, _ = _app_section_annotations(short_df, monkeypatch)
    assert fig_short.layout.xaxis.tickangle == 0


def test_app_section_header_offset_does_not_depend_on_the_label_text(tmp_path, monkeypatch):
    """The header sits above the group labels now, so nothing about its placement may be
    derived from how far those labels reach. The old placement dropped it past them and
    had to predict their extent from the character count — a calibrated guess that
    collided with the labels whenever it ran short.
    """
    _, five = _app_section_annotations(
        _sectioned_df(["1", "2"], ["AAAAA", "BBBBB"]), monkeypatch)
    _, twelve = _app_section_annotations(
        _sectioned_df(["1", "2"], ["AAAAAAAAAAAA", "BBBBBBBBBBBB"]), monkeypatch)
    _, short = _app_section_annotations(
        _sectioned_df(["1", "2"], ["A", "B"]), monkeypatch)

    # Same font size, so same offset — regardless of label length, and regardless of the
    # tick angle that length selects.
    assert five[0].yshift == twelve[0].yshift == short[0].yshift


def test_app_reserves_the_header_slot_with_ticklabelstandoff(tmp_path, monkeypatch):
    """The group labels are pushed down by the space the header occupies. That space is
    one line of header text, whose height this code sets, so it is arithmetic on the font
    size rather than a measurement of anything.
    """
    df = _sectioned_df(["1", "2"], ["0-control", "Antimycin"])
    fig_small, small = _app_section_annotations(df, monkeypatch, axis_label_size=10)
    fig_large, large = _app_section_annotations(df, monkeypatch, axis_label_size=34)

    assert fig_small.layout.xaxis.ticklabelstandoff == round(1.6 * 10)
    assert fig_large.layout.xaxis.ticklabelstandoff == round(1.6 * 34)
    # The slot has to be deeper than the header's own pad from the axis, or the header
    # would run into the labels it is supposed to sit above.
    assert fig_small.layout.xaxis.ticklabelstandoff > abs(small[0].yshift)
    assert fig_large.layout.xaxis.ticklabelstandoff > abs(large[0].yshift)
    # The tick labels are the lowest element now, so the bottom margin is Plotly's to
    # size from what it rendered rather than something computed from the label text here.
    assert fig_large.layout.xaxis.automargin is True


def test_app_section_header_is_theme_aware(tmp_path, monkeypatch):
    """The header colour was hardcoded "black", which apply_plot_styling does not
    override (it only rewrites annotation *size*), so it vanished in dark mode.
    """
    df = _sectioned_df(["1", "2"], ["A", "B"])
    _, headers = _app_section_annotations(df, monkeypatch, axis_label_size=24)
    assert all(a.font.color == "black" for a in headers)  # the patched theme colour

    import streamlit as st

    from src.vis import univar
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "white")
    monkeypatch.setattr(st, "session_state", {"plot_axis_label_size": 24}, raising=False)
    fig = univar.feature_comparison_plot(
        df, "cell_id", "image_name", "feature_a",
        color_by=["treatment"], separate_by="day")
    dark = [a for a in fig.layout.annotations if a.yref == "paper"]
    assert all(a.font.color == "white" for a in dark)


def test_app_bottom_margin_is_a_floor_not_a_text_measurement(tmp_path, monkeypatch):
    """Placing the header correctly is useless if the margin clips it off the figure, but
    the margin is no longer what guarantees that — automargin is. What is asserted here
    is that the floor stopped depending on the label text: it used to be
    max(120, len(longest_label) * 5), a character count standing in for a pixel height.
    """
    short = _sectioned_df(["1", "2"], ["A", "B"])
    long_ = _sectioned_df(["1", "2"], ["a-very-long-treatment-name", "another-long-one"])
    fig_short, headers = _app_section_annotations(short, monkeypatch, axis_label_size=34)
    fig_long, _ = _app_section_annotations(long_, monkeypatch, axis_label_size=34)

    assert fig_short.layout.margin.b == fig_long.layout.margin.b
    assert fig_short.layout.margin.b > abs(headers[0].yshift)
    # Both branches hand the sizing to Plotly, which is the only party that can measure.
    assert fig_long.layout.xaxis.automargin is True


def _section_state(**overrides):
    return _base_state(
        "Feature Comparison",
        categorical_cols=["treatment", "day"],
        color_by=["treatment"],
        separate_by="day",
        method_params=_feature_comparison_params(),
        **overrides,
    )


def test_section_name_is_a_header_not_repeated_in_every_tick(tmp_path, monkeypatch):
    """The app labels each tick with the group alone and names the section once, in a
    centred header below the axis (univar.py separate_sections_info). The export used
    to fold the section into every tick as "{section}\\n{group}", which collided into
    an unreadable smear as soon as a section held more than about three groups.
    """
    df = _sectioned_df(["1", "2"], ["Antimycin", "Cyanide", "0-control"])
    ns = _run_script(tmp_path, _section_state(), df, monkeypatch)

    tick_labels = [t.get_text() for t in ns["ax"].get_xticklabels()]
    assert sorted(set(tick_labels)) == ["0-control", "Antimycin", "Cyanide"]
    assert not any("\n" in label for label in tick_labels)

    headers = [t.get_text() for t in ns["ax"].texts]
    assert headers == ["1", "2"]


def test_section_headers_are_centred_under_their_own_groups(tmp_path, monkeypatch):
    df = _sectioned_df(["1", "2"], ["A", "B", "C"])
    ns = _run_script(tmp_path, _section_state(), df, monkeypatch)

    positions = sorted(float(v) for v in ns["x_positions"].values())
    # Headers are annotations anchored at (x, axes-bottom) with a point offset, so the
    # x to check is the anchor `xy`, not get_position() (which returns the offset).
    header_x = [t.xy[0] for t in ns["ax"].texts]
    assert header_x == pytest.approx([
        sum(positions[:3]) / 3,   # centre of section 1
        sum(positions[3:]) / 3,   # centre of section 2
    ])


@pytest.mark.parametrize("labels,expected_angle", [
    (["0-control", "Antimycin", "Cyanide", "2DG", "IAA"], 45),  # mpl +45 == plotly -45
    (["A", "B"], 0),
])
def test_export_tick_angle_follows_the_same_rule_as_the_app(
        tmp_path, monkeypatch, labels, expected_angle):
    """The app pins the tick angle (0 for labels of up to four characters, 90 beyond)
    so it can offset its section headers exactly; the export has to use the same rule or
    the two figures disagree on something the reader sees immediately. Matplotlib never
    rotates on its own, so without this the long names collide head-on.
    """
    df = _sectioned_df(["1", "2"], labels)
    ns = _run_script(tmp_path, _section_state(axis_label_size=24), df, monkeypatch)
    assert all(t.get_rotation() == expected_angle for t in ns["ax"].get_xticklabels())


def test_export_tick_angle_matches_the_app_figure(tmp_path, monkeypatch):
    """Pin the two together directly rather than against a copied constant.

    The signs are opposite on purpose: Plotly measures tickangle clockwise, Matplotlib
    measures rotation counter-clockwise, so -45 there and +45 here are the same uphill
    slant on screen. Comparing the raw numbers would enforce mirrored figures.
    """
    df = _sectioned_df(["1", "2"], ["0-control", "Antimycin", "Cyanide"])
    ns = _run_script(tmp_path, _section_state(axis_label_size=24), df, monkeypatch)
    app_fig, _ = _app_section_annotations(df, monkeypatch, axis_label_size=24)

    export_angles = {t.get_rotation() for t in ns["ax"].get_xticklabels()}
    assert export_angles == {abs(float(app_fig.layout.xaxis.tickangle))}
    # And the export anchors the label's end at the tick, which is what makes
    # Matplotlib's +45 read the same way round as Plotly's -45.
    assert {t.get_ha() for t in ns["ax"].get_xticklabels()} == {"right"}


def test_section_headers_sit_between_the_axis_and_the_tick_labels(tmp_path, monkeypatch):
    """The header goes directly under the axis line with the group labels pushed below
    it, mirroring the app's xaxis.ticklabelstandoff. Ordering them this way is what makes
    the placement exact — the reserved gap is one line of header text, whose height the
    script sets, rather than a guess at how far the rotated labels reach.
    """
    df = _sectioned_df(["1", "2"], ["0-control", "Antimycin", "Cyanide", "2DG", "IAA"])
    ns = _run_script(tmp_path, _section_state(axis_label_size=24), df, monkeypatch)

    ax = ns["ax"]
    ax.figure.canvas.draw()
    axes_bottom = ax.get_window_extent().y0
    headers_top = max(t.get_window_extent().y1 for t in ax.texts)
    headers_bottom = min(t.get_window_extent().y0 for t in ax.texts)
    labels_top = max(t.get_window_extent().y1 for t in ax.get_xticklabels())

    assert headers_top <= axes_bottom      # below the axis line
    assert labels_top <= headers_bottom    # and above the group labels


def test_separate_by_divider_sits_in_the_middle_of_the_gap(tmp_path, monkeypatch):
    """The app centres the dashed divider between the two sections it separates
    (univar.py: midpoint of the previous section's last point and the next section's
    first). Shrinking the gap without moving the divider would leave it hugging the
    left-hand section.
    """
    df = _sectioned_df(["1", "2"], ["A", "B", "C"])
    state = _base_state(
        "Feature Comparison",
        categorical_cols=["treatment", "day"],
        color_by=["treatment"],
        separate_by="day",
        method_params=_feature_comparison_params(),
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)

    positions = sorted(float(v) for v in ns["x_positions"].values())
    boundary = float(ns["section_boundaries"][0])
    left_end, right_start = positions[2], positions[3]
    assert boundary == pytest.approx((left_end + right_start) / 2)


# ---------------------------------------------------------------------------
# Annotation font size — apply_plot_styling() overrides every annotation's size
# ---------------------------------------------------------------------------

def _app_effective_annotation_size(axis_label_size):
    """The size the app actually renders annotations at, taken from the real styling
    pass rather than from the literal each plot function happens to pass.
    """
    import plotly.graph_objects as go

    from src.vis.helpers import apply_plot_styling

    fig = go.Figure()
    fig.add_annotation(x=0, y=0, text="x", showarrow=False,
                       font={"size": 12, "color": "black"})
    fig = apply_plot_styling(fig, 5, axis_label_size, 10)
    return fig.layout.annotations[0].font.size


def test_app_styling_overrides_every_annotation_size():
    """The premise of the tests below, asserted directly: the size a plot function
    writes on an annotation is discarded — src/vis/helpers.py apply_plot_styling
    rewrites it to plot_axis_label_size after the figure is built. Any exported
    counterpart that copies the literal from the source is therefore wrong.
    """
    assert _app_effective_annotation_size(24) == 24
    assert _app_effective_annotation_size(11) == 11


def test_effect_size_bracket_text_matches_the_app_size(tmp_path, monkeypatch):
    """The export hardcoded 10 while the app renders these at plot_axis_label_size."""
    df = _grouped_df({"A": 1.0, "B": 3.0}, n_per_group=15)
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        axis_label_size=24,
        method_params=_feature_comparison_params(statistical_test="Welch's t-test"))
    ns = _run_script(tmp_path, state, df, monkeypatch)

    brackets = [t for t in ns["ax"].texts if "*" in t.get_text()]
    assert brackets, "expected at least one significance bracket"
    assert {t.get_fontsize() for t in brackets} == {_app_effective_annotation_size(24)}


def test_phasor_annotation_sizes_match_the_app(tmp_path, monkeypatch):
    """Lifetime marker labels (app literal 12) and the frequency text (app literal 15)
    are both annotations, so both render at plot_axis_label_size.
    """
    ns = _run_script(tmp_path, _phasor_state(1, axis_label_size=24), _phasor_df(), monkeypatch)
    expected = _app_effective_annotation_size(24)

    marker_labels = [t for t in ns["ax"].texts if t.get_text().endswith(" ns")]
    freq_text = [t for t in ns["ax"].texts if t.get_text().startswith("f = ")]
    assert len(marker_labels) == 6
    assert len(freq_text) == 1
    assert {t.get_fontsize() for t in marker_labels + freq_text} == {expected}


def test_gmm_threshold_text_matches_the_app_size(tmp_path, monkeypatch):
    """The GMM threshold label is an annotation in the app (univar.py) with no size of
    its own, so styling gives it plot_axis_label_size; the export hardcoded 9.
    """
    state = _base_state(
        "Feature Histogram", categorical_cols=["treatment"], axis_label_size=24,
        method_params={"selected_var": "feature_a", "log_x": False, "bin_width": None,
                       "apply_gmm": True, "intersection_threshold": True,
                       "gmm_max_components": 2, "gmm_min_weight_threshold": 0.2})
    ns = _run_script(tmp_path, state, _bimodal_df(), monkeypatch)

    thresholds = [t for t in ns["ax"].texts if t.get_text().startswith("Threshold")]
    assert thresholds, "expected a threshold label"
    assert {t.get_fontsize() for t in thresholds} == {_app_effective_annotation_size(24)}


def test_no_hardcoded_annotation_font_sizes_remain():
    """Regression net for the whole family: every text the app draws as an annotation
    must derive its exported size from AXIS_LABEL_SIZE, never from a copied literal.
    """
    source = (Path(__file__).resolve().parents[1] / "src" / "export_script.py").read_text()
    import re
    leftovers = re.findall(r"fontsize=\d+", source)
    assert leftovers == [], f"hardcoded annotation font sizes left: {leftovers}"


def test_exported_tick_font_size_derives_from_axis_label_size():
    """Ticks follow the app's single rule — `axis_label_size - 2`, applied centrally
    in `apply_plot_styling` — not the export's former three competing rules
    (LEGEND_SIZE, AXIS_LABEL_SIZE-6, AXIS_LABEL_SIZE-4). Legend size is an
    independent user control, so the old export ticks drifted from the app
    whenever the two sizes differed.
    """
    source = (Path(__file__).resolve().parents[1] / "src" / "export_script.py").read_text()
    assert "labelsize=LEGEND_SIZE" not in source
    assert "AXIS_LABEL_SIZE - 6" not in source
    assert "AXIS_LABEL_SIZE - 4" not in source
    assert "labelsize=AXIS_LABEL_SIZE - 2" in source


# ---------------------------------------------------------------------------
# "Show group counts (n) in legend" must reach the exported script
#
# The toggle (visualization_widgets.py) writes plot_show_group_counts, which every
# app plot path passes to format_group_label(). The export captured neither the flag
# nor the helper, so a legend that read "2DG / n=2522" on screen exported as "2DG".
# ---------------------------------------------------------------------------

def _stub_streamlit_for_plots(monkeypatch, show_counts):
    """Run the app's plot functions outside a Streamlit runtime.

    In-plot widgets return their own `value=` defaults, which is the state the export
    capture assumes; only the count toggle is set explicitly. session_state is a dict
    with attribute access because multivar.dimension_reduction stores a lock on it.
    """
    import contextlib
    import importlib

    import streamlit as st

    class _Session(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = value

    session = _Session(plot_show_group_counts=show_counts)
    monkeypatch.setattr(st, "columns", lambda spec, **kw: tuple(
        contextlib.nullcontext() for _ in (spec if isinstance(spec, list) else range(spec))))
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kw: value)
    monkeypatch.setattr(st, "number_input", lambda label, value=0, **kw: value)
    monkeypatch.setattr(st, "slider", lambda label, *a, value=None, **kw: value)
    monkeypatch.setattr(st, "selectbox", lambda label, options, index=0, **kw: options[index])
    monkeypatch.setattr(st, "markdown", lambda *a, **kw: None)
    monkeypatch.setattr(st, "write", lambda *a, **kw: None)
    monkeypatch.setattr(st, "warning", lambda *a, **kw: None)
    monkeypatch.setattr(st, "session_state", session, raising=False)
    for mod_name in ("univar", "bivar", "multivar"):
        mod = importlib.import_module(f"src.vis.{mod_name}")
        monkeypatch.setattr(mod, "get_context_theme_color", lambda: "black")
    return session


def _app_legend_labels(fig):
    """Legend entries of the app's Plotly figure, normalised to the export's text.

    Plotly renders markup in legend entries, so the app puts the count on a second
    line with <br> and a 0.75em span; Matplotlib renders none, so the exported label
    is the same text with a newline. Comparing the two means stripping the markup —
    the words and the count are what have to agree.
    """
    import re

    labels = []
    for trace in fig.data:
        if getattr(trace, "showlegend", None) is False:
            continue
        name = getattr(trace, "name", None)
        if name is None or name in labels:
            continue
        labels.append(re.sub(r"<[^>]+>", "", name.replace("<br>", "\n")))
    return labels


def _export_legend_labels(ax):
    labels = []
    for label in ax.get_legend_handles_labels()[1]:
        if label and label not in labels:
            labels.append(label)
    return labels


def _counts_df(counts, group_col="treatment", seed=3):
    """One colour group per entry, with a deliberately uneven, recognisable size."""
    rng = np.random.default_rng(seed)
    frames = []
    for i, (group, n) in enumerate(counts.items()):
        frames.append(pd.DataFrame({
            "cell_id": [f"img{i + 1}_{j}" for j in range(n)],
            "image_name": [f"img{i + 1}"] * n,
            group_col: [group] * n,
            "cell_line": [("lineA", "lineB")[j % 2] for j in range(n)],
            "feature_a": rng.normal(1.0 + i, 0.3, n),
            "feature_b": rng.normal(2.0 - i, 0.3, n),
        }))
    return pd.concat(frames, ignore_index=True)


_COUNTS = {"ctrl": 17, "drug": 23}


@pytest.mark.parametrize("show_counts", [True, False])
def test_feature_comparison_legend_counts_match_the_app(tmp_path, monkeypatch, show_counts):
    from src.vis import univar

    df = _counts_df(_COUNTS)
    _stub_streamlit_for_plots(monkeypatch, show_counts)
    app_fig = univar.feature_comparison_plot(
        df.copy(), "cell_id", "image_name", "feature_a", color_by=["treatment"])
    state = _base_state("Feature Comparison", categorical_cols=["treatment"],
                        color_by=["treatment"], show_group_counts=show_counts,
                        method_params=_feature_comparison_params())
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert _export_legend_labels(ns["ax"]) == _app_legend_labels(app_fig)
    assert ("n=17" in "".join(_export_legend_labels(ns["ax"]))) is show_counts


@pytest.mark.parametrize("show_counts", [True, False])
def test_separate_by_legend_counts_are_group_totals_not_section_subtotals(
        tmp_path, monkeypatch, show_counts):
    """With separate_by, a colour group appears in several sections but keeps one
    legend entry. The app counts it once over the whole frame (univar.py builds
    group_counts before splitting), so the entry shows the total — counting the
    section a group happens to be drawn in first would halve it here.
    """
    from src.vis import univar

    df = _counts_df(_COUNTS)
    _stub_streamlit_for_plots(monkeypatch, show_counts)
    app_fig = univar.feature_comparison_plot(
        df.copy(), "cell_id", "image_name", "feature_a",
        color_by=["treatment"], separate_by="cell_line")
    state = _base_state("Feature Comparison", categorical_cols=["treatment", "cell_line"],
                        color_by=["treatment"], separate_by="cell_line",
                        show_group_counts=show_counts,
                        method_params=_feature_comparison_params())
    ns = _run_script(tmp_path, state, df, monkeypatch)
    exported = _export_legend_labels(ns["ax"])
    assert exported == _app_legend_labels(app_fig)
    if show_counts:
        # Totals (17/23), not the per-cell_line halves each section draws.
        assert exported == ["ctrl\nn=17", "drug\nn=23"]


@pytest.mark.parametrize("show_counts", [True, False])
def test_histogram_legend_counts_match_the_app(tmp_path, monkeypatch, show_counts):
    from src.vis import univar

    df = _counts_df(_COUNTS)
    _stub_streamlit_for_plots(monkeypatch, show_counts)
    app_fig = univar.feature_histogram_plot(df.copy(), "feature_a", color_by=["treatment"])
    state = _base_state("Feature Histogram", categorical_cols=["treatment"],
                        color_by=["treatment"], show_group_counts=show_counts,
                        method_params={"selected_var": "feature_a", "log_x": False,
                                       "apply_gmm": False, "bin_width": None})
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert _export_legend_labels(ns["ax"]) == _app_legend_labels(app_fig)


@pytest.mark.parametrize("show_counts", [True, False])
def test_gmm_legend_counts_sit_inside_the_gmm_suffixed_label(tmp_path, monkeypatch, show_counts):
    """The GMM curve is labelled "<group> GMM"; the count belongs to that whole label,
    and the per-component curves below it carry no count at all — as in the app.
    """
    from src.vis import univar

    df = _bimodal_df(n_per_mode=25)
    _stub_streamlit_for_plots(monkeypatch, show_counts)
    app_fig, _ = univar.feature_gmm_plot(df.copy(), "feature_a", color_by=["treatment"])
    state = _base_state("Feature Histogram", categorical_cols=["treatment"],
                        color_by=["treatment"], show_group_counts=show_counts,
                        method_params={"selected_var": "feature_a", "log_x": False,
                                       "apply_gmm": True, "intersection_threshold": False,
                                       "bin_width": None, "gmm_max_components": 3,
                                       "gmm_min_weight_threshold": 0.1})
    ns = _run_script(tmp_path, state, df, monkeypatch)
    exported = _export_legend_labels(ns["ax"])
    assert exported == _app_legend_labels(app_fig)
    components = [label for label in exported if "Component" in label]
    assert components, "expected per-component curves in this fixture"
    assert all("n=" not in label for label in components)


@pytest.mark.parametrize("show_counts", [True, False])
def test_2d_legend_counts_match_the_app(tmp_path, monkeypatch, show_counts):
    from src.vis import bivar

    df = _counts_df(_COUNTS)
    _stub_streamlit_for_plots(monkeypatch, show_counts)
    app_fig, _, _ = bivar.feature_2d_distribution_plot(
        df.copy(), "cell_id", "image_name", "feature_a", "feature_b",
        color_by=["treatment"], marginal_plot_type="gaussian fit")
    state = _base_state("2D Feature Distribution", categorical_cols=["treatment"],
                        color_by=["treatment"], show_group_counts=show_counts,
                        method_params={"selected_x": "feature_a", "selected_y": "feature_b",
                                       "log_x": False, "log_y": False,
                                       "marginal_plot_type": "gaussian fit",
                                       "fit_regression": False, "fit_gmm_2d": False})
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert _export_legend_labels(ns["ax_main"]) == _app_legend_labels(app_fig)


@pytest.mark.parametrize("show_counts", [True, False])
def test_phasor_legend_counts_match_the_app(tmp_path, monkeypatch, show_counts):
    """Phasor is the one point plot data_analysis.py hands over unfiltered; the app
    drops the missing coordinates inside phasor_plot instead (bivar.py: notna on both
    G and S). The count therefore excludes them, and the export has to apply the same
    dropna before counting — this fixture blanks 4 of 30 G values to pin that down.
    """
    from src.vis import bivar

    df = _phasor_df()
    df.loc[df.index[:4], "Lifetime fit free_Ch1: G(1st)"] = np.nan
    _stub_streamlit_for_plots(monkeypatch, show_counts)
    app_fig, _ = bivar.phasor_plot(df.copy(), "cell_id", "image_name", "Ch1",
                                   color_by=["treatment"], f=0.08, harmonic=1)
    ns = _run_script(tmp_path, _phasor_state(1, show_group_counts=show_counts), df, monkeypatch)
    exported = _export_legend_labels(ns["ax"])
    assert exported == _app_legend_labels(app_fig)
    if show_counts:
        assert exported == ["ctrl\nn=26"], "26 of 30 rows have both coordinates"


@pytest.mark.parametrize("show_counts", [True, False])
def test_dimension_reduction_legend_counts_match_the_app(tmp_path, monkeypatch, show_counts):
    from src.vis import multivar

    df = _counts_df(_COUNTS)
    _stub_streamlit_for_plots(monkeypatch, show_counts)
    app_fig = multivar.dimension_reduction_plot(
        df.copy(), "cell_id", "image_name", ["feature_a", "feature_b"],
        colored_by=["treatment"], method="PCA")
    state = _base_state("Dimension Reduction", categorical_cols=["treatment"],
                        color_by=["treatment"], show_group_counts=show_counts,
                        method_params={"selected_features": ["feature_a", "feature_b"],
                                       "dr_method": "PCA", "hyperParam_dict": {}})
    ns = _run_script(tmp_path, state, df, monkeypatch)
    assert _export_legend_labels(ns["ax"]) == _app_legend_labels(app_fig)


def test_shape_and_opacity_legend_entries_never_carry_counts(tmp_path, monkeypatch):
    """Only colour groups are counted. The shape/opacity legend entries name a value,
    not a sample — the app adds them as separate empty traces (helpers.py
    add_point_legend_traces) that never see format_group_label.
    """
    df = _encoding_df()
    state = _base_state(
        "2D Feature Distribution",
        categorical_cols=["treatment", "cell_line", "day"],
        color_by=["treatment"], shape_by="cell_line", opacity_by="day",
        show_group_counts=True,
        method_params={"selected_x": "feature_a", "selected_y": "feature_b",
                       "log_x": False, "log_y": False, "marginal_plot_type": "none",
                       "fit_regression": False, "fit_gmm_2d": False},
    )
    ns = _run_script(tmp_path, state, df, monkeypatch)
    labels = _export_legend_labels(ns["ax_main"])
    assert {"lineA", "lineB", "d1", "d2"} <= set(labels)
    assert [label for label in labels if "n=" in label] == ["A\nn=48", "B\nn=48"]


def test_export_inlines_the_apps_group_label_helper_rather_than_copying_it():
    """The "n=" wording lives in one place. export_script.py inlines the app's
    format_group_label via _extract_source, so re-wording it in src/vis/helpers.py
    changes both renderers at once; a template that spelled the label out itself
    would drift the moment the app's wording changed.
    """
    import inspect
    import textwrap

    from src.vis.helpers import format_group_label

    script = generate_script(_base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        show_group_counts=True, method_params=_feature_comparison_params()))
    assert textwrap.dedent(inspect.getsource(format_group_label)).strip() in script
    assert "engine='mpl'" in script


@pytest.mark.parametrize("method,method_params", [
    ("Feature Comparison", None),
    ("Feature Histogram", {"selected_var": "feature_a", "log_x": False,
                           "apply_gmm": False, "bin_width": None}),
    ("2D Feature Distribution", {"selected_x": "feature_a", "selected_y": "feature_b",
                                 "log_x": False, "log_y": False,
                                 "marginal_plot_type": "none", "fit_regression": False,
                                 "fit_gmm_2d": False}),
    ("Phasor Plot", {"selected_channel": "Ch1", "phasor_harmonic": 1, "phasor_f": 0.08,
                     "k_means": False, "k_means_clusters": 2}),
    ("Dimension Reduction", {"selected_features": ["feature_a", "feature_b"],
                             "dr_method": "PCA", "hyperParam_dict": {}}),
])
def test_show_group_counts_reaches_every_plot_method(method, method_params):
    """One capture, every method: the flag becomes a constant the reader can flip."""
    params = _feature_comparison_params() if method_params is None else method_params
    for flag in (True, False):
        script = generate_script(_base_state(
            method, categorical_cols=["treatment"], color_by=["treatment"],
            show_group_counts=flag, method_params=params))
        assert f"SHOW_GROUP_COUNTS = {flag}" in script
        assert "format_group_label(" in script


def test_show_group_counts_defaults_off_when_not_captured():
    """Older capture dicts (and uncoloured plots, where the toggle never renders)
    carry no flag; the script must fall back to the app's own default rather than
    inventing counts.
    """
    script = generate_script(_base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        method_params=_feature_comparison_params()))
    assert "SHOW_GROUP_COUNTS = False" in script


# ---------------------------------------------------------------------------
# Collapse by — one point per replicate, in the app and in the script
# ---------------------------------------------------------------------------

def _replicate_df(n_per_dish=8, seed=17):
    """Three dishes per treatment, two FOVs per dish, one day per dish.

    `day` is constant within a dish (coarser, so a decoration survives the collapse);
    `image_name` takes two values inside every dish (finer, so it cannot).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i, (dish, treatment, day) in enumerate([
        ("D1", "ctrl", "Day 1"), ("D2", "drug", "Day 1"),
        ("D3", "ctrl", "Day 2"), ("D4", "drug", "Day 2"),
        ("D5", "ctrl", "Day 3"), ("D6", "drug", "Day 3"),
    ]):
        for j in range(n_per_dish):
            rows.append({
                "cell_id": f"{dish}_c{j}",
                "dish": dish,
                "treatment": treatment,
                "day": day,
                "image_name": f"{dish}_f{j % 2}",
                "feature_a": float(rng.normal(1.0 + 0.3 * (treatment == "drug"), 0.2)),
            })
    return pd.DataFrame(rows)


def _collapse_state(**mp):
    return _base_state(
        "Feature Comparison",
        categorical_cols=["treatment", "dish", "day"],
        color_by=["treatment"],
        method_params=_feature_comparison_params(collapse_by="dish", **mp),
    )


def test_collapse_by_reaches_the_generated_script():
    script = generate_script(_collapse_state())
    assert "COLLAPSE_BY = 'dish'" in script
    assert "def collapse_rows" in script


def test_no_collapse_emits_no_constant_and_no_inlined_source():
    """Off must stay byte-for-byte the old path, so no existing parity check moves."""
    script = generate_script(_base_state(
        "Feature Comparison", categorical_cols=["treatment"], color_by=["treatment"],
        method_params=_feature_comparison_params()))
    # On the emitted CODE, not the bare tokens: the shared bracket template carries a
    # comment naming COLLAPSE_BY to explain its own nan guard, and a comment is not a
    # constant. What must not appear is a definition or a call.
    assert "COLLAPSE_BY =" not in script
    assert "def collapse_rows" not in script
    assert "collapse_rows(" not in script


def test_the_script_inlines_the_apps_collapse_verbatim():
    """Not a re-implementation: the same source, so the two cannot drift."""
    import inspect
    import textwrap

    from src.collapse import collapse_rows

    script = generate_script(_collapse_state())
    assert textwrap.dedent(inspect.getsource(collapse_rows)).strip() in script


def test_the_collapsed_points_match_the_app(tmp_path, monkeypatch):
    """The headline check: same dots, same values, app and script."""
    from src.collapse import collapse_rows

    df = _replicate_df()
    _stub_streamlit_for_plots(monkeypatch, show_counts=False)
    collapsed, label_col, _varied = collapse_rows(
        df.copy(), "dish", ["treatment"], "cell_id")
    app_fig = _app_feature_comparison_fig_collapsed(collapsed, label_col, monkeypatch)

    ns = _run_script(tmp_path, _collapse_state(), df, monkeypatch)

    app_y = sorted(round(float(y), 9)
                   for trace in app_fig.data if getattr(trace, "mode", None)
                   for y in (trace.y if trace.y is not None else []))
    exported = sorted(round(float(y), 9)
                      for coll in _nonempty_collections(ns["ax"])
                      for y in coll.get_offsets()[:, 1])
    assert len(exported) == 6                     # six dishes, not 48 cells
    assert exported == app_y


def _app_feature_comparison_fig_collapsed(collapsed, label_col, monkeypatch):
    import contextlib

    import streamlit as st

    from src.vis import univar

    monkeypatch.setattr(st, "columns", lambda spec, **kw: tuple(
        contextlib.nullcontext() for _ in (spec if isinstance(spec, list) else range(spec))))
    monkeypatch.setattr(st, "checkbox", lambda label, value=False, **kw: value)
    monkeypatch.setattr(univar, "get_context_theme_color", lambda: "black")
    return univar.feature_comparison_plot(
        collapsed, label_col, None, "feature_a", color_by=["treatment"],
        row_id_label="dish")


def test_the_collapse_runs_after_the_nan_drop(tmp_path, monkeypatch):
    """`n` must count the cells that actually contributed to the mean, and the mean
    must be over those cells only. Collapse first and both are wrong -- silently, on
    a plot that still looks entirely reasonable."""
    df = _replicate_df()
    df.loc[df["cell_id"] == "D1_c0", "feature_a"] = np.nan
    survivors = df[(df["dish"] == "D1") & df["feature_a"].notna()]["feature_a"]

    ns = _run_script(tmp_path, _collapse_state(), df, monkeypatch)

    row = ns["df"][ns["df"]["dish"] == "D1"].iloc[0]
    assert row["feature_a"] == pytest.approx(survivors.mean())
    assert "(n=7)" in row[[c for c in ns["df"].columns if c.startswith("dish (n")][0]]


def test_the_collapse_runs_before_the_log(tmp_path, monkeypatch):
    """The app logs inside feature_comparison_plot, i.e. after the page collapsed:
    log10(mean), never mean(log10). The two differ by Jensen's inequality."""
    df = _replicate_df()
    plain = _run_script(tmp_path, _collapse_state(), df, monkeypatch)
    logged = _run_script(tmp_path, _collapse_state(log_y=True), df, monkeypatch)

    got = logged["df"].set_index("dish")["feature_a"]
    expected = np.log10(plain["df"].set_index("dish")["feature_a"] + 1e-6)
    pd.testing.assert_series_equal(got, expected)


def test_the_encoding_maps_are_built_on_the_collapsed_frame(tmp_path, monkeypatch):
    """_build_visual_encoding must run AFTER the collapse -- otherwise the colour and
    count maps describe cells while the points are replicates."""
    ns = _run_script(tmp_path, _collapse_state(), _replicate_df(), monkeypatch)
    counts = ns["df"].groupby("_color_group").size().to_dict()
    assert counts == {"ctrl": 3, "drug": 3}


def test_a_hand_edited_collapse_explains_a_channel_it_has_to_switch_off(tmp_path, monkeypatch, capsys):
    """The app resolves the decoration channels before capture, so this guard is a
    no-op on generated state. It is here for the "standalone, EDITABLE script"
    promise: someone who sets COLLAPSE_BY by hand gets a printed reason rather than a
    KeyError out of df[SHAPE_BY].unique()."""
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment", "dish", "day"],
        color_by=["treatment"], shape_by="image_name",
        method_params=_feature_comparison_params(collapse_by="dish"))
    ns = _run_script(tmp_path, state, _replicate_df(), monkeypatch)

    out = capsys.readouterr().out
    assert "SHAPE_BY is off" in out
    assert "covers several image_name values" in out
    assert "cannot be further divided" in out
    assert ns["SHAPE_BY"] is None


def test_a_decoration_coarser_than_the_replicate_still_applies(tmp_path, monkeypatch):
    """`day` is one value per dish, so a collapsed dot can carry it -- the case that
    makes Shape by worth keeping on offer at all."""
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment", "dish", "day"],
        color_by=["treatment"], shape_by="day",
        method_params=_feature_comparison_params(collapse_by="dish"))
    ns = _run_script(tmp_path, state, _replicate_df(), monkeypatch)

    assert ns["SHAPE_BY"] == "day"
    assert sorted(ns["df"]["day"].unique()) == ["Day 1", "Day 2", "Day 3"]


def test_a_collapsed_script_compiles_and_runs(tmp_path, monkeypatch):
    ns = _run_script(tmp_path, _collapse_state(add_boxplot=True), _replicate_df(), monkeypatch)
    assert len(ns["df"]) == 6


def _thin_group_df(seed=23):
    """`ctrl` has one dish, `drug` has three -- so collapsing by dish leaves the CONTROL
    group with a single point. Control rather than treatment on purpose: Glass's delta
    divides by the control group's spread alone, so only this way is it undefined for
    both statistics rather than just Cohen's d."""
    rng = np.random.default_rng(seed)
    rows = []
    for dish, treatment in [("D1", "ctrl"), ("D2", "drug"), ("D3", "drug"), ("D4", "drug")]:
        for j in range(8):
            rows.append({"cell_id": f"{dish}_c{j}", "dish": dish, "treatment": treatment,
                         "image_name": f"{dish}_f{j % 2}",
                         "feature_a": float(rng.normal(1.0, 0.2))})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["Glass's Delta", "Absolute Cohen's d"])
def test_an_undefined_effect_size_draws_nothing_in_the_script_either(tmp_path, monkeypatch, method):
    """The app's guard is POSITIVE (`draw if abs(es) >= threshold`) so a nan falls
    through it; the script's was NEGATIVE (`skip if abs(es) < threshold`), and
    `abs(nan) < t` is *also* False -- so the two are opposites everywhere except nan,
    where both are False and the script drew a bracket reading "Δ=nan" that the app
    never showed. Collapse by makes this routine: one dish in a treatment is one point."""
    state = _base_state(
        "Feature Comparison", categorical_cols=["treatment", "dish"],
        color_by=["treatment"],
        method_params=_feature_comparison_params(
            collapse_by="dish", effect_size_method=method, mean_or_median="Mean",
            effect_size_threshold=0.0))
    ns = _run_script(tmp_path, state, _thin_group_df(), monkeypatch)

    drawn = [t.get_text() for t in ns["ax"].texts]
    assert not any("nan" in text.lower() for text in drawn), drawn
