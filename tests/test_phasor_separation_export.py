"""Standalone Phasor exports preserve app faceting and clustering semantics."""

import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.export_script import generate_script


G_COL = "Lifetime fit free_Ch1: G(1st)"
S_COL = "Lifetime fit free_Ch1: S(1st)"


def _state(
    *,
    separate_by="day",
    k_means=False,
    show_counts=True,
    phasor_category=None,
):
    return {
        "csv_filename": "phasor.csv",
        "unique_row_id_col": "cell_id",
        "fov_name_col": "image_name",
        "method": "Phasor Plot",
        "categorical_filters": {},
        "numerical_filters": [],
        "color_by": ["treatment"],
        "opacity_by": "dose",
        "shape_by": "cell_line",
        "separate_by": separate_by,
        "point_size": 5,
        "axis_label_size": 12,
        "legend_size": 10,
        "show_group_counts": show_counts,
        "colormap": "tab10",
        "categorical_cols": ["treatment", "cell_line", "dose", "day"],
        "method_params": {
            "selected_channel": "Ch1",
            "phasor_harmonic": 1,
            "phasor_f": 0.08,
            "k_means": k_means,
            "k_means_clusters": 2,
            "phasor_category": phasor_category,
        },
    }


def _run(tmp_path, monkeypatch, state, df, *, save_derived=False):
    df.to_csv(tmp_path / state["csv_filename"], index=False)
    script = generate_script(state)
    if save_derived:
        assert "SAVE_DERIVED_DATA = False" in script
        script = script.replace("SAVE_DERIVED_DATA = False", "SAVE_DERIVED_DATA = True")
    path = tmp_path / "analysis.py"
    path.write_text(script)
    monkeypatch.chdir(tmp_path)
    try:
        return runpy.run_path(str(path))
    finally:
        plt.close("all")


def _faceted_df():
    return pd.DataFrame(
        {
            "cell_id": [f"row{i}" for i in range(8)],
            "image_name": ["image"] * 8,
            "treatment": ["ctrl", "drug", "ctrl", "drug", "ctrl", "drug", "ctrl", "drug"],
            "cell_line": ["A", "B"] * 4,
            "dose": ["low", "high"] * 4,
            "day": ["Day 10", "Day 10", "Day 2", "Day 2", None, None, "Day 2", "Day 10"],
            G_COL: [0.20, 0.25, 0.40, 0.45, 0.60, 0.65, np.nan, 0.80],
            S_COL: [0.10, 0.12, 0.20, 0.22, 0.30, 0.32, 0.40, np.nan],
            "metadata": list("abcdefgh"),
        }
    )


def test_separated_phasor_exports_one_large_selected_category_with_context_points(
    tmp_path, monkeypatch
):
    ns = _run(
        tmp_path,
        monkeypatch,
        _state(phasor_category="Day 10"),
        _faceted_df(),
    )

    fig = ns["fig"]
    assert [path.name for path in tmp_path.glob("*.svg")] == ["phasor_plot.svg"]
    assert len(fig.axes) == 1
    assert fig.get_size_inches().tolist() == pytest.approx([10, 6])
    ax = fig.axes[0]
    assert ax.get_aspect() == 1.0
    assert ax.get_title() == "Ch1 1st Harmonic Phasor"
    assert ns["phasor_category"] == "Day 10"
    labels = [text for text in ax.texts if text.get_text() == "day: Day 10"]
    assert len(labels) == 1
    assert labels[0].get_ha() == "center"
    assert labels[0].get_va() == "top"

    assert not fig.legends
    assert ax.get_legend() is not None
    legend_labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_labels.count("ctrl\nn=1") == 1
    assert legend_labels.count("drug\nn=1") == 1

    # Maps still cover every retained category, so switching categories does not
    # change the meaning of a color, shape, or opacity.
    assert set(ns["color_map"]) == {"ctrl", "drug"}
    assert set(ns["shape_map"]) == {"A", "B"}
    assert set(ns["opacity_map"]) == {"high", "low"}

    assert len([line for line in ax.lines if line.get_marker() == "o"]) == 11
    background = [
        collection
        for collection in ax.collections
        if len(collection.get_offsets()) == 4
        and collection.get_alpha() == pytest.approx(0.18)
    ]
    assert len(background) == 1
    assert background[0].get_sizes().tolist() == pytest.approx([9])
    assert background[0].get_facecolors()[0, :3].tolist() == pytest.approx(
        matplotlib.colors.to_rgb("#b8b8b8")
    )
    foreground = [
        collection
        for collection in ax.collections
        if len(collection.get_offsets())
        and not np.isscalar(collection.get_alpha())
    ]
    assert len(foreground) == 2
    assert all(collection.get_sizes().tolist() == pytest.approx([25])
               for collection in foreground)
    assert sorted(float(collection.get_alpha()[0]) for collection in foreground) == pytest.approx(
        [0.18, 0.6]
    )
    assert ns["opacity_map"] == {"high": 0.3, "low": 1.0}


def test_separated_phasor_falls_back_to_first_and_always_shows_other_categories(
    tmp_path, monkeypatch
):
    ns = _run(
        tmp_path,
        monkeypatch,
        _state(phasor_category="unavailable"),
        _faceted_df(),
    )

    assert ns["phasor_category"] == "Day 2"
    assert any(
        collection.get_alpha() == pytest.approx(0.18)
        for collection in ns["ax"].collections
    )
    assert "PHASOR_SHOW_OTHER_CATEGORIES" not in generate_script(
        _state(phasor_category="Day 2")
    )


def test_phasor_viewer_places_large_legend_outside_right_and_category_below_g(
    tmp_path, monkeypatch
):
    state = _state()
    state["legend_size"] = 18
    state["axis_label_size"] = 18
    ns = _run(tmp_path, monkeypatch, state, _faceted_df())

    fig = ns["fig"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0]
    legend_box = ax.get_legend().get_window_extent(renderer)
    axes_box = ax.get_window_extent(renderer)
    category_label = next(text for text in ax.texts if text.get_text() == "day: Day 2")
    xlabel_box = ax.xaxis.label.get_window_extent(renderer)
    category_box = category_label.get_window_extent(renderer)
    axes_bottom = ax.get_tightbbox(renderer).y0
    assert legend_box.x0 >= 0
    assert legend_box.x0 >= axes_box.x1
    assert legend_box.y1 == pytest.approx(axes_box.y1, abs=1)
    assert category_box.y1 + 0.1 * fig.dpi <= xlabel_box.y0
    assert axes_bottom >= 0


def test_saved_bounds_include_a_long_active_category_label(tmp_path, monkeypatch):
    category = "Sample collection 24 hours after treatment with mitochondrial inhibitor"
    source = _faceted_df()
    source["day"] = category
    state = _state(phasor_category=category)
    state["legend_size"] = 18
    state["axis_label_size"] = 18
    ns = _run(tmp_path, monkeypatch, state, source)

    fig = ns["fig"]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label = next(text for text in ns["ax"].texts if text.get_text() == f"day: {category}")
    label_box = label.get_window_extent(renderer)
    saved_box = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    assert saved_box.x0 <= label_box.x0
    assert saved_box.x1 >= label_box.x1
    assert saved_box.y0 <= label_box.y0


def test_shared_legend_keeps_equal_text_from_distinct_encoding_channels(
    tmp_path, monkeypatch
):
    source = _faceted_df()
    source["dose"] = source["cell_line"]
    ns = _run(tmp_path, monkeypatch, _state(), source)

    labels = [text.get_text() for text in ns["ax"].get_legend().get_texts()]
    assert labels.count("A") == 2
    assert labels.count("B") == 2


def _cluster_df():
    rows = []
    for day, offset in [("Day 1", 0.0), ("Day 2", 0.2)]:
        for treatment, shift in [("ctrl", 0.0), ("drug", 0.04)]:
            for i, (g, s) in enumerate([(0.20, 0.10), (0.22, 0.11), (0.70, 0.35), (0.72, 0.36)]):
                rows.append(
                    {
                        "cell_id": f"{day}-{treatment}-{i}",
                        "image_name": "image",
                        "treatment": treatment,
                        "cell_line": "A" if i % 2 == 0 else "B",
                        "dose": "low" if i < 2 else "high",
                        "day": day,
                        G_COL: g + offset + shift,
                        S_COL: s + offset / 2 + shift,
                        "metadata": f"meta-{i}",
                    }
                )
    rows.extend(
        [
            {
                "cell_id": "missing-g",
                "image_name": "image",
                "treatment": "ctrl",
                "cell_line": "A",
                "dose": "low",
                "day": "Day 1",
                G_COL: np.nan,
                S_COL: 0.2,
                "metadata": "keep-original",
            },
            {
                "cell_id": "missing-s",
                "image_name": "image",
                "treatment": "drug",
                "cell_line": "B",
                "dose": "high",
                "day": "Day 2",
                G_COL: 0.4,
                S_COL: np.nan,
                "metadata": "keep-original",
            },
        ]
    )
    return pd.DataFrame(rows)


def test_separated_kmeans_fits_each_panel_and_color_group_and_exports_retained_rows(
    tmp_path, monkeypatch
):
    source = _cluster_df()
    ns = _run(
        tmp_path,
        monkeypatch,
        _state(k_means=True, phasor_category="Day 2"),
        source,
        save_derived=True,
    )
    saved = pd.read_csv(tmp_path / "kmeans_clustered_data.csv")

    assert saved["cell_id"].tolist() == source.iloc[:16]["cell_id"].tolist()
    assert saved["metadata"].tolist() == source.iloc[:16]["metadata"].tolist()
    assert saved[G_COL].tolist() == pytest.approx(source.iloc[:16][G_COL].tolist())
    assert saved[S_COL].tolist() == pytest.approx(source.iloc[:16][S_COL].tolist())
    assert len(ns["phasor_fits"]) == 4
    assert all(len(fit["positions"]) == 4 for fit in ns["phasor_fits"])
    centroids = [line for line in ns["ax"].lines if line.get_marker() == "x"]
    assert len(centroids) == 4  # two clusters for each of two active color groups

    labels = saved["k_means_cluster"].dropna().astype(str)
    assert len(labels) == 16
    assert set(label.split(" | ", 1)[0] for label in labels) == {"day=Day 1", "day=Day 2"}
    assert all(label.split(" | ", 1)[1].startswith(("ctrl_group", "drug_group")) for label in labels)


def test_kmeans_skips_too_few_distinct_pairs_but_retains_rows_and_reports_group(
    tmp_path, monkeypatch, capsys
):
    df = _cluster_df().iloc[:8].copy()
    mask = df["treatment"] == "drug"
    df.loc[mask, [G_COL, S_COL]] = [0.5, 0.25]

    _run(tmp_path, monkeypatch, _state(k_means=True), df, save_derived=True)
    output = capsys.readouterr().out
    saved = pd.read_csv(tmp_path / "kmeans_clustered_data.csv")

    skipped = saved.loc[saved["treatment"] == "drug", "k_means_cluster"]
    fitted = saved.loc[saved["treatment"] == "ctrl", "k_means_cluster"]
    assert skipped.isna().all()
    assert fitted.notna().all()
    assert "Day 1" in output and "drug" in output and "distinct" in output


def test_separator_named_like_internal_color_column_is_preserved(
    tmp_path, monkeypatch
):
    source = _cluster_df().iloc[:16].rename(columns={"day": "_color_group"})
    state = _state(separate_by="_color_group", k_means=True)
    state["categorical_cols"] = [
        "treatment",
        "cell_line",
        "dose",
        "_color_group",
    ]

    ns = _run(tmp_path, monkeypatch, state, source, save_derived=True)
    saved = pd.read_csv(tmp_path / "kmeans_clustered_data.csv")

    assert [panel[0] for panel in ns["phasor_panels"]] == ["Day 1", "Day 2"]
    assert saved["_color_group"].tolist() == source["_color_group"].tolist()
    assert saved["k_means_cluster"].str.startswith("_color_group=Day ").all()


@pytest.mark.parametrize(
    ("separate_by", "color_by"),
    [(["day"], ["treatment"]), ("day", ["day"])],
)
def test_phasor_separator_must_be_scalar_present_and_disjoint_from_color(
    tmp_path, monkeypatch, separate_by, color_by
):
    state = _state(separate_by=separate_by)
    state["color_by"] = color_by
    with pytest.raises(ValueError):
        _run(tmp_path, monkeypatch, state, _faceted_df())
