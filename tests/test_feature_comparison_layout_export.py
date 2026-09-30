"""Feature Comparison exports lay out and style the figure as the app does.

Plotly draws its legend beside the plot and grows the y range to hold annotation text,
centring each label on its computed height; Matplotlib does none of that by default.
Streamlit's chart theme adds the title style, gridlines and absent frame.
"""
import runpy
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import font_manager

from src.export_script import generate_script

GROUP_MEANS = {"2DG": 70, "0-control": 73, "Antimycin": 77, "Cyanide": 77, "IAA": 66}
LEGEND_SIZE = 18
# A FLIM feature whose axis label needs a subscript glyph: "nadh α₁ (%)".
FEATURE = "Lifetime fit_nadh: a1"


@pytest.fixture
def export(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = [{"cell_id": f"{line}-{group}-{i}", "cell_line": line, "treatment": group,
             FEATURE: mean + shift + rng.normal(0, 3)}
            for line, shift in (("Panc1", 0), ("MCF7", -8))
            for group, mean in GROUP_MEANS.items() for i in range(60)]
    pd.DataFrame(rows).to_csv(tmp_path / "cells.csv", index=False)
    state = {
        "csv_filename": "cells.csv", "delimiter": ",", "unique_row_id_col": "cell_id",
        "fov_name_col": None, "method": "Feature Comparison",
        "categorical_filters": {}, "numerical_filters": [], "color_by": ["treatment"],
        "separate_by": "cell_line", "shape_by": None, "opacity_by": None, "subcolor_by": None,
        "point_size": 5, "axis_label_size": 20, "legend_size": LEGEND_SIZE,
        "show_group_counts": True, "colormap": "tab10",
        "categorical_cols": ["cell_line", "treatment"],
        "method_params": {
            "selected_var": FEATURE, "effect_size_method": "Glass's Delta",
            "mean_or_median": "Mean", "statistical_test": "None", "effect_size_threshold": 0.0,
            # Pairs sharing one group stack three brackets in each section.
            "selected_pairs": ["0-control vs Antimycin", "0-control vs Cyanide", "0-control vs IAA"],
            "custom_order": {"compare_groups": list(GROUP_MEANS)},
            "overlay": "None", "connect_means": False, "log_y": False,
        },
    }
    (tmp_path / "analysis.py").write_text(generate_script(state), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(plt, "show", lambda: None)
    # The script sets fonts in rcParams; restore them so other tests keep the defaults.
    with matplotlib.rc_context():
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                namespace = runpy.run_path(str(tmp_path / "analysis.py"))
                namespace["fig"].canvas.draw()
            namespace["warnings"] = [str(warning.message) for warning in caught]
            yield namespace
        finally:
            plt.close("all")


def test_legend_sits_beside_the_axes_with_markers_sized_by_the_legend_font(export):
    ax = export["ax"]
    legend = ax.get_legend()
    assert legend.get_window_extent().x0 >= ax.get_window_extent().x1
    assert {float(handle.get_sizes()[0]) for handle in legend.legend_handles} == {LEGEND_SIZE ** 2}


def test_effect_size_labels_stay_inside_the_axes_and_below_the_next_bracket(export):
    ax = export["ax"]
    labels = [text for text in ax.texts if text.get_text().startswith("Δ")]
    brackets = [line for line in ax.lines if len(line.get_xdata()) == 4]
    assert len(labels) == len(brackets) == 6
    axes_top = ax.get_window_extent().y1
    for label in labels:
        box = label.get_window_extent()
        assert box.y1 <= axes_top, f"{label.get_text()} extends past the top of the axes"
        x, y = label.get_position()
        for bracket in brackets:
            xs, ys = bracket.get_xdata(), bracket.get_ydata()
            if xs[0] <= x <= xs[-1] and ys[1] > y:
                bracket_px = ax.transData.transform((x, ys[1]))[1]
                assert box.y1 <= bracket_px, f"{label.get_text()} crosses the bracket above it"


def test_title_frame_and_gridlines_follow_the_app_chart_theme(export):
    ax = export["ax"]
    # Streamlit's theme titles every chart at 16 px, bold, left-aligned.
    assert ax.get_title() == ""
    assert ax.get_title(loc="left").startswith("Distribution of")
    assert (ax._left_title.get_fontsize(), ax._left_title.get_fontweight()) == (16, "bold")
    assert not any(spine.get_visible() for spine in ax.spines.values())
    assert ax.get_axisbelow() is True
    assert all(line.get_visible() for line in ax.get_ygridlines())
    assert not any(line.get_visible() for line in ax.get_xgridlines())


def test_text_uses_arial_when_installed_with_dejavu_subscripts(export, caplog):
    installed = {font.name for font in font_manager.fontManager.ttflist}
    # Helvetica's subscript digits hang below the baseline, so the fallback skips it.
    expected = ["Arial", "DejaVu Sans"] if "Arial" in installed else ["sans-serif"]
    assert plt.rcParams["font.family"] == expected
    assert not [message for message in export["warnings"] if "missing from font" in message]
    assert not [record for record in caplog.get_records("setup") if "findfont" in record.getMessage()]
