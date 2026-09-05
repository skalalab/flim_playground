"""The 2D distribution's per-group analysis (correlation, regression, marginal
densities) must key on COLOR groups only — never fan out across shape/opacity
sub-groups. This matches the plot's own GMM block (bivar.py), the phasor
k-means block, feature-comparison statistics, and the exported script.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vis.bivar import feature_2d_distribution_plot


def _two_color_two_shape_df():
    """2 color groups (A, B), each split into 2 shape sub-groups (s1, s2).

    Each (color, shape) cell holds a clean linear x↔y relationship so Pearson r
    and the marginal densities are always defined.
    """
    rows = []
    for color_group in ["A", "B"]:
        for shape_group in ["s1", "s2"]:
            for idx in range(8):
                rows.append(
                    {
                        "cell_id": f"{color_group}_{shape_group}_{idx}",
                        "image_name": "fov_1",
                        "x_feat": float(idx),
                        "y_feat": float(idx) * 2.0 + (0.0 if shape_group == "s1" else 5.0),
                        "group": color_group,
                        "shape_group": shape_group,
                    }
                )
    return pd.DataFrame(rows)


def _run_2d_plot(df, **kwargs):
    st.session_state.plot_point_size = 5
    st.session_state.plot_axis_label_size = 18
    st.session_state.plot_legend_size = 16
    return feature_2d_distribution_plot(
        df,
        unique_row_id_col="cell_id",
        fov_name_col="image_name",
        selected_x="x_feat",
        selected_y="y_feat",
        color_by=["group"],
        shape_by="shape_group",
        **kwargs,
    )


def test_correlation_is_reported_once_per_color_group():
    fig, table_md, _ = _run_2d_plot(_two_color_two_shape_df())

    # Two color groups → exactly two correlation entries, regardless of the
    # two shape sub-groups inside each. Fanning out per shape would give four.
    assert table_md.count("Correlation Coefficient b/w") == 2
    assert "**A:**" in table_md
    assert "**B:**" in table_md


def test_marginal_densities_drawn_once_per_color_group():
    fig, _, _ = _run_2d_plot(_two_color_two_shape_df())

    x_marginals = [
        t for t in fig.data
        if getattr(t, "name", None) and str(t.name).endswith("_x_density")
    ]
    y_marginals = [
        t for t in fig.data
        if getattr(t, "name", None) and str(t.name).endswith("_y_density")
    ]
    # One X- and one Y-marginal per color group (A, B); not one per sub-group.
    assert len(x_marginals) == 2
    assert len(y_marginals) == 2


@pytest.mark.parametrize("marginal_plot_type", ["gaussian fit", "boxplot", "violin"])
def test_y_marginals_share_the_data_scale_without_the_main_grid(marginal_plot_type):
    fig, _, _ = _run_2d_plot(
        _two_color_two_shape_df(), marginal_plot_type=marginal_plot_type)
    marginals = [trace for trace in fig.data if trace.xaxis == "x2"]
    assert len(marginals) == 2
    assert fig.layout.yaxis.showgrid is True
    for trace in marginals:
        axis = fig.layout["yaxis" + (trace.yaxis or "y")[1:]]
        assert axis.showgrid is False
        assert axis.matches == "y"
        assert axis.domain == fig.layout.yaxis.domain
