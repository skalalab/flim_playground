"""Tests for src.feature_labels.format_feature_label.

The function turns FLIM-Playground feature column names into human-readable axis
titles using proper FLIM notation (channel + Greek/scientific symbol + unit).
Unknown / non-FP columns must pass through unchanged.
"""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_labels import format_feature_label


@pytest.mark.parametrize(
    "column, expected",
    [
        # --- Lifetime fit: lifetimes (ps) ---
        ("Lifetime fit_nadh: t1", "nadh τ₁ (ps)"),       # nadh τ₁ (ps)
        ("Lifetime fit_nadh: t2", "nadh τ₂ (ps)"),       # nadh τ₂ (ps)
        ("Lifetime fit_nadh: t3", "nadh τ₃ (ps)"),       # nadh τ₃ (ps)
        ("Lifetime fit_nadh: tm", "nadh τₘ (ps)"),            # subscript m, like τ₁
        ("Lifetime fit_nadh: tm_iw", "nadh τₘ,ᵢ (ps)"),       # intensity-weighted mean
        # --- Lifetime fit: amplitude fractions (%) ---
        ("Lifetime fit_nadh: a1", "nadh α₁ (%)"),        # nadh α₁ (%)
        ("Lifetime fit_nadh: a2", "nadh α₂ (%)"),        # nadh α₂ (%)
        ("Lifetime fit_nadh: a3", "nadh α₃ (%)"),        # nadh α₃ (%)
        # --- Phasor coordinates (unitless) ---
        ("Lifetime fit free_nadh: G(1st)", "nadh g"),
        ("Lifetime fit free_nadh: S(1st)", "nadh s"),
        ("Lifetime fit free_nadh: G(2nd)", "nadh g (2nd harm.)"),
        ("Lifetime fit free_nadh: S(2nd)", "nadh s (2nd harm.)"),
        # --- Phasor-derived lifetimes (ns, not ps) ---
        ("Lifetime fit free_nadh: Tau_phase", "nadh τᵩ (ns)"),   # subscript phi
        ("Lifetime fit free_nadh: Tau_mod", "nadh τ<sub>mod</sub> (ns)"),  # modulation: 'mod' subscript
        ("Lifetime fit free_nadh: Tau_m", "nadh τ<sub>mod</sub> (ns)"),    # legacy alias (pre-rename CSVs)
        # --- Morphology ---
        ("Intensity morphology_nadh: area", "nadh Area (px²)"),       # px²
        ("Intensity morphology_nadh: perimeter", "nadh Perimeter (px)"),
        ("Intensity morphology_nadh: major_axis_length", "nadh Major axis (px)"),
        ("Intensity morphology_nadh: minor_axis_length", "nadh Minor axis (px)"),
        ("Intensity morphology_nadh: solidity", "nadh Solidity"),
        ("Intensity morphology_nadh: eccentricity", "nadh Eccentricity"),
        ("Intensity morphology_nadh: circularity", "nadh Circularity"),
        # --- Texture ---
        ("Intensity texture_nadh: intensity_sum", "nadh Intensity (photons)"),
        ("Intensity texture_nadh: granularity_1", "nadh Granularity r=1 (%)"),
        ("Intensity texture_nadh: granularity_3", "nadh Granularity r=3 (%)"),
        ("Intensity texture_nadh: granularity_7", "nadh Granularity r=7 (%)"),
        ("Intensity texture_nadh: radial_distribution_ring1", "nadh Radial dist. ring 1"),
        ("Intensity texture_nadh: radial_distribution_ring4", "nadh Radial dist. ring 4"),
        ("Intensity texture_nadh: mass_displacement", "nadh Mass displacement (px)"),
        # --- Uncategorized (channel_suffix, no ": ") ---
        ("nadh_amp1", "nadh A₁ (photons)"),       # photon counts
        ("nadh_amp2", "nadh A₂ (photons)"),
        ("nadh_amp3", "nadh A₃ (photons)"),
        ("nadh_offset", "nadh Offset (photons)"),
        ("nadh_reduced_chi_square", "nadh χ²ᵣ"),    # subscript r
        ("nadh_centroid_x", "nadh Centroid x (px)"),
        ("nadh_centroid_y", "nadh Centroid y (px)"),
    ],
)
def test_known_features_map_to_flim_notation(column, expected):
    assert format_feature_label(column) == expected


def test_channel_name_is_preserved_verbatim():
    # different channel string flows through unchanged (case + content)
    assert format_feature_label("Lifetime fit_fad: t1") == "fad τ₁ (ps)"
    assert format_feature_label("Lifetime fit_FAD: a1") == "FAD α₁ (%)"


def test_channel_name_with_underscore_is_kept():
    # extractor/channel split is on the FIRST underscore only (mirrors dataset_io)
    assert format_feature_label("Lifetime fit_ch_1: t1") == "ch_1 τ₁ (ps)"


def test_digit_subscripts_are_real_unicode():
    # τ₁ must use the unicode subscript, not "t1" / "tau1"
    assert format_feature_label("Lifetime fit_nadh: t1").endswith("τ₁ (ps)")


@pytest.mark.parametrize(
    "column",
    [
        "treatment",
        "cell_id",
        "image_name",
        "some_random_measurement",
        "weird name with spaces",
        # recognised extractor prefix but UNKNOWN feature -> unchanged
        "Lifetime fit_nadh: not_a_feature",
        "",
    ],
)
def test_unknown_columns_pass_through_unchanged(column):
    assert format_feature_label(column) == column


def test_every_flim_feature_in_inhibitors_csv_is_relabelled():
    """Every numerical FLIM feature column in the real extracted dataset should be
    recognised (i.e. changed), while identifiers/categoricals pass through."""
    csv = Path(__file__).resolve().parents[1] / "example_data" / "Data_Analysis" / "inhibitors.csv"
    cols = list(pd.read_csv(csv, nrows=0).columns)

    identifiers_and_categoricals = {"cell_id", "image_name", "cell_line", "treatment", "dish"}
    flim_cols = [c for c in cols if c not in identifiers_and_categoricals]

    assert flim_cols, "expected FLIM feature columns in inhibitors.csv"
    for c in flim_cols:
        assert format_feature_label(c) != c, f"feature column not relabelled: {c!r}"

    # identifiers / categoricals must be left alone
    for c in identifiers_and_categoricals & set(cols):
        assert format_feature_label(c) == c


# ---------------------------------------------------------------------------
# Hover tooltips: FLIM notation + theme-aware styling (not Plotly default gray)
# ---------------------------------------------------------------------------

def test_2d_distribution_hover_uses_flim_notation():
    """The 2D scatter hover labels x/y with Greek FLIM notation, not raw column names."""
    import streamlit as st
    st.session_state.plot_point_size = 5
    st.session_state.plot_axis_label_size = 18
    st.session_state.plot_legend_size = 16
    from src.vis.bivar import feature_2d_distribution_plot

    rows = []
    for i, grp in enumerate(["a", "b"]):
        for j in range(20):
            rows.append({"cell_id": f"img{i}_{j}", "image_name": f"img{i}", "treatment": grp,
                         "Lifetime fit_nadh: t1": 380 + j, "Lifetime fit_fad: tm": 600 + j})
    df = pd.DataFrame(rows)
    fig, _, _ = feature_2d_distribution_plot(
        df, "cell_id", "image_name",
        "Lifetime fit_nadh: t1", "Lifetime fit_fad: tm", ["treatment"])

    point_tmpl = next(t.hovertemplate for t in fig.data
                      if getattr(t, "hovertemplate", None) and "%{x}" in t.hovertemplate)
    assert format_feature_label("Lifetime fit_nadh: t1") in point_tmpl  # nadh τ₁ (ps)
    assert format_feature_label("Lifetime fit_fad: tm") in point_tmpl   # fad τₘ (ps)
    assert "Lifetime fit_nadh: t1" not in point_tmpl                    # raw name gone


def test_apply_plot_styling_themes_hover_tooltip():
    """Hover tooltip is theme-aware (black/white), centrally applied — not Plotly gray."""
    import plotly.graph_objects as go
    from src.vis.helpers import apply_plot_styling

    fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2]))
    fig = apply_plot_styling(fig, 5, 18, 16)
    hl = fig.layout.hoverlabel
    assert hl.font.color in ("black", "white")
    assert hl.bgcolor in ("white", "rgb(30, 30, 30)")
    assert hl.bordercolor in ("black", "white")


def test_feature_comparison_hover_uses_flim_notation():
    """The Feature Comparison point hover labels the value with FLIM notation, not raw name."""
    import streamlit as st
    st.session_state.plot_point_size = 5
    st.session_state.plot_axis_label_size = 18
    st.session_state.plot_legend_size = 16
    from src.vis.univar import feature_comparison_plot

    rows = []
    for grp in ["a", "b"]:
        for j in range(8):
            rows.append({"cell_id": f"{grp}_{j}", "image_name": "fov1", "group": grp,
                         "Lifetime fit_nadh: t1": 380 + j})
    fig = feature_comparison_plot(pd.DataFrame(rows), unique_row_id_col="cell_id",
                                  fov_name_col="image_name",
                                  selected_var="Lifetime fit_nadh: t1", color_by=["group"])
    tmpl = next(t.hovertemplate for t in fig.data
                if getattr(t, "hovertemplate", None) and "%{y:.3f}" in t.hovertemplate)
    assert format_feature_label("Lifetime fit_nadh: t1") in tmpl  # nadh τ₁ (ps)
    assert "Lifetime fit_nadh: t1" not in tmpl                    # raw name gone


def test_no_tau_label_uses_underscore_subscript():
    """All tau-based lifetime labels use real subscripts, never 'τ_' underscore notation."""
    for col in ["Lifetime fit_nadh: tm", "Lifetime fit_nadh: tm_iw",
                "Lifetime fit free_nadh: Tau_phase",
                "Lifetime fit free_nadh: Tau_mod", "Lifetime fit free_nadh: Tau_m"]:
        assert "τ_" not in format_feature_label(col), col


def test_modulation_lifetime_uses_subscript_markup_per_engine():
    """Modulation lifetime renders 'mod' as a subscript: <sub> for Plotly, mathtext for mpl.

    Tau_mod is the canonical extracted key; Tau_m is the legacy alias kept so that
    pre-rename CSVs still render identically.
    """
    for col in ["Lifetime fit free_nadh: Tau_mod", "Lifetime fit free_nadh: Tau_m"]:
        assert format_feature_label(col) == "nadh τ<sub>mod</sub> (ns)"              # plotly (default)
        assert format_feature_label(col, engine="mpl") == r"nadh $τ_{\mathrm{mod}}$ (ns)"  # matplotlib


def test_mpl_engine_only_affects_markup_labels():
    """engine='mpl' leaves plain-unicode labels (no <sub>) byte-for-byte identical."""
    for col in ["Lifetime fit_nadh: t1", "Lifetime fit_nadh: tm", "Lifetime fit_nadh: a1",
                "Lifetime fit free_nadh: G(1st)", "nadh_reduced_chi_square"]:
        assert format_feature_label(col, engine="mpl") == format_feature_label(col)


def test_modulation_mpl_label_renders_without_tofu():
    """The matplotlib mathtext form of modulation lifetime parses and renders cleanly."""
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    label = format_feature_label("Lifetime fit free_nadh: Tau_mod", engine="mpl")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fig, ax = plt.subplots()
        ax.set_ylabel(label)
        fig.canvas.draw()
        miss = [str(x.message) for x in w
                if "missing" in str(x.message).lower() or "Glyph" in str(x.message)]
        plt.close("all")
    assert not miss, miss


def test_histogram_and_gmm_wrap_log_x_axis_label():
    """Histogram & GMM x-axis reads log₁₀(pretty label) when Log X is on, else the plain label.

    The page log-transforms the data before calling these plots, so the only job here is
    the label — mirroring feature_comparison_plot (Log Y) and the 2D plot (Log X/Y).
    """
    import numpy as np
    import streamlit as st
    st.session_state.plot_point_size = 5
    st.session_state.plot_axis_label_size = 18
    st.session_state.plot_legend_size = 16
    from src.vis.univar import feature_histogram_plot, feature_gmm_plot

    col = "Lifetime fit_nadh: t1"
    pretty = format_feature_label(col)  # "nadh τ₁ (ps)"
    rng = np.random.default_rng(0)
    rows = []
    for i, grp in enumerate(["a", "b"]):
        for j in range(40):
            rows.append({"cell_id": f"{grp}_{j}", "image_name": f"img{i}", "group": grp,
                         col: float(rng.normal(390, 20))})
    df = pd.DataFrame(rows)

    hist_log = feature_histogram_plot(df.copy(), col, ["group"], log_x=True)
    hist_lin = feature_histogram_plot(df.copy(), col, ["group"], log_x=False)
    assert hist_log.layout.xaxis.title.text == f"log₁₀({pretty})"
    assert hist_lin.layout.xaxis.title.text == pretty

    gmm_log, _ = feature_gmm_plot(df.copy(), col, ["group"], log_x=True)
    gmm_lin, _ = feature_gmm_plot(df.copy(), col, ["group"], log_x=False)
    assert gmm_log.layout.xaxis.title.text == f"log₁₀({pretty})"
    assert gmm_lin.layout.xaxis.title.text == pretty
