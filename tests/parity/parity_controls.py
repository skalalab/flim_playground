"""Walk every Data Analysis control and check the app and the exported script agree.

Exercise control options from per-method baselines on a filtered subset (SUBSET)
to keep the matrix quick. parity_methods.py covers full-data default settings.

Run:  uv run python tests/parity/parity_controls.py [all|shared|filters|enc|fc|hist|2d|phasor|dr|clf]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from harness_common import (
    EXAMPLES,
    WORK_ROOT,
    Results,
    app_point_traces,
    apply_filters,
    base_state,
    load_app_df,
    page_collectors,
    run_export,
    scatter_points,
    sorted_rows,
)
from harness_widgets import patch_streamlit

from src.widgets.multiselect_modes import EXCEPT_LABEL

CATS = ["cell_line", "treatment", "dish", "image_name"]
CSV = EXAMPLES / "inhibitors.csv"
WORK = WORK_ROOT / "controls"
VAR = "Lifetime fit_nadh: tm"
VAR2 = "Lifetime fit free_nadh: Tau_m"
FEATS = ["Lifetime fit_nadh: t1", "Lifetime fit_nadh: t2", "Lifetime fit_nadh: a1",
         "Lifetime fit_nadh: tm", "Intensity morphology_nadh: area"]

# Keeps every run in the matrix small. Also exercises the categorical-filter control
# on every single case rather than only in the one case that targets it.
SUBSET = {"treatment": ["IAA", "2DG"]}

R = Results()
_CASE = {"n": 0}


def _wd(tag):
    _CASE["n"] += 1
    return WORK / f"{_CASE['n']:03d}_{tag}"


# ---------------------------------------------------------------------------
# Comparators shared by every case
# ---------------------------------------------------------------------------

def legend_labels_app(fig):
    return [t.name for t in fig.data if t.name and getattr(t, "showlegend", None) is not False]


def legend_labels_exp(ax):
    return list(ax.get_legend_handles_labels()[1])


def compare_points(tag, fig, ax, main_axis_only=False, known_gap=False, detail="",
                   exclude_names=()):
    """Same plotted cloud: count, positions, values.

    Also checks the y values alone, which must agree even where the x jitter is known
    to differ — that separates "plots different data" from "places points differently".
    """
    traces = [t for t in app_point_traces(fig, main_axis_only=main_axis_only)
              if t.name not in exclude_names]
    if not traces:
        return
    app_pts = np.vstack([np.column_stack([t.x, t.y]) for t in traces])
    exp_pts = scatter_points(ax)
    exp_pts = exp_pts[np.isfinite(exp_pts).all(axis=1)]

    same_y = (app_pts.shape == exp_pts.shape
              and np.allclose(np.sort(app_pts[:, 1]), np.sort(exp_pts[:, 1]), atol=1e-6))
    R.check(f"{tag}: plotted values ({len(app_pts)} points)", same_y,
            "" if same_y else f"app={app_pts.shape} exp={exp_pts.shape}")

    same = app_pts.shape == exp_pts.shape and np.allclose(sorted_rows(app_pts),
                                                          sorted_rows(exp_pts), atol=1e-6)
    R.check(f"{tag}: point cloud ({len(app_pts)})", same,
            detail if not same else "", known_gap=known_gap)


def _alphas_app(fig, main_axis_only=False, include_color_alpha=False):
    """Per-point marker opacity, optionally including DR's translucent color."""
    vals = []
    for t in app_point_traces(fig, main_axis_only=main_axis_only):
        opacity = getattr(getattr(t, "marker", None), "opacity", None)
        count = len(t.x) if t.x is not None else 0
        if not count:
            continue
        point_opacity = (list(opacity) if np.ndim(opacity)
                         else [1.0 if opacity is None else opacity] * count)
        if include_color_alpha:
            # DR uses one rgba color per trace. Plotly multiplies its alpha by
            # marker.opacity; Matplotlib stores that product in each facecolor.
            color = t.marker.color
            if isinstance(color, str) and color.startswith("rgba("):
                color_alpha = float(color.rstrip(")").rsplit(",", 1)[1])
                point_opacity = np.asarray(point_opacity) * color_alpha
        vals.extend(point_opacity)
    return np.sort(np.asarray(vals, float))


def _alphas_exp(ax):
    """Per-point alpha on the export side, where opacity is a per-point alpha array."""
    vals = []
    for coll in ax.collections:
        count = len(coll.get_offsets())
        face = coll.get_facecolor()
        if not count or not len(face):
            continue
        vals.extend(face[:, 3].tolist() if len(face) == count else [face[0][3]] * count)
    return np.sort(np.asarray(vals, float))


def compare_alphas(tag, fig, ax, main_axis_only=False, include_color_alpha=False):
    """The opacity channel, point by point.

    Compare Plotly's per-trace marker.opacity with Matplotlib's per-point alpha
    arrays as multisets, because the renderers use different paint orders. DR
    additionally includes color alpha to compare the effective rendered opacity.
    """
    app = _alphas_app(fig, main_axis_only=main_axis_only,
                     include_color_alpha=include_color_alpha)
    exp = _alphas_exp(ax)
    same = app.shape == exp.shape and np.allclose(app, exp, atol=1e-9)
    R.check(f"{tag}: per-point opacity ({len(app)} points)", same,
            "" if same else f"app={np.unique(app)[:5]} exp={np.unique(exp)[:5]}")


def compare_styling(tag, state, ax):
    """Plot Styling controls: axis label size, legend size, point size."""
    als = state["axis_label_size"]
    # Only axes that actually carry a label: Feature Comparison has no x label (the x
    # axis is categorical), and an unlabelled axis keeps Matplotlib's default size.
    labelled = [a for a in (ax.xaxis, ax.yaxis) if a.label.get_text()]
    R.check(f"{tag}: axis label font size = {als}",
            bool(labelled) and all(a.label.get_fontsize() == als for a in labelled),
            f"{[(a.axis_name, a.label.get_fontsize()) for a in labelled]}")
    leg = ax.get_legend()
    if leg is not None and leg.get_texts():
        R.check(f"{tag}: legend font size = {state['legend_size']}",
                all(t.get_fontsize() == state["legend_size"] for t in leg.get_texts()))


def _to_hex(c):
    """Plotly hands back 'rgba(r, g, b, a)' strings, which Matplotlib cannot parse."""
    import matplotlib.colors as mcolors

    if isinstance(c, str) and c.startswith(("rgb(", "rgba(")):
        nums = [float(v) for v in c[c.index("(") + 1:c.rindex(")")].split(",")]
        return mcolors.to_hex([v / 255 for v in nums[:3]])
    return mcolors.to_hex(c)


def compare_colors(tag, fig, ax):
    """Colormap control: the same group must get the same RGB on both sides."""
    import matplotlib.colors as mcolors

    app_colors = {}
    for t in fig.data:
        if not t.name:
            continue
        c = getattr(getattr(t, "marker", None), "color", None) or \
            getattr(getattr(t, "line", None), "color", None)
        if isinstance(c, str):
            app_colors.setdefault(t.name, _to_hex(c))
    exp_colors = {}
    for coll in ax.collections:
        lbl = coll.get_label()
        fc = coll.get_facecolor()
        if lbl and not lbl.startswith("_") and len(fc):
            exp_colors.setdefault(lbl, mcolors.to_hex(fc[0][:3]))  # noqa: keep
    shared = set(app_colors) & set(exp_colors)
    if not shared:
        return
    same = all(app_colors[k] == exp_colors[k] for k in shared)
    R.check(f"{tag}: group colors ({len(shared)} groups)", same,
            "" if same else f"{[(k, app_colors[k], exp_colors[k]) for k in sorted(shared)][:3]}")


# ---------------------------------------------------------------------------
# Per-method runners. Each takes a control dict and returns (fig, ax, ns, state).
# ---------------------------------------------------------------------------

def _state(method, ctrl, mp):
    return base_state(method, ctrl.get("csv_name", "inhibitors.csv"), ctrl.get("cats", CATS),
                      color_by=ctrl.get("color_by", ["treatment"]),
                      opacity_by=ctrl.get("opacity_by"),
                      shape_by=ctrl.get("shape_by"),
                      separate_by=ctrl.get("separate_by"),
                      subcolor_by=ctrl.get("subcolor_by"),
                      categorical_filters=ctrl.get("categorical_filters", SUBSET),
                      numerical_filters=ctrl.get("numerical_filters", []),
                      point_size=ctrl.get("point_size", 5),
                      axis_label_size=ctrl.get("axis_label_size", 12),
                      legend_size=ctrl.get("legend_size", 10),
                      colormap=ctrl.get("colormap", "tab10"),
                      analysis_columns=ctrl["analysis_columns"],
                      method_params=mp)


def _load(ctrl):
    df, _ = load_app_df(ctrl.get("csv_path", CSV), ctrl.get("cats", CATS),
                        "cell_id", "image_name")
    ctrl["analysis_columns"] = list(df.columns)
    return apply_filters(df, {"categorical_filters": ctrl.get("categorical_filters", SUBSET),
                              "numerical_filters": ctrl.get("numerical_filters", [])})


def run_hist(ctrl, tag):
    widgets = {"Use intersection as threshold": ctrl.get("intersection_threshold", False),
               "Max Components": ctrl.get("gmm_max_components", 3),
               "Min Weight Threshold": ctrl.get("gmm_min_weight_threshold", 0.1)}
    if ctrl.get("bin_width") is not None:
        # The app reads bin width from its own widget; the export from BIN_WIDTH. Set
        # both or the two sides bin differently for reasons unrelated to parity.
        widgets["Bin Width"] = ctrl["bin_width"]
    patch_streamlit(widgets)
    df = _load(ctrl)
    log_x = ctrl.get("log_x", False)
    plot_df = df.copy()
    if log_x:
        # data_analysis.py log-transforms upstream of the plot function
        plot_df[VAR] = np.log10(plot_df[VAR] + 1e-6)
    if ctrl.get("apply_gmm"):
        from src.vis.univar import feature_gmm_plot

        fig, _out = feature_gmm_plot(plot_df, VAR, ctrl.get("color_by", ["treatment"]),
                                     colormap=ctrl.get("colormap", "tab10"), log_x=log_x)
    else:
        from src.vis.univar import feature_histogram_plot

        fig = feature_histogram_plot(plot_df, VAR, ctrl.get("color_by", ["treatment"]),
                                     colormap=ctrl.get("colormap", "tab10"), log_x=log_x)
    state = _state("Feature Histogram", ctrl, {
        "selected_var": VAR, "log_x": log_x,
        "apply_gmm": ctrl.get("apply_gmm", False),
        "intersection_threshold": ctrl.get("intersection_threshold", False),
        "bin_width": ctrl.get("bin_width"),
        "gmm_max_components": ctrl.get("gmm_max_components", 3),
        "gmm_min_weight_threshold": ctrl.get("gmm_min_weight_threshold", 0.1)})
    ns, _ = run_export(state, ctrl.get("csv_path", CSV), _wd(tag))
    return fig, ns["ax"], ns, state


def run_fc(ctrl, tag):
    patch_streamlit()
    from src.vis.univar import feature_comparison_plot

    df = _load(ctrl)
    plot_df = df.copy()
    # Match the page: collapse before logging or plotting, then disable encoding
    # channels whose columns varied within a collapsed group.
    collapse_by = ctrl.get("collapse_by")
    row_id_col, row_id_label, fov_col = "cell_id", "ID", "image_name"
    shape_by, opacity_by = ctrl.get("shape_by"), ctrl.get("opacity_by")
    subcolor_by = ctrl.get("subcolor_by")
    if collapse_by:
        from src.collapse import collapse_rows
        from src.dataset_io import resolve_effective_fov_col
        from src.widgets.encoding_state import drop_varying_channels

        plot_df, row_id_col, varied = collapse_rows(
            plot_df, collapse_by,
            [*ctrl.get("color_by", ["treatment"]), ctrl.get("separate_by")], "cell_id")
        row_id_label = collapse_by
        fov_col = resolve_effective_fov_col(plot_df, "image_name")
        channels, _dropped = drop_varying_channels(
            {"shape": shape_by, "opacity": opacity_by, "subcolor": subcolor_by}, varied)
        shape_by, opacity_by, subcolor_by = (
            channels["shape"], channels["opacity"], channels["subcolor"])
    log_y = ctrl.get("log_y", False)
    if log_y:
        plot_df[VAR] = np.log10(plot_df[VAR] + 1e-6)
    fig = feature_comparison_plot(
        plot_df, unique_row_id_col=row_id_col, fov_name_col=fov_col, selected_var=VAR,
        color_by=ctrl.get("color_by", ["treatment"]),
        opacity_by=opacity_by, shape_by=shape_by,
        separate_by=ctrl.get("separate_by"), colormap=ctrl.get("colormap", "tab10"),
        effect_size_method=ctrl.get("effect_size_method", "None"),
        mean_or_median=ctrl.get("mean_or_median"),
        statistical_test=ctrl.get("statistical_test", "None"),
        custom_order=ctrl.get("custom_order"),
        subcolor_by=subcolor_by, row_id_label=row_id_label)
    from src.export_script import get_effect_size_threshold_capture

    state = _state("Feature Comparison", ctrl, {
        "selected_var": VAR,
        "effect_size_method": ctrl.get("effect_size_method", "None"),
        "mean_or_median": ctrl.get("mean_or_median"),
        "statistical_test": ctrl.get("statistical_test", "None"),
        "log_y": log_y,
        "add_boxplot": ctrl.get("add_boxplot", False),
        "connect_means": ctrl.get("connect_means", False),
        "effect_size_threshold": get_effect_size_threshold_capture(
            {}, ctrl.get("effect_size_method", "None"), VAR, ctrl.get("separate_by")),
        "selected_pairs": ctrl.get("selected_pairs"),
        "custom_order": ctrl.get("custom_order"),
        "collapse_by": collapse_by})
    state["shape_by"], state["opacity_by"], state["subcolor_by"] = (
        shape_by, opacity_by, subcolor_by)
    ns, _ = run_export(state, ctrl.get("csv_path", CSV), _wd(tag))
    return fig, ns["ax"], ns, state


def run_2d(ctrl, tag):
    patch_streamlit({"2D Gaussian Mixture Model": ctrl.get("fit_gmm_2d", False),
                     "Regression line": ctrl.get("fit_regression", False),
                     "Marginal Plot Type": ctrl.get("marginal_plot_type", "gaussian fit"),
                     "Log X": ctrl.get("log_x", False),
                     "Log Y": ctrl.get("log_y", False)})
    from src.vis.bivar import feature_2d_distribution_plot

    df = _load(ctrl)
    fig, _md, _out = feature_2d_distribution_plot(
        df.copy(), unique_row_id_col="cell_id", fov_name_col="image_name",
        selected_x=VAR, selected_y=VAR2,
        color_by=ctrl.get("color_by", ["treatment"]),
        shape_by=ctrl.get("shape_by"), opacity_by=ctrl.get("opacity_by"),
        marginal_plot_type=ctrl.get("marginal_plot_type", "gaussian fit"),
        colormap=ctrl.get("colormap", "tab10"))
    state = _state("2D Feature Distribution", ctrl, {
        "selected_x": VAR, "selected_y": VAR2,
        "log_x": ctrl.get("log_x", False), "log_y": ctrl.get("log_y", False),
        "marginal_plot_type": ctrl.get("marginal_plot_type", "gaussian fit"),
        "fit_regression": ctrl.get("fit_regression", False),
        "fit_gmm_2d": ctrl.get("fit_gmm_2d", False),
        "gmm_max_components": ctrl.get("gmm_max_components", 3),
        "gmm_min_weight_threshold": ctrl.get("gmm_min_weight_threshold", 0.1)})
    ns, _ = run_export(state, ctrl.get("csv_path", CSV), _wd(tag))
    return fig, ns["ax_main"], ns, state


def run_phasor(ctrl, tag):
    patch_streamlit()
    from src.vis.bivar import phasor_plot

    df = _load(ctrl)
    fig, _out = phasor_plot(df.copy(), "cell_id", "image_name", "nadh",
                            color_by=ctrl.get("color_by", ["treatment"]),
                            shape_by=ctrl.get("shape_by"), opacity_by=ctrl.get("opacity_by"),
                            colormap=ctrl.get("colormap", "tab10"),
                            f=ctrl.get("phasor_f", 0.08),
                            harmonic=ctrl.get("phasor_harmonic", 1))
    state = _state("Phasor Plot", ctrl, {
        "selected_channel": "nadh",
        "phasor_harmonic": ctrl.get("phasor_harmonic", 1),
        "phasor_f": ctrl.get("phasor_f", 0.08)})
    ns, _ = run_export(state, ctrl.get("csv_path", CSV), _wd(tag))
    return fig, ns["ax"], ns, state


def run_dr(ctrl, tag):
    patch_streamlit()
    from src.vis.multivar import dimension_reduction_plot

    df = _load(ctrl)
    method = ctrl.get("dr_method", "PCA")
    hp = ctrl.get("hyperParam_dict", {})
    fig = dimension_reduction_plot(df.copy(), unique_row_id_col="cell_id",
                                   fov_name_col="image_name", selected_features=FEATS,
                                   colored_by=ctrl.get("color_by", ["treatment"]),
                                   opacity_by=ctrl.get("opacity_by"),
                                   shape_by=ctrl.get("shape_by"),
                                   colormap=ctrl.get("colormap", "tab10"),
                                   method=method, hyperParam_dict=hp)
    state = _state("Dimension Reduction", ctrl, {
        "selected_features": FEATS, "dr_method": method, "hyperParam_dict": hp})
    ns, _ = run_export(state, ctrl.get("csv_path", CSV), _wd(tag))
    return fig, ns["ax"], ns, state


def run_clf(ctrl, tag):
    patch_streamlit()
    from src.classify import run_classification

    df = _load(ctrl).dropna(subset=FEATS)
    df["classes"] = df[["cell_line"]].astype(str).agg("_".join, axis=1)
    classes = sorted(df["classes"].unique())
    err, app_res = run_classification(
        df[FEATS + ["classes"]], ctrl["classification_method"], ctrl.get("splits", 0.7),
        ctrl.get("sampling_method", "None"), ctrl.get("class_weight", "None"),
        ctrl.get("threshold_method", "None"),
        classifier_params=ctrl.get("classifier_params", {}), random_state=42)
    assert not err, err
    state = _state("Classification", ctrl, {
        "selected_features": FEATS,
        "classification_method": ctrl["classification_method"],
        "splits": ctrl.get("splits", 0.7),
        "sampling_method": ctrl.get("sampling_method", "None"),
        "class_weight": ctrl.get("class_weight", "None"),
        "threshold_method": ctrl.get("threshold_method", "None"),
        "classifier_params": ctrl.get("classifier_params", {}),
        "classify_by": ["cell_line"], "classify_classes": classes})
    ns, _ = run_export(state, ctrl.get("csv_path", CSV), _wd(tag))
    return app_res, ns, state


# ---------------------------------------------------------------------------
# The control matrix
# ---------------------------------------------------------------------------

# Phasor's 11 lifetime markers are drawn as marker traces too.
# The export draws them with ax.plot (Line2D), so they never reach
# scatter_points() and would otherwise read as points the app has and the export lacks.
PHASOR_NON_DATA = ("Lifetime Markers",)


def case(runner, tag, ctrl, points=True, colors=False, main_axis_only=False,
         exclude_names=()):
    print(f"\n-- {tag} --")
    fig, ax, ns, state = runner(dict(ctrl), tag)
    if points:
        compare_points(tag, fig, ax, main_axis_only=main_axis_only,
                       exclude_names=exclude_names)
        if ctrl.get("opacity_by"):
            compare_alphas(tag, fig, ax, main_axis_only=main_axis_only,
                           include_color_alpha=state["method"] == "Dimension Reduction")
    compare_styling(tag, state, ax)
    if colors:
        compare_colors(tag, fig, ax)
    return fig, ax, ns, state


def shared_controls():
    """Controls that appear on every (or almost every) method."""
    print("\n=== Shared controls (encoding, filters, styling) ===")

    case(run_fc, "color_by=1", {"color_by": ["treatment"]}, colors=True)
    case(run_fc, "color_by=2", {"color_by": ["treatment", "dish"]}, colors=True)
    case(run_2d, "opacity_by", {"opacity_by": "dish"}, main_axis_only=True)
    case(run_2d, "shape_by", {"shape_by": "dish"}, main_axis_only=True)
    case(run_2d, "shape+opacity", {"shape_by": "dish", "opacity_by": "cell_line"},
         main_axis_only=True)
    case(run_fc, "separate_by", {"separate_by": "cell_line"})

    # Filters. Every other case already runs with SUBSET applied; these vary it.
    case(run_2d, "categorical_filter", {"categorical_filters": {"treatment": ["IAA"]}},
         main_axis_only=True)
    case(run_2d, "numerical_filter",
         {"numerical_filters": [(VAR, ">", 900.0)]}, main_axis_only=True)
    case(run_2d, "cat+num_filters",
         {"categorical_filters": {"treatment": ["IAA"], "cell_line": ["MCF7"]},
          "numerical_filters": [(VAR, ">", 850.0), (VAR2, "<=", 3.0)]}, main_axis_only=True)

    # Plot Styling
    case(run_2d, "point_size=12", {"point_size": 12}, main_axis_only=True)
    case(run_2d, "axis_label_size=20", {"axis_label_size": 20}, main_axis_only=True)
    case(run_2d, "legend_size=18", {"legend_size": 18}, main_axis_only=True)
    case(run_2d, "colormap=Set2", {"colormap": "Set2"}, colors=True, main_axis_only=True)

    # Both app plots and base_state() read the group-count toggle from session state.
    print("\n-- show_group_counts --")
    import re

    import streamlit as st

    from src.vis.helpers import format_group_label

    def _plain(label):
        """Convert Plotly legend markup to plain text with Matplotlib-style newlines."""
        return re.sub(r"<[^>]+>", "", label.replace("<br>", "\n"))

    st.session_state["plot_show_group_counts"] = True
    try:
        fig, ax, _ns, _state = run_fc({"color_by": ["treatment"]}, "show_counts")
        app_legend = [_plain(t.name) for t in fig.data if t.name]
        exp_legend = legend_labels_exp(ax)
        counted = [lbl for lbl in app_legend if "n=" in lbl]
        R.check("show_group_counts: app renders counts", bool(counted),
                f"{counted[:2]}")
        R.check("show_group_counts: export legend matches app legend",
                set(exp_legend) == set(app_legend),
                f"app={sorted(app_legend)[:2]} exp={sorted(exp_legend)[:2]}")
        R.check("show_group_counts: export legend carries the counts",
                all("n=" in lbl for lbl in exp_legend), f"{sorted(exp_legend)[:2]}")
        assert format_group_label("g", 5, True) != "g"  # guards the helper's contract
    finally:
        st.session_state["plot_show_group_counts"] = False

    # ...and with the toggle off, neither side invents one.
    fig, ax, _ns, _state = run_fc({"color_by": ["treatment"]}, "no_counts")
    R.check("show_group_counts off: no counts on either side",
            not any("n=" in lbl for lbl in
                    legend_labels_exp(ax) + [t.name for t in fig.data if t.name]))


def _encoding_csv():
    """Add numeric and missing-value encoding columns to inhibitors.csv.

    passage uses 2, 4, and 10 to distinguish numeric from lexical ordering; batch
    contains NaNs. Cases using this fixture verify that both paths normalize
    category values to the same strings, including "N/A" for missing values.
    """
    path = WORK / "encodings.csv"
    if not path.exists():
        import pandas as pd
        path.parent.mkdir(parents=True, exist_ok=True)
        src = pd.read_csv(CSV, index_col=False)
        src["passage"] = np.take([2, 10, 4], np.arange(len(src)) % 3)
        batch = np.take(np.array(["b1", "b2"], dtype=object), np.arange(len(src)) % 2)
        batch[::7] = np.nan
        src["batch"] = batch
        src.to_csv(path, index=False)
    return path


def encoding_controls():
    """Encoding columns that are not plain strings: numeric-valued, and NaN-bearing."""
    print("\n=== Encoding column types ===")
    common = {"csv_path": _encoding_csv(), "csv_name": "encodings.csv",
              "cats": [*CATS, "passage", "batch"]}

    # Feature Comparison because it is the only method that jitters, and the jitter is
    # split per (colour, shape, opacity) subgroup — so a mismatched key on either side
    # moves points rather than just relabelling a legend.
    _fig, ax, _ns, _state = case(run_fc, "enc: numeric-valued shape_by",
                                 {**common, "shape_by": "passage"})
    exp_legend = legend_labels_exp(ax)
    order = [lbl for lbl in exp_legend if lbl in ("2", "4", "10")]
    R.check("enc: numeric-valued groups sort numerically, not lexically",
            order == ["2", "4", "10"], f"{order}")

    _fig, ax, _ns, _state = case(run_fc, "enc: shape_by with missing values",
                                 {**common, "shape_by": "batch"})
    exp_legend = legend_labels_exp(ax)
    R.check("enc: missing values became one shared 'N/A' group",
            "N/A" in exp_legend and "nan" not in exp_legend, f"{sorted(exp_legend)}")

    fig, ax, _ns, _state = case(run_fc, "enc: numeric shape + missing opacity",
                                {**common, "shape_by": "passage", "opacity_by": "batch"})
    # The shared opacity mapping assigns "N/A" 0.15, below the ordinal ramp.
    R.check("enc: missing opacity level is held below the ramp on both sides",
            np.isclose(_alphas_app(fig).min(), 0.15)
            and np.isclose(_alphas_exp(ax).min(), 0.15),
            f"app={_alphas_app(fig).min()} exp={_alphas_exp(ax).min()}")


def hist_controls():
    print("\n=== Feature Histogram controls ===")
    for tag, ctrl in [
        ("hist: default", {}),
        ("hist: log_x", {"log_x": True}),
        ("hist: bin_width", {"bin_width": 25.0}),
        ("hist: gmm", {"apply_gmm": True}),
        ("hist: gmm+intersection", {"apply_gmm": True, "intersection_threshold": True}),
        ("hist: gmm max_components=5", {"apply_gmm": True, "gmm_max_components": 5}),
        ("hist: gmm min_weight=0.3", {"apply_gmm": True, "gmm_min_weight_threshold": 0.3}),
    ]:
        print(f"\n-- {tag} --")
        fig, ax, _ns, state = run_hist(dict(ctrl), tag)
        app_counts = {t.name: np.asarray(t.y, float) for t in fig.data
                      if t.name and t.y is not None and len(t.y)}
        exp_counts = {ln.get_label(): np.asarray(ln.get_ydata(), float)
                      for ln in ax.lines if not ln.get_label().startswith("_")}
        shared = set(app_counts) & set(exp_counts)
        R.check(f"{tag}: histogram/GMM curves ({len(shared)} shared)",
                bool(shared) and all(len(app_counts[k]) == len(exp_counts[k]) for k in shared),
                f"app={sorted(app_counts)[:3]} exp={sorted(exp_counts)[:3]}")
        compare_styling(tag, state, ax)


def fc_controls():
    print("\n=== Feature Comparison controls ===")
    for tag, ctrl in [
        ("fc: default", {}),
        ("fc: log_y", {"log_y": True}),
        ("fc: add_boxplot", {"add_boxplot": True}),
        ("fc: connect_means", {"connect_means": True, "mean_or_median": "mean"}),
        ("fc: effect=Cohen mean", {"effect_size_method": "Absolute Cohen's d",
                                   "mean_or_median": "mean"}),
        ("fc: effect=Cohen median", {"effect_size_method": "Absolute Cohen's d",
                                     "mean_or_median": "median"}),
        ("fc: effect=Glass", {"effect_size_method": "Glass's Delta",
                              "mean_or_median": "mean"}),
        ("fc: Independent t-test", {"statistical_test": "Independent t-test"}),
        ("fc: Welch's t-test", {"statistical_test": "Welch's t-test"}),
        ("fc: selected_pairs", {"statistical_test": "Welch's t-test",
                                "selected_pairs": ["IAA vs 2DG"]}),
        ("fc: custom_order", {"custom_order": {"compare_groups": ["IAA", "2DG"]}}),
        ("fc: shape_by", {"shape_by": "dish"}),
        ("fc: opacity_by", {"opacity_by": "dish"}),
    # Exercise sina jitter split jointly by colour, shape, and opacity.
        ("fc: shape+opacity", {"shape_by": "dish", "opacity_by": "cell_line"}),
        ("fc: everything", {"log_y": True, "add_boxplot": True, "connect_means": True,
                            "effect_size_method": "Absolute Cohen's d",
                            "mean_or_median": "mean",
                            "statistical_test": "Welch's t-test",
                            "separate_by": "cell_line", "shape_by": "dish"}),
    # Collapse to replicate means within x groups; check counts, quartiles, and effect sizes.
        ("fc: collapse=dish", {"collapse_by": "dish"}),
        ("fc: collapse+separate_by", {"collapse_by": "dish", "separate_by": "cell_line"}),
        # The SuperPlot: one colour per replicate, held across every x group.
        ("fc: collapse+subcolor same column", {"collapse_by": "dish", "subcolor_by": "dish"}),
        ("fc: collapse+log_y", {"collapse_by": "dish", "log_y": True}),
        ("fc: collapse+boxplot+effect", {"collapse_by": "dish", "add_boxplot": True,
                                         "effect_size_method": "Absolute Cohen's d",
                                         "mean_or_median": "mean"}),
    # Disable encoding channels that vary within a replicate.
        ("fc: collapse drops a finer decoration", {"collapse_by": "dish",
                                                   "shape_by": "image_name"}),
    ]:
        case(run_fc, tag, ctrl)


def subcolor_controls():
    """Check one global color and legend entry per subcolor value in Feature Comparison.

    Both renderers must assign the same colors across x groups. The page offers
    subcolor only for sina plots and shares its picker with shape, so the two
    encodings cannot be selected together.
    """
    print("\n=== Subcolor channel (Feature Comparison) ===")
    import re

    import streamlit as st

    def _plain(label):
        return re.sub(r"<[^>]+>", "", label.replace("<br>", "\n"))

    # Cases hold (tag, controls, subcolor values, x groups). X groups must appear
    # only in ticks; the legend may also contain entries for other encodings.
    cases = [
        # dish under SUBSET: two values, each present in both colour groups.
        ("subcolor: dish", {"subcolor_by": "dish"}, {"dish1", "dish2"}, {"IAA", "2DG"}),
        ("subcolor: +separate_by", {"subcolor_by": "dish", "separate_by": "cell_line"},
         {"dish1", "dish2"}, {"IAA", "2DG"}),
        ("subcolor: +opacity_by", {"subcolor_by": "dish", "opacity_by": "cell_line"},
         {"dish1", "dish2"}, {"IAA", "2DG"}),
        ("subcolor: +custom_order",
         {"subcolor_by": "dish", "custom_order": {"compare_groups": ["IAA", "2DG"]}},
         {"dish1", "dish2"}, {"IAA", "2DG"}),
        # Unfiltered, and more values than colour groups: five treatments nested inside
        # two cell lines. The palette is generated for the value count, so this is the
        # case where a per-group palette would disagree with a global one.
        ("subcolor: 5 values in 2 groups",
         {"color_by": ["cell_line"], "subcolor_by": "treatment",
          "categorical_filters": {}},
         {"0-control", "2DG", "Antimycin", "Cyanide", "IAA"}, {"MCF7", "Panc1"}),
    ]
    for tag, ctrl, values, groups in cases:
        fig, ax, _ns, _state = case(run_fc, tag, ctrl, colors=True)
        app_legend = {_plain(t.name) for t in fig.data if t.name}
        exp_legend = set(legend_labels_exp(ax))
        R.check(f"{tag}: legend is one entry per value on both sides",
                app_legend == exp_legend and values <= app_legend
                and not (groups & app_legend),
                f"app={sorted(app_legend)} exp={sorted(exp_legend)} want={sorted(values)}")
        app_title = fig.layout.title.text or ""
        R.check(f"{tag}: title names the subcolor column on both sides",
                f"subcolor: {ctrl['subcolor_by']}" in app_title
                and f"subcolor: {ctrl['subcolor_by']}" in ax.get_title(),
                f"app={app_title!r} exp={ax.get_title()!r}")

    # Subcolor legend counts span all x groups and include only plotted, non-NaN rows.
    st.session_state["plot_show_group_counts"] = True
    try:
        fig, ax, _ns, _state = run_fc({"subcolor_by": "dish"}, "subcolor_counts")
        app_legend = {_plain(t.name) for t in fig.data if t.name}
        exp_legend = set(legend_labels_exp(ax))
        R.check("subcolor: counted legend matches, and counts the whole figure",
                app_legend == exp_legend and all("n=" in lbl for lbl in exp_legend),
                f"app={sorted(app_legend)} exp={sorted(exp_legend)}")
    finally:
        st.session_state["plot_show_group_counts"] = False

    # A subcolor column with real NaNs (see _encoding_csv): both sides fold them to one
    # "N/A" value rather than one side inventing a "nan" level with its own colour.
    _fig, ax, _ns, _state = case(
        run_fc, "subcolor: missing values",
        {"csv_path": _encoding_csv(), "csv_name": "encodings.csv",
         "cats": [*CATS, "passage", "batch"], "subcolor_by": "batch"}, colors=True)
    exp_legend = legend_labels_exp(ax)
    R.check("subcolor: missing values became one shared 'N/A' value",
            "N/A" in exp_legend and "nan" not in exp_legend, f"{sorted(exp_legend)}")


def twod_controls():
    print("\n=== 2D Feature Distribution controls ===")
    for tag, ctrl in [
        ("2d: default", {}),
        ("2d: log_x", {"log_x": True}),
        ("2d: log_y", {"log_y": True}),
        ("2d: marginal=boxplot", {"marginal_plot_type": "boxplot"}),
        ("2d: marginal=violin", {"marginal_plot_type": "violin"}),
        ("2d: fit_regression", {"fit_regression": True}),
        ("2d: fit_gmm", {"fit_gmm_2d": True}),
        ("2d: gmm max_components=5", {"fit_gmm_2d": True, "gmm_max_components": 5}),
        ("2d: gmm min_weight=0.3", {"fit_gmm_2d": True, "gmm_min_weight_threshold": 0.3}),
        ("2d: everything", {"log_x": True, "log_y": True, "marginal_plot_type": "violin",
                            "fit_regression": True, "fit_gmm_2d": True,
                            "shape_by": "dish", "opacity_by": "cell_line"}),
    ]:
        case(run_2d, tag, ctrl, main_axis_only=True)


def phasor_controls():
    print("\n=== Phasor controls ===")
    for tag, ctrl in [
        ("phasor: harmonic=1", {"phasor_harmonic": 1}),
        ("phasor: harmonic=2", {"phasor_harmonic": 2}),
        ("phasor: f=0.05", {"phasor_f": 0.05}),
        ("phasor: h2+shape", {"phasor_harmonic": 2, "shape_by": "dish"}),
    ]:
        _fig, ax, _ns, _state = case(run_phasor, tag, ctrl,
                                     exclude_names=PHASOR_NON_DATA)
        # marker geometry must track BOTH the harmonic and the laser rate control
        wt = 2 * np.pi * ctrl.get("phasor_f", 0.08) * ctrl.get("phasor_harmonic", 1)
        want = (1 / (1 + wt**2), wt / (1 + wt**2))
        exp_marks = np.array([(ln.get_xdata()[0], ln.get_ydata()[0]) for ln in ax.lines
                              if ln.get_marker() == "o" and len(ln.get_xdata()) == 1])
        R.check(f"{tag}: 1 ns marker at n*2pi*f", bool(len(exp_marks))
                and np.any(np.all(np.isclose(exp_marks, want), axis=1)))


def dr_controls():
    print("\n=== Dimension Reduction controls ===")
    for tag, ctrl in [
        ("dr: PCA", {"dr_method": "PCA"}),
        ("dr: UMAP defaults", {"dr_method": "UMAP",
                               "hyperParam_dict": {"n_neighbors": 15, "min_dist": 0.1}}),
        ("dr: UMAP n_neighbors=5", {"dr_method": "UMAP",
                                    "hyperParam_dict": {"n_neighbors": 5, "min_dist": 0.5}}),
        # widget defaults: perplexity 15, early_exaggeration 1
        ("dr: t-SNE defaults", {"dr_method": "t-SNE",
                                "hyperParam_dict": {"perplexity": 15,
                                                    "early_exaggeration": 1}}),
        ("dr: t-SNE perplexity=40", {"dr_method": "t-SNE",
                                     "hyperParam_dict": {"perplexity": 40,
                                                         "early_exaggeration": 4}}),
        ("dr: shape+opacity", {"dr_method": "PCA", "shape_by": "dish",
                               "opacity_by": "cell_line"}),
    ]:
        fig, ax, _ns, _state = case(run_dr, tag, ctrl)
        R.check(f"{tag}: axis labels", ax.get_xlabel() == fig.layout.xaxis.title.text,
                f"app={fig.layout.xaxis.title.text!r} exp={ax.get_xlabel()!r}")


def clf_controls():
    print("\n=== Classification controls ===")
    for tag, ctrl in [
        ("clf: Random Forest", {"classification_method": "Random Forest"}),
        ("clf: Gradient Boosting", {"classification_method": "Gradient Boosting"}),
        ("clf: SVM", {"classification_method": "SVM"}),
        ("clf: Logistic Regression", {"classification_method": "Logistic Regression"}),
        ("clf: splits=0.5", {"classification_method": "Random Forest", "splits": 0.5}),
        ("clf: undersampling", {"classification_method": "Random Forest",
                                "sampling_method": "Undersampling"}),
        ("clf: oversampling", {"classification_method": "Random Forest",
                               "sampling_method": "Oversampling"}),
        ("clf: class_weight", {"classification_method": "Random Forest",
                               "class_weight": "Balanced"}),
        ("clf: threshold=BalAcc", {"classification_method": "Logistic Regression",
                                   "threshold_method": "Balanced Accuracy"}),
        ("clf: threshold=F1", {"classification_method": "Logistic Regression",
                               "threshold_method": "F1 Score"}),
        ("clf: hyperparams", {"classification_method": "Random Forest",
                              "classifier_params": {"n_estimators": 50, "max_depth": 4}}),
    ]:
        print(f"\n-- {tag} --")
        app_res, ns, _state = run_clf(dict(ctrl), tag)
        app_m, exp_m = app_res["metrics"], ns["metrics"]
        R.check(f"{tag}: accuracy", np.isclose(app_m["accuracy"], exp_m["accuracy"]),
                f"app={app_m['accuracy']:.6f} exp={exp_m['accuracy']:.6f}")
        R.check(f"{tag}: predictions ({len(ns['y_pred'])} rows)",
                (np.asarray(app_res["y_pred"]) == np.asarray(ns["y_pred"])).all())



# ---------------------------------------------------------------------------
# Filters, driven through the app's OWN widget and the page's OWN capture helpers
# ---------------------------------------------------------------------------

def _seed_filter_session(cat_selections, num_filters):
    """Put the widgets into the state the described UI interaction would leave them in.

    Categorical: `<col>_multiselect` holds the picks, or ["All"] for no filter.
    Numerical: one row per filter, chained by `add_another_num_filter_<i>`, keyed exactly
    as filter_widgets.py writes them.
    """
    import streamlit as st

    for key in [k for k in list(st.session_state.keys())
                if k.endswith("_multiselect") or k.startswith(("num_filter_",
                                                               "add_another_num_filter_"))]:
        del st.session_state[key]
    for col in CATS:
        st.session_state[f"{col}_multiselect"] = list(cat_selections.get(col, ["All"]))
    for i, (feat, op, thresh) in enumerate(num_filters):
        st.session_state[f"num_filter_feature_{i}"] = feat
        st.session_state[f"num_filter_operator_{i}_{feat}"] = op
        st.session_state[f"num_filter_threshold_{i}_{feat}"] = float(thresh)
        st.session_state[f"add_another_num_filter_{i}"] = i < len(num_filters) - 1
    st.session_state[f"num_filter_feature_{len(num_filters)}"] = "None"


def filter_controls():
    """Compare filters_widget() output with exports using the page's filter collectors.

    Exercise widget, capture, and replay together instead of using apply_filters().
    """
    print("\n=== Filter controls (real widget + real capture) ===")
    import streamlit as st

    from src.widgets.filter_widgets import filters_widget

    collect_cat, collect_num = page_collectors()
    full, _ = load_app_df(CSV, CATS, "cell_id", "image_name")

    cases = [
        ("no filter (All)", {}, []),
        ("cat: one value", {"treatment": ["IAA"]}, []),
        ("cat: several values", {"treatment": ["IAA", "2DG", "Cyanide"]}, []),
        ("cat: two columns", {"treatment": ["IAA"], "cell_line": ["MCF7"]}, []),
        ("num: >", {}, [(VAR, ">", 900.0)]),
        ("num: <=", {}, [(VAR, "<=", 900.0)]),
        ("num: two chained", {}, [(VAR, ">", 800.0), (VAR2, "<=", 2.0)]),
        ("cat+num", {"treatment": ["IAA", "2DG"]}, [(VAR, ">", 850.0)]),
        ("cat+num: two of each",
         {"treatment": ["IAA", "2DG"], "dish": ["dish1"]},
         [(VAR, ">", 800.0), (VAR2, "<=", 2.0)]),
    # Capture thresholds after the widget clamps them to the current filtered range.
        ("num: threshold clamped", {}, [(VAR, ">", 800.0), (VAR2, "<=", 99.0)]),
    # "Except:" stores exclusions but exports a keep-list. Explicit expectations
    # verify that full-frame complements agree with the widget's narrowed choices,
    # including when combined with another filter.
        ("cat: except one", {"treatment": [EXCEPT_LABEL, "IAA"]}, [],
         {"treatment": ["Cyanide", "Antimycin", "0-control", "2DG"]}),
        ("cat: except several", {"treatment": [EXCEPT_LABEL, "IAA", "2DG"]}, [],
         {"treatment": ["Cyanide", "Antimycin", "0-control"]}),
        ("cat: except + second column",
         {"treatment": [EXCEPT_LABEL, "IAA"], "dish": ["dish1"]}, [],
         {"treatment": ["Cyanide", "Antimycin", "0-control", "2DG"], "dish": ["dish1"]}),
        ("cat: except + num", {"treatment": [EXCEPT_LABEL, "Cyanide", "Antimycin"]},
         [(VAR, ">", 850.0)], {"treatment": ["IAA", "0-control", "2DG"]}),
    ]

    for tag, cat_sel, num_f, *expected in cases:
        expected_cat = expected[0] if expected else cat_sel
        print(f"\n-- filters: {tag} --")
        patch_streamlit()
        _seed_filter_session(cat_sel, num_f)

        app_df = filters_widget(full.copy(), CATS)

        # what the export button would capture from that same session state
        cap_cat = collect_cat(CATS, full)
        cap_num = collect_num()
        # Capture the final widget state, including threshold clamps.
        settled = [(f, st.session_state[f"num_filter_operator_{i}_{f}"],
                    float(st.session_state[f"num_filter_threshold_{i}_{f}"]))
                   for i, (f, _o, _t) in enumerate(num_f)]
        # Sorted: the capture becomes an isin() list on both sides, so which values are in
        # it is the contract and their order is not.
        R.check(f"filters {tag}: capture matches the widget's settled state",
                {k: sorted(v) for k, v in cap_cat.items()}
                == {k: sorted(v) for k, v in expected_cat.items()}
                and [(f, o, float(t)) for f, o, t in cap_num] == settled,
                f"cat={cap_cat} num={cap_num} settled={settled}")
        if tag == "num: threshold clamped":
            R.check("filters: out-of-range threshold was clamped, not passed through",
                    bool(cap_num) and cap_num[-1][2] < 99.0, f"{cap_num[-1][2]}")

        state = base_state("2D Feature Distribution", "inhibitors.csv", CATS,
                           color_by=["treatment"],
                           categorical_filters=cap_cat, numerical_filters=cap_num,
                           analysis_columns=list(full.columns),
                           method_params={"selected_x": VAR, "selected_y": VAR2,
                                          "log_x": False, "log_y": False,
                                          "marginal_plot_type": "gaussian fit",
                                          "fit_regression": False, "fit_gmm_2d": False,
                                          "gmm_max_components": 3,
                                          "gmm_min_weight_threshold": 0.1})
        ns, _ = run_export(state, CSV, _wd(f"filter_{tag}".replace(" ", "_")))

        app_ids = set(app_df["cell_id"])
        exp_ids = set(ns["df"]["cell_id"])
        R.check(f"filters {tag}: same rows survive ({len(app_ids)} app / {len(exp_ids)} export)",
                app_ids == exp_ids,
                f"app-only={len(app_ids - exp_ids)} exp-only={len(exp_ids - app_ids)}")
        R.check(f"filters {tag}: actually filtered something",
                len(app_ids) < len(full) or tag == "no filter (All)",
                f"{len(app_ids)} of {len(full)}")

    # With every value excluded there is no plot/export button. Check that capture
    # preserves the empty keep-list, which must filter out every row.
    print("\n-- filters: except everything --")
    patch_streamlit()
    all_treatments = full["treatment"].unique().tolist()
    _seed_filter_session({"treatment": [EXCEPT_LABEL, *all_treatments]}, [])
    app_df = filters_widget(full.copy(), CATS)
    R.check("filters except everything: app frame is empty", app_df.empty, f"{len(app_df)} rows")
    R.check("filters except everything: capture is isin([]), not 'no filter'",
            collect_cat(CATS, full) == {"treatment": []}, f"{collect_cat(CATS, full)}")

    _seed_filter_session({}, [])


SECTIONS = {
    "shared": shared_controls,
    "filters": filter_controls,
    "enc": encoding_controls,
    "hist": hist_controls,
    "fc": fc_controls,
    "subcolor": subcolor_controls,
    "2d": twod_controls,
    "phasor": phasor_controls,
    "dr": dr_controls,
    "clf": clf_controls,
}


def main(which="all"):
    for name, fn in SECTIONS.items():
        if which in ("all", name):
            fn()
    return 0 if R.summary("Controls") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
