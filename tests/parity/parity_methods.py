"""App-vs-export parity for analysis methods on the example datasets.

Covers histogram bin edges, GMM subpopulation numbering, the sina jitter +
effect-size defaults, 2D marginals for every marginal type, PCA/UMAP embeddings, and
the ANALYSIS_COLUMNS prune.

Run:  uv run python tests/parity/parity_methods.py [all|hist|gmm|fc|2d|dr|prune]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness_common import (
    EXAMPLES,
    WORK_ROOT,
    Results,
    app_point_traces,
    apply_filters,
    base_state,
    enable_derived,
    load_app_df,
    mpl_label,
    run_export,
    scatter_points,
    sorted_rows,
)
from harness_widgets import patch_streamlit

CATS = ["cell_line", "treatment", "dish", "image_name"]
CSV = EXAMPLES / "inhibitors.csv"
WORK = WORK_ROOT / "methods"
VAR = "Lifetime fit_nadh: tm"
VAR2 = "Lifetime fit free_nadh: Tau_m"

R = Results()


# ---------------------------------------------------------------- Histogram
def _app_bin_edges(df, var):
    from src.widgets.visualization_widgets import histogram_bin_width_widget

    patch_streamlit()
    return histogram_bin_width_widget(df[var], key=f"hist_bin_width_{var}")


def histogram():
    print("\n=== Feature Histogram — inhibitors.csv ===")
    patch_streamlit()
    from src.vis.univar import feature_histogram_plot

    df, _ = load_app_df(CSV, CATS, "cell_id", "image_name")
    fig = feature_histogram_plot(df.copy(), VAR, ["treatment"], colormap="tab10", log_x=False)

    # The app's bin-width widget was never touched, so nothing is in session state and
    # the export captures BIN_WIDTH=None — exactly what _export_script_button would do.
    state = base_state("Feature Histogram", "inhibitors.csv", CATS, color_by=["treatment"],
                       analysis_columns=list(df.columns),
                       method_params={"selected_var": VAR, "log_x": False, "apply_gmm": False,
                                      "intersection_threshold": False, "bin_width": None,
                                      "gmm_max_components": 3, "gmm_min_weight_threshold": 0.1})
    ns, _ = run_export(state, CSV, WORK / "hist")
    ax = ns["ax"]

    app_edges = _app_bin_edges(df, VAR)
    R.check("bin edges identical", np.allclose(ns["bin_edges"], app_edges),
            f"app n={len(app_edges)} exp n={len(ns['bin_edges'])}")

    app_counts = {t.name: np.asarray(t.y, float) for t in fig.data}
    exp_counts = {ln.get_label(): np.asarray(ln.get_ydata(), float) for ln in ax.lines}
    same = set(app_counts) == set(exp_counts) and all(
        np.allclose(app_counts[k], exp_counts[k]) for k in app_counts)
    R.check(f"per-group counts ({len(app_counts)} groups)", same,
            "" if same else f"app={list(app_counts)} exp={list(exp_counts)}")
    R.check("title", ax.get_title() ==
            f"Frequency histogram of {mpl_label(VAR)} by treatment", ax.get_title())


# ---------------------------------------------------------------- Histogram + GMM
def histogram_gmm(intersection):
    print(f"\n=== Feature Histogram + GMM (intersection={intersection}) — inhibitors.csv ===")
    patch_streamlit({"Use intersection as threshold": intersection})
    from src.vis.univar import feature_gmm_plot

    df, _ = load_app_df(CSV, CATS, "cell_id", "image_name")
    _fig, app_out = feature_gmm_plot(df.copy(), VAR, ["treatment"], colormap="tab10", log_x=False)

    state = base_state("Feature Histogram", "inhibitors.csv", CATS, color_by=["treatment"],
                       analysis_columns=list(df.columns),
                       method_params={"selected_var": VAR, "log_x": False, "apply_gmm": True,
                                      "intersection_threshold": intersection, "bin_width": None,
                                      "gmm_max_components": 3, "gmm_min_weight_threshold": 0.1})
    wd = WORK / f"gmm_{int(intersection)}"
    ns, _ = run_export(state, CSV, wd, transform=enable_derived)
    ax = ns["ax"]

    exp_out = pd.read_csv(wd / "gmm_grouped_data.csv")
    a = app_out.set_index("cell_id")["GMM_group"]
    e = exp_out.set_index("cell_id")["GMM_group"]
    common = a.index.intersection(e.index)
    same = len(common) == len(a) == len(e) and a.loc[common].equals(e.loc[common])
    n_diff = int((a.loc[common].fillna("~") != e.loc[common].fillna("~")).sum())
    R.check(f"GMM_group label for all {len(common)} cells", same, f"{n_diff} differ")
    R.check("subpopulation labels present", a.notna().any(),
            f"{a.nunique()} distinct: {sorted(a.dropna().unique())[:6]}")
    R.check("title", ax.get_title() ==
            f"Gaussian Mixture Model fit of {mpl_label(VAR)} by treatment", ax.get_title())


# ---------------------------------------------------------------- Feature Comparison
def _group_signatures(groups):
    """(n, sorted y, jitter offsets about the group's own centre) per plotted group,
    ordered so the app and export lists line up without relying on x positions."""
    sigs = []
    for x, y in groups:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if not len(x):
            continue
        centre = (x.min() + x.max()) / 2
        sigs.append((len(y), np.sort(y), np.sort(x - centre)))
    return sorted(sigs, key=lambda s: (s[0], float(s[1][0])))


def feature_comparison(separate_by, effect_size, statistical_test="Welch's t-test",
                       collapse_by=None):
    print(f"\n=== Feature Comparison (separate_by={separate_by}, effect={effect_size}, "
          f"test={statistical_test}, collapse_by={collapse_by}) ===")
    patch_streamlit()
    from src.vis.univar import feature_comparison_plot

    df, _ = load_app_df(CSV, CATS, "cell_id", "image_name")
    # Match the page's replicate collapse before calling the plot function.
    plot_df = df.copy()
    row_id_col, row_id_label, fov_col = "cell_id", "ID", "image_name"
    if collapse_by:
        from src.collapse import collapse_rows
        from src.dataset_io import resolve_effective_fov_col

        plot_df, row_id_col, _varied = collapse_rows(
            plot_df, collapse_by, ["treatment", separate_by], "cell_id")
        row_id_label = collapse_by
        fov_col = resolve_effective_fov_col(plot_df, "image_name")
    fig = feature_comparison_plot(
        plot_df, unique_row_id_col=row_id_col, fov_name_col=fov_col, selected_var=VAR,
        color_by=["treatment"], separate_by=separate_by, colormap="tab10",
        effect_size_method=effect_size, mean_or_median="mean",
        statistical_test=statistical_test, custom_order=None,
        row_id_label=row_id_label)

    from src.export_script import get_effect_size_threshold_capture

    thresh = get_effect_size_threshold_capture({}, effect_size, VAR, separate_by)

    state = base_state("Feature Comparison", "inhibitors.csv", CATS, color_by=["treatment"],
                       separate_by=separate_by, analysis_columns=list(df.columns),
                       method_params={"selected_var": VAR, "effect_size_method": effect_size,
                                      "mean_or_median": "mean",
                                      "statistical_test": statistical_test,
                                      "log_y": False, "add_boxplot": False,
                                      "connect_means": False,
                                      "effect_size_threshold": thresh,
                                      "selected_pairs": None, "custom_order": None,
                                      "collapse_by": collapse_by})
    tag = f"fc_{separate_by}_{effect_size.replace(chr(39), '').replace(' ', '')[:6]}"
    tag += "_" + statistical_test.replace(chr(39), "").replace(" ", "")[:8]
    tag += f"_collapse{collapse_by}" if collapse_by else ""
    ns, _ = run_export(state, CSV, WORK / tag)
    ax = ns["ax"]

    app_sigs = _group_signatures([(t.x, t.y) for t in app_point_traces(fig)])
    exp_sigs = _group_signatures([(o[:, 0], o[:, 1]) for o in
                                  (np.asarray(c.get_offsets(), float) for c in ax.collections)
                                  if len(o)])
    same = len(app_sigs) == len(exp_sigs) and all(
        a[0] == e[0] and np.allclose(a[1], e[1]) and np.allclose(a[2], e[2])
        for a, e in zip(app_sigs, exp_sigs))
    n_pts = sum(s[0] for s in app_sigs)
    R.check(f"sina groups: counts, y values, jitter offsets ({len(app_sigs)} groups, {n_pts} pts)",
            same, "" if same else f"app={[s[0] for s in app_sigs]} exp={[s[0] for s in exp_sigs]}")

    # Absolute x positions must use the app's section_spacing for separate_by groups.
    app_x = np.sort(np.concatenate([np.asarray(t.x, float) for t in app_point_traces(fig)]))
    exp_x = np.sort(scatter_points(ax)[:, 0])
    R.check("absolute x positions (section spacing)",
            app_x.shape == exp_x.shape and np.allclose(app_x, exp_x),
            "" if app_x.shape == exp_x.shape and np.allclose(app_x, exp_x)
            else f"app centres {np.unique(np.round(app_x, 2))[:8]} "
                 f"exp {np.unique(np.round(exp_x, 2))[:8]}")

    # Tick labels: the group alone on both sides — the section name is a separate
    # centred header, not folded into every tick.
    app_ticks = list(fig.layout.xaxis.ticktext or [])
    R.check(f"x tick labels ({len(app_ticks)})", app_ticks == list(ns["x_labels"]),
            f"app={app_ticks[:4]} exp={list(ns['x_labels'])[:4]}")

    if separate_by:
        # One bold header per section, centred under its groups.
        app_headers = sorted(
            (round(float(a.x), 4), a.text.replace("<b>", "").replace("</b>", ""))
            for a in fig.layout.annotations if a.text.startswith("<b>"))
        exp_headers = sorted((round(float(x), 4), lbl) for x, lbl in ns["section_headers"])
        R.check(f"section headers ({len(app_headers)})", app_headers == exp_headers,
                f"app={app_headers} exp={exp_headers}")

        # Dashed divider drawn at the centre of the inter-section gap. Filter on the
        # dash style: significance brackets are vertical line shapes too, so matching
        # only on x0 == x1 sweeps them in.
        app_div = sorted(round(float(s.x0), 4) for s in (fig.layout.shapes or [])
                         if s.x0 == s.x1 and getattr(s.line, "dash", None) == "dash")
        exp_div = sorted(round(float(x), 4) for x in ns["section_boundaries"])
        R.check(f"section dividers ({len(app_div)})", app_div == exp_div,
                f"app={app_div} exp={exp_div}")

    expected = 0.5 if effect_size == "Absolute Cohen's d" else 0.7 if effect_size == "Glass's Delta" else 0.0
    R.check("effect-size threshold default", thresh == expected, f"{thresh}")

    want = f"Distribution of {mpl_label(VAR)} by treatment"
    if separate_by:
        want += f" (separated by: {separate_by})"
    R.check("title", ax.get_title() == want, ax.get_title())


# ---------------------------------------------------------------- 2D distribution
def two_d(marginal, fit_gmm):
    print(f"\n=== 2D Feature Distribution (marginal={marginal}, gmm={fit_gmm}) ===")
    # Exercise only marginal types offered by the app.
    patch_streamlit({"2D Gaussian Mixture Model": fit_gmm, "Marginal Plot Type": marginal})
    from src.vis.bivar import feature_2d_distribution_plot

    df, _ = load_app_df(CSV, CATS, "cell_id", "image_name")
    fig, _table_md, _app_out = feature_2d_distribution_plot(
        df.copy(), unique_row_id_col="cell_id", fov_name_col="image_name",
        selected_x=VAR, selected_y=VAR2, color_by=["treatment"], colormap="tab10")

    state = base_state("2D Feature Distribution", "inhibitors.csv", CATS,
                       color_by=["treatment"], analysis_columns=list(df.columns),
                       method_params={"selected_x": VAR, "selected_y": VAR2,
                                      "log_x": False, "log_y": False,
                                      "marginal_plot_type": marginal,
                                      "fit_regression": False, "fit_gmm_2d": fit_gmm,
                                      "gmm_max_components": 3,
                                      "gmm_min_weight_threshold": 0.1})
    wd = WORK / f"2d_{marginal.replace(' ', '')}_{int(fit_gmm)}"
    ns, _ = run_export(state, CSV, wd, transform=enable_derived if fit_gmm else None)

    ax_main = ns["ax_main"]
    exp_pts = scatter_points(ax_main)
    app_pts = np.vstack([np.column_stack([t.x, t.y])
                         for t in app_point_traces(fig, main_axis_only=True)])
    same = app_pts.shape == exp_pts.shape and np.allclose(sorted_rows(app_pts), sorted_rows(exp_pts))
    R.check(f"scatter point cloud ({len(app_pts)})", same,
            "" if same else f"app={app_pts.shape} exp={exp_pts.shape}")

    def n_artists(axis):
        """One marginal per group, drawn as a line / box / violin depending on type."""
        if axis is None:
            return 0
        if marginal == "gaussian fit":
            return len([ln for ln in axis.lines if len(ln.get_xdata()) > 2])
        if marginal == "boxplot":
            return len(axis.patches)
        # violinplot emits body + cmin/cmax/cbar/cmedian artists per violin; only the
        # PolyCollection bodies correspond to the app's traces.
        from matplotlib.collections import PolyCollection

        return len([c for c in axis.collections
                    if isinstance(c, PolyCollection) and len(c.get_paths())])

    n_top, n_right = n_artists(ns.get("ax_top")), n_artists(ns.get("ax_right"))
    app_top = len([t for t in fig.data if t.yaxis == "y2"])
    app_right = len([t for t in fig.data if t.xaxis == "x2"])
    R.check(f"marginal count top ({app_top} app / {n_top} export)", app_top == n_top)
    R.check(f"marginal count right ({app_right} app / {n_right} export)", app_right == n_right)
    R.check("title", ax_main.get_title() ==
            f"2D Distribution of {mpl_label(VAR)} and {mpl_label(VAR2)} by treatment",
            ax_main.get_title())


# ---------------------------------------------------------------- Dimension reduction
def dimension_reduction(method, cat_filters=None):
    print(f"\n=== Dimension Reduction ({method}) — inhibitors.csv"
          f"{' filtered ' + str(cat_filters) if cat_filters else ''} ===")
    patch_streamlit()
    from src.vis.multivar import dimension_reduction_plot

    df, _ = load_app_df(CSV, CATS, "cell_id", "image_name")
    feats = ["Lifetime fit_nadh: t1", "Lifetime fit_nadh: t2", "Lifetime fit_nadh: a1",
             "Lifetime fit_nadh: tm", "Intensity morphology_nadh: area"]
    hp = ({} if method == "PCA"
          else {"n_neighbors": 15, "min_dist": 0.1} if method == "UMAP"
          else {"perplexity": 30})
    df = apply_filters(df, {"categorical_filters": cat_filters or {}})
    fig = dimension_reduction_plot(df.copy(), unique_row_id_col="cell_id",
                                   fov_name_col="image_name", selected_features=feats,
                                   colored_by=["treatment"], colormap="tab10",
                                   method=method, hyperParam_dict=hp)

    state = base_state("Dimension Reduction", "inhibitors.csv", CATS, color_by=["treatment"],
                       analysis_columns=list(df.columns),
                       categorical_filters=cat_filters or {},
                       method_params={"selected_features": feats, "dr_method": method,
                                      "hyperParam_dict": hp})
    ns, _ = run_export(state, CSV, WORK / f"dr_{method}")
    ax = ns["ax"]

    app_pts = np.vstack([np.column_stack([t.x, t.y]) for t in fig.data
                         if t.x is not None and len(t.x)])
    exp_pts = scatter_points(ax)
    same = app_pts.shape == exp_pts.shape and np.allclose(sorted_rows(app_pts),
                                                          sorted_rows(exp_pts), atol=1e-6)
    R.check(f"{method} embedding coordinates ({len(app_pts)} pts)", same,
            "" if same else f"app={app_pts.shape} exp={exp_pts.shape}")
    R.check("axis labels", ax.get_xlabel() == fig.layout.xaxis.title.text,
            f"app={fig.layout.xaxis.title.text!r} exp={ax.get_xlabel()!r}")


# ---------------------------------------------------------------- ANALYSIS_COLUMNS
def analysis_columns_prune():
    """A narrower categorical config makes get_features() drop real columns; the export
    must drop the same ones so its derived CSV carries the app's columns, not the file's."""
    print("\n=== ANALYSIS_COLUMNS prune — inhibitors.csv, categorical_cols=['treatment'] ===")
    narrow = ["treatment"]
    patch_streamlit({"Use intersection as threshold": False})
    from src.vis.univar import feature_gmm_plot

    df, _ = load_app_df(CSV, narrow, "cell_id", "image_name")
    dropped = {"cell_line", "dish"}
    R.check("app pruned cell_line/dish", dropped.isdisjoint(df.columns),
            f"kept={sorted(dropped & set(df.columns))}")

    _fig, app_out = feature_gmm_plot(df.copy(), VAR, ["treatment"], colormap="tab10", log_x=False)
    state = base_state("Feature Histogram", "inhibitors.csv", narrow,
                       color_by=["treatment"], analysis_columns=list(df.columns),
                       method_params={"selected_var": VAR, "log_x": False, "apply_gmm": True,
                                      "intersection_threshold": False, "bin_width": None,
                                      "gmm_max_components": 3, "gmm_min_weight_threshold": 0.1})
    wd = WORK / "prune"
    run_export(state, CSV, wd, transform=enable_derived)
    exp_out = pd.read_csv(wd / "gmm_grouped_data.csv")

    R.check("export derived CSV drops the same columns",
            dropped.isdisjoint(exp_out.columns), f"kept={sorted(dropped & set(exp_out.columns))}")
    R.check("export derived CSV column set == app columns",
            list(exp_out.columns) == list(app_out.columns),
            f"app_only={set(app_out.columns) - set(exp_out.columns)} "
            f"exp_only={set(exp_out.columns) - set(app_out.columns)}")


def main(which="all"):
    if which in ("all", "hist"):
        histogram()
    if which in ("all", "gmm"):
        histogram_gmm(False)
        histogram_gmm(True)
    if which in ("all", "fc"):
        feature_comparison(None, "Absolute Cohen's d")
        feature_comparison("cell_line", "Absolute Cohen's d")
        feature_comparison(None, "Glass's Delta")
        feature_comparison(None, "Absolute Cohen's d", "Independent t-test")
        # Collapse by on the full frame: 14k cells become one point per dish, so the
        # point count, the jitter and the effect-size n all move together.
        feature_comparison(None, "Absolute Cohen's d", "Welch's t-test", collapse_by="dish")
        feature_comparison("cell_line", "Absolute Cohen's d", "Welch's t-test",
                           collapse_by="dish")
    if which in ("all", "2d"):
        two_d("gaussian fit", False)
        two_d("gaussian fit", True)
        two_d("boxplot", False)
        two_d("violin", False)
    if which in ("all", "dr"):
        dimension_reduction("PCA")
        dimension_reduction("UMAP")
        # t-SNE on a filtered subset: covers the third DR method and the filter path
        dimension_reduction("t-SNE", {"treatment": ["IAA"]})
    if which in ("all", "prune"):
        analysis_columns_prune()
    return 0 if R.summary("Methods") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
