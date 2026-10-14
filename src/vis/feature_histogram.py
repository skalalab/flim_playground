"""Feature Histogram: populations, GMM subpopulations and the Plotly figure.

Individual observations have already been filtered and transformed. The pure half (bin
settings, populations, GMM assignment, the Gaussian intersection) is shared with
standalone export and embedded with inspect.getsource(), along with its dependencies;
membership and GMM assignment use row positions, and the caller's dataframe is never
modified. The Plotly half draws it in the app.
"""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import brentq
from scipy.stats import norm

from src.column_roles import code_span
from src.export_labels import available_label_column, format_export_group_labels
from src.feature_labels import format_feature_label
from src.natural_sort import natural_tuple_sort
from src.widgets.visualization_widgets import gmm_hyperParams_widget, histogram_bin_width_widget
from .gmm_tables import gmm_component_table, gmm_tables_html
from .helpers import _find_best_gmm, create_color_map, hover_field
from .plot_defaults import get_context_theme_color


def histogram_bin_settings(values):
    """Return numpy's automatic edges and the allowed default/common bin width."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    edges = np.histogram_bin_edges(values, bins="auto")
    if len(edges) <= 2:
        return edges, None, None
    max_width = float(np.ptp(values)) / 3
    return edges, min(float(edges[1] - edges[0]), max_width), max_width


def histogram_bin_edges(values, bin_width=None):
    """One consistent bin calculation, including constant and empty populations."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    edges, default_width, max_width = histogram_bin_settings(values)
    if default_width is None:
        return edges
    width = default_width if bin_width is None else float(bin_width)
    if not np.isfinite(width) or not 0 < width <= max_width:
        width = default_width
    return np.arange(values.min(), values.max() + width + 1e-9, width)


def histogram_skewness(values):
    """Bias-corrected numeric skewness, undefined for sparse or constant groups."""
    if len(values) < 3 or np.unique(values).size < 2:
        return np.nan
    skewness = float(pd.Series(values).skew())
    return skewness if np.isfinite(skewness) else np.nan


def histogram_legend_label(label, count, skewness, show_counts, *, show_skewness=True):
    """Compact local statistics shared by the app and exported panel legends."""
    name = str(label) + (f" (n={count})" if show_counts else "")
    if not show_skewness:
        return name
    skew = f"{skewness:.3f}" if np.isfinite(skewness) else "undefined"
    return f"{name}\nskew={skew}"


def histogram_labels(prepared, log_x, engine="plotly"):
    """Axis labels and title shared by the app's figure and exported scripts.

    Colour names are escaped for Plotly's HTML text; the feature label carries
    the engine's own markup and is left as the formatter made it.
    """
    import html

    pretty_var = format_feature_label(prepared["selected_var"], engine=engine)
    gmm_mode = prepared["apply_gmm"]
    names = ", ".join(prepared["color_by"])
    if engine == "plotly":
        names = html.escape(names)
    prefix = "Gaussian Mixture Model fit of " if gmm_mode else "Frequency histogram of "
    return {
        "x": f"log₁₀({pretty_var})" if log_x else pretty_var,
        "y": ("Density" if prepared["separate_by"] else "Probability Density") if gmm_mode else "Count",
        "title": prefix + pretty_var + (f" by {names}" if prepared["color_by"] else ""),
    }


def _assign_subpopulation_labels(values, best_gmm, thresholds, color_group, *,
                                 separate_by=None, category=None, color_by=None):
    """Label components by ascending mean, for both threshold and posterior rules."""
    values = np.asarray(values)
    if thresholds is not None:
        ranks = np.digitize(values, bins=thresholds)
    else:
        sorted_indices = np.argsort(best_gmm.means_.flatten())
        rank_of = {int(orig): rank for rank, orig in enumerate(sorted_indices)}
        ranks = [rank_of[int(c)] for c in best_gmm.predict(values.reshape(-1, 1))]
    return format_export_group_labels(ranks, color_group, separate_by=separate_by,
                                      category=category, color_by=color_by)


def histogram_gmm(values, label, max_components=3, min_weight_threshold=0.1,
                  intersection_threshold=False, *, separate_by=None, category=None,
                  color_by=None):
    """Fit one local population; failures and unsupported analyses stay local."""
    result = dict(gmm=None, x=np.array([]), pdf=np.array([]), components=[],
                  thresholds=None, h_index=None, assignments=None, notices=[])
    distinct = np.unique(values).size
    if len(values) < 2 or distinct < 2:
        result["notices"].append("GMM requires at least two distinct observations.")
        return result
    try:
        model = _find_best_gmm(values, max_components=min(max_components, distinct),
                               min_weight_threshold=min_weight_threshold)
        if model is None:
            result["notices"].append("No valid GMM found with current constraints.")
            return result
        x = np.linspace(values.min(), values.max(), 1000)
        pdf = np.exp(model.score_samples(x.reshape(-1, 1)))
        individual = model.predict_proba(x.reshape(-1, 1)) * pdf[:, np.newaxis]
        means = model.means_.flatten()
        weights = model.weights_
        sigmas = np.sqrt(model.covariances_.ravel())
        order = np.argsort(means)
        components = [dict(rank=rank + 1, mean=float(means[i]), std=float(sigmas[i]),
                           weight=float(weights[i]), density=individual[:, i])
                      for rank, i in enumerate(order)]
        result.update(gmm=model, x=x, pdf=pdf, components=components, h_index=0.0)
        if model.n_components == 1:
            return result

        overall_mean = np.sum(weights * means)
        means_std = np.std(means, ddof=1)
        if means_std > 0:
            entropy = -weights * np.log(np.maximum(weights, np.finfo(float).tiny))
            result["h_index"] = float(np.sum(entropy * np.abs(means - overall_mean) / means_std))
        thresholds = None
        if intersection_threshold:
            try:
                thresholds = np.array([
                    find_intersection(weights[i], means[i], sigmas[i],
                                      weights[j], means[j], sigmas[j])
                    for i, j in zip(order[:-1], order[1:])
                ])
                if not np.isfinite(thresholds).all() or (np.diff(thresholds) <= 0).any():
                    raise ValueError("thresholds are not finite and increasing")
                thresholds.sort()
            except Exception:
                thresholds = None
                result["notices"].append("Intersection threshold is unavailable; using hard assignment in this group.")
        result["thresholds"] = thresholds
        result["assignments"] = _assign_subpopulation_labels(
            values, model, thresholds, label, separate_by=separate_by,
            category=category, color_by=color_by)
    except Exception as error:
        # Discard incomplete fit output, retaining the group's observations/counts.
        result.update(gmm=None, x=np.array([]), pdf=np.array([]), components=[],
                      thresholds=None, h_index=None, assignments=None)
        result["notices"].append(f"GMM fitting failed: {error}")
    return result


def prepare_histogram(df, selected_var, color_by=None, separate_by=None, bin_width=None,
                      bin_edges=None, apply_gmm=False, max_components=3,
                      min_weight_threshold=0.1, intersection_threshold=False, label_column=None):
    """Prepare all category × color populations, bins, fits, and shared ranges.

    Labels are descriptive strings, while membership and assignment are positional.
    No internal grouping columns are added to the returned analyzed dataframe.
    """
    label_column = available_label_column(df.columns, label_column or "GMM_group")
    color_by = [color_by] if isinstance(color_by, str) else list(color_by or [])
    if separate_by is not None:
        if not isinstance(separate_by, str) or separate_by not in df.columns:
            raise ValueError("Separate by must be one categorical column present in the data.")
        if separate_by in color_by:
            raise ValueError("Separate by cannot also be used for Color by.")
    df = df.loc[df[selected_var].notna()].copy()
    values = df[selected_var].to_numpy(dtype=float)
    # Categorical normalization is confined to membership arrays, preserving CSV metadata.
    if color_by:
        categories = df[color_by].astype(str).where(df[color_by].notna(), "N/A")
        colors = categories.agg("::".join, axis=1).to_numpy()
    else:
        colors = np.full(len(df), "all_data", dtype=object)
    color_groups = natural_tuple_sort(list(dict.fromkeys(colors)))
    color_counts = {group: int(np.sum(colors == group)) for group in color_groups}
    if separate_by:
        series = df[separate_by]
        categories = series.astype(str).where(series.notna(), "N/A").to_numpy()
        levels = natural_tuple_sort(list(dict.fromkeys(categories)))
    else:
        categories = np.full(len(df), None, dtype=object)
        levels = [None]
    if apply_gmm:
        # Density curves do not use histogram bins. A single extent bin keeps the
        # local sample totals without allocating potentially enormous auto bins
        # for outliers, or inheriting a width chosen in count mode.
        edges = np.histogram_bin_edges(values[np.isfinite(values)], bins=1)
    else:
        edges = histogram_bin_edges(values, bin_width) if bin_edges is None else np.asarray(bin_edges)
    panels = []
    assignments = np.full(len(df), None, dtype=object)
    x_min, x_max = float(edges[0]), float(edges[-1])
    y_max = 0.0
    for category in levels:
        positions = np.flatnonzero(categories == category) if separate_by else np.arange(len(df))
        panel = dict(category=category, positions=positions, groups=[])
        for color in color_groups:
            local_positions = positions[colors[positions] == color]
            if not len(local_positions):
                continue
            local = values[local_positions]
            skewness = histogram_skewness(local)
            # One combined title per population: no separate heading names the
            # level, so the level and the colour group travel together here.
            label = f"{category} × {color}" if separate_by else color
            group = dict(category=category, color_group=color, label=label,
                         positions=local_positions, values=local, count=len(local),
                         counts=np.histogram(local, bins=edges)[0], skewness=skewness)
            if apply_gmm:
                group.update(histogram_gmm(local, color, max_components,
                                          min_weight_threshold, intersection_threshold,
                                          separate_by=separate_by, category=category,
                                          color_by=color_by))
                if group["assignments"] is not None:
                    assignments[local_positions] = group["assignments"]
                if len(group["pdf"]):
                    y_max = max(y_max, float(np.max(group["pdf"])),
                                *(float(np.max(c["density"])) for c in group["components"]))
                    x_min = min(x_min, float(group["x"][0]))
                    x_max = max(x_max, float(group["x"][-1]))
                if group["thresholds"] is not None and len(group["thresholds"]):
                    x_min = min(x_min, float(min(group["thresholds"])))
                    x_max = max(x_max, float(max(group["thresholds"])))
            else:
                y_max = max(y_max, float(max(group["counts"], default=0)))
            panel["groups"].append(group)
        panels.append(panel)
    if apply_gmm:
        df[label_column] = assignments
    return dict(df=df, selected_var=selected_var, color_by=color_by,
                separate_by=separate_by, apply_gmm=apply_gmm, panels=panels,
                color_groups=color_groups, color_counts=color_counts, bin_edges=edges,
                bin_centers=(edges[:-1] + edges[1:]) / 2,
                x_range=[x_min, x_max], y_range=[0, y_max * 1.1 if y_max else 1.0])


# Plotly figure and Streamlit rendering for Feature Histogram and its GMM overlay.
def feature_histogram_plot(df, selected_var, color_by=None, colormap="tab10", log_x=False,
                           separate_by=None):
    """Count curves in full-width category rows, with one set of bin controls."""
    from src.widgets.analysis_widget_state import preserve_analysis_controls
    from src.widgets.visualization_widgets import histogram_bin_width_key

    width_keys = [histogram_bin_width_key(selected_var, scale) for scale in (False, True)]
    width_key = histogram_bin_width_key(selected_var, log_x)
    legacy_key = f"hist_bin_width_{selected_var}"
    if not any(key in st.session_state for key in width_keys) and legacy_key in st.session_state:
        # The previous shared value belongs to the scale currently on screen.
        st.session_state[width_key] = st.session_state[legacy_key]
    preserve_analysis_controls(st.session_state, width_keys)
    edges = histogram_bin_width_widget(df[selected_var].dropna(), key=width_key)
    prepared = prepare_histogram(df, selected_var, color_by, separate_by, bin_edges=edges)
    return _histogram_figure(prepared, colormap, log_x)


def feature_gmm_plot(df, selected_var, color_by=None, colormap="tab10", log_x=False,
                     separate_by=None, label_column=None):
    """Category-local GMM curves and a positionally labeled analyzed dataframe."""
    max_components, min_weight = gmm_hyperParams_widget()
    intersection = st.checkbox(
        "Use intersection as threshold", value=False, key="intersection_threshold",
        help="Use intersections between adjacent components as thresholds within each "
             "category and color group. If unavailable, use the highest posterior probability.")
    prepared = prepare_histogram(
        df, selected_var, color_by, separate_by, apply_gmm=True,
        max_components=max_components, min_weight_threshold=min_weight,
        intersection_threshold=intersection, label_column=label_column)
    return _histogram_figure(prepared, colormap, log_x), prepared["df"]


def _histogram_figure(prepared, colormap, log_x):
    """Render panel legends and serialize optional GMM details for display below."""
    theme = get_context_theme_color()
    panels = prepared["panels"]
    separate_by = prepared["separate_by"]
    rows = max(1, len(panels))
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=min(0.16, 80 / (300 * rows)) if rows > 1 else 0,
        subplot_titles=[html.escape(str(p['category']))
                        for p in panels] if separate_by else None)
    fig.update_annotations(font_color=theme)
    colors = create_color_map(prepared["color_groups"], overlap_point=False, colormap=colormap)
    show_counts = st.session_state.get("plot_show_group_counts", False)
    gmm_mode = prepared["apply_gmm"]
    dash_styles = ["dash", "dot", "dashdot", "longdash", "longdashdot"]
    summaries = []
    component_tables = []
    for row, panel in enumerate(panels, 1):
        summary = dict(category=panel["category"], groups=[])
        legend_id = "legend" + (str(row) if row > 1 else "")
        yaxis = "yaxis" + (str(row) if row > 1 else "")
        fig.update_layout({legend_id: dict(
            orientation="v", x=1.02 if gmm_mode else 1, y=fig.layout[yaxis].domain[1],
            xanchor="left" if gmm_mode else "right", yanchor="top", xref="paper", yref="paper",
            font=dict(color=theme),
            bgcolor="rgba(255,255,255,0.85)" if theme == "black" else "rgba(30,30,30,0.85)",
            groupclick="togglegroup", tracegroupgap=4)})
        for group in panel["groups"]:
            color = group["color_group"]
            rank = prepared["color_groups"].index(color) * 10
            label = f"{color} GMM" if gmm_mode else color
            name = html.escape(histogram_legend_label(
                label, group["count"], group["skewness"], show_counts,
                show_skewness=not gmm_mode)).replace("\n", "<br>")
            legend_group = f"{row}:{color}"
            hover = hover_field("Group", html.escape(str(group["label"]))) + f"n={group['count']}<br>"
            if not gmm_mode or len(group["pdf"]):
                fig.add_trace(go.Scatter(
                    x=group["x"] if gmm_mode else prepared["bin_centers"],
                    y=group["pdf"] if gmm_mode else group["counts"],
                    mode="lines+markers" if not gmm_mode and len(prepared["bin_centers"]) == 1 else "lines",
                    marker=dict(color=colors[color], size=6), name=name,
                    line=dict(color=colors[color], width=2), legendgroup=legend_group,
                    legend=legend_id, showlegend=True, legendrank=rank,
                    hovertemplate=hover + hover_field("Density" if gmm_mode else "Count", "%{y}") + "<extra></extra>"),
                    row=row, col=1)
            else:
                # Sparse fits still contribute their optional local count.
                fig.add_trace(go.Scatter(
                    x=[None], y=[None], mode="lines", name=name,
                    legendgroup=legend_group, legend=legend_id, legendrank=rank,
                    line=dict(color=colors[color], width=2), showlegend=True,
                    hoverinfo="skip"), row=row, col=1)
            component_rows = []
            if gmm_mode:
                editable_rows = []
                for component in group["components"]:
                    component_rank = component["rank"]
                    mean_sd = f"{component['mean']:.2f} ± {component['std']:.2f}"
                    component_rows.append((component_rank,
                                           mean_sd,
                                           f"{component['weight']:.2f}"))
                    if len(group["components"]) <= 1:
                        continue
                    editable_rows.append(dict(
                        source_label=format_export_group_labels(
                            [component_rank - 1], color, separate_by=separate_by,
                            category=panel["category"], color_by=prepared["color_by"])[0],
                        component=component_rank, x_mean_sd=mean_sd,
                        weight=float(component["weight"])))
                    fig.add_trace(go.Scatter(
                        x=group["x"], y=component["density"], mode="lines",
                        name=f"{html.escape(str(color))} Component {component_rank}",
                        legendgroup=legend_group, legend=legend_id,
                        legendrank=rank + component_rank, showlegend=True,
                        line=dict(color=colors[color], width=1,
                                  dash=dash_styles[(component_rank - 1) % len(dash_styles)]),
                        hovertemplate=hover + f"Component {component_rank}<br>Density: %{{y}}<extra></extra>"),
                        row=row, col=1)
                if editable_rows and group.get("gmm") is not None:
                    component_tables.append(dict(
                        category=panel["category"], group=color,
                        features=[prepared["selected_var"]], rows=editable_rows,
                        h_index=group["h_index"]))
                for threshold in group["thresholds"] if group["thresholds"] is not None else []:
                    fig.add_shape(type="line", x0=threshold, x1=threshold,
                                  y0=0, y1=max(group["pdf"]),
                                  line=dict(color=colors[color], width=2, dash="dash"), opacity=0.5,
                                  row=row, col=1)
                    if not separate_by:
                        fig.add_annotation(x=threshold, y=max(group["pdf"]) * 1.05,
                                           text=f"Threshold ({threshold:.2f})", showarrow=False,
                                           font=dict(color=theme), row=row, col=1)
            summary["groups"].append(dict(
                label=group["label"], color_group=color,
                count=group["count"], color=colors[color],
                skewness=float(group["skewness"]) if np.isfinite(group["skewness"]) else None,
                components=component_rows, h_index=group.get("h_index"),
                thresholds=list(group["thresholds"]) if group.get("thresholds") is not None else [],
                notices=group.get("notices", [])))
        summaries.append(summary)
    labels = histogram_labels(prepared, log_x)
    fig.update_xaxes(title=dict(text="", font=dict(color=theme)),
                     tickfont=dict(color=theme), showgrid=False, zeroline=False,
                     range=prepared["x_range"], visible=False)
    fig.update_xaxes(title_text=labels["x"], visible=True, showticklabels=True, row=rows, col=1)
    fig.update_yaxes(title=dict(text=labels["y"], font=dict(color=theme)),
                     tickfont=dict(color=theme), showgrid=True, zeroline=False,
                     range=prepared["y_range"])
    fig.update_layout(
        title=dict(text=labels["title"], font=dict(color=theme)),
        height=300 * rows + 130 if separate_by else 450,
        hovermode="x unified", margin=dict(l=60, r=20, t=80, b=100, autoexpand=True),
        meta=dict(histogram=True, histogram_summaries=summaries,
                  histogram_gmm=gmm_mode, histogram_separator=separate_by,
                  histogram_feature=prepared["selected_var"],
                  gmm_component_tables=component_tables))
    return fig


def render_histogram_summaries(fig, component_editor=None):
    """Render category-local GMM details and return optional component name edits."""
    meta = fig.layout.meta
    value_names = {}
    if not meta["histogram_gmm"]:
        return value_names
    component_tables = {
        (table["category"], table["group"]): table
        for table in meta.get("gmm_component_tables", []) if len(table["rows"]) > 1
    } if component_editor is not None else {}
    for panel in meta["histogram_summaries"]:
        # No per-level heading: every title below already names its level.
        tables = []
        editable_tables = []
        for group in panel["groups"]:
            for notice in group["notices"]:
                st.info(f"{code_span(group['label'])}: {notice}")
            component_table = component_tables.get((panel["category"], group.get("color_group")))
            if component_table is not None and len(group["components"]) > 1:
                editable_tables.append(component_table)
            elif group["components"]:
                tables.append(gmm_component_table(
                    group["label"], group["components"], [meta["histogram_feature"]],
                    h_index=group["h_index"]))
            for i, threshold in enumerate(group["thresholds"], 1):
                st.markdown(f"Threshold for {code_span(group['label'])} between component {i} and {i+1}: **{threshold:.2f}**")
        if editable_tables:
            value_names.update(component_editor(editable_tables))
        if tables:
            st.markdown(gmm_tables_html(tables), unsafe_allow_html=True)
    return value_names


# Intersection of two weighted Gaussians, the GMM threshold.
def find_intersection(pi1, mu1, sigma1, pi2, mu2, sigma2):
    """
    Find the intersection point between two weighted Gaussian components where
    pi1 * N(x; mu1, sigma1) = pi2 * N(x; mu2, sigma2).
    """
    f = lambda x: pi1 * norm.pdf(x, mu1, sigma1) - pi2 * norm.pdf(x, mu2, sigma2)
    # Search for the intersection between the two means.
    return brentq(f, min(mu1, mu2), max(mu1, mu2))
