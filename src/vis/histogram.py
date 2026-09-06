"""Numerical histogram populations shared by Plotly and standalone export.

Individual observations have already been filtered and transformed. Membership
and GMM assignment use row positions; the caller's dataframe is never modified.
The functions are embedded with inspect.getsource(), along with their dependencies.
"""
import numpy as np
import pandas as pd

from .helpers import _find_best_gmm, find_intersection, natural_tuple_sort


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


def _assign_subpopulation_labels(values, best_gmm, thresholds, color_group):
    """Label components by ascending mean, for both threshold and posterior rules."""
    values = np.asarray(values)
    if thresholds is not None:
        ranks = np.digitize(values, bins=thresholds)
    else:
        sorted_indices = np.argsort(best_gmm.means_.flatten())
        rank_of = {int(orig): rank for rank, orig in enumerate(sorted_indices)}
        ranks = [rank_of[int(c)] for c in best_gmm.predict(values.reshape(-1, 1))]
    return [f"{color_group}_group{int(rank) + 1}" for rank in ranks]


def histogram_gmm(values, label, max_components=3, min_weight_threshold=0.1,
                  intersection_threshold=False):
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
        result["assignments"] = _assign_subpopulation_labels(values, model, thresholds, label)
    except Exception as error:
        # Discard incomplete fit output, retaining the group's observations/counts.
        result.update(gmm=None, x=np.array([]), pdf=np.array([]), components=[],
                      thresholds=None, h_index=None, assignments=None)
        result["notices"].append(f"GMM fitting failed: {error}")
    return result


def prepare_histogram(df, selected_var, color_by=None, separate_by=None, bin_width=None,
                      bin_edges=None, apply_gmm=False, max_components=3,
                      min_weight_threshold=0.1, intersection_threshold=False):
    """Prepare all category × color populations, bins, fits, and shared ranges.

    Labels are descriptive strings, while membership and assignment are positional.
    No internal grouping columns are added to the returned analyzed dataframe.
    """
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
            label = f"{separate_by}={category} | {color}" if separate_by else color
            group = dict(category=category, color_group=color, label=label,
                         positions=local_positions, values=local, count=len(local),
                         counts=np.histogram(local, bins=edges)[0], skewness=skewness)
            if apply_gmm:
                group.update(histogram_gmm(local, label, max_components,
                                          min_weight_threshold, intersection_threshold))
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
        df["GMM_group"] = assignments
    return dict(df=df, selected_var=selected_var, color_by=color_by,
                separate_by=separate_by, apply_gmm=apply_gmm, panels=panels,
                color_groups=color_groups, color_counts=color_counts, bin_edges=edges,
                bin_centers=(edges[:-1] + edges[1:]) / 2,
                x_range=[x_min, x_max], y_range=[0, y_max * 1.1 if y_max else 1.0])
