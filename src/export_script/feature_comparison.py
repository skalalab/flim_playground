"""Feature Comparison export: its builder and the helpers only it pastes."""
from src.export_script.mpl_helpers import scatter_with_encodings, style_axes_like_app
from src.export_script.skeleton import (
    MethodSpec,
    _build_visual_encoding,
    _extract_module_source,
    _extract_source,
    prepare_rows,
)
from src.vis.plot_defaults import (
    SUPERPLOT_OBSERVATION_MIN_SIZE,
    SUPERPLOT_OBSERVATION_OPACITY_SCALE,
    SUPERPLOT_OBSERVATION_SIZE_SCALE,
    SUPERPLOT_REPLICATE_JITTER_WIDTH,
    SUPERPLOT_REPLICATE_LINE_WIDTH,
    SUPERPLOT_REPLICATE_SIZE_SCALE,
)


def get_effect_size_threshold_capture(session_state, effect_size_method, selected_var, separate_by):
    """Read the effect-size threshold the app's widgets wrote to session state.

    Mirrors the widget keys and defaults in src/vis/feature_comparison.py, read by
    _add_effect_size_annotations (non-separate path) and feature_comparison_plot
    (separate path): the key suffix is the selected
    variable; Glass's Delta defaults to 0.7 and Absolute Cohen's d to 0.5 on
    both paths. `separate_by` changes neither default.
    """
    if effect_size_method == "Glass's Delta":
        return float(session_state.get(f"glass_delta_thresh_{selected_var}", 0.7))
    if effect_size_method == "Absolute Cohen's d":
        return float(session_state.get(f"cohens_d_thresh_{selected_var}", 0.5))
    return 0.0


def _build_feature_comparison(state: dict) -> str:
    from src.export_labels import available_label_column
    from src.vis import subcolor_palette
    from src.vis.helpers import _palette_rgb, _sorted_levels
    from src.vis.feature_comparison import (
        _calculate_effect_size,
        _compute_bracket_position,
        _density_at_points,
        _estimate_density_1d,
        box_statistics,
        bracket_positioning_metrics,
        cohens_d,
        comparison_labels,
        comparison_layout,
        create_subcolor_map,
        glass_delta,
        interleave_point_batches,
        p_value_stars,
        plan_effect_size_brackets,
        sina_offsets,
        summarize_superplot,
        superplot_display_frame,
    )

    # Inline the app's density helpers for identical sina jitter and degenerate-data
    # handling. Include the palette module before its callers and name _sorted_levels
    # explicitly: _extract_source does not resolve transitive dependencies.
    subcolor_src = _extract_module_source(subcolor_palette) + "\n" + _extract_source(
        _palette_rgb, _sorted_levels, interleave_point_batches, create_subcolor_map,
    )
    effect_size_src = _extract_source(
        glass_delta, cohens_d, _compute_bracket_position, _estimate_density_1d,
        _density_at_points, summarize_superplot, superplot_display_frame,
        sina_offsets, box_statistics, comparison_layout, comparison_labels,
        _calculate_effect_size, p_value_stars, bracket_positioning_metrics, plan_effect_size_brackets,
    )

    from src.feature_labels import format_feature_label

    # The app's own axis-label helper and chart-theme styling, so the plot renders identical
    # FLIM notation (e.g. "nadh τ₁ (ps)") and theme; scatter_with_encodings draws the sina cloud.
    group_preparation = _extract_source(available_label_column, format_feature_label,
                                        style_axes_like_app, scatter_with_encodings) + """
# Reserve the grouping column across both replicate and observation layers.
_fc_columns = list(df.columns) + (list(source_df.columns) if OVERLAY == "SuperPlot" else [])
FC_GROUP_COLUMN = available_label_column(_fc_columns, "_color_group")
"""
    return prepare_rows(state, _header(state), ("SHAPE_BY", "OPACITY_BY", "SUBCOLOR_BY"),
                        _LOG_Y_BLOCK) + group_preparation + _build_visual_encoding(
        overlap_point=False, group_column_expr="FC_GROUP_COLUMN", byte_colors=False
    ) + f"""
# ============================================================
# Feature Comparison — Sina Plot
# ============================================================

# Subcolor palette (module extracted from FLIM Playground source)
{subcolor_src}

# Effect size + bracket positioning functions (extracted from FLIM Playground source)
{effect_size_src}

# The main frame remains replicate means; observations and SEM ends affect geometry only.
display_df = df
if OVERLAY == "SuperPlot":
    source_df[FC_GROUP_COLUMN] = (source_df[COLOR_BY].astype(str).agg("::".join, axis=1)
                                 if COLOR_BY else "all_data")
    _summary_groups = ([SEPARATE_BY] if SEPARATE_BY else []) + [FC_GROUP_COLUMN]
    superplot_summary_df = summarize_superplot(df, SELECTED_VAR, _summary_groups)
    display_df = superplot_display_frame(source_df, superplot_summary_df, SELECTED_VAR, _summary_groups)
    if (superplot_summary_df["count"] == 1).any():
        print("NOTE: SuperPlot shows the mean only for groups with one replicate; SEM requires at least two.")

# --- Separate_by logic ---
separate_groups = [None]
if SEPARATE_BY:
    separate_groups = natural_tuple_sort(df[SEPARATE_BY].unique().astype(str).tolist())

# --- Layout: order, slots, ticks, sections and dividers (the app's own rule) ---
_present_cells = set()
if SEPARATE_BY:
    _present_cells = {{(sec_group, cg) for sec_group in separate_groups
                      for cg in df.loc[df[SEPARATE_BY] == sec_group, FC_GROUP_COLUMN].unique()}}
layout = comparison_layout(color_groups, [] if separate_groups == [None] else separate_groups,
                           _present_cells, CUSTOM_ORDER)
ordered_color_groups = layout["compare_groups"]
ordered_separate_groups = layout["separate_groups"] or [None]
x_positions = layout["x_positions"]
tick_positions, x_labels = layout["tick_positions"], layout["tick_labels"]
section_boundaries = layout["dividers"]
section_headers = [(section["center"], section["group"])
                   for section in layout["sections"] if section["combinations"]]

fig, ax = plt.subplots(figsize=(max(10, len(tick_positions) * 1.2), 6))

# Count groups across all sections of the retained frame for shared legend entries.
group_counts = df.groupby(FC_GROUP_COLUMN).size().to_dict()
# Subcolor identifies nested values across the figure; x positions, boxes, and
# statistics continue to describe comparison groups.
subcolor_of = create_subcolor_map(
    df, SUBCOLOR_BY, FC_GROUP_COLUMN, ordered_color_groups, engine='mpl', colormap=COLORMAP)
subcolor_counts = {{}}
if subcolor_of:
    subcolor_counts = df[SUBCOLOR_BY].fillna("N/A").astype(str).value_counts().to_dict()
legend_entries = set()
# The app's Point Size is a diameter; Matplotlib scatter takes an area.
primary_point_area = (POINT_SIZE * {SUPERPLOT_REPLICATE_SIZE_SCALE}) ** 2 if OVERLAY == "SuperPlot" else POINT_SIZE ** 2
point_layers = [(df, primary_point_area, 1.0, 2, True)]
if OVERLAY == "SuperPlot":
    point_layers.insert(0, (source_df, max({SUPERPLOT_OBSERVATION_MIN_SIZE}, POINT_SIZE * {SUPERPLOT_OBSERVATION_SIZE_SCALE}) ** 2,
                           {SUPERPLOT_OBSERVATION_OPACITY_SCALE}, 1, False))
for layer_df, point_area, opacity_scale, layer_zorder, is_primary in point_layers:
    layer_opacity_map = {{level: alpha * opacity_scale for level, alpha in opacity_map.items()}}
    layer_alpha = opacity_scale if OVERLAY == "SuperPlot" else 0.7
    layer_linewidth = ({SUPERPLOT_REPLICATE_LINE_WIDTH} if is_primary else 0) if OVERLAY == "SuperPlot" else 0.5
    layer_jitter = {SUPERPLOT_REPLICATE_JITTER_WIDTH} if OVERLAY == "SuperPlot" and is_primary else 0.35
    for sec_group in ordered_separate_groups:
        if sec_group is not None:
            sec_df = layer_df[layer_df[SEPARATE_BY] == sec_group]
            sec_color_groups = [cg for cg in ordered_color_groups if cg in sec_df[FC_GROUP_COLUMN].unique()]
        else:
            sec_df = layer_df
            sec_color_groups = ordered_color_groups

        for cg in sec_color_groups:
            key = (sec_group, cg) if sec_group is not None else cg
            x_pos = x_positions[key]
            group_df = sec_df[sec_df[FC_GROUP_COLUMN] == cg]
            y_data = group_df[SELECTED_VAR].values

            # Fit independent layer jitter once per section/color group; encodings keep the same positions.
            x_vals = x_pos + sina_offsets(y_data, layer_jitter)

            if not subcolor_of:
                cg_label = (format_group_label(cg, group_counts.get(cg), SHOW_GROUP_COUNTS, engine='mpl')
                            if is_primary and cg not in legend_entries else None)
                scatter_with_encodings(ax, x_vals, y_data, color_map[cg][:3],
                                       cg_label, point_area,
                                       shape_vals=group_df[SHAPE_BY] if SHAPE_BY else None, shape_map=shape_map,
                                       opacity_vals=group_df[OPACITY_BY] if OPACITY_BY else None,
                                       opacity_map=layer_opacity_map, base_alpha=layer_alpha,
                                       linewidths=layer_linewidth, zorder=layer_zorder)
                if is_primary:
                    legend_entries.add(cg)
            else:
                # Reuse group-level jitter and give each subcolor value one figure-wide legend entry.
                _subcolor_vals = group_df[SUBCOLOR_BY].fillna("N/A").astype(str).values
                _shape_vals = group_df[SHAPE_BY].values if SHAPE_BY else None
                _opacity_vals = group_df[OPACITY_BY].values if OPACITY_BY else None
                # Interleave subcolor batches to limit occlusion bias, skipping absent values.
                for _value, _mask in interleave_point_batches({{
                        _v: np.flatnonzero(_subcolor_vals == _v) for _v in subcolor_of}}):
                    _label = (format_group_label(_value, subcolor_counts.get(_value),
                                                 SHOW_GROUP_COUNTS, engine='mpl')
                              if is_primary and _value not in legend_entries else None)
                    # Batch indices are positional, so index arrays instead of Series labels.
                    scatter_with_encodings(ax, x_vals[_mask], y_data[_mask], subcolor_of[_value][:3],
                                           _label, point_area,
                                           shape_vals=_shape_vals[_mask] if SHAPE_BY else None, shape_map=shape_map,
                                           opacity_vals=_opacity_vals[_mask] if OPACITY_BY else None,
                                           opacity_map=layer_opacity_map, base_alpha=layer_alpha,
                                           linewidths=layer_linewidth, zorder=layer_zorder)
                    if is_primary:
                        legend_entries.add(_value)

# --- SuperPlot mean and sample SEM across the replicate points ---
if OVERLAY == "SuperPlot":
    for _summary_key, _summary in superplot_summary_df.iterrows():
        _summary_key = _summary_key if isinstance(_summary_key, tuple) else (_summary_key,)
        _groups = dict(zip(superplot_summary_df.index.names, _summary_key))
        _section = _groups[SEPARATE_BY] if SEPARATE_BY else None
        _color = _groups[FC_GROUP_COLUMN]
        _key = (_section, _color) if _section is not None else _color
        _x = x_positions[_key]
        _mean, _sem = _summary["mean"], _summary["sem"]
        ax.plot([_x - 0.2, _x + 0.2], [_mean, _mean], color='black', linewidth=2, zorder=1.5)
        if np.isfinite(_sem):
            _low, _high = _mean - _sem, _mean + _sem
            ax.plot([_x, _x], [_low, _high], color='black', linewidth=2, zorder=1.5)
            ax.plot([_x - 0.1, _x + 0.1], [_low, _low], color='black', linewidth=2, zorder=1.5)
            ax.plot([_x - 0.1, _x + 0.1], [_high, _high], color='black', linewidth=2, zorder=1.5)

# --- Boxplot overlay ---
if OVERLAY == "Boxplot":
    for sec_group in ordered_separate_groups:
        if sec_group is not None:
            sec_df = df[df[SEPARATE_BY] == sec_group]
            sec_color_groups = [cg for cg in ordered_color_groups if cg in sec_df[FC_GROUP_COLUMN].unique()]
        else:
            sec_df = df
            sec_color_groups = ordered_color_groups

        for cg in sec_color_groups:
            key = (sec_group, cg) if sec_group is not None else cg
            x_pos = x_positions[key]
            gdata = sec_df[sec_df[FC_GROUP_COLUMN] == cg][SELECTED_VAR].dropna().values
            if len(gdata) == 0:
                continue
            # Match the app's IQR fences, capped at observed extrema.
            # Matplotlib's automatic whiskers instead snap to in-fence observations.
            stats = box_statistics(gdata)
            box_stats = dict(q1=stats["q1"], med=stats["median"], q3=stats["q3"],
                             whislo=stats["lower_fence"], whishi=stats["upper_fence"],
                             mean=stats["mean"], fliers=[])
            bp = ax.bxp([box_stats], positions=[x_pos], widths=0.5, patch_artist=True,
                        manage_ticks=False, showfliers=False, showmeans=True, meanline=True, zorder=1)
            bp['boxes'][0].set_facecolor('none')
            bp['boxes'][0].set_edgecolor('black')
            bp['medians'][0].set_color('black')
            bp['medians'][0].set_linewidth(2)
            bp['means'][0].set_color('black')
            bp['means'][0].set_linestyle('--')

# --- Connect means ---
if CONNECT_MEANS:
    for sec_group in ordered_separate_groups:
        if sec_group is not None:
            sec_df = df[df[SEPARATE_BY] == sec_group]
            sec_color_groups = [cg for cg in ordered_color_groups if cg in sec_df[FC_GROUP_COLUMN].unique()]
        else:
            sec_df = df
            sec_color_groups = ordered_color_groups

        means_x, means_y = [], []
        for cg in sec_color_groups:
            key = (sec_group, cg) if sec_group is not None else cg
            x_pos = x_positions[key]
            gdata = sec_df[sec_df[FC_GROUP_COLUMN] == cg][SELECTED_VAR].dropna()
            if len(gdata) > 0:
                means_x.append(x_pos)
                means_y.append(gdata.mean())
        if len(means_x) > 1:
            ax.plot(means_x, means_y, 'k-o', linewidth=2, markersize=6, zorder=3)

# --- Section dividers ---
# The app's divider is a 2 px dashed line in the theme colour.
for boundary in section_boundaries:
    ax.axvline(x=boundary, linestyle='--', color='black', linewidth=2)

# --- Effect size / statistical test annotations ---
bracket_labels = []
if EFFECT_SIZE_METHOD != "None" or STATISTICAL_TEST != "None":
    from itertools import combinations
    for sec_group in ordered_separate_groups:
        if sec_group is not None:
            sec_df = df[df[SEPARATE_BY] == sec_group]
            display_sec_df = display_df[display_df[SEPARATE_BY] == sec_group]
            sec_color_groups = [cg for cg in ordered_color_groups if cg in sec_df[FC_GROUP_COLUMN].unique()]
        else:
            sec_df = df
            display_sec_df = display_df
            sec_color_groups = ordered_color_groups

        if len(sec_color_groups) < 2:
            continue

        pairs = list(combinations(sec_color_groups, 2))
        if SELECTED_PAIRS is not None:
            pairs = [p for p in pairs
                     if f"{{p[0]}} vs {{p[1]}}" in SELECTED_PAIRS
                     or f"{{p[1]}} vs {{p[0]}}" in SELECTED_PAIRS]
        # Bracket spacing uses the GLOBAL data range (one scale across all sections), like the app.
        all_y = df[SELECTED_VAR].dropna()
        if OVERLAY == "SuperPlot":
            all_y = display_df[SELECTED_VAR].dropna()
        data_range = all_y.max() - all_y.min()
        if data_range == 0:
            data_range = 1.0

        # Plan the section's brackets with the app's own rule, then draw them in Matplotlib.
        _section_positions = {{cg: x_positions[(sec_group, cg) if sec_group is not None else cg]
                              for cg in sec_color_groups}}
        _brackets, _ = plan_effect_size_brackets(
            sec_df, SELECTED_VAR, FC_GROUP_COLUMN, pairs, _section_positions,
            effect_size_method=EFFECT_SIZE_METHOD, mean_or_median=MEAN_OR_MEDIAN,
            threshold=EFFECT_SIZE_THRESHOLD, statistical_test=STATISTICAL_TEST,
            data_range=data_range, fallback_y=all_y.min(), display_df=display_sec_df)
        for bracket in _brackets:
            es, pval = bracket["effect_size"], bracket["p_value"]
            x_start, x_end, y_bracket_top = bracket["x_start"], bracket["x_end"], bracket["y_top"]
            bracket_h = bracket["height"]
            ax.plot([x_start, x_start, x_end, x_end],
                   [y_bracket_top - bracket_h, y_bracket_top, y_bracket_top, y_bracket_top - bracket_h],
                   color='black', linewidth=1.5, zorder=4)
            # Match the font size applied by the app's final styling pass. Plotly centres
            # data-anchored annotations on y, which keeps each label below the next bracket.
            bracket_labels.append(ax.text((x_start + x_end) / 2, bracket["y_text"], bracket["text"],
                                          ha='center', va='center', fontsize=AXIS_LABEL_SIZE, zorder=4))

# --- Axis setup ---
ax.set_xticks(tick_positions)
ax.set_xticklabels(x_labels, fontsize=AXIS_LABEL_SIZE - 2)
# Title and y label name the feature and every active encoding (the app's own rule).
_labels = comparison_labels(SELECTED_VAR, COLOR_BY, log_y=LOG_Y, separate_by=SEPARATE_BY,
                            opacity_by=OPACITY_BY, shape_by=SHAPE_BY, subcolor_by=SUBCOLOR_BY,
                            engine='mpl')
ax.set_ylabel(_labels["y"], fontsize=AXIS_LABEL_SIZE)
ax.set_title(_labels["title"], fontsize=TITLE_SIZE, fontweight='bold', loc='left')
ax.tick_params(axis='y', labelsize=AXIS_LABEL_SIZE - 2)
style_axes_like_app(ax)
add_encoding_legend_entries(ax, shape_map, opacity_map, LEGEND_SIZE ** 2)
# Like the app: the legend sits beside the plot, its markers sized by the legend font.
legend = ax.legend(fontsize=LEGEND_SIZE, loc='upper left', frameon=False,
                   bbox_to_anchor=(1.02, 1), borderaxespad=0)
for handle in legend.legend_handles:
    handle.set_sizes([LEGEND_SIZE ** 2])

# Match the app's tick angles: slant labels longer than four characters.
if max((len(str(lbl)) for lbl in x_labels), default=0) > 4:
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

# Center one bold header over each section and reserve space above the tick labels.
if section_headers:
    # Reserve the app's header-height ratio plus Matplotlib's default 3.5-point tick padding.
    ax.tick_params(axis='x', pad=1.6 * AXIS_LABEL_SIZE + 3.5)
    for _header_x, _header_label in section_headers:
        # Use the app's final annotation size. Point-based offsets survive tight_layout resizing.
        ax.annotate(_header_label, xy=(_header_x, 0), xycoords=('data', 'axes fraction'),
                    xytext=(0, -0.25 * AXIS_LABEL_SIZE), textcoords='offset points',
                    ha='center', va='top', fontsize=AXIS_LABEL_SIZE, fontweight='bold')

# Plotly's autorange makes room for annotation text; Matplotlib's ignores text. Raise the
# y maximum until the highest label, plus the headers' padding, fits inside the axes.
if bracket_labels:
    plt.tight_layout()  # the final axes height sets each label's height in data units
    _axes_px = ax.get_window_extent().height
    _y_low, _y_high = ax.get_ylim()
    for _label in bracket_labels:
        _reach_px = (_label.get_window_extent().height / 2
                     + 0.25 * AXIS_LABEL_SIZE * fig.dpi / 72)
        if _reach_px < _axes_px:
            _y_high = max(_y_high, _y_low + (_label.get_position()[1] - _y_low)
                          * _axes_px / (_axes_px - _reach_px))
    ax.set_ylim(_y_low, _y_high)
"""


def _header(state: dict) -> str:
    """Selected-feature NaN removal, then the SuperPlot guard that matches the export's collapse."""
    header = """
df = df[df[SELECTED_VAR].notna()].copy()
"""
    if state.get("method_params", {}).get("collapse_by"):
        return header + """
if OVERLAY == "SuperPlot":
    if not globals().get("COLLAPSE_BY") or COLLAPSE_BY not in df.columns:
        raise ValueError("SuperPlot requires a valid Collapse by column.")
    # Keep filtered observations before aggregation, after selected-feature NaN removal.
    source_df = df.copy()
"""
    return header + """
if OVERLAY == "SuperPlot":
    # This export has no collapse logic; adding settings alone would leave cells as primary data.
    raise ValueError("SuperPlot requires Collapse by. Regenerate this export with Collapse by selected.")
"""


# Log transform after optional replicate aggregation; SuperPlot logs its observations too.
_LOG_Y_BLOCK = """
if LOG_Y:
    _log_values = source_df[SELECTED_VAR] if OVERLAY == "SuperPlot" else df[SELECTED_VAR]
    if (_log_values < 0).any():
        print("WARNING: Cannot apply log to " + str(SELECTED_VAR) + ": contains negative values.")
        LOG_Y = False
    else:
        df = df.copy()
        df[SELECTED_VAR] = np.log10(df[SELECTED_VAR] + 1e-6)
        if OVERLAY == "SuperPlot":
            source_df[SELECTED_VAR] = np.log10(source_df[SELECTED_VAR] + 1e-6)
"""


def _config_lines(state: dict) -> list[str]:
    mp = state.get("method_params", {})
    lines = [f"SELECTED_VAR = {mp.get('selected_var')!r}"]
    if mp.get("collapse_by"):
        lines.append(f"COLLAPSE_BY = {mp['collapse_by']!r}  # one point per replicate within each plot group, holding the MEAN of its cells")
    lines.append(f"SEPARATE_BY = {state.get('separate_by')!r}")
    lines.append(f"SUBCOLOR_BY = {state.get('subcolor_by')!r}")
    lines.append(f"EFFECT_SIZE_METHOD = {mp.get('effect_size_method', 'None')!r}")
    lines.append(f"MEAN_OR_MEDIAN = {mp.get('mean_or_median')!r}")
    lines.append(f"STATISTICAL_TEST = {mp.get('statistical_test', 'None')!r}")
    lines.append(f"LOG_Y = {mp.get('log_y', False)!r}")
    overlay = mp.get("overlay", "Boxplot" if mp.get("add_boxplot", False) else "None")
    lines.append(f"OVERLAY = {overlay!r}  # 'None', 'Boxplot', or 'SuperPlot'; SuperPlot requires an export generated with Collapse by")
    lines.append(f"CONNECT_MEANS = {mp.get('connect_means', False)!r}")
    lines.append(f"EFFECT_SIZE_THRESHOLD = {mp.get('effect_size_threshold', 0.0)!r}")
    lines.append(f"SELECTED_PAIRS = {mp.get('selected_pairs')!r}  # None → annotate all pairs; else list of 'group1 vs group2' labels")
    custom_order = mp.get("custom_order")
    if custom_order:
        lines.append(f"CUSTOM_ORDER = {custom_order!r}  # Reorder: compare_groups and/or separate_groups")
    else:
        lines.append("CUSTOM_ORDER = None  # Set to {'compare_groups': ['group1', 'group2', ...]} to reorder x-axis")
    return lines


SPEC = MethodSpec(
    name="Feature Comparison",
    build=_build_feature_comparison,
    config_lines=_config_lines,
    imports=lambda state: ["from scipy.stats import gaussian_kde, ttest_ind, median_abs_deviation"],
    packages=lambda state: {"scipy"},
    footer=lambda state: "plt.tight_layout()",
)
