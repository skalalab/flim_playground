"""Keep analysis settings while the column-review screen replaces their widgets."""

_SETTING_KEYS = {
    "compare_pairs", "intersection_threshold", "classify_by_multiselect",
    "plot_point_size", "plot_axis_label_size", "plot_legend_size",
    "plot_colormap", "plot_show_group_counts",
}
_SETTING_PREFIXES = (
    "analysis_control_", "_menu_", "2d_x_menu_", "2d_y_menu_", "ms_",
    "vis_encoding_", "num_filter_", "add_another_num_filter_",
    "hist_bin_width_", "fit_gmm_", "fit_regression_", "log_x_", "log_y_",
    "add_boxplot_", "comparison_overlay_", "connect_means_", "marginal_plot_type_selector_",
    "glass_delta_thresh_",
    "cohens_d_thresh_", "clf_", "sort_cmp_",
)


def analysis_control_keys(state):
    """Capture setting keys before review widgets can introduce matching names."""
    return tuple(key for key in state if (
        key in _SETTING_KEYS or key.startswith(_SETTING_PREFIXES)
        or key.endswith("_multiselect")))


def derived_fit_control_keys(state):
    """Retain inputs to exported fits when switching analysis methods.

    Unlike the review-time snapshot, this runs on every page visit. Match only
    fit and feature controls so review buttons and dataset filters stay outside
    this module-switch lifecycle.
    """
    prefixes = (
        "_menu_", "2d_x_menu_", "2d_y_menu_", "analysis_control_phasor_",
        "fit_gmm_", "log_x_hist_", "log_x_2d_", "log_y_2d_",
    )
    keys = {"analysis_control_apply_gmm", "intersection_threshold"}
    return tuple(key for key in state if key in keys or key.startswith(prefixes))


def preserve_analysis_controls(state, keys):
    """Interrupt widget cleanup using the public Session State API.

    Called before the review gate can rerun, throughout review and its closing
    transition. Reassigning a value preserves it even when its widget is absent.
    The fixed key set comes from the analysis screen: review buttons may also end
    in `_multiselect` when a profile's name does, but must never be reassigned.
    """
    for key in keys:
        if key in state:
            state[key] = state[key]


def control_default(state, key, default):
    """Use a constructor default only when Session State has no saved value.

    Restored widgets must receive their value through Session State to update the
    browser. Also passing a default makes Streamlit display a duplication warning.
    """
    return None if key is not None and key in state else default


def number_input_default(state, key, default):
    """Omit a numeric default without making the input nullable."""
    return "min" if key is not None and key in state else default
