"""Background-correction calibration for QPI channels.

One settings row, then two rows: a per-FOV box plot of the corrected background to pick a
FOV from with that FOV's readouts beside it, then a cell picker with that cell's dry mass
beside it above a full-width line scan through it. Every number comes
from the two cached wrappers in ``src/fov_extraction.py``, so a settings change recomputes
once per FOV and re-picking costs nothing. The widget renders no button: the page's single
Confirm reads the returned settings.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.fov_extraction import corrected_qpi_image, qpi_fov_diagnostics
from src.qpi import DEFAULT_BACKGROUND, KEPT_FRACTION_WARNING, line_scan, mass_factor
from src.vis.helpers import get_context_theme_color

_METHOD_LABELS = {"none": "already corrected", "polynomial": "polynomial surface", "inpainting": "inpainting"}
_DEGREES = [2, 4, 6]


def _settings_controls(channel_name, defaults):
    """The recipe row; returns ``{"method", "degree", "expand_pct"}``."""
    cols = st.columns([1.6, 1, 1])
    with cols[0]:
        method = st.radio(
            "Background", list(_METHOD_LABELS), index=list(_METHOD_LABELS).index(defaults["method"]),
            format_func=_METHOD_LABELS.get, horizontal=True, key=f"qpi_method_{channel_name}",
            help="Polynomial: a robust-clipped surface through the background pixels. Inpainting: fill the cell regions from the surrounding fringes, then smooth. Already corrected: subtract nothing.",
        )
    degree = None
    with cols[1]:
        if method == "polynomial":
            default_degree = defaults["degree"] if defaults["degree"] in _DEGREES else DEFAULT_BACKGROUND["degree"]
            degree = st.selectbox("Degree", _DEGREES, index=_DEGREES.index(default_degree),
                                  key=f"qpi_degree_{channel_name}",
                                  help="Higher degrees follow curved illumination; watch the line scan for a surface bending into cells.")
    with cols[2]:
        expand_pct = 0.0
        if method != "none":
            expand_pct = st.number_input("Cell exclusion (+% area)", min_value=0.0, max_value=100.0,
                                         value=float(defaults["expand_pct"]), step=5.0,
                                         key=f"qpi_expand_{channel_name}",
                                         help="Grow each cell mask by this share of its area before choosing background pixels.")
    return {"method": method, "degree": degree, "expand_pct": float(expand_pct)}


def _theme_axes(fig, color):
    """Axis titles and ticks follow the theme, the way ``apply_plot_styling`` does elsewhere."""
    fig.update_xaxes(title_font_color=color, tickfont_color=color)
    fig.update_yaxes(title_font_color=color, tickfont_color=color)


def _fov_box_plot(fov_names, stats, color):
    """Per-FOV box of the corrected background in nm from precomputed percentiles; medians are clickable."""
    fig = go.Figure()
    x = list(range(1, len(fov_names) + 1))
    fig.add_trace(go.Box(
        x=x, lowerfence=[s[0] for s in stats], q1=[s[1] for s in stats], median=[s[2] for s in stats],
        q3=[s[3] for s in stats], upperfence=[s[4] for s in stats], name="background", showlegend=False,
        hoverinfo="skip", marker_color=color, line_color=color,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=[s[2] for s in stats], mode="markers", marker=dict(size=9, color="crimson"), name="median",
        customdata=fov_names, hovertemplate="%{customdata}<br>median %{y:.2f} nm<extra></extra>", showlegend=False,
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=color)
    fig.update_layout(
        title=dict(text="Corrected background per FOV (p5 · p25 · p50 · p75 · p95, nm). Click a FOV to inspect it.", font=dict(color=color, size=13)),
        xaxis=dict(title="FOV", tickmode="array", tickvals=x, ticktext=[str(i) for i in x]),
        yaxis=dict(title="OPD (nm)"), margin=dict(l=40, r=10, t=40, b=40), height=320,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    _theme_axes(fig, color)
    return fig


def _numbers(diag, method):
    """Readouts for the picked FOV: a surface too low (fit kept) or too high (negative interiors).

    The box plot already shows the background median and spread per FOV; the rim negatives
    are expected edge halos, so neither is repeated here. The caller renders the kept-fraction
    warning below the row, where the message has the full width. Only the polynomial method
    fits anything: the others report a hard-coded 1.0, so showing them a confident 100 % would
    claim a check that never ran.
    """
    if method == "polynomial":
        st.metric("Fit kept", f"{100 * diag['kept_fraction']:.0f} %", delta=f"{diag['rounds']} rounds", delta_color="off",
                  help=f"Background pixels inside the clip after the last round. Below {100 * KEPT_FRACTION_WARNING:.0f} % the surface may sit too low, inflating masses.")
    st.metric("Negative pixels inside cells", f"{diag['neg_interior_pct']:.1f} %",
              help="Below-zero pixels two rings in from every cell edge. More than a few percent means the surface sits above the cells.")


def _line_scan_plot(scan, label, fov_title, color):
    """Raw, surface and corrected along the cell's major axis, on one axis so the zero line reads true.

    ``scan["mass_pg"]`` is the corrected profile times a constant, so a fourth trace added no
    shape; on its own autoranged secondary axis its zero sat at a different height than the
    dotted line, inviting negatives to be read off the wrong curve. The magnitude check it
    existed for is the dry-mass metric beside the cell picker.
    """
    fig = go.Figure()
    x = scan["position_px"]
    fig.add_trace(go.Scatter(x=x, y=scan["raw_nm"], name="raw", line=dict(color="gray", dash="dot")))
    fig.add_trace(go.Scatter(x=x, y=scan["surface_nm"], name="surface", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=x, y=scan["corrected_nm"], name="corrected", line=dict(color=color)))
    inside = np.flatnonzero(scan["inside"])
    if inside.size:
        for edge in (x[inside[0]], x[inside[-1]]):
            fig.add_vline(x=edge, line_dash="dash", line_color=color, opacity=0.4)
    fig.add_hline(y=0, line_dash="dot", line_color=color)
    fig.update_layout(title=dict(text=f"{fov_title}: line scan along the major axis of cell {label}", font=dict(color=color, size=13)),
                      xaxis=dict(title="Position (px)"), yaxis=dict(title="OPD (nm)"),
                      legend=dict(orientation="h", y=-0.25, font=dict(color=color)),
                      margin=dict(l=40, r=10, t=40, b=40), height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    _theme_axes(fig, color)
    return fig


def choose_background_widget(metadata_df, metadata_dict, fov_name_col, channel_name):
    """Render the calibration block for one QPI channel; ``("", settings)`` or ``(error_msg, None)``."""
    constants = metadata_dict[channel_name]["qpi"]
    defaults = metadata_dict[channel_name].get("background_defaults", DEFAULT_BACKGROUND)
    settings = _settings_controls(channel_name, defaults)
    # Draft choices survive hidden widgets and recalibration without confirming
    # the recipe or changing the last saved metadata record.
    metadata_dict[channel_name]["background_defaults"] = dict(settings)
    args = (constants["opd_unit"], settings["method"], settings["degree"], settings["expand_pct"])
    fov_names = metadata_df[fov_name_col].astype(str).tolist()
    image_paths = metadata_df[f"{channel_name}_QPI (2D)"].tolist()
    mask_paths = metadata_df[f"{channel_name}_Mask"].tolist()
    stats, diags = [], []
    for image_path, mask_path in zip(image_paths, mask_paths):
        error_msg, diag = qpi_fov_diagnostics(image_path, mask_path, *args)
        if error_msg:
            return error_msg, None
        stats.append(diag["bg_percentiles_after_nm"])
        diags.append(diag)
    color = get_context_theme_color()
    # Row 1: the per-FOV box plot with the picked FOV's readouts beside it, so the pick and
    # what it reads sit together. Row 2: the cell picker with that cell's dry mass beside it,
    # then the line scan full width, since it runs over the cell's whole major axis.
    plot_col, stats_col = st.columns([3, 1])
    with plot_col:
        event = st.plotly_chart(_fov_box_plot(fov_names, stats, color), on_select="rerun", selection_mode="points", key=f"qpi_fov_plot_{channel_name}")
    picked = 0
    if event and event.selection and event.selection.points:
        # Clamp: the chart key does not include the CSV, so a pick kept from a table with
        # more FOVs would otherwise index past the end of this one.
        picked = min(int(event.selection.points[0]["point_index"]), len(diags) - 1)
    with stats_col:
        _numbers(diags[picked], settings["method"])
    if diags[picked]["kept_fraction"] < KEPT_FRACTION_WARNING:
        st.warning(f"The fit kept only {100 * diags[picked]['kept_fraction']:.0f} % of the background pixels, below {100 * KEPT_FRACTION_WARNING:.0f} %. The surface may sit too low, inflating masses; try a lower degree or a larger exclusion.")

    error_msg, result = corrected_qpi_image(image_paths[picked], mask_paths[picked], *args)
    if error_msg:
        return error_msg, None
    corrected, surface, _, mask, _ = result
    labels, counts = np.unique(mask[mask > 0], return_counts=True)
    if labels.size == 0:
        st.info("No cells in this FOV's mask.")
    else:
        pick_col, mass_col, _ = st.columns([1, 1, 2])
        with pick_col:
            label = st.selectbox("Cell for the line scan", [int(v) for v in labels], index=int(np.argmax(counts)),
                                 key=f"qpi_cell_{channel_name}_{picked}", help="Largest cell first. The scan runs along the cell's major axis; dashed lines mark where it enters and leaves the mask.")
        with mass_col:
            # The same signed sum over the label that cell_features emits as dry_mass_pg: the one
            # place the calibration step shows what pixel size, OPD unit and alpha add up to.
            # Three significant digits so a unit off by a millionfold reads as 3.33e-05, not 0.0.
            mass_pg = float(corrected[mask == label].sum()) * mass_factor(constants["pixel_size_um"], constants["alpha_um3_per_pg"])
            st.metric("Cell dry mass", f"{mass_pg:.3g} pg",
                      help="Signed sum of the corrected OPD over this cell, the dry_mass_pg feature extraction will emit. Tens of picograms is typical for a mammalian cell; orders of magnitude away points at the pixel size, OPD unit or alpha rather than the background recipe.")
        scan = line_scan(corrected + surface, corrected, surface, mask, label, constants["pixel_size_um"], constants["alpha_um3_per_pg"])
        fov_title = f"FOV {picked + 1} ({fov_names[picked]})"
        st.plotly_chart(_line_scan_plot(scan, label, fov_title, color), key=f"qpi_scan_{channel_name}_{picked}_{label}")
    return "", settings
