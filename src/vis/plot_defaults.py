"""Default plot styling values (Streamlit session state and exported matplotlib scripts),
and the theme-resolved axis colour. Imports nothing from the project.
"""
import streamlit as st

DEFAULT_POINT_SIZE = 5
DEFAULT_AXIS_LABEL_FONT_SIZE = 24
DEFAULT_LEGEND_FONT_SIZE = 18
DEFAULT_COLORMAP = "tab10"

# Keep replicate means prominent over the original observations in SuperPlots.
SUPERPLOT_OBSERVATION_SIZE_SCALE = 0.75
SUPERPLOT_OBSERVATION_MIN_SIZE = 3
SUPERPLOT_OBSERVATION_OPACITY_SCALE = 0.3
SUPERPLOT_REPLICATE_SIZE_SCALE = 1.5
SUPERPLOT_REPLICATE_LINE_WIDTH = 1.5
SUPERPLOT_REPLICATE_JITTER_WIDTH = 0.1

# Switch large Plotly point sets to WebGL to avoid one SVG DOM node per point.
# Smaller sets use SVG without allocating a WebGL context. Script exports use Matplotlib.
WEBGL_POINT_THRESHOLD = 5000

# Faceted maps and grey context points: the panel geometry the app and exported scripts share.
PANEL_CONTEXT_COLOR = "#b8b8b8"
PANEL_CONTEXT_ALPHA = 0.25
PANEL_MODEL_WIDTH = 1      # a model drawn in its own category map
OVERVIEW_MODEL_WIDTH = 2   # a model drawn on the whole-dataset axes


def panel_point_size(point_size):
    """Marker size of faceted and context points: two below the main size, at least 1."""
    return max(1, point_size - 2)


def get_context_theme_color():
    """Plot color for the current theme: 'black' in light mode, 'white' otherwise.

    ``st.context.theme.type`` is a plain server-side read, so it costs no rerun. It is
    None until the browser reports in, which falls through to the dark-mode color.
    """
    return "black" if st.context.theme.type == "light" else "white"
