"""Default plot styling values (Streamlit session state and exported matplotlib scripts)."""

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
