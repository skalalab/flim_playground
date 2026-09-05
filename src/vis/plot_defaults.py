"""Default plot styling values (Streamlit session state and exported matplotlib scripts)."""

DEFAULT_POINT_SIZE = 5
DEFAULT_AXIS_LABEL_FONT_SIZE = 24
DEFAULT_LEGEND_FONT_SIZE = 18
DEFAULT_COLORMAP = "tab10"

# Switch large Plotly point sets to WebGL to avoid one SVG DOM node per point.
# Smaller sets use SVG without allocating a WebGL context. Script exports use Matplotlib.
WEBGL_POINT_THRESHOLD = 5000
