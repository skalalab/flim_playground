"""Default plot styling values (Streamlit session state and exported matplotlib scripts)."""

DEFAULT_POINT_SIZE = 5
DEFAULT_AXIS_LABEL_FONT_SIZE = 24
DEFAULT_LEGEND_FONT_SIZE = 24
DEFAULT_COLORMAP = "tab10"

# Above this many drawn points, point traces switch from go.Scatter to go.Scattergl.
# Plotly's SVG renderer emits one <path> node per point, and a figure holding tens of
# thousands of them makes the browser re-rasterise that tree on every scroll; WebGL keeps
# the points in typed-array buffers and adds no DOM nodes at all. Below the threshold SVG
# is kept, so ordinary datasets render exactly as before and need no WebGL context.
# Plotly-only: the exported matplotlib scripts never see this.
WEBGL_POINT_THRESHOLD = 5000
