"""Checks for exported figures whose app chart is Plotly under Streamlit's chart theme.

The theme draws light gridlines and no tick marks, and anchors a bold title at the left
(Streamlit wraps every Plotly title in <b>). Axis lines appear only where a figure asks.
The saved SVG is cropped to whatever was drawn, but plt.show() draws only the canvas.
"""


def assert_theme_axes(ax, grid_axis="y", lines=()):
    assert {side for side, spine in ax.spines.items() if spine.get_visible()} == set(lines)
    for axis, gridlines in (("x", ax.get_xgridlines()), ("y", ax.get_ygridlines())):
        assert all(line.get_visible() == (axis == grid_axis) for line in gridlines), axis
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        assert not tick.tick1line.get_visible() or tick.tick1line.get_markersize() == 0


def assert_theme_title(title, size=16):
    assert title.get_text()
    assert (title.get_fontsize(), title.get_fontweight(), title.get_horizontalalignment()) == (
        size, "bold", "left")


def assert_canvas_holds_everything(fig):
    fig.canvas.draw()
    content = fig.get_tightbbox(fig.canvas.get_renderer())
    width, height = fig.get_size_inches()
    assert 0 <= content.x0 and content.x1 <= width, (content.x0, content.x1, width)
    assert 0 <= content.y0 and content.y1 <= height, (content.y0, content.y1, height)
