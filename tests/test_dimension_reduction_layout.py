"""The overview and full facet grid share height and undistorted coordinates."""
import pandas as pd
import pytest

from src.vis.dimension_facets import dimension_facet_groups, dimension_facet_layout, dimension_ranges


@pytest.mark.parametrize("rows,columns", [(1, 1), (1, 3), (1, 4), (2, 3), (3, 1), (5, 2), (8, 4)])
@pytest.mark.parametrize("aspect", [0.4, 1., 2.])
def test_full_grid_aligns_with_overview_and_every_panel_has_same_data_aspect(rows, columns, aspect):
    frame = pd.DataFrame([{"row": f"row{r}", "column": f"col{c}"}
                          for r in range(rows) for c in range(columns)])
    separation = ["row"] if columns == 1 else ["row", "column"]
    groups = dimension_facet_groups(frame, separation)
    layout = dimension_facet_layout(groups, [0., 10.], [0., 10. * aspect])
    overview, panels = layout["overview"], layout["panels"]
    assert min(panel["y_domain"][0] for panel in panels) == pytest.approx(overview["y_domain"][0])
    assert max(panel["y_domain"][1] for panel in panels) == pytest.approx(overview["y_domain"][1])
    # Domains are fractions of the inner canvas, whose physical height is
    # plot_height times its width. All panels must show equal x/y unit scales.
    for panel in [overview, *panels]:
        width = panel["x_domain"][1] - panel["x_domain"][0]
        height = (panel["y_domain"][1] - panel["y_domain"][0]) * layout["plot_height"]
        assert height / width == pytest.approx(aspect)
    # Plotly tests overlap with strict comparisons, so shared edges must be
    # identical even at floating-point precision to avoid background overdraw.
    for first, second in zip(panels, panels[1:]):
        if first["row"] == second["row"]:
            assert first["x_domain"][1] == second["x_domain"][0]
    for column in range(columns):
        vertical = [panel for panel in panels if panel["col"] == column]
        for upper, lower in zip(vertical, vertical[1:]):
            assert upper["y_domain"][0] == lower["y_domain"][1]


def test_single_overview_uses_the_embedding_aspect():
    layout = dimension_facet_layout({"panels": []}, [-1., 1.], [-2., 2.])
    assert layout["plot_height"] == pytest.approx(2.)


@pytest.mark.parametrize("x,y", [
    ([-50., 50.], [100., 102.]),
    ([100., 102.], [-50., 50.]),
    ([0., 0.], [-5., 5.]),
    ([7., 7.], [0., 0.]),
])
def test_coordinate_bounds_keep_equal_units_in_a_compact_frame(x, y):
    x_range, y_range = dimension_ranges(x, y)
    assert (y_range[1] - y_range[0]) / (x_range[1] - x_range[0]) == pytest.approx(0.72)
    for values, bounds in ((x, x_range), (y, y_range)):
        assert bounds[0] < min(values) <= max(values) < bounds[1]
        assert sum(bounds) / 2 == pytest.approx((min(values) + max(values)) / 2)


@pytest.mark.parametrize("separation", [[], ["row"], ["row", "column"]])
def test_frame_dimensions_do_not_depend_on_embedding_extent(separation):
    frame = pd.DataFrame({"row": ["a", "a", "b", "b"], "column": ["c", "d", "c", "d"]})
    groups = dimension_facet_groups(frame, separation)
    heights = [dimension_facet_layout(groups, *dimension_ranges(x, y))["plot_height"]
               for x, y in (([-3., 3.], [-1., 1.]), ([5., 6.], [-8., 8.]),
                            ([-50., 80.], [-10., 100.]))]
    assert heights == pytest.approx([heights[0]] * len(heights))
