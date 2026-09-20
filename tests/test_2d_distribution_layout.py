"""The 2D separation grid keeps the overview and every small map square."""

from itertools import pairwise

import pytest

from src.vis.dimension_facets import category_facet_groups, dimension_facet_layout


@pytest.mark.parametrize("levels", [1, 2, 3, 5, 8])
def test_square_aspect_keeps_every_panel_and_the_overview_square(levels):
    groups = category_facet_groups([(f"level{index}", []) for index in range(levels)])
    layout = dimension_facet_layout(groups, [0., 10.], [100., 103.], aspect=1.)
    overview, panels = layout["overview"], layout["panels"]
    assert len(panels) == levels
    # X and Y carry independent units, so the frame must not follow the data ratio.
    for panel in [overview, *panels]:
        width = panel["x_domain"][1] - panel["x_domain"][0]
        height = (panel["y_domain"][1] - panel["y_domain"][0]) * layout["plot_height"]
        assert height / width == pytest.approx(1.)


@pytest.mark.parametrize("levels", [1, 2, 3, 5, 8])
def test_the_panel_column_shares_the_overviews_edges_and_the_full_width(levels):
    groups = category_facet_groups([(f"level{index}", []) for index in range(levels)])
    layout = dimension_facet_layout(groups, [0., 10.], [0., 10.], aspect=1.)
    overview, panels = layout["overview"], layout["panels"]
    assert min(panel["y_domain"][0] for panel in panels) == pytest.approx(overview["y_domain"][0])
    assert max(panel["y_domain"][1] for panel in panels) == pytest.approx(overview["y_domain"][1])
    assert panels[0]["x_domain"][0] == pytest.approx(overview["x_domain"][1] + .04)
    assert panels[-1]["x_domain"][1] == 1.
    # Plotly detects overlapping subplots with strict comparisons.
    for upper, lower in pairwise(panels):
        assert upper["y_domain"][0] == lower["y_domain"][1]
        assert upper["x_domain"] == lower["x_domain"]


def test_membership_order_becomes_panel_order_in_one_column():
    groups = category_facet_groups([("Day 2", []), ("Day 10", []), ("N/A", [])])
    assert groups["nrows"] == 3
    assert groups["ncols"] == 1
    assert [panel["row"] for panel in groups["panels"]] == [0, 1, 2]
    assert all(panel["col"] == 0 for panel in groups["panels"])
    assert [panel["values"] for panel in groups["panels"]] == [("Day 2",), ("Day 10",), ("N/A",)]


def test_no_panels_keeps_a_full_frame_overview():
    layout = dimension_facet_layout(category_facet_groups([]), [0., 1.], [0., 3.], aspect=1.)
    assert layout["panels"] == []
    assert layout["overview"] == {"x_domain": [0., 1.], "y_domain": [0., 1.]}
    assert layout["plot_height"] == pytest.approx(1.)


def test_omitting_aspect_keeps_the_dimension_reduction_range_ratio():
    groups = category_facet_groups([("a", []), ("b", [])])
    layout = dimension_facet_layout(groups, [0., 10.], [0., 5.])
    assert layout["plot_height"] == pytest.approx(layout["overview"]["x_domain"][1] * .5)
