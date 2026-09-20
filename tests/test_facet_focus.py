"""Promoting a facet panel swaps contents between slots, never the geometry."""

import pytest

from src.vis.dimension_facets import (
    facet_focus_from_click,
    focus_facet_figure,
    focus_slot_keys,
)


def test_without_a_focus_the_overview_owns_the_main_slot():
    assert focus_slot_keys(["Day 2", "Day 10"]) == [None, "Day 2", "Day 10"]


def test_a_promoted_key_trades_places_with_the_overview():
    assert focus_slot_keys(["Day 2", "Day 10", "N/A"], "Day 10") == [
        "Day 10", "Day 2", None, "N/A"]


def test_a_key_absent_from_the_grid_leaves_the_arrangement_alone():
    # A filter can remove the promoted level between renders.
    assert focus_slot_keys(["Day 2"], "Day 99") == [None, "Day 2"]


def test_the_only_panel_can_still_be_promoted():
    assert focus_slot_keys(["Day 2"], "Day 2") == ["Day 2", None]


def test_tuple_keys_from_a_two_column_matrix_are_compared_whole():
    keys = [("Day 2", "ctrl"), ("Day 2", "drug")]
    assert focus_slot_keys(keys, ("Day 2", "drug")) == [
        ("Day 2", "drug"), ("Day 2", "ctrl"), None]


def grid_figure(separate_by="day", keys=("Day 2", "Day 10"), slot_labels=(0, 1)):
    """A minimal grid carrying the contract: two panels, strips, one stray trace."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1], y=[1], xaxis="x", yaxis="y",
                             meta=dict(facet_role="points", facet_slot=0)))
    for slot, axis in enumerate(["x4", "x5"], start=1):
        fig.add_trace(go.Scatter(x=[1], y=[1], xaxis=axis, yaxis=axis.replace("x", "y"),
                                 meta=dict(facet_role="context", facet_slot=slot)))
        fig.add_trace(go.Scatter(x=[1], y=[1], xaxis=axis, yaxis=axis.replace("x", "y"),
                                 meta=dict(facet_role="points", facet_slot=slot)))
    for slot in (0, 1, 2):
        fig.add_trace(go.Scatter(x=[1], y=[1], xaxis="x", yaxis="y2", visible=slot == 0,
                                 meta=dict(facet_role="marginal", facet_slot=slot)))
    fig.add_trace(go.Scatter(x=[None], y=[None], name="swatch"))  # legend only, untagged
    labels = [key if isinstance(key, str) else " ".join(key) for key in keys]
    for label in [*labels, ""]:
        fig.add_annotation(x=1., y=.5, xref="paper", yref="paper", text=label,
                           showarrow=False)
    axes = {"xaxis": [0., .48], "yaxis": [0., 1.],
            "xaxis4": [.52, 1.], "yaxis4": [.5, 1.],
            "xaxis5": [.52, 1.], "yaxis5": [0., .5]}
    fig.update_layout(
        title=dict(text="A grid"),
        meta={"distribution_facet_layout": {
                  "plot_height": .5, "axes": axes,
                  "annotations": [{"x": item.x, "y": item.y,
                                   "xref": item.xref, "yref": item.yref}
                                  for item in fig.layout.annotations]},
              "facet_focus": {
                  "layout_key": "distribution_facet_layout",
                  "separate_by": separate_by,
                  "keys": list(keys),
                  "labels": labels,
                  "axes": [["x", "y"], ["x4", "y4"], ["x5", "y5"]],
                  "slot_labels": list(slot_labels),
                  "title": "A grid",
                  "stamp_annotation": 2,
                  "applied": None}})
    return fig


def slot_axes(fig, role, slot):
    trace = next(t for t in fig.data if isinstance(t.meta, dict)
                 and t.meta["facet_role"] == role and t.meta["facet_slot"] == slot)
    return trace.xaxis, trace.yaxis


def test_promoting_a_panel_moves_its_traces_onto_the_main_axes():
    fig = focus_facet_figure(grid_figure(), "Day 10")
    assert slot_axes(fig, "points", 2) == ("x", "y")
    assert slot_axes(fig, "context", 2) == ("x", "y")


def test_promoting_a_panel_sends_the_overview_to_the_vacated_slot():
    fig = focus_facet_figure(grid_figure(), "Day 10")
    assert slot_axes(fig, "points", 0) == ("x5", "y5")
    assert slot_axes(fig, "points", 1) == ("x4", "y4")


def test_strips_stay_on_the_main_block_and_show_the_promoted_level():
    fig = focus_facet_figure(grid_figure(), "Day 10")
    visible = {t.meta["facet_slot"]: t.visible for t in fig.data
               if isinstance(t.meta, dict) and t.meta["facet_role"] == "marginal"}
    assert visible == {0: False, 1: False, 2: True}
    assert slot_axes(fig, "marginal", 2) == ("x", "y2")


def test_the_vacated_panel_is_named_main_plot_and_the_title_names_the_promotion():
    fig = focus_facet_figure(grid_figure(), "Day 10")
    assert [item.text for item in fig.layout.annotations][:2] == ["Day 2", "Main plot"]
    assert fig.layout.title.text == "A grid (day: Day 10)"


def test_a_matrix_cell_keeps_its_row_labels_and_takes_a_stamp():
    fig = focus_facet_figure(
        grid_figure(separate_by=["day", "treatment"],
                    keys=[("Day 2", "ctrl"), ("Day 2", "drug")],
                    slot_labels=[None, None]),
        ("Day 2", "drug"))
    stamp = fig.layout.annotations[2]
    assert stamp.text == "Main plot"
    # Centred at the top of the cell the overview now occupies (xaxis5/yaxis5).
    assert (stamp.x, stamp.y) == (pytest.approx(.76), pytest.approx(.5))
    canonical = fig.layout.meta["distribution_facet_layout"]["annotations"][2]
    assert (canonical["x"], canonical["y"]) == (pytest.approx(.76), pytest.approx(.5))
    assert fig.layout.title.text == "A grid (day: Day 2 · treatment: drug)"


def test_restoring_puts_every_trace_and_label_back():
    fig = focus_facet_figure(grid_figure(), None)
    assert slot_axes(fig, "points", 0) == ("x", "y")
    assert slot_axes(fig, "points", 2) == ("x5", "y5")
    assert [item.text for item in fig.layout.annotations] == ["Day 2", "Day 10", ""]
    assert fig.layout.title.text == "A grid"


def test_a_level_a_filter_removed_leaves_the_grid_unpromoted():
    fig = focus_facet_figure(grid_figure(), "Day 99")
    assert slot_axes(fig, "points", 0) == ("x", "y")
    assert fig.layout.meta["facet_focus"]["applied"] is None


def test_a_grid_without_a_title_is_titled_by_the_promotion_alone():
    """Dimension Reduction has no title of its own to augment."""
    fig = grid_figure()
    fig.update_layout(title=dict(text=None))
    fig.layout.meta["facet_focus"]["title"] = ""
    assert focus_facet_figure(fig, "Day 10").layout.title.text == "day: Day 10"


def test_untagged_traces_are_never_moved():
    fig = focus_facet_figure(grid_figure(), "Day 10")
    swatch = next(t for t in fig.data if t.name == "swatch")
    assert (swatch.xaxis, swatch.yaxis) == (None, None)


def test_promotion_leaves_the_base_figure_alone():
    import plotly.graph_objects as go

    base = grid_figure()
    focus_facet_figure(go.Figure(base), "Day 10")
    assert slot_axes(base, "points", 0) == ("x", "y")
    assert base.layout.meta["facet_focus"]["applied"] is None
    assert base.layout.annotations[1].text == "Day 10"


def test_clicking_a_panels_points_promotes_that_panel():
    selection = {"points": [{"curve_number": 4, "point_index": 0}]}
    assert facet_focus_from_click(grid_figure(), selection, None) == "Day 10"


def test_clicking_the_overview_restores_the_default_arrangement():
    selection = {"points": [{"curve_number": 0, "point_index": 0}]}
    assert facet_focus_from_click(grid_figure(), selection, "Day 10") is None


def test_clicking_an_untagged_trace_changes_nothing():
    selection = {"points": [{"curve_number": 8, "point_index": 0}]}
    assert facet_focus_from_click(grid_figure(), selection, "Day 2") == "Day 2"


def test_an_empty_selection_changes_nothing():
    assert facet_focus_from_click(grid_figure(), {"points": []}, "Day 2") == "Day 2"
    assert facet_focus_from_click(grid_figure(), None, "Day 2") == "Day 2"
