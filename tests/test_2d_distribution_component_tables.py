"""Direct tests for ``render_distribution_component_tables``: split and merge.

No existing test in ``tests/test_2d_distribution_separation.py`` calls this helper
directly -- it is only reached through ``feature_2d_distribution_plot`` there, and
only when GMM produces multi-component tables. These tests call it directly with a
fake editor so the per-level table split and the name-merge behaviour are each
pinned on their own, along with the absence of any heading above them.
"""
import pytest

from src.vis import bivar


def _events(monkeypatch):
    """Record every ``st.markdown`` call the function makes."""
    events = []
    monkeypatch.setattr(bivar.st, "markdown", lambda value, **kwargs: events.append(value))
    return events


def _table(category, group):
    return {"category": category, "group": group, "features": ["x", "y"],
            "rows": [{"source_label": f"{category}::{group}"}]}


def test_no_heading_is_rendered_above_the_levels(monkeypatch):
    """``gmm_group_title`` names the level in each table, so a heading would repeat it."""
    events = _events(monkeypatch)
    tables = [_table("Day 2", "ctrl"), _table("Day 2", "drug"), _table("Day 10", "ctrl")]

    bivar.render_distribution_component_tables(tables, "day", lambda level_tables: {})

    assert events == []


def test_each_editor_call_receives_exactly_that_levels_tables(monkeypatch):
    _events(monkeypatch)
    tables = [_table("Day 2", "ctrl"), _table("Day 2", "drug"), _table("Day 10", "ctrl")]
    calls = []

    def editor(level_tables):
        calls.append(level_tables)
        return {}

    bivar.render_distribution_component_tables(tables, "day", editor)

    assert len(calls) == 2
    assert calls[0] == [tables[0], tables[1]]
    assert calls[1] == [tables[2]]


def test_returned_dict_merges_every_levels_names_not_just_the_last(monkeypatch):
    _events(monkeypatch)
    tables = [_table("Day 2", "ctrl"), _table("Day 10", "ctrl")]

    def editor(level_tables):
        category = level_tables[0]["category"]
        return {f"name_{category}": f"renamed {category}"}

    result = bivar.render_distribution_component_tables(tables, "day", editor)

    # Both levels' names must survive together: a `names = component_editor(...)`
    # implementation (instead of `names.update(...)`) would drop the first level's
    # entry as soon as the second level's call returned.
    assert result == {"name_Day 2": "renamed Day 2", "name_Day 10": "renamed Day 10"}


def test_empty_tables_emits_no_heading_and_returns_empty_dict(monkeypatch):
    events = _events(monkeypatch)

    def editor(_tables):
        pytest.fail("editor must not be called when there are no tables")

    result = bivar.render_distribution_component_tables([], "day", editor)

    assert result == {}
    assert events == []


def test_none_editor_renders_nothing_and_returns_empty_dict(monkeypatch):
    """The no-assignments export path calls this with ``component_editor=None``.

    ``export_labels_widget`` renders read-only fit details that way; the
    reviewed sibling (``render_histogram_summaries``) guards the same case with
    ``if component_editor is not None``, and this function must match it. With
    no heading left to draw, that guard leaves nothing behind but the notice
    ``export_labels_widget`` writes itself.
    """
    events = _events(monkeypatch)
    tables = [_table("Day 2", "ctrl"), _table("Day 10", "ctrl")]

    result = bivar.render_distribution_component_tables(tables, "day", None)

    assert result == {}
    assert events == []
