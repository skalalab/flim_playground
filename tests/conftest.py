"""Put the repository root on sys.path so each test module can import src independently."""
import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _single_selection_button_group_indices(monkeypatch):
    """Bridge Streamlit 1.54 AppTest's scalar single-select serialization.

    A real segmented control stores a scalar, while AppTest's ButtonGroup.indices
    assumes a list (iterating a string character by character on the next rerun).
    Normalize only that scalar case; native widgets and multi-select tests are
    unchanged. Remove when Streamlit's test driver supports single selection.
    """
    from streamlit.proto.ButtonGroup_pb2 import ButtonGroup as ButtonGroupProto
    from streamlit.testing.v1.element_tree import ButtonGroup

    original = ButtonGroup.indices.fget

    def indices(group):
        value = group.value
        if (group.proto.click_mode == ButtonGroupProto.SINGLE_SELECT
                and not isinstance(value, list)):
            return ([] if value is None else
                    [group.options.index(group.format_func(value))])
        return original(group)

    monkeypatch.setattr(ButtonGroup, "indices", property(indices))


@pytest.fixture(autouse=True)
def _forget_bare_mode_containers():
    """Clear form state left on Streamlit generators by bare-mode widget rendering.

    Without a ScriptRunContext, forms can mark the shared main DeltaGenerator.
    Restore the default stack and clear form marks on generators from both stacks
    so later AppTests start outside a form. Import private internals during teardown
    and tolerate renames without failing collection.
    """
    yield
    try:
        from streamlit.delta_generator_singletons import (
            context_dg_stack,
            get_default_dg_stack_value,
        )
    except ImportError:
        return
    left_standing = context_dg_stack.get()
    default = get_default_dg_stack_value()
    context_dg_stack.set(default)
    for dg in (*left_standing, *default):
        dg._form_data = None
