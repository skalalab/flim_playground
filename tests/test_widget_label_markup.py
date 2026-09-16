"""Widget labels and options must not open with Markdown block syntax.

Streamlit 1.55 (streamlit/streamlit#13887) backslash-escapes a leading heading,
list, or quote marker in label Markdown, so ``### **Univariate**`` renders as the
literal text ``### Univariate``. Streamlit 1.54 unwrapped the heading instead and
showed only the bold text. No version renders a heading inside a label, so the
marker is dead weight on the pinned version and visible garbage on newer ones.
The lock file pins 1.54; conda or pip installs from ``pyproject.toml`` get newer.
"""
import re
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROOT = Path(__file__).resolve().parents[1]
_PAGES = ["main.py", "pages/data_analysis.py", "pages/data_extraction.py"]

# Mirrors the two substitutions Streamlit's StreamlitMarkdown applies when isLabel is set.
_BLOCK_MARKER = re.compile(r"^\s*((?:[+\-*]|#+)(?=\s|$)|>)", re.MULTILINE)
_ORDERED_MARKER = re.compile(r"^\s*\d+[.)](?=\s|$)", re.MULTILINE)

# AppTest accessors whose elements expose a Markdown-rendered label, and, for the
# option widgets, Markdown-rendered options.
_LABELLED = [
    "radio", "checkbox", "selectbox", "multiselect", "toggle", "button",
    "text_input", "number_input", "slider", "select_slider", "text_area", "expander",
]


def _escaped_labels(at):
    hits = []
    for kind in _LABELLED:
        for element in getattr(at, kind):
            texts = [getattr(element, "label", None), *(getattr(element, "options", None) or [])]
            for text in texts:
                if isinstance(text, str) and (_BLOCK_MARKER.search(text) or _ORDERED_MARKER.search(text)):
                    hits.append((kind, text))
    return hits


@pytest.mark.parametrize("page", _PAGES)
def test_no_widget_label_opens_with_a_block_marker(page):
    from streamlit.testing.v1 import AppTest

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        at = AppTest.from_file(str(_ROOT / page)).run(timeout=60)
    assert not at.exception, [str(e.message) for e in at.exception]
    assert _escaped_labels(at) == []
