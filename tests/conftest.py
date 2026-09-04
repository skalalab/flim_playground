"""Put the repo root on sys.path so `import src...` resolves in every test module.

pytest only prepends the test file's own directory, so a module that imports
`src.*` without inserting the root itself is importable only when it happens to
be collected alongside one that does -- it fails when run on its own. Doing it
here once covers every test module regardless of how it is invoked; the existing
per-file `sys.path.insert` calls remain harmless.
"""
import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def _forget_bare_mode_containers():
    """Undo a container a bare-mode render left standing on Streamlit's block stack.

    Widget code called outside `streamlit run` -- which is how the gate suites drive it --
    opens each `with st.form(...)` / `st.popover(...)` against Streamlit's block stack, and
    with no ScriptRunContext to build a child block against, the form id is written onto
    the *main* DeltaGenerator itself. Popping the stack therefore restores an object that
    is still marked as being a form, and it is the singleton every later run starts from.
    AppTest runs the page in this same thread, its first `st.button` reads that mark, and
    the page dies inside `st.form()` -- in a test that never mentioned a form, in a file
    after the one that rendered it. Suites that pass alone and fail together, so the
    reset lives here rather than in whichever module happens to trip it.

    **The mark is cleared on the generators themselves, never on a copy.** The poison is
    an attribute of that process-wide singleton, so a copy would be cleaned and thrown
    away with the singleton still marked. Both the current stack and the default one are
    swept, because either may hold a generator a `with` block left marked.

    **Imported inside the body, and tolerant of an ImportError.** These are Streamlit
    private internals with no API guarantee; at module scope a rename would fail
    *collection* for the whole suite, with a traceback pointing nowhere near the cause.
    In here it degrades to a no-op -- the suites that need it fail on the symptom above,
    which at least names a form.
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
