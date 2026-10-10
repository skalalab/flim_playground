"""Put the repository root on sys.path so each test module can import src independently."""
import sys
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def isolated_config_paths(tmp_path, monkeypatch):
    """Isolate both config files and guard the repository's saved settings."""
    from src import config
    from src.widgets import analysis_config_widgets

    real_paths = [Path(ROOT) / name for name in ("config.toml", "analysis_config.toml")]
    before = [path.read_bytes() if path.exists() else None for path in real_paths]
    paths = (tmp_path / "config.toml", tmp_path / "analysis_config.toml")
    monkeypatch.setattr(config, "_CONFIG_PATH", paths[0])
    monkeypatch.setattr(analysis_config_widgets, "_ANALYSIS_CONFIG_PATH", paths[1])
    yield paths
    assert [path.read_bytes() if path.exists() else None for path in real_paths] == before


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


@pytest.fixture(autouse=True)
def _forget_pages_directory_probe():
    """Let each AppTest decide for itself whether its script has a ``pages/`` directory.

    Streamlit 1.54 records that answer once per process on the PagesManager class. A
    test that runs ``main.py`` (which has ``pages/``) would otherwise turn every later
    ``AppTest.from_function`` script into a legacy multipage app whose generated file
    name fails the page-title check. Reset the probe around each test.
    """
    try:
        from streamlit.runtime.pages_manager import PagesManager
    except ImportError:
        yield
        return
    if hasattr(PagesManager, "uses_pages_directory"):
        PagesManager.uses_pages_directory = None
    yield
    if hasattr(PagesManager, "uses_pages_directory"):
        PagesManager.uses_pages_directory = None


_REAL_MAIN_MODULE = sys.modules["__main__"]


@pytest.fixture(autouse=True)
def _restore_main_module():
    """Put the real ``__main__`` back after each test.

    Streamlit's script runner installs the AppTest script as ``sys.modules["__main__"]``
    and never restores it. A later spawn-based ``multiprocessing.Pool`` (the lifetime
    fitter's parallel path) re-executes ``__main__`` in every worker; when that script is
    an AppTest body the workers die at startup and ``Pool.imap`` waits forever.
    """
    yield
    sys.modules["__main__"] = _REAL_MAIN_MODULE


@pytest.fixture
def install_page_table(monkeypatch, isolated_config_paths):
    """Opt-in page fixture: raw reader plus ordinary exact profile and real gate.

    Existing plot-control tests can state their intended groups without bypassing
    review or forcing confirmation. The callback-capable uploader holds a table.
    """
    import io
    import streamlit as st
    import toml
    from src import dataset_io

    def install(frame, groups, row_id="cell_id", delimiter=",", categories=None):
        numeric = [column for columns in groups.values() for column in columns]
        if categories is None:
            categories = [c for c in frame.columns if c not in numeric and c != row_id]
        profile = {"unique_row_id_col": row_id, "categorical_cols": categories,
                   "all_numerical_features": numeric, "feature_groups": groups}
        isolated_config_paths[1].write_text(toml.dumps({
            "current_profile": "fixture", "profiles": {"fixture": profile}}))
        payload = frame.to_csv(index=False, sep=delimiter).encode()
        class Upload(io.BytesIO):
            name = "fixture.csv"
        seen = [False]
        def uploader(*args, **kwargs):
            if not seen[0]:
                seen[0] = True
                callback = kwargs.get("on_change")
                if callback:
                    callback()
            return Upload(payload)
        monkeypatch.setattr(st, "file_uploader", uploader)
        monkeypatch.setattr(dataset_io, "read_table", lambda _upload: (
            frame.copy(), {}, delimiter, "", ""))
    return install
