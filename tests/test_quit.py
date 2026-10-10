"""The power button ends only the launch that rendered it."""
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)

_TOKEN_VAR = "FLIM_PLAYGROUND_QUIT_TOKEN"

_MENU_SCRIPT = (
    "import sys\n"
    f"sys.path.insert(0, r'{ROOT}')\n"
    "import streamlit as st\n"
    "from src.navigation import render_top_menu\n"
    "render_top_menu()\n"
    "st.write('PAGE-BODY-RENDERED')\n"
)

_TRANSLOCATED = (
    "/private/var/folders/xk/lqlzjrs12yb7vscb0r20jdy00000gn/T/AppTranslocation/"
    "448EDF53-C8C9-4A62-9CA1-6412113CDF60/d/Flim-Playground 2.app/"
    "Contents/MacOS/Flim-Playground"
)


def _render(monkeypatch, quit_param=None):
    """Render the bar, recording exits instead of exiting.

    The exit timer captures os._exit when it starts, so the recorder must be in
    place before the script runs; otherwise a quit would end pytest itself.
    """
    from streamlit.testing.v1 import AppTest

    exits = []
    monkeypatch.setattr(os, "_exit", exits.append)
    at = AppTest.from_string(_MENU_SCRIPT)
    if quit_param is not None:
        at.query_params["quit"] = quit_param
    at.run(timeout=60)
    assert not at.exception, [e.value for e in at.exception]
    return at, exits


def _bar(at):
    bars = [m.value for m in at.markdown if "background-color:#f0f0f0" in m.value]
    return bars[0] if bars else None


def _body_rendered(at):
    return "PAGE-BODY-RENDERED" in " ".join(m.value for m in at.markdown)


def _exits_after(exits, seconds):
    """Wait up to ``seconds`` for the exit timer, returning the exits recorded."""
    deadline = time.monotonic() + seconds
    while not exits and time.monotonic() < deadline:
        time.sleep(0.05)
    return exits


def test_launcher_turns_on_the_power_button(monkeypatch):
    import launcher

    # setup_environment looks for ./main.py and writes os.environ; a copy keeps
    # its variables out of later tests.
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(os, "environ", os.environ.copy())
    launcher.setup_environment()

    at, exits = _render(monkeypatch)

    token = os.environ[_TOKEN_VAR]
    # Same tab: without a target, Streamlit opens markdown links in a new one.
    assert f"<a href='/?quit={token}' target='_self'" in _bar(at)
    assert _body_rendered(at)
    assert not _exits_after(exits, 1.5)

    launcher.setup_environment()  # a later launch
    assert os.environ[_TOKEN_VAR] != token, "a stale tab must not hold a later launch's token"


@pytest.mark.parametrize(("token", "quit_param"), [
    ("t0k3n", "stale"),  # a goodbye tab left open from an earlier launch
    (None, ""),  # online or `streamlit run`: no launcher, so no token
    (None, None),
])
def test_page_renders_without_this_launchs_token(monkeypatch, token, quit_param):
    if token is None:
        monkeypatch.delenv(_TOKEN_VAR, raising=False)
    else:
        monkeypatch.setenv(_TOKEN_VAR, token)

    at, exits = _render(monkeypatch, quit_param)

    bar = _bar(at)
    assert bar is not None and _body_rendered(at)
    assert ("?quit=" in bar) == (token is not None), "the button appears only under the launcher"
    assert not _exits_after(exits, 1.5)


def test_quit_with_this_launchs_token_says_goodbye_and_exits(monkeypatch):
    monkeypatch.setenv(_TOKEN_VAR, "t0k3n")

    at, exits = _render(monkeypatch, "t0k3n")

    assert len(at.info) == 1, "a goodbye replaces the page"
    assert _bar(at) is None and not _body_rendered(at)
    assert _exits_after(exits, 3) == [0]


def test_translocated_launch_quits_by_itself(monkeypatch):
    monkeypatch.setenv(_TOKEN_VAR, "t0k3n")
    monkeypatch.setattr(sys, "executable", _TRANSLOCATED)

    at, exits = _render(monkeypatch)

    assert any("xattr -dr com.apple.quarantine" in e.value for e in at.error)
    assert _exits_after(exits, 3) == [0]
