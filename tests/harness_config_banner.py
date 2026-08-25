"""AppTest harness that exercises the stale-config banner check in isolation.

Calls the inner checker directly (not the run_every fragment) so a test can drive
the stale/fresh branches by pre-seeding the recorded baseline mtime in session
state — the outer notify function resets that baseline on every full run, so the
stale branch can only be reached by calling the checker on its own.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.config_watch import _render_stale_banner_if_needed

_render_stale_banner_if_needed()
st.write("banner check ran")
