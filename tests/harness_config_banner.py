"""Exercise the stale-config checker against a pre-seeded mtime baseline.

Call the inner checker because the outer notifier resets the baseline each full run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.config_watch import _render_stale_banner_if_needed

_render_stale_banner_if_needed()
st.write("banner check ran")
