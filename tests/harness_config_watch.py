"""AppTest harness that exercises notify_on_config_change() in isolation.

Kept as a tiny standalone page so a test can drive the config-change notifier
without booting the whole Data Extraction page and its heavy imports.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.config_watch import notify_on_config_change

notify_on_config_change()
st.write("notify ran")
