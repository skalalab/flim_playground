"""Exercise config-change notifications without loading the Data Extraction page."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.config_watch import notify_on_config_change

notify_on_config_change()
st.write("notify ran")
