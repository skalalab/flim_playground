"""Minimal Streamlit app exercising twod_single_feature_select_widget under AppTest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.widgets.selection_widgets import twod_single_feature_select_widget

# Rebuilt every rerun on purpose: the widget mutates the dict in place, removing
# the x-axis pick so it cannot also be chosen for y.
feature_groups_dict = {
    "Lifetime fit_nadh": ["Lifetime fit_nadh: t1", "Lifetime fit_nadh: t2"],
    "Derived Features": ["Derived: ratio", "Derived: sum"],
}

selected_x, selected_y = twod_single_feature_select_widget(
    feature_groups_dict, data_extraction=True, n_per_row=2
)
st.text(f"x={selected_x}|y={selected_y}")
